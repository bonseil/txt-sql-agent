"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 2

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


class VerificationResult(BaseModel):
    """Pydantic model for LLM verification response."""
    valid: bool = Field(description="Whether the SQL query and results are valid and answer the question")
    issue: str = Field(default="", description="Description of the issue if valid is False, empty otherwise")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    row_samples: str = ""

#"""Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
# we use the singleton pattern to avoid creating a new client for each call
_LLM = ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

def llm() -> ChatOpenAI:
    return _LLM

def _referenced_tables(sql: str) -> set[str]:
    """Best-effort extraction of table names referenced via FROM/JOIN."""
    return {m.group(1).strip('"') for m in re.finditer(r'\b(?:FROM|JOIN)\s+"?(\w+)"?', sql, re.IGNORECASE)}


def _known_tables(schema: str) -> set[str]:
    """Table names declared in the rendered schema text."""
    return {m.group(1) for m in re.finditer(r'CREATE TABLE\s+"([^"]+)"', schema)}

# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()

def sample_rows_node(state: AgentState) -> dict:
    tables = _referenced_tables(state.sql) & _known_tables(state.schema)
    chunks = []
    for t in sorted(tables):
        # Sample only 2 rows, all columns, but truncate aggressively
        result = execute_sql(state.db_id, f'SELECT * FROM "{t}" LIMIT 2')
        if result.ok:
            chunks.append(f"-- Sample from {t}:\n{result.render(max_rows=2)}")
    text = "\n\n".join(chunks) if chunks else "No sample rows available."
    return {
        "row_samples": text,
        "history": state.history + [{"node": "sample_rows", "tables": sorted(tables)}],
    }

def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    response = llm().invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration,       # we don't want to increment the iteration here, only on revise node to allow for the full MAX_ITERATIONS check
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: build messages from the VERIFY_*
    prompts, call llm(), parse the reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.
    """

    # 1. Deterministic check: did the SQL even execute?
    if state.execution and state.execution.error:
        return {
            "verify_ok": False,
            "verify_issue": state.execution.error,
            "history": state.history + [{"node": "verify", "valid": False, "issue": state.execution.error}],
        }

    # 2. Deterministic check: does it only reference real tables?
    unknown = _referenced_tables(state.sql) - _known_tables(state.schema)
    if unknown:
        issue = f"Query references unknown table(s): {', '.join(sorted(unknown))}"
        return {
            "verify_ok": False,
            "verify_issue": issue,
            "history": state.history + [{"node": "verify", "valid": False, "issue": issue}],
        }

    # 3. LLM judgment: does the result plausibly answer the question?
    structured_llm = llm().with_structured_output(VerificationResult)
    verification = structured_llm.invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            schema=state.schema,
            question=state.question,
            sql_query=state.sql,
            execution_result=state.execution.render() if state.execution else "No execution result",
            iteration=state.iteration,
        )),
    ])
    return {
        "verify_ok": verification.valid,
        "verify_issue": verification.issue,
        "history": state.history + [{"node": "verify", "valid": verification.valid, "issue": verification.issue}],
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    execution_result = state.execution.render()
    response = llm().invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            invalid_sql_query=state.sql,
            verification_issue=state.verify_issue,
            execution_result=execution_result,
            row_samples=state.row_samples,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }



def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok:
        return "end"

    if state.iteration >= MAX_ITERATIONS:
        return "end"

    return "revise"

# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("sample_rows", sample_rows_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "sample_rows", "end": END},   # route to sample_rows, not revise
    )
    g.add_edge("sample_rows", "revise")
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
