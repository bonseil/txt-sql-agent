import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

llm = ChatOpenAI(
    model=VLLM_MODEL,
    base_url=VLLM_BASE_URL,
    api_key="not-needed",
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)


class VerificationResult(BaseModel):
    valid: bool = Field(description="Whether the SQL query and results are valid and answer the question")
    issue: str = Field(default="", description="Description of the issue if valid is False, empty otherwise")


structured_llm = llm.with_structured_output(VerificationResult)

result = structured_llm.invoke([
    ("system", "You are reviewing a SQL query."),
    ("user", "Schema:\nCREATE TABLE t (id INTEGER);\n\nQuestion: how many rows?\nSQL: SELECT COUNT(*) FROM t;\nExecution Result: OK: 1 rows.\nCOLUMNS: count\nFIRST ROWS:\n5\n\nJudge whether this is valid."),
])

print("RESULT:", result)
print("\n--- Now let's see the raw underlying call (bypass structured output) ---\n")

# Compare: does plain invoke with the SAME llm object still suppress thinking?
plain_result = llm.invoke([
    ("system", "You are an expert SQL generator. Return only SQL."),
    ("user", "Schema:\nCREATE TABLE t (id INTEGER);\n\nQuestion: count rows"),
])
print("plain content:", repr(plain_result.content)[:200])
print("plain metadata:", plain_result.response_metadata)
