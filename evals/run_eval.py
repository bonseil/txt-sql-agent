"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx
# Notes on BIRD eval corrections:
# - Several gold SQL queries in the original BIRD-based eval set were logically misaligned
#   with their natural-language questions, including issues with date logic, inclusive
#   ranges, fragile string parsing of times, and reliance on hard-coded ID values.
# - Concretely, we corrected:
#   * The crimes-in-1995 query to filter regions with A15 > 4000 and require at least one
#     account opened in or after 1997 via an EXISTS subquery (no over-counting regions).
#   * The disqualified-finishers query to use an inclusive race range (BETWEEN 50 AND 100)
#     instead of excluding races 50 and 100.
#   * The uric-acid query to compute per-patient latest lab results with a correlated
#     MAX(Date) subquery and to apply the normal-range thresholds correctly to both sexes.
#   * The fastest-lap query to order lap times by a straightforward conversion from
#     m:ss.xxx to seconds, avoiding brittle nested INSTR/SUBSTR logic.
# - All other question/SQL pairs were preserved as-is, and the corrected eval set keeps
#   the same JSONL structure (one object per line with question, db_id, gold_sql) so it
#   can be used as a drop-in replacement for the original evaluation file.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set_corrected.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    question_text = question["question"]

    # Run gold SQL to get expected rows
    gold_ok, gold_rows, gold_error = run_sql(db_id, gold_sql)
    if not gold_ok:
        return {
            "question": question_text,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "gold_ok": False,
            "gold_error": gold_error,
            "per_iteration": [],
            "final_ok": False,
            "final_error": "gold SQL failed",
        }

    gold_canonical = canonicalize(gold_rows)

    # Call agent over HTTP
    try:
        response = httpx.post(
            agent_url,
            json={"question": question_text, "db": db_id},
            timeout=30.0,
        )
        response.raise_for_status()
        agent_result = response.json()
    except Exception as e:  # noqa: BLE001
        return {
            "question": question_text,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "gold_ok": True,
            "per_iteration": [],
            "final_ok": False,
            "final_error": f"agent request failed: {type(e).__name__}: {e}",
        }

    # Extract agent's final SQL and execution
    agent_sql = agent_result.get("sql", "")
    agent_ok = agent_result.get("ok", False)
    agent_error = agent_result.get("error")
    agent_rows = agent_result.get("rows")
    agent_iterations = agent_result.get("iterations", 0)
    history = agent_result.get("history", [])

    if not agent_ok:
        return {
            "question": question_text,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "agent_sql": agent_sql,
            "gold_ok": True,
            "per_iteration": [],
            "final_ok": False,
            "final_error": agent_error or "agent execution failed",
            "revisions": agent_iterations,
        }

    # Compare canonicalized results
    agent_canonical = canonicalize(agent_rows)
    correct = matches(gold_canonical, agent_canonical)

    # Build per-iteration correctness by re-executing each SQL attempt from history.
    # Each "generate_sql" or "revise" node represents one SQL attempt (iteration).
    per_iteration: list[bool] = []
    for entry in history:
        if entry.get("node") in ("generate_sql", "revise"):
            sql = entry.get("sql", "")
            if sql:
                ok, rows, _ = run_sql(db_id, sql)
                per_iteration.append(ok and matches(gold_canonical, canonicalize(rows)))
            else:
                per_iteration.append(False)

    # Safety fallback: if history had no SQL-generating nodes, derive from final answer
    if not per_iteration:
        per_iteration = [correct]

    return {
        "question": question_text,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "agent_sql": agent_sql,
        "gold_ok": True,
        "per_iteration": per_iteration,
        "final_ok": correct,
        "revisions": agent_iterations,
    }

def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    if not results:
        return {
            "total_questions": 0,
            "final_pass_rate": 0.0,
            "per_iteration_pass_rates": {},
        }
    total_questions = len(results)
    final_correct = sum(1 for r in results if r.get("final_ok", False))
    final_pass_rate = final_correct / total_questions

    # Find the maximum number of SQL attempts across all questions
    max_iterations = max(
        (len(r.get("per_iteration", [])) for r in results),
        default=0,
    )

    per_iteration_pass_rates: dict[int, float] = {}
    for i in range(max_iterations):
        correct_at_i = 0
        for r in results:
            per_iter = r.get("per_iteration", [])
            if i < len(per_iter):
                if per_iter[i]:
                    correct_at_i += 1
            elif per_iter:
                # Carry-forward: agent stopped before iteration i, use its last result
                if per_iter[-1]:
                    correct_at_i += 1
            # else: agent failed completely → counts as wrong at all iterations

        per_iteration_pass_rates[i] = correct_at_i / total_questions

    # Tally revision distribution
    revision_counts: dict[int, int] = {}
    for r in results:
        n = r.get("revisions")
        if n is None:
            per_iter = r.get("per_iteration", [])
            n = len(per_iter) - 1 if per_iter else 0
        revision_counts[n] = revision_counts.get(n, 0) + 1

    return {
        "total_questions": total_questions,
        "final_pass_rate": final_pass_rate,
        "per_iteration_pass_rates": per_iteration_pass_rates,
        "revision_counts": revision_counts,
    }

# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
