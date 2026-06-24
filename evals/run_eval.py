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
    """Score one question across all iterations using carry-forward."""
    payload = {"question": question["question"], "db": question["db_id"]}
    try:
        resp = httpx.post(agent_url, json=payload, timeout=120.0)
        resp.raise_for_status()
        agent_result = resp.json()
    except Exception as e:
        return {
            "question": question["question"],
            "db_id": question["db_id"],
            "error": f"{type(e).__name__}: {e}",
            "final_correct": False,
            "iteration_correct": {},
        }

    gold_ok, gold_rows, gold_err = run_sql(question["db_id"], question["gold_sql"])

    # Replay history to reconstruct per-iteration SQL attempts
    sql_by_iter: dict[int, str] = {}
    iter_counter = 0
    for h in agent_result.get("history", []):
        if h["node"] == "generate_sql":
            sql_by_iter[0] = h["sql"]
        elif h["node"] == "revise":
            iter_counter += 1
            sql_by_iter[iter_counter] = h["sql"]

    final_iteration = agent_result.get("iterations", 0)

    # Score each iteration's SQL against gold, independently
    iteration_correct: dict[int, bool] = {}
    for it, sql in sql_by_iter.items():
        pred_ok, pred_rows, pred_err = run_sql(question["db_id"], sql)
        iteration_correct[it] = bool(gold_ok and pred_ok and matches(gold_rows, pred_rows))

    return {
        "question": question["question"],
        "db_id": question["db_id"],
        "final_iteration": final_iteration,
        "iteration_correct": iteration_correct,
        "final_correct": iteration_correct.get(final_iteration, False),
        "agent_ok": agent_result.get("ok", False),
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate with per-iteration carry-forward.

    For each question, build a correctness value at every iteration 0..MAX_ITERATIONS
    by carrying forward the result at the question's actual termination point.
    """
    MAX_ITERATIONS = 3  # mirrors agent.graph.MAX_ITERATIONS

    per_iteration_correct: dict[int, int] = {k: 0 for k in range(MAX_ITERATIONS + 1)}
    total = len(results)
    errors = 0

    for r in results:
        if "error" in r:
            errors += 1
            continue
        term = r["final_iteration"]
        # carry forward: the value at term applies to every iteration >= term
        last_known = r["iteration_correct"].get(term, False)
        for k in range(MAX_ITERATIONS + 1):
            if k <= term:
                per_iteration_correct[k] += r["iteration_correct"].get(k, last_known if k == term else False)
            else:
                per_iteration_correct[k] += last_known

    # for it,sql in sql_by_iter.items():
    #     pred_ok, pred_rows, pred_err = run_sql(question["db_id"], sql)
    #     print(f"DEBUG iter={it} sql={sql!r}")
    #     print(f"DEBUG iter={it} pred_ok={pred_ok} pred_rows={pred_rows} pred_err={pred_err}")
    #     print(f"DEBUG gold_ok={gold_ok} gold rows={gold_rows} gold err={gold_err}")
    #     iteration_correct[it]=bool(gold_ok and pred_ok and matches(gold_rows,pred_rows))
    #     print(f"DEBUG iter={it} correct={per_iteration_correct[it]}")

    accuracy_by_iteration = {
        k: round(v / total, 4) if total else 0.0
        for k, v in per_iteration_correct.items()
    }

    return {
        "total_questions": total,
        "errors": errors,
        "final_accuracy": accuracy_by_iteration[MAX_ITERATIONS],
        "accuracy_by_iteration": accuracy_by_iteration,
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
