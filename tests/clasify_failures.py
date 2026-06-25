"""Classify each eval question by what the revise loop did to it.

For each question, compute:
- iter_0_correct: did the first SQL match gold?
- iter_0_verified_ok: did the verifier pass it at iter 0?
- iter_final_correct: did the final SQL match gold?
- final_iter: where did the loop terminate?

Then categorize each question into one of:
  A. CORRECT_AT_0_AND_KEPT: iter0=True, verifier passed, terminated at 0. Working as intended.
  B. CORRECT_AT_0_BUT_LOOPED: iter0=True, verifier rejected, terminated later (either still correct or broken).
  C. WRONG_AT_0_FIXED: iter0=False, ended correct. Revise worked.
  D. WRONG_AT_0_NOT_FIXED: iter0=False, ended wrong, but loop ran. Revise tried and failed.
  E. WRONG_AT_0_NOT_TRIED: iter0=False, verifier accepted, terminated at 0. Verifier missed the bug.
  F. ERRORED_AT_0: agent crashed before producing any SQL.

This is the actual failure-mode distribution the report needs.
"""
import json
import sys
import sqlite3
from pathlib import Path

EVAL_RESULT_FILE = Path("results/eval_baseline.json")
EVAL_SET_FILE = Path("evals/eval_set.jsonl")
DB_DIR = Path("data/bird")


def run_sql(db_id, sql, timeout=5.0):
    try:
        with sqlite3.connect(f"file:{DB_DIR / f'{db_id}.sqlite'}?mode=ro", uri=True, timeout=timeout) as conn:
            return True, conn.execute(sql).fetchall(), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows):
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold, pred):
    if gold is None or pred is None:
        return False
    return canonicalize(gold) == canonicalize(pred)


def classify(r, gold_sql):
    if "error" in r:
        return "F. ERRORED_AT_0", None, None
    gold_ok, gold_rows, _ = run_sql(r["db_id"], gold_sql)
    final = r["final_iteration"]
    correct_per_iter = r["iteration_correct"]
    iter0_correct = correct_per_iter.get("0", False)
    final_correct = correct_per_iter.get(str(final), False)

    # was iter 0 verified ok? (i.e. did the loop stop there because verifier passed?)
    iter0_verified_ok = (final == 0)

    if iter0_correct and iter0_verified_ok:
        return "A. CORRECT_AT_0_AND_KEPT", iter0_correct, final_correct
    if iter0_correct and not iter0_verified_ok:
        if final_correct:
            return "B1. CORRECT_AT_0_LOOPED_STILL_CORRECT", iter0_correct, final_correct
        else:
            return "B2. CORRECT_AT_0_LOOPED_BROKEN", iter0_correct, final_correct
    if not iter0_correct and final_correct:
        return "C. WRONG_AT_0_FIXED", iter0_correct, final_correct
    if not iter0_correct and not iter0_verified_ok and not final_correct:
        return "D. WRONG_AT_0_NOT_FIXED", iter0_correct, final_correct
    if not iter0_correct and iter0_verified_ok and not final_correct:
        return "E. WRONG_AT_0_NOT_TRIED", iter0_correct, final_correct
    return "?. UNCLASSIFIED", iter0_correct, final_correct


def main():
    results = json.loads(EVAL_RESULT_FILE.read_text())["results"]
    gold_lookup = {}
    for line in EVAL_SET_FILE.read_text().splitlines():
        row = json.loads(line)
        gold_lookup[row["question"]] = row["gold_sql"]

    buckets = {}
    for r in results:
        gold_sql = gold_lookup.get(r["question"], "")
        category, iter0, final = classify(r, gold_sql)
        buckets.setdefault(category, []).append({
            "q": r["question"][:60],
            "db": r["db_id"],
            "iter0": iter0,
            "final": final,
            "final_iter": r.get("final_iteration"),
        })

    print("\n=== FAILURE MODE DISTRIBUTION ===\n")
    for cat in sorted(buckets):
        rows = buckets[cat]
        print(f"{cat}: {len(rows)} questions")
        for r in rows:
            print(f"  [iter0={r['iter0']}, final={r['final']}, term@{r['final_iter']}] {r['db']}: {r['q']}")
        print()


if __name__ == "__main__":
    main()