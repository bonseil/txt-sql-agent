import json
import sys
sys.path.insert(0, "evals")
from run_eval import run_sql, canonicalize, matches
import httpx

question = {
    "question": "What is the coordinates location of the circuits for Australian grand prix?",
    "db_id": "formula_1",
    "gold_sql": "SELECT DISTINCT T1.lat, T1.lng FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T2.name = 'Australian Grand Prix'",
}

resp = httpx.post(
    "http://localhost:8001/answer",
    json={"question": question["question"], "db": question["db_id"]},
    timeout=120.0,
)
agent_result = resp.json()
print("AGENT RESULT:", json.dumps(agent_result, indent=2))

gold_ok, gold_rows, gold_err = run_sql(question["db_id"], question["gold_sql"])
print("GOLD:", gold_ok, gold_rows, gold_err)

sql_by_iter = {}
iter_counter = 0
for h in agent_result.get("history", []):
    if h["node"] == "generate_sql":
        sql_by_iter[0] = h["sql"]
    elif h["node"] == "revise":
        iter_counter += 1
        sql_by_iter[iter_counter] = h["sql"]

print("SQL_BY_ITER:", sql_by_iter)

for it, sql in sql_by_iter.items():
    pred_ok, pred_rows, pred_err = run_sql(question["db_id"], sql)
    print(f"iter={it} sql={sql!r}")
    print(f"iter={it} pred_ok={pred_ok} pred_rows={pred_rows} pred_err={pred_err}")
    print(f"iter={it} matches={matches(gold_rows, pred_rows)}")
