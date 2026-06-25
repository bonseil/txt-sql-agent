import json
from pathlib import Path
import httpx

EVAL_SET = [json.loads(l) for l in Path("evals/eval_set.jsonl").read_text().splitlines() if l.strip()]

E_BUCKET_PREFIXES = [
    "List the top five schools, by descending order, from the hig",
    "What is the average fastest lap time in seconds for Lewis Ha",
    "From race no. 50 to 100, how many finishers have been disqua",
    "What is the complete address of the school with the lowest e",
    "How many users received commentator badges in 2014?",
    "Among the patients with a normal Ig G level, how many of the",
    "For all patients with normal uric acid (UA), what is the ave",
    "Among all the lap records set on various circuits, what is t",
    "In superheroes with missing weight data, calculate the diffe",
    "Mention the display name and location of the user who owned",
    "List all the mythic rarity print cards banned in gladiator f",
]

targets = [q for q in EVAL_SET if any(q["question"].startswith(p) for p in E_BUCKET_PREFIXES)]

for q in targets:
    resp = httpx.post(
        "http://localhost:8001/answer",
        json={"question": q["question"], "db": q["db_id"]},
        timeout=120.0,
    )
    agent_result = resp.json()
    print(f"Q: {q['question']}")
    print(f"DB: {q['db_id']}")
    print(f"GOLD SQL: {q['gold_sql']}")
    print(f"AGENT SQL: {agent_result.get('sql')}")
    print(f"AGENT ROWS: {agent_result.get('rows')}")
    for h in agent_result.get("history", []):
        if h["node"] == "verify":
            print(f"  verify -> valid={h['valid']}, issue={h.get('issue','')[:200]}")
    print()