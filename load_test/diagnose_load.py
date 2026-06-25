"""Diagnostic load driver - same as driver.py but logs error response bodies.

Run:
    uv run python diagnose_load.py --rps 2 --duration 60
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent
PERF_POOL = ROOT / "load_test" / "perf_pool.jsonl"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


async def fire_one(session, url, question, results):
    payload = {"question": question["question"], "db": question["db_id"]}
    t0 = time.monotonic()
    status = "ok"
    err = None
    body_snippet = None
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            text = await resp.text()
            if resp.status != 200:
                status = "http_error"
                err = f"HTTP {resp.status}"
                body_snippet = text[:500]
    except asyncio.TimeoutError:
        status = "timeout"
    except Exception as e:
        status = "client_error"
        err = f"{type(e).__name__}: {e}"
    results.append({
        "latency_seconds": time.monotonic() - t0,
        "status": status,
        "error": err,
        "body_snippet": body_snippet,
        "question": question["question"][:60],
    })


async def drive(args):
    questions = [json.loads(l) for l in PERF_POOL.read_text().splitlines() if l.strip()]
    rnd = random.Random(0)
    results = []
    interval = 1.0 / args.rps

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.monotonic()
        deadline = start + args.duration
        tasks = []
        next_fire = start
        while time.monotonic() < deadline:
            q = rnd.choice(questions)
            tasks.append(asyncio.create_task(fire_one(session, args.agent_url, q, results)))
            next_fire += interval
            sleep_for = next_fire - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        if tasks:
            await asyncio.wait(tasks, timeout=60.0)

    errors = [r for r in results if r["status"] != "ok"]
    print(f"\n=== {len(errors)} non-ok out of {len(results)} ===\n")
    for r in errors:
        print(f"status={r['status']} err={r['error']}")
        if r["body_snippet"]:
            print(f"  body: {r['body_snippet']}")
        print(f"  question: {r['question']}")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rps", type=float, default=2.0)
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = p.parse_args()
    asyncio.run(drive(args))


if __name__ == "__main__":
    main()
