"""Check whether enable_thinking=False via extra_body is actually being honored
by vLLM, by calling the model directly (bypassing LangChain) and inspecting
the raw response - including whether 'reasoning_content' is present and how
long it is relative to 'content'.
"""
import json
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"

# A question similar to the ones that failed - long-ish schema, asks for an aggregate
payload = {
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "messages": [
        {"role": "system", "content": "You are an expert SQL generator. Return only SQL."},
        {"role": "user", "content": "Schema:\nCREATE TABLE Player_Attributes (id INTEGER PRIMARY KEY, player_api_id INTEGER, overall_rating INTEGER, potential INTEGER);\nCREATE TABLE Player (player_api_id INTEGER PRIMARY KEY, player_name TEXT);\n\nQuestion: Calculate the average overall rating of Pietro Marino."}
    ],
    "temperature": 0,
    "max_tokens": 500,
    "chat_template_kwargs": {"enable_thinking": False},
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.loads(resp.read())

print(json.dumps(result, indent=2))

msg = result["choices"][0]["message"]
print("\n--- SUMMARY ---")
print("content:", repr(msg.get("content"))[:200])
print("reasoning_content present:", "reasoning_content" in msg)
if "reasoning_content" in msg:
    rc = msg["reasoning_content"]
    print("reasoning_content length:", len(rc) if rc else 0)
print("finish_reason:", result["choices"][0].get("finish_reason"))
print("completion_tokens:", result.get("usage", {}).get("completion_tokens"))
