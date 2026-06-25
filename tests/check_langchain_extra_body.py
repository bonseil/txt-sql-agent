"""Call ChatOpenAI exactly like graph.py does, with extra_body for
enable_thinking, and inspect the raw response to see whether thinking
was actually suppressed."""
import os
from langchain_openai import ChatOpenAI

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

llm = ChatOpenAI(
    model=VLLM_MODEL,
    base_url=VLLM_BASE_URL,
    api_key="not-needed",
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

response = llm.invoke([
    ("system", "You are an expert SQL generator. Return only SQL."),
    ("user", "Schema:\nCREATE TABLE Player_Attributes (id INTEGER PRIMARY KEY, player_api_id INTEGER, overall_rating INTEGER);\nCREATE TABLE Player (player_api_id INTEGER PRIMARY KEY, player_name TEXT);\n\nQuestion: Calculate the average overall rating of Pietro Marino."),
])

print("content:", repr(response.content)[:300])
print("\nresponse_metadata:", response.response_metadata)
print("\nadditional_kwargs:", response.additional_kwargs)
