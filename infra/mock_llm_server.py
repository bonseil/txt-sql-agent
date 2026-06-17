"""Mock LLM server for local development (Phases 3-5).

This server mimics the OpenAI-compatible vLLM API without needing:
- GPU/CUDA infrastructure
- Large model downloads
- Complex dependency resolution

Phases covered:
- Phase 3 (Agent): Full SQL generation, verification, revision flow
- Phase 4 (Tracing): Compatible with Langfuse callbacks
- Phase 5 (Evals): Returns realistic mock SQL responses

Switch to real vLLM on H100 for Phase 6 (SLO testing).
"""
import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# ============================================================================
# DEVELOPMENT STATUS & STAGE INFO
# ============================================================================
DEVELOPMENT_STAGE = """
Mock LLM Server - Development Progress
=====================================

STAGE: Local Development (Phases 3-5)
Current Date: 2026-06-17

OBSTACLES ENCOUNTERED:
1. vLLM 0.10.2 tokenizer incompatibility
   - Qwen3 tokenizer lacks 'all_special_tokens_extended' attribute
   - xformers build failures on WSL2
   
2. vLLM version conflicts
   - v0.4.2 requires xformers which fails to build without torch
   - Modern versions have different architecture incompatibilities
   
3. Windows Subsystem for Linux (WSL) issues
   - CRLF line endings in .env files causing parsing errors
   - CUDA/GPU compatibility challenges

SOLUTION: Mock Server
   - Eliminates GPU/CUDA requirements
   - Provides OpenAI-compatible API
   - Allows rapid agent development without infrastructure delays
   - Real vLLM on H100 will be used for Phase 6 (final SLO testing)

MOCK SERVER CAPABILITIES:
✓ POST /v1/chat/completions - SQL generation & verification
✓ GET /status - Development stage information (this endpoint)
✓ GET /metrics - Prometheus-compatible metrics (stub)
✓ Langfuse integration support
✓ Deterministic responses for reproducible testing
"""

# Counter for metrics
REQUEST_COUNT = 0
GENERATION_TOKENS = 0
PROMPT_TOKENS = 0

# ============================================================================
# MOCK RESPONSE TEMPLATES
# ============================================================================

SQL_EXAMPLES = [
    "SELECT * FROM employees WHERE department = 'Sales' LIMIT 100",
    "SELECT COUNT(*) as count, department FROM employees GROUP BY department",
    "SELECT * FROM orders WHERE order_date >= '2024-01-01' AND status != 'cancelled'",
    "SELECT * FROM products WHERE price > 100 AND category = 'Electronics'",
    "SELECT * FROM customers WHERE country = 'USA' ORDER BY customer_id DESC",
]

VERIFICATION_RESPONSES = [
    {"ok": True, "issue": ""},
    {"ok": False, "issue": "Query returned no results, but question implies data exists"},
    {"ok": False, "issue": "Columns returned don't match the question asked"},
    {"ok": True, "issue": "Result looks reasonable"},
]

REVISED_SQL = [
    "SELECT * FROM employees WHERE department = 'Sales' AND hire_date >= '2024-01-01'",
    "SELECT DISTINCT department FROM employees WHERE salary > 50000",
    "SELECT * FROM orders WHERE total_amount > 1000 ORDER BY order_date DESC",
]


class MockLLMHandler(BaseHTTPRequestHandler):
    """Mock OpenAI-compatible LLM API handler."""

    def do_GET(self):
        """Handle GET requests for status and metrics."""
        global REQUEST_COUNT

        if self.path == "/status":
            # Return development stage information
            response = {
                "status": "running",
                "version": "mock-0.1",
                "stage": "local-development-phases-3-5",
                "date": datetime.now().isoformat(),
                "development_info": DEVELOPMENT_STAGE,
                "endpoints": {
                    "POST /v1/chat/completions": "OpenAI-compatible chat endpoint",
                    "GET /status": "This endpoint - development stage info",
                    "GET /metrics": "Prometheus-compatible metrics (stub)",
                    "GET /health": "Health check",
                },
                "note": (
                    "This mock server is for Phases 3-5 (agent development). "
                    "Phase 6 (SLO testing) requires real vLLM on H100."
                ),
            }
            self._send_json(200, response)

        elif self.path == "/metrics":
            # Return Prometheus-compatible metrics (stub)
            metrics_text = f"""# HELP mock_requests_total Total requests handled
# TYPE mock_requests_total counter
mock_requests_total {REQUEST_COUNT}

# HELP mock_tokens_generated Total tokens generated
# TYPE mock_tokens_generated counter
mock_tokens_generated {GENERATION_TOKENS}

# HELP mock_tokens_prompted Total prompt tokens processed
# TYPE mock_tokens_prompted counter
mock_tokens_prompted {PROMPT_TOKENS}

# HELP mock_server_running Server uptime indicator
# TYPE mock_server_running gauge
mock_server_running 1
"""
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(metrics_text.encode())))
                self.end_headers()
                self.wfile.write(metrics_text.encode())
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected, ignore
        elif self.path == "/health":
            self._send_json(200, {"status": "healthy"})

        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>Welcome to the Mock LLM Server</h1>"
                b"<p>This is a mock server for local development.</p>"
                b"<p>Use /v1/chat/completions for LLM API calls.</p>"
                b"<p>Use /status for development stage information.</p>"
                b"<p>Use /metrics for Prometheus metrics.</p>"
                b"<p>Use /health for health check.</p>"
                b"<p>Use / for this page.</p>"
                b"<p>"+DEVELOPMENT_STAGE.encode()+
                b"</body></html>"
            )

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests for LLM API calls."""
        global REQUEST_COUNT, GENERATION_TOKENS, PROMPT_TOKENS

        if self.path == "/v1/chat/completions":
            REQUEST_COUNT += 1
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            # Extract the user message
            messages = body.get("messages", [])
            user_msg = messages[-1]["content"].lower() if messages else ""

            # Determine response type based on message content
            if "verify" in user_msg or "plausible" in user_msg:
                # Verification request
                response_text = json.dumps(random.choice(VERIFICATION_RESPONSES))
            elif "revise" in user_msg or "rewrite" in user_msg:
                # Revision request
                response_text = random.choice(REVISED_SQL)
            else:
                # Default: SQL generation
                response_text = random.choice(SQL_EXAMPLES)

            prompt_tokens = len(user_msg.split())
            generation_tokens = len(response_text.split())

            PROMPT_TOKENS += prompt_tokens
            GENERATION_TOKENS += generation_tokens

            response = {
                "id": f"chatcmpl-{REQUEST_COUNT}",
                "object": "text_completion",
                "created": int(datetime.now().timestamp()),
                "model": "mock-sql-generator",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": generation_tokens,
                    "total_tokens": prompt_tokens + generation_tokens,
                },
            }

            self._send_json(200, response)
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, status_code, data):
        """Helper to send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    """Start the mock LLM server."""
    server_address = ("0.0.0.0", 8000)
    httpd = HTTPServer(server_address, MockLLMHandler)

    print("=" * 70)
    print("Mock LLM Server Starting")
    print("=" * 70)
    print(DEVELOPMENT_STAGE)
    print("\n" + "=" * 70)
    print("SERVER RUNNING")
    print("=" * 70)
    print("\nEndpoints:")
    print("  POST http://localhost:8000/v1/chat/completions  - LLM API")
    print("  GET  http://localhost:8000/status              - Development info")
    print("  GET  http://localhost:8000/metrics             - Prometheus metrics")
    print("  GET  http://localhost:8000/health              - Health check")
    print("\nPress CTRL+C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    main()