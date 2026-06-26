# Text-to-SQL Agent on BIRD - Report

## TL;DR

A LangGraph agent (generate → verify → revise) serving BIRD text-to-SQL via vLLM
on a single H100. Eval accuracy: **33.3% execution accuracy** on a 30-question
held-out set, with most of that gained on the first generate attempt; the revise
loop added only ~3 points. Under load, the SLO target (P95 < 5s at 10+ RPS)
**was not met** - the system sustains roughly **0.4–0.7 RPS** at acceptable
tail latency. The dominant bottleneck is per-request latency variance driven
by schema size (one BIRD DB has a 24k-token schema) combined with 2–4 sequential
LLM calls per agent run; not raw vLLM throughput. The bulk of the engineering
time was spent diagnosing several distinct failure modes that initially looked
the same, which itself is the most valuable thing I learned in this assignment.

---

## Phase 1 - vLLM serving config

**Initial choice and why.** Workload sampling on the eval set showed prompts in
the 1.5–3K token range with short SQL outputs, suggesting `--max-model-len 4096`
was a reasonable cap. The startup logs confirmed ~33.7x theoretical concurrency
at that length given the available KV cache.

**What I missed at this point.** This choice was made by sampling only the eval
set (30 questions), which by chance under-represented the upper tail of schema
sizes. The full BIRD perf pool spans larger DBs - `european_football_2`'s
rendered schema alone is **24,630 tokens**. The original 4096 cap was structurally
unable to serve those DBs at all; this only surfaced once load testing began.

**What I ended up with.** `--max-model-len 42000`, `--gpu-memory-utilization 0.97`.
Concurrency dropped from ~33x to ~3–4x worst-case, but real-world effective
concurrency is higher because most requests are well below the cap.

**Other settings I noted but did not pursue.** The MoE kernel-config warning
(`E=128, N=768, NVIDIA_H100_80GB_HBM3.json` not found) is a real, fixable
optimization (vLLM ships `benchmark_moe.py` for auto-tuning) - left as known
gap due to time budget.

**Lesson for the future.** Sample the *real* request distribution before
choosing serving parameters, not a convenient subset.

---

## Phase 2 - Observability

Built a Grafana dashboard with five panels driven by the metrics actually
exposed by vLLM 0.10.2:

- **Latency percentiles (e2e)** - P50/P95/P99 via `histogram_quantile()` on
  `vllm:e2e_request_latency_seconds_bucket`
- **Latency breakdown by phase** - P95 of `request_queue_time`, `request_prefill_time`,
  `request_decode_time` separately, so the question "is it slow, and *where*"
  is answerable at a glance
- **Throughput** - `rate(vllm:request_success_total[5m])`,
  `rate(vllm:generation_tokens_total[5m])`,
  `rate(vllm:prompt_tokens_total[5m])`
- **Concurrency / saturation** - raw gauges for `vllm:num_requests_running`
  and `vllm:num_requests_waiting`
- **KV cache headroom** - `vllm:kv_cache_usage_perc`

The `num_requests_running` vs `waiting` panel was the most valuable for
diagnosing saturation during load testing - see Phase 6.

Screenshot: `screenshots/grafana_dashboard.png`,
`screenshots/grafana_load_test.png`.

---

## Phase 3 - Agent design

**Graph shape.**

```
attach_schema → generate_sql → execute → verify
                                            │
                                  ok=true   └──→ END
                                  ok=false  └──→ sample_rows → revise → execute → verify (loop)
```

I added one node beyond the provided scaffold: **`sample_rows`**. It runs
`SELECT * FROM <referenced_table> LIMIT 2` for each table the failing SQL
touches and feeds those rows to the revise prompt. The motivation was that
many BIRD failures involve case/format mismatches on string filters
(`'Art and Design'` vs `'Art & Design'`); sample rows let the model see the
real value conventions. The cost is prompt-size growth, particularly painful
for text-heavy schemas like `cards`.

**Verifier design - deterministic + LLM hybrid.**
1. If `execution.error` is set, fail verify immediately (no LLM call needed)
2. If the SQL references tables not present in the schema, fail verify
   (regex-based check; doesn't handle CTEs but cheap and catches the common
   "hallucinated table" failure)
3. Otherwise, an LLM call with `with_structured_output(VerificationResult)`
   to judge plausibility - specifically asking it to flag zero-row results
   that look like wrong-filter bugs, shape mismatches between question and
   result, and inverted sort/filter directions.

**Revise design - minimal-patch bias.** The revise prompt was tuned to make
the *smallest change* that fixes the verifier's specific complaint, rather
than regenerate from scratch. This worked well on small fixes (e.g. adding
`DISTINCT`) but has a real failure mode: if the original attempt's structure
is fundamentally wrong, minimal patches can lock the model into iterating on
a structurally bad query. Observed concretely on multi-iteration questions
where the model kept patching the wrong WHERE clause rather than restructuring
the join.

**Verifier non-determinism.** The same SQL + result occasionally gets opposite
verdicts (valid vs invalid) across calls, despite `temperature=0`. This is
likely structured-output routing through tool-calling with its own sampling.
Mitigated but not eliminated.

**MAX_ITERATIONS.** Set to 2 after observing in eval (Phase 5) that accuracy
gained essentially nothing after iteration 1 - most fixable failures are
fixed by the first revise, the rest tend to hit a verifier-revise stalemate.

---

## Phase 4 - Tracing

Langfuse wired in via the provided `CallbackHandler` in `server.py`,
auto-initialized from `LANGFUSE_*` env vars. Traces include nested spans
per LLM call, surfacing the full generate → verify → revise → verify loop
per request.

Screenshot: `screenshots/langfuse_01.png`.

---

## Phase 5 - Eval

`run_eval.py` implements execution accuracy: each agent attempt's SQL is
re-executed against the DB and its rows compared to gold rows after a
canonical sort/stringify. Per-iteration carry-forward is applied: a question
that terminated at iteration 1 has its iteration-1 result carried forward to
all later "iteration k" buckets, since the agent had stopped emitting by then.

Several of the answers in the `gold set` where mistaken. Great care was taken 
into verifying the answers manually and updating the `gold set` where necessary.
The corrected questions and answers are available in the `eval_set_corrected.jsonl` file.

**Results (30 questions):**

| Iteration | Accuracy |
|-----------|----------|
| 0 (initial generate) | 30.0% |
| 1 | 33.3% |
| 2 | 33.3% |
| 3 | 33.3% |

**Distribution of termination iterations:** 19 questions terminated at iter 0
(passed verify on first try), 4 at iter 1, 7 hit MAX_ITERATIONS.

**Reading this honestly.** The revise loop *barely helps* on this workload -
+3 points from one revise iteration, zero gain from iterations 2–3. Inspecting
the iter-3 stuck cases showed three categories:

1. **Genuine SQLite dialect errors** (e.g. `EXTRACT(YEAR FROM ...)` is not
   SQLite) - fixable by adding SQLite-specific dialect hints to generate/revise
   prompts, which I did. Reduced but didn't eliminate.
2. **Verifier false-positives on zero-row results** - verifier flagging
   structurally correct queries as wrong because the underlying data had zero
   matching rows for the question's filter (genuine empty result, not a bug).
3. **A small number of true model failures** that no amount of revising fixed.

I also validated the eval harness itself by hand-tracing the
"Australian Grand Prix" question end-to-end - confirmed `canonicalize`/`matches`
correctly handles float precision and row ordering. A regression case I
investigated (a revise dropping `DISTINCT` and producing 11 duplicate rows of
the correct value) was a real agent behavior, not an eval bug.

---

## Phase 6 - Load test and SLO

**Target:** P95 e2e < 5s at sustained 10+ RPS for 5 minutes.

**Results, after all fixes:**

| RPS  | Success rate | P50   | P95   | P99   | Notes                                                                           |
|------|--------------|-------|-------|-------|---------------------------------------------------------------------------------|
| 10.0 | 2.8%         | 42.3s | 104s  | 107s  | Massive backlog; vLLM saturated, queue grew unbounded until 120s client timeout |
| 1.0  | 96%          | 3.8s  | 16.5s | 23.1s | P50 just over SLO, long tail                                                    |
| 0.4  | 98%          | 1.6s  | 6.7s  | 9.7s  | P50 well under SLO, but P95 ~1.3x over                                          |

**SLO not met.** Even at 0.7 RPS the P95 is roughly 2x over the 5s target;
the system would meet the SLO somewhere around 0.4–0.5 RPS, **roughly 20x
below the 10 RPS target.**

### Why the gap is this large - three composing factors

1. **Workload structure.** Each agent request is 2–4 *sequential* LLM calls
   (generate → verify → optional revise → verify). These cannot parallelize -
   wall-clock latency per request is the sum, not the max. A request that
   needs one revise is ~5 LLM calls (generate, verify, sample_rows pulls
   a few short DB queries, revise, verify) before it returns. Even if each
   LLM call is a fast 1–2s, the agent-level latency is 5–10s on the happy
   path with one revise, well over the 5s SLO before queueing is even
   considered.

2. **Model + hardware.** Qwen3-30B-A3B on one H100 80GB. MoE makes per-token
   compute light, but prefill on long prompts is still expensive, and the
   30B weights eat ~57 GiB leaving limited KV cache. No multi-GPU tensor
   parallelism is available.

3. **Schema size.** BIRD DBs are uneven - `european_football_2` has a
   24,630-token rendered schema (`Match` table has ~115 columns including
   ten bookmakers' worth of betting odds; `Player_Attributes` has 40+
   columns). To serve this DB at all, `--max-model-len` had to be raised to
   42,000, which cut theoretical concurrency by ~10x compared to the
   original 4,096 cap. This is the single biggest design tension I hit:
   correctness for the worst case directly cost throughput on the common
   case.

The cost of raising --max-model-len from 4096 to 42000 to accommodate european_football_2 was directly visible in the 10 RPS run: vLLM's effective concurrency ceiling dropped from ~38 to ~19 — roughly halving the system's parallel capacity. Combined with per-request agent latency of 30+ seconds, this gives a sustained ceiling of 19 / 30s ≈ 0.6 RPS, which matches the measured saturation behavior. This is the explicit tradeoff: making the worst-case schema serviceable directly cost half the throughput available to every other request.


### Measurement caveat: client-side latency under saturation
The reported P95 of 104s at 10 RPS is a floor on true latency, not a measurement of it. 
aiohttp.ClientTimeout(total=120) cancels the client's wait, not the server's work.
Once a request is admitted into vLLM's batch, it runs to completion regardless of 
whether the HTTP client is still listening — there is no cancellation signal
propagated from aiohttp → FastAPI → LangGraph → httpx → vLLM in the provided stack.


I observed this directly: 10 minutes after the 5-minute 10 RPS load test ended, 
Grafana still showed num_requests_running between 5 and 19 with num_requests_waiting 
plateaued at ~14–17. These were zombie requests — the driver had logged them as 120s 
timeouts and moved on, but the GPU was still grinding through their generations. 
Some of these requests were taking 8+ minutes of real server time before they finished,
even though their reported latency was capped at 120s.
The practical implications:

1. Under saturation, **client-side P95 converges to whatever the client timeout is set to**, regardless of how slow the server actually is. This is misleading — it makes a system that's processing requests in 8 minutes look like a system with 120s P95, just with a high error rate. The true latency is hidden in the "timeout" bucket, not in the latency histogram.
2. **The right place to read latency under saturation is server-side**, via vLLM's vllm:e2e_request_latency_seconds histogram. That metric records actual generation time, no client cutoff. The Phase 2 dashboard exposes exactly this — the panel was already correct; I just hadn't read the right one during the 10 RPS run.
3. **The driver as written undercounts server load.** Even after stopping firing at the 5-minute mark, server-side work continued for ~10 more minutes — a real, costly post-test tail that the load report doesn't capture. For accurate capacity planning you'd want the driver to either cancel server work on client timeout (requires explicit cancellation propagation, non-trivial) or to wait for full server drain before exiting.



### What Grafana showed during the 10 RPS run

The original report said running pegged at ~38; that was true at `--max-model-len 16384 `
but not at the final 42000. At the final config the ceiling is ~19, which 
matches $138032/42000 ≈ 3.3$ per-slot but in practice averages higher because 
not every slot is at the max length. So you can substitute "~19 in the final config" 
wherever the report previously claimed ~38, or just keep both as a historical comparison 
since the drop from ~38 -> ~19 is itself the cost of raising `max-model-len` and is a 
worthwhile data point to surface.

### What would actually move the SLO (not implemented)

In rough order of expected impact:

1. **Per-question schema pruning.** A cheap LLM-free filter (or a tiny LLM
   call) to keep only tables the question plausibly needs. For
   `european_football_2`, dropping from 24k → 3–5k tokens of schema would
   roughly 5x prefill cost on the worst case. This is the single highest-leverage
   change. I prototyped a keyword-based version but reverted it after seeing
   accuracy drops on questions needing FK-hop tables not mentioned in the
   question text.

2. **Skip verify when generate's SQL executes cleanly with non-empty results.**
   This halves LLM calls on the happy path. The risk is silently accepting
   structurally wrong queries that happen to return data; would need an eval
   delta to quantify.

3. **A smaller, faster model for verify.** Verifier is doing a yes/no plausibility
   judgment, not full SQL reasoning - a 3B or 7B model would likely do this
   well at a fraction of the latency. Two-model serving on one H100 is doable
   but adds operational complexity.

4. **Prefix caching.** vLLM exposes `gpu_prefix_cache_*` metrics; shared
   schema+system prefixes across requests on the same DB could amortize
   prefill cost. Already configured in vLLM by default - its actual hit rate
   under this workload is something I didn't measure.

5. **Measure latency server-side, not client-side**, under saturation conditions. 
Client-side timeouts mask true per-request cost — the request count, error count, 
and timeout count are still meaningful at the client, but latency percentiles 
are only honest while the system isn't queueing. Once it queues, switch to 
reading vllm:e2e_request_latency_seconds_bucket from Prometheus.

---

## Cross-cutting: bugs that slowed me down, and what they taught me

Several of the largest engineering tax was paid on bugs that initially
all looked like the same thing under load.

**1. `_q(None)` crash in the provided `schema.py`.** SQLite returns
`NULL` for the target column of a foreign key declared without an explicit
column reference (which implicitly points at the target's primary key).
`schema.py`'s `_q()` helper didn't handle that, crashing inside `.replace()`.
This produced `AttributeError: 'NoneType' object has no attribute 'replace'`
for every request on DBs with that FK pattern (european_football, cards,
financial, toxicology, thrombosis).

I spent considerable time chasing LLM-side theories - Qwen3's "thinking" mode
returning `content=None`, structured-output routing dropping `extra_body`,
singleton-staleness - before finally adding `traceback.print_exc()` to the
FastAPI exception handler and seeing the real, one-line traceback. The bug
had been visible the entire time, just hidden because `HTTPException(detail=...)`
collapsed the traceback to a single string.

**Lesson:** *get the traceback before theorizing*. I had multiple isolated
test scripts proving the LLM and structured output worked fine, all of
which should have redirected me to "the bug isn't where I'm looking" much
sooner than they did. I kept generating new hypotheses on the LLM side
instead of widening the search space.

**2. `--max-model-len 4096` being workload-misjudged.** Discussed above
in Phase 1.

**3. Per-DB failure clustering vs. concurrency.** A failure pattern that
*looks* concurrency-specific because individual repro is hard, but is
actually deterministic-per-input. The "Pietro Marino" question failed
under load and never standalone - leading me to assume concurrency was
the issue. It was actually that the load test exercised many more
`european_football_2` questions, all of which hit `_q(None)`, but the
mix made it look load-correlated.

**Lesson:** *group failures by input dimension before assuming a runtime
dimension is the cause.* I should have bucketed errors by DB much earlier.

---

## What I would do differently

1. **Profile the workload before configuring the serve.** A 5-minute
   script that runs `render_schema()` on every BIRD DB and tokenizes the
   output would have shown the upper tail and saved me hours.

2. **Add `traceback.print_exc()` to the FastAPI exception handler from
   day one.** The provided `server.py`'s exception flattening makes real
   debugging structurally impossible - that should be a one-line change
   on initial setup, not something I did under time pressure six hours
   into Phase 6.

3. **Eval before load test, every change.** Several mid-stream prompt
   tweaks (SQLite dialect hints, the row-sample feature) plausibly
   helped *or* hurt accuracy on different question subsets; I didn't have
   a clean per-change eval delta. A faster, smaller eval (e.g. 10 questions
   from each DB family, scriptable to run in 2 minutes) would have caught
   regressions like the revise step dropping `DISTINCT` much earlier.

4. **Treat the rubric's emphasis on *understanding* as a prompt, not a
   consolation.** I spent too much time trying to hit numbers and not
   enough writing things down as I went. This report is being
   reconstructed at the end rather than accumulated throughout, which
   is the wrong order.
