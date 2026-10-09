# Test plan for sglang-torchtpu #1161: priority scheduling with PD, prefix caching and KV offload

Ask (wyzhang): validate `--enable-priority-scheduling` combined with PD disaggregation, prefix caching and KV offload to host RAM, on
GPT-OSS-120B (sliding-window attention, 128K) and Qwen3-VL (images, video; 128K and 256K). Goal: no correctness or stability issues.
Constraint from the issue thread (and confirmed by the user on 2026-10-09): the launch runs **without preemption**
(`--disable-priority-preemption`), so the preemption cases in §3.C are **out of scope** and not queued; they stay listed for when preemption is enabled.

## 1. How the feature works in this build (what the tests target)

| Mechanism | Where | Boundary it creates |
|---|---|---|
| `priority` integer on every request (`/generate`, chat completions); higher value = higher priority unless `--schedule-low-priority-values-first` | `io_struct.GenerateReqInput.priority`, tokenizer manager `_set_default_priority` | missing / negative / huge / non-integer values; direction flag |
| Requests without a priority get `--default-priority-value`; with scheduling disabled a priority is ignored unless `--abort-on-priority-when-disabled` (then 400) | tokenizer manager | enabled vs disabled × priority present vs absent × abort flag |
| Waiting queue ordered by priority, then arrival (`calc_priority`) | `schedule_policy` | ties, FIFO within a priority, starvation of low priority |
| **Preemption**: a new request evicts running requests whose priority is lower by **more than** `--priority-scheduling-preemption-threshold` (default 10), lowest first, until its tokens fit; nothing is preempted if the freed tokens would not suffice | `schedule_policy.preempt_to_schedule` | diff = threshold (no), threshold+1 (yes); token budget just short / just enough; already-finished requests in the batch |
| Preempted requests are released (`release_req`), requeued with `is_retracted`, and re-prefill later; on the PD decode they re-enter the prealloc queue (`is_rebootstrap`) and the KV is transferred again | `disaggregation/utils.py`, `decode.py` | prefix-cache hit on re-prefill (Config B), host reload (Config C), SWA window pages, multimodal embeddings, chunked re-prefill |
| Queue full (`--max-queued-requests`): with priority scheduling the **lowest-priority waiting** request is evicted with 503 "aborted by a higher priority request" if the newcomer is strictly higher; otherwise the newcomer gets 503; ties evict the newer | `scheduler._abort_on_queued_limit` | equal priority, newer vs older, direction flag |
| PD decode: prealloc queue sorted by priority before admission; transfer queue unchanged | `decode.py` ~L1392 | high-priority arrival behind a long low-priority transfer |
| Metrics gain a `priority` label when enabled | `metrics_reporter` | dashboards that select `priority=""` show nothing (ours) |

## 2. Configurations

| Id | Pair | Flags added to the Config B / C recipes |
|---|---|---|
| PS-off | B (radix on) | none; control |
| PS-np | B, C | `--enable-priority-scheduling --disable-priority-preemption` (launch configuration) |
| ~~PS-pre~~ | – | preemption on: **out of scope** (not used in production) |
| PS-low | B | `--enable-priority-scheduling --schedule-low-priority-values-first` |
| PS-q | C | PS-np plus `--max-queued-requests 4` |
| PS-dis | B | scheduling disabled, `--abort-on-priority-when-disabled` |

Models: gpt-oss-120b at the 128k shape (fp8 KV; SWA), Qwen3-VL-30B-A3B at 128k and a 256k recipe (text + image + video). HiCache = Config C
(raiden host offload). Both engines carry the flags; the decode is where preemption would hit running decodes.

## 3. Test cases

Pass criteria that apply everywhere: no engine exception, both `/health` 200, `num_used_tokens` back to 0 and all queues empty after each case,
no recompile after warm-up (`fail_on_recompile`), greedy output of an undisturbed request identical to its PS-off golden at c=1.

### A. API and validation boundaries (any model, PS-np unless stated)

| # | Case | Expect |
|---|---|---|
| A1 | `priority` absent, no default | accepted; scheduled as priority 0 (verify via metrics label) |
| A2 | `priority` absent with `--default-priority-value 5` | scheduled as 5 |
| A3 | `priority` 0, 1, −1, 2^31−1, −2^31 | accepted; ordering follows the integer. **Result:** all accepted |
| A4 | `priority` 2^63, float 1.5, string "high", null | clean 400 from the router and the engine, same message; engines unaffected. **Result:** float and string → 400 with a validation message; **2^63 accepted** (no range check; low severity, O-5); engines healthy |
| A5 | PS-dis: priority given | 400 "priority scheduling is disabled"; without the flag: ignored, served. **Result:** rejected with a clear message but HTTP 503 (O-4: should be 4xx); plain request served |
| A6 | priority on chat completions, `/generate` batch (list of prompts), `n=1`, streaming | honoured on every API shape; batch items share the priority |
| A7 | priority on a health-check request (`/health_generate`) | never preempts or evicts |

### B. Ordering without preemption (PS-np, Config B, gpt-oss then Qwen)

| # | Case | Expect |
|---|---|---|
| B1 | Fill all slots with priority 0 decodes (long outputs); submit priority 100 | runs when a slot frees, **no running request preempted** (`num_retracted_reqs` 0, no retraction/requeue log). **Result (PS-np, PS-low):** PASS, TTFT 10–11 s = wait for a free slot, 0 retractions |
| B2 | 8 waiting requests with priorities 0…7 submitted low-first while slots are busy | admitted high-first; TTFT monotone in priority. **Result (PS-np):** PASS, admission order 7→0 once the first slot freed (probe needs staggered holder lengths; see §5) |
| B3 | Same priority, 8 requests | FIFO (arrival order). **Result (PS-np):** PASS, 4 equal-priority probes admitted FIFO between the priority-3 and priority-2 probes |
| B4 | PS-low: priorities 0…7 | admitted low-value-first (inverted B2). **Result:** PASS (0→7, ties FIFO) |
| B5 | Starvation: continuous priority-10 stream at full occupancy plus one priority-0 request | the priority-0 request is served once the stream pauses; never lost; its TTFT recorded. **Result:** served in 1.9 s under a 10-client stream (170 requests): no starvation; high-priority arrivals wait in prefill/transfer, so the decode queue is rarely contended |
| B6 | PD: high-priority request arrives while a low-priority 131k transfer is in the decode prealloc/transfer queue | prealloc queue reorders; high-priority request admitted first once pre-allocable; the long transfer still completes. **Result:** both got their first token at 6.9 s: the running 131k chunked prefill is not interrupted → prefill-side head-of-line blocking (O-7) |
| B7 | Prefix cache (Config B): low-priority request warms a 32k prefix; high-priority request with the same prefix | `cached_tokens` ≈ 32k; correctness vs golden. **Result:** cached_tokens 32,704, TTFT 1.38 → 0.57 s; PASS (the first run's 'text differed' was a probe artefact: `ignore_eos` mismatch; the identity check in §6 shows cold and cached outputs byte-identical) |
| B8 | HiCache (Config C): prefix evicted to host under pressure; high-priority request with that prefix | `#host-hit-token` in the prefill log; identical output. **Result:** `#host-hit-token: 131072`, cached_tokens 131,072, TTFT 0.91 s; PASS (same `ignore_eos` probe artefact as B7) |

### C. Preemption on — OUT OF SCOPE (kept for reference; production disables preemption)

| # | Case | Expect |
|---|---|---|
| C1 | Running priority 0 decode (long output); new request priority 10 (diff = threshold) | **not** preempted (strict >) |
| C2 | Same with priority 11 | running request preempted, requeued, resumed later; its final greedy output identical to the uninterrupted golden |
| C3 | PS-pre-t0: priority 1 vs running 0 | preempted (threshold 0) |
| C4 | Token budget: running request small, newcomer needs more tokens than preemption would free | no preemption, newcomer waits |
| C5 | Cascade: 8 running priority-0 decodes; newcomer priority 50 needs 2 slots' tokens | exactly the lowest/oldest 2 preempted, not all 8 |
| C6 | Preempt a request that is mid **chunked prefill** on the prefill engine (131k prompt, radix on) | abort/requeue bookkeeping clean (this is the F-8/F-11 code area); re-prefill hits the radix cache |
| C7 | Preempted request's re-admission on the **decode**: KV re-transferred from the prefill (re-bootstrap) | handshake completes; `is_rebootstrap` path; output identical; no handshake hold (F-12 watch) |
| C8 | Preempt while the victim is in the transfer window (prefilled, not yet pulled) | no double release; pages freed once |
| C9 | Client aborts the **preempting** request right after preemption | victims resume; nothing leaks. **Result (monolithic):** PASS |
| C10 | Client aborts the **victim** while it is requeued | its state fully released; no "state was deleted" storm. **Result (monolithic):** PASS |
| C11 | Preemption under HiCache (Config C): victim's prefix is in host memory | re-prefill reloads from host; `#host-hit-token`; output identical |
| C12 | gpt-oss SWA: victim at seqlen beyond the window | sliding-window pages freed and re-created correctly (needle check on resumed output) |
| C13 | Preempt a request with `max_new_tokens` 1 and one with `n`… (n>1 is F-6, skip) | single-token request handled |
| C14 | Preemption with speculative decoding (Config D) | draft/verify state reset; acceptance stats sane |
| C15 | Repeated preemption of the same victim (3×) | resumes each time; completion identical. **Result (monolithic):** PASS (three newcomers admitted in ≤0.12 s, all holders completed) |

### D. Queue limit (PS-q: `--max-queued-requests 4`)

**Result 2026-10-09:** not enforceable in PD. `_abort_on_queued_limit` only runs when `disaggregation_mode == NULL`, so D1–D3 returned 200 with no
eviction on the pair (F-16). D1–D3 re-run on a monolithic server (`_phase_1161d.sh`): **all PASS** (503 "queue is full" for lower and equal priority; the newest lowest-priority
waiting request evicted with "aborted by a higher priority request" for a higher newcomer). The behaviour exists; PD simply never reaches it.

| # | Case | Expect |
|---|---|---|
| D1 | Queue full of priority 5; newcomer priority 3 | newcomer 503 "queue is full" |
| D2 | Newcomer priority 9 | lowest-priority waiting request (newest among ties) gets 503 "aborted by a higher priority request"; newcomer queued |
| D3 | Newcomer equal to the lowest (5) | newcomer rejected (strictly better required) |
| D4 | PS-low direction | eviction direction inverted |
| D5 | Evicted request was mid-handshake on the PD pair (prefill bootstrap queue) | both engines release its room; no F-12-style hold |

### E. Multimodal, Qwen3-VL (PS-np and PS-pre; 128k and 256k recipes)

| # | Case | Expect |
|---|---|---|
| E1 | Image request priority 0 running; text request priority 100 arrives (PS-np) | no preemption; admitted when a slot frees; both outputs identical to golden |
| E2 | Video request (many frames, near the compiled patch bucket) with high priority behind a full batch | no recompile (F-10 area), identical output |
| E3 | Mixed batch: 4 image + 4 text, priorities interleaved | ordering as specified; mm cache hits reported. **Result:** 3 concurrent 4-image requests at priorities 0/10/50 all served |
| E4 | 256k context: 200k-token text + image prompt, priority high, arriving while a 256k low-priority decode runs (PS-np) | no preemption; admitted when the slot frees. **Result:** 255k low-priority text + priority-50 image request both served, no preemption, image TTFT 33 s behind the long prefill (O-7), 0 exceptions |
| E5 | Image prefix reuse across priorities (same image, different question) | `cached_tokens` covers the image tokens. **Result:** 768 of 818 tokens cached at both priorities |

### F. Stability and resource return

| # | Case | Expect |
|---|---|---|
| F1 | 30 min open-loop mixed priorities (0/5/10/50), PS-np, both models | no error responses, no stuck requests, used tokens return to 0 at idle, no recompiles. **Result (gpt-oss, 15 min, 12 clients):** 5,035 requests, 0 errors, idle at end; mean TTFT 3.3 s (p0) vs 1.7–1.9 s (p5/10/50) |
| F2 | Same soak with `num_retracted_reqs` watched | must stay 0 (no preemption, no retraction) |
| F3 | Metrics | `priority` label present on queue/running gauges; dashboard panels adjusted (`priority=~".*"`). **Result:** label present on both engines |
| F4 | Disconnect storm (F-8 fix verified) with priorities | no engine exception. **Result:** 20 aborts, 10 concurrent, mixed priorities: 0 errors, used tokens back to 0 |

## 4. Harness

- Recipes: copies of `gpt-oss_B_128k.json`, `gpt-oss_C_128k.json`, `qwen3-vl_B_128k.json`, `qwen3-vl_C_128k.json`, plus a Qwen 256k recipe, with the
  flags of §2 added to both engines (`_ps_np`, `_ps_pre`, `_ps_pre_t0`, `_ps_low`, `_ps_q`, `_ps_dis` suffixes).
- Driver: `scripts/priority_probe.py` (to write): submits requests with explicit priorities and timing offsets, records TTFT/order/HTTP status,
  polls `/metrics` for `num_retracted_reqs` and queue gauges, and compares greedy outputs with the PS-off golden (`golden_diff.py`).
- Phase script `_phase_1161.sh`: one block per configuration, `ensure_up` relaunch, waits on `tpu_free.sh`; results into `results/1161/`.
- Oracles: golden diff at c=1 for undisturbed requests; needle check for resumed requests; identical-output check for preempted requests
  (greedy; batch-composition caveat F-3 noted, so compare after resumption at c=1).

## 5. Probe design notes

- Admission order is observable only when slots free one at a time and every probe is already queued: use holders of staggered length
  (2,500 + 500·i tokens) and submit the probes before the shortest holder ends. With equal-length holders all slots free in the same step and
  the probes are admitted in one batch, which made the first B2/B3 runs meaningless.
- Each 8-token probe cycles through a freed slot in ~70 ms, so twelve queued probes are admitted within a second in queue order; TTFT at the
  client is the first token emitted at decode admission.
- `--max-queued-requests` cannot be exercised on a PD pair (F-16); use the monolithic control.

## 6. Cold vs cached output identity (`_phase_1161i.sh`)

Config B, concurrency 1, 256 greedy tokens, 8k and 32k prompts, with priority scheduling on (priority 0 cold, 50 cached) and off: the cached
run is byte-identical to the cold run in all four cases, and a second hit is identical again. Prefix caching does not change greedy output in
this build, with or without priorities. The B7/B8 differences seen in the second battery came from an `ignore_eos` mismatch between the two
requests in the probe, not from the server.
