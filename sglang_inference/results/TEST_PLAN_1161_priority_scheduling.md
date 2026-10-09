# Test plan for sglang-torchtpu #1161: priority scheduling with PD, prefix caching and KV offload

Ask (wyzhang): validate `--enable-priority-scheduling` combined with PD disaggregation, prefix caching and KV offload to host RAM, on
GPT-OSS-120B (sliding-window attention, 128K) and Qwen3-VL (images, video; 128K and 256K). Goal: no correctness or stability issues.
Constraint from the issue thread: for the 10/15 launch a new request must **not** preempt a running decode, i.e. production runs with
`--disable-priority-preemption`; preemption-on behaviour is tested as well so its failure modes are known.

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
| PS-pre | B, C | `--enable-priority-scheduling` (preemption on, threshold 10) |
| PS-pre-t0 | B | `--enable-priority-scheduling --priority-scheduling-preemption-threshold 0` (any higher priority preempts) |
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
| A3 | `priority` 0, 1, −1, 2^31−1, −2^31 | accepted; ordering follows the integer |
| A4 | `priority` 2^63, float 1.5, string "high", null | clean 400 from the router and the engine, same message; engines unaffected |
| A5 | PS-dis: priority given | 400 "priority scheduling is disabled"; without the flag: ignored, served |
| A6 | priority on chat completions, `/generate` batch (list of prompts), `n=1`, streaming | honoured on every API shape; batch items share the priority |
| A7 | priority on a health-check request (`/health_generate`) | never preempts or evicts |

### B. Ordering without preemption (PS-np, Config B, gpt-oss then Qwen)

| # | Case | Expect |
|---|---|---|
| B1 | Fill all slots with priority 0 decodes (long outputs); submit priority 100 | runs when a slot frees, **no running request preempted** (`num_retracted_reqs` 0, no retraction/requeue log) |
| B2 | 8 waiting requests with priorities 0…7 submitted low-first while slots are busy | admitted high-first; TTFT monotone in priority |
| B3 | Same priority, 8 requests | FIFO (arrival order) |
| B4 | PS-low: priorities 0…7 | admitted low-value-first (inverted B2) |
| B5 | Starvation: continuous priority-10 stream at full occupancy plus one priority-0 request | the priority-0 request is served once the stream pauses; never lost; its TTFT recorded |
| B6 | PD: high-priority request arrives while a low-priority 131k transfer is in the decode prealloc/transfer queue | prealloc queue reorders; high-priority request admitted first once pre-allocable; the long transfer still completes |
| B7 | Prefix cache (Config B): low-priority request warms a 32k prefix; high-priority request with the same prefix | `cached_tokens` ≈ 32k; correctness vs golden |
| B8 | HiCache (Config C): prefix evicted to host under pressure; high-priority request with that prefix | `#host-hit-token` in the prefill log; identical output |

### C. Preemption on (PS-pre, PS-pre-t0; Config B then C)

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
| C9 | Client aborts the **preempting** request right after preemption | victims resume; nothing leaks |
| C10 | Client aborts the **victim** while it is requeued | its state fully released; no "state was deleted" storm |
| C11 | Preemption under HiCache (Config C): victim's prefix is in host memory | re-prefill reloads from host; `#host-hit-token`; output identical |
| C12 | gpt-oss SWA: victim at seqlen beyond the window | sliding-window pages freed and re-created correctly (needle check on resumed output) |
| C13 | Preempt a request with `max_new_tokens` 1 and one with `n`… (n>1 is F-6, skip) | single-token request handled |
| C14 | Preemption with speculative decoding (Config D) | draft/verify state reset; acceptance stats sane |
| C15 | Repeated preemption of the same victim (3×) | resumes each time; completion identical |

### D. Queue limit (PS-q: `--max-queued-requests 4`)

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
| E1 | Image request priority 0 running; text request priority 11 preempts it (PS-pre) | vision embeddings recomputed or cached on resume; output identical to golden |
| E2 | Video request (many frames, near the compiled patch bucket) preempted and resumed | no recompile (F-10 area), identical output |
| E3 | Mixed batch: 4 image + 4 text, priorities interleaved | ordering as specified; mm cache hits reported |
| E4 | 256k context: 200k-token text + image prompt, priority high, arriving while a 256k low-priority decode runs (PS-np) | no preemption; admitted when the slot frees |
| E5 | Image prefix reuse across priorities (same image, different question) | `cached_tokens` covers the image tokens |

### F. Stability and resource return

| # | Case | Expect |
|---|---|---|
| F1 | 30 min open-loop mixed priorities (0/5/10/50), PS-np, both models | no error responses, no stuck requests, used tokens return to 0 at idle, no recompiles |
| F2 | Same with PS-pre, preemption rate recorded | all requests complete; `num_retracted_reqs` matches preemption log count; no leak |
| F3 | Metrics | `priority` label present on queue/running gauges; dashboard panels adjusted (`priority=~".*"`) |
| F4 | Disconnect storm (F-8 fix verified) with priorities | no engine exception |

## 4. Harness

- Recipes: copies of `gpt-oss_B_128k.json`, `gpt-oss_C_128k.json`, `qwen3-vl_B_128k.json`, `qwen3-vl_C_128k.json`, plus a Qwen 256k recipe, with the
  flags of §2 added to both engines (`_ps_np`, `_ps_pre`, `_ps_pre_t0`, `_ps_low`, `_ps_q`, `_ps_dis` suffixes).
- Driver: `scripts/priority_probe.py` (to write): submits requests with explicit priorities and timing offsets, records TTFT/order/HTTP status,
  polls `/metrics` for `num_retracted_reqs` and queue gauges, and compares greedy outputs with the PS-off golden (`golden_diff.py`).
- Phase script `_phase_1161.sh`: one block per configuration, `ensure_up` relaunch, waits on `tpu_free.sh`; results into `results/1161/`.
- Oracles: golden diff at c=1 for undisturbed requests; needle check for resumed requests; identical-output check for preempted requests
  (greedy; batch-composition caveat F-3 noted, so compare after resumption at c=1).
