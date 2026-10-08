# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [TPU/PD] Chunked prefill deadlocks when the SWA continuation budget reaches 0 before window eviction runs (hit by gpt-oss-120b with `--max-running-requests 1`)

**Labels:** bug, tpu, pd

---

## Summary

For a hybrid sliding-window model in chunked prefill, the scheduler sizes the next chunk from the free sliding-window (SWA) pool minus
`window + page`. The SWA pages that already slid behind the window are freed only inside the preparation of a batch that contains the request.
If the free pool drops below `window + 2 pages` (256 tokens here) while those pages are still occupying it, the next chunk is sized to 0,
no batch is formed, the eviction never runs, and the request waits forever. Both engines report healthy and no watchdog fires, because no
forward pass is in flight.

The number of resident chunks does not matter; only the free pool at the continuation point does. Our configuration reaches exactly 0 free
budget after two chunks because of how the pool is sized:

1. With the radix cache off and `--max-running-requests` explicit, the SWA pool is sized from a worst-case cap: two prefill chunks plus 384
   tokens per request slot plus one page. With one slot that is 16,832 tokens.
2. A TPU-only rule rounds the pool **down** to a multiple of 8 pages (512 tokens): 16,832 becomes 16,384, exactly two chunks of 8,192.
3. Chunk 1 takes 8,192, chunk 2 is cut to 8,000 to keep `window + page` free, 192 tokens remain, and the budget for chunk 3 is 192 − 192 = 0.

With two or more slots the rounding leaves at least 512 free tokens at that point, a small third chunk is admitted, its preparation evicts the
pages behind the window, and the pool cycles normally. So in practice gpt-oss-120b with one request slot, which is what a 1M-context recipe
uses, cannot serve any prompt longer than 16,384 tokens.

| Pool | Occupied when chunk 3 is planned | Free | Chunk 3 budget | Result |
|---|---|---|---|---|
| 16,384 (1 slot) | 16,192 | 192 | 0 | deadlock, 4 of 4 prompts above 16,384 tokens |
| 16,896 (2 slots, observed) | 16,384 | 512 | 320 | chunk of exactly 320 tokens logged, pool 0.97 → 0.51, 32k prompt completes in 6.5 s |
| 17,920 (4 slots) | 16,256 | 1,664 | 1,472 | chunk of 1,344 tokens observed, eviction runs; 8k, 32k, 131k complete in 7 to 13 s |

Pool sizes as logged (`aligning swa_max_total_num_tokens`):

| `--max-running-requests` | SWA pool from the cap | After TPU rounding | Free above two chunks | Observed |
|---|---|---|---|---|
| 1 | 16,832 | 16,384 | 0 | hang on every prompt > 16,384 tokens (4 of 4) |
| 2 | 17,216 | 16,896 | 512 | 8k and 32k complete; third chunk 320 tokens |
| 4 | 17,984 | 17,920 | 1,536 | 8k, 32k, 131k prompts complete in 7 to 13 s (6 of 6) |
| 8 | 19,520 | 19,456 | 3,072 | 128k prompts complete; pool cycles 0.58 to 0.84 |

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips / 8 TensorCores each). Host a: prefill engine. Host b: decode engine. TP 8, EP 8 on each |
| Model | `openai/gpt-oss-120b` (MXFP4 weights, `sliding_window` 128), KV cache fp8_e4m3, page size 64 |
| Prefill flags that matter | `--disaggregation-mode prefill --disaggregation-transfer-backend raiden --disable-radix-cache --chunked-prefill-size 8192 --max-running-requests 1 --context-length 1056768 --enable-torch-compile` (overlap scheduling on) |

## Code trace

All links at commit 09d1244c26.

**1. Pool sizing.** `SWAChunkCapPoolConfigurator` is selected when the radix cache is off, `max_running_requests` is explicit and the model has
a sliding window. It sizes the SWA pool from `compute_swa_request_cap`:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/model_executor/pool_configurator.py#L1053-L1086

The cap for a prefill engine is `per_request * slots + chunks_in_flight * chunked_prefill_size + page_size`, where
`per_request = window + eviction_interval * draft_tokens + page_size + page_size` = 128 + 128 + 64 + 64 = 384, and `chunks_in_flight` is 2
with overlap scheduling:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/model_executor/pool_configurator.py#L1007-L1050
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/model_executor/pool_configurator.py#L679-L684

For one slot: 384 + 2 × 8192 + 64 = **16,832**. For four: 17,984. For eight: 19,520. The server log confirms each value.

**2. TPU rounding.** The pool is then rounded down to a multiple of 8 pages for the compiled decode kernel:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/mem_cache/kv_cache_configurator.py#L517-L545

Log line on the prefill engine with one slot:
```
[hybrid-SWA] aligning swa_max_total_num_tokens 16832 -> 16384 (SWA page count must be a multiple of 8 for correct torch.compile decode on TPU)
```
16,384 is exactly `2 × chunked_prefill_size`. The 448 tokens of headroom the cap added are removed.

**3. Continuation chunk budget.** For a chunked request the next chunk is capped at `remaining_swa - (window + page)`, floored to a page:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/mem_cache/prefill_budget.py#L220-L227
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/mem_cache/prefill_budget.py#L278-L288

Chunk 1: 8,192 (pool empty, no cap). Chunk 2: remaining 8,192 − 192 = **8,000** (matches the log). After chunk 2,
16,192 tokens are in use and 192 are free. Chunk 3: cap = 192 − 192 = 0, `fit_chunk` returns `None`, and `add_chunked_req` returns the
request without scheduling it:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/schedule_policy.py#L1157-L1200
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/scheduler.py#L4164-L4172

**4. Eviction only runs with a batch.** SWA pages outside the window are freed in `maybe_evict_swa`, which runs while a batch that contains
the request is prepared. For a chunked request under overlap scheduling the first eviction happens on the third chunk
(`extend_batch_idx < 2` is skipped):
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/schedule_batch.py#L3895-L3960

No chunk is scheduled, so no eviction runs; no eviction runs, so no chunk can be scheduled. The scheduler loop idles with one running request.

**Why the monolithic server is fine.** With the radix cache on, the pool is sized by `swa_full_tokens_ratio`, which gives a far larger pool.
The monolithic 1M-context server served the same 256k, 512k and 1M prompts in our tests.

## How we detected it

We run a 1-prefill + 1-decode validation campaign on two TPU v7x hosts. The long-context sweep sends 32k, 128k, 256k and 1M prompts through
the mini load balancer with a 1,800 s client timeout and a watchdog that polls both engines' `/metrics` every few seconds.

- The 32k request produced no output for 30 minutes. The watchdog showed, on the prefill engine, `num_running_reqs 1`,
  `num_used_tokens 16192`, `full_token_usage 0.01` and **`swa_token_usage 0.99`**, with `max_total_num_tokens_swa 16384`. The decode engine was idle.
- The prefill log showed two `Prefill batch` lines for the request, `#new-token: 8192` then `#new-token: 8000`, and nothing afterwards. When the
  client aborted, usage dropped to 0 and the same two lines repeated for the next request.
- The same recipe with `--max-running-requests 4` served 8k, 32k and 131k prompts in 7 to 13 s, and the SWA usage cycled between 0.54 and
  0.91 as chunks were admitted and windows were evicted.
- The three pool sizes in the log (16,384 / 17,920 / 19,456) and the chunk sizes (8,192 / 8,000 / 1,344) match the formulas in the code trace
  exactly, which is how we located the mechanism.

## How to reproduce

Two hosts, prefill on a, decode on b, raiden transfer. Only the prefill flags matter for the hang; the decode can use the same recipe.

1. Start the prefill engine with: `--disaggregation-mode prefill --disaggregation-transfer-backend raiden --disable-radix-cache
   --chunked-prefill-size 8192 --max-running-requests 1 --context-length 1056768 --page-size 64 --kv-cache-dtype fp8_e4m3
   --attention-backend pallas_tpu_rpa_v3 --tp-size 8 --ep-size 8 --enable-torch-compile` on `openai/gpt-oss-120b`.
2. Confirm the pool size: `grep 'aligning swa_max_total_num_tokens' <prefill log>` prints `16832 -> 16384`.
3. Start the decode engine and the load balancer, then send one 32,768-token prompt with `max_new_tokens 64`.
4. After 60 s: `curl -s http://<prefill>:30000/metrics | grep -E 'swa_token_usage|num_running_reqs'`.
5. `grep 'Prefill batch' <prefill log> | tail -3`.

**Observed.** Step 3 never returns. Step 4 shows `swa_token_usage 0.99`, `num_running_reqs 1`. Step 5 shows `#new-token: 8192`, then `8000`,
then nothing. An 8k prompt (one chunk) completes normally.
**Expected.** The 32k prompt completes in a few seconds, as it does with `--max-running-requests 4`, or the request fails with an error.

Control (run): repeat with `--max-running-requests 2`. Log: `aligning swa_max_total_num_tokens 17216 -> 16896`, 512 tokens of headroom. The 32k prompt's
chunks are `8192, 8192, 320, 7680, ...` with `swa token usage` 0.97 before the 320-token chunk and 0.51 after it, and the prompt completes in 6.5 s.
The third-chunk size matches the budget arithmetic to the token.

## Fix proposal

The defect is that the continuation budget can reach 0 before the eviction that would raise it runs, and nothing else triggers that eviction.
Fix 1 removes the defect for every pool size. Fix 2 keeps this configuration away from the edge. Fix 3 turns a silent hang into an error.

1. **Evict before budgeting (primary).** In `get_new_batch_prefill`, when `self.chunked_req` is set and the cache supports SWA, run the
   request's out-of-window eviction (`tree_cache.evict_sliding_windows(req, len(req.prefix_indices))`, respecting the overlap safety rule that
   the previous chunk's result must have landed) before constructing the `PrefillAdder`, so `remaining_swa` reflects the pages the completed
   chunks no longer need. Chunk sizes then stop depending on pool headroom, which is also what makes PD and monolithic chunk schedules differ
   for this model.
2. **Do not let the TPU alignment eat the cap's headroom.** In `kv_cache_configurator.py`, round a chunk-capped SWA pool **up** to the next
   multiple of 8 pages instead of down (at most 7 extra pages), or apply the 8-page alignment inside
   `SWAChunkCapPoolConfigurator.calculate_pool_sizes` before the full pool is sized, so the cap is honoured after alignment.
3. **Fail instead of waiting.** When `add_chunked_req` cannot fit any tokens, nothing else is running and no eviction is possible, abort the
   request with a clear error (`SWA pool too small for chunked prefill continuation`) rather than returning it to the next iteration forever.
   A startup check that the aligned SWA pool leaves at least `window + 2 pages` above a whole number of chunks would catch the configuration at launch.

## Related

- #1067 (long-context multi-turn PD stress test for GPT-OSS-120B) would hit this with one request slot.
- Not the same as #1113 / #1078: those are raiden transfer and reshard waits that expire after their deadline. This wait has no deadline.
- Side observation from the same campaign: because the chunk-capped SWA pool is small, the chunk schedule depends on pool headroom, so PD and
  monolithic outputs differ numerically for long gpt-oss prompts (identical with `--disable-hybrid-swa-memory`; no measurable GSM8K impact).

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`:
`results/F11/` (isolation runs, server-info snapshots), `results/logs/runs/A_gptoss_1m_run2/` (prefill and decode logs of the hang),
`results/FINDINGS.md` (F-11 entries), `results/bug_reports/F-11.md`.
