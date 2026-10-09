# PR: [TPU/PD] Keep the chunked-prefill SWA continuation budget above zero

**Branch:** `pd-swa-chunk-cap-deadlock` (1 commit, rebased on `2cec1cf194`) · **Fixes:** the chunked-prefill deadlock with the radix cache off (our F-11 report, both triggers)

## Problem

gpt-oss-120b with `--disable-radix-cache` uses `SWAChunkCapPoolConfigurator`, which sizes the sliding-window (SWA) pool as two chunks plus a 384-token hold per slot. A chunked request's next chunk is capped at `remaining_swa − (window + page)`. Pages outside the window are only evicted in `maybe_evict_swa`, which runs when a batch containing the request runs. When the cap reaches 0 first, no batch forms, no eviction runs, and the request waits forever with both engines reporting healthy. Two triggers observed on a raiden pair:

1. `--max-running-requests 1`, 1M-context shape: `kv_cache_configurator` rounds the SWA pool **down** to a multiple of 8 pages for the compiled TPU decode (19,520 → 19,456 tokens at 8 slots; 16,832 → 16,384 at 1 slot). The removed tokens are exactly the third chunk's headroom. 256k, 512k and 1M prompts never finish (4 of 4).
2. Default 8 slots, decode under back-pressure (full decode pool, slow decode, held handshake): a request whose prefill finished but whose KV the decode has not pulled has no next chunk, so nothing evicts its last chunk's SWA pages. It holds ~8k tokens instead of 384; one such request plus the chunk in flight fills the pool, the next request's budget hits 0, and prefill and decode wait on each other (py-spy: prefill spins in `process_prefill_chunk`, decode in `_start_chunked_pull`). 1 of 4 requests completed; the rest hung until the client timeout.

## Change (4 files, +31/−8 in `srt`)

- `hardware_backend/tpu/kv_pool.py`: one constant, `SWA_POOL_PAGE_MULTIPLE = 8`, the sublane tiling factor the compiled decode needs the SWA page count to be a multiple of.
- `model_executor/pool_configurator.py` (`SWAChunkCapPoolConfigurator`): on TPU, align the chunk cap **up** to that multiple (`_swa_cap_tokens`), in both sizing paths (free-memory sizing and the `--max-total-tokens` constrained path), before the full pool is sized, so the later round-down is a no-op and the cap is honoured after alignment (at most 7 extra pages, accounted in the memory budget). Other devices unchanged.
- `mem_cache/kv_cache_configurator.py`: the TPU rounding block uses the same constant instead of a literal 8 (no behaviour change).
- `disaggregation/prefill.py`: free a finished prefill's SWA pages beyond the window on every raiden transfer, not only pipeline-parallel ones. Raiden's SWA payload is the window only (`_swa_payload` rounds the window start down to a page), so those pages are dead once prefill finishes; `_release_swa_beyond_window` already did this for `pp_size > 1` (#748).

Why not "evict before budgeting" in the scheduler: it is the fuller fix for every pool size, but it touches the shared `PrefillAdder` path for all models; the two changes here are TPU- and raiden-local, remove both observed triggers, and keep the shared code change to one condition.

## Tests

Added to the existing files, CPU-only:

- `test/registered/unit/model_executor/test_pool_configurator.py`: `test_chunk_cache_cap_tpu_aligns_up_to_eight_pages`: the same configuration that yields a cap of 76 tokens on CPU yields 96 on TPU (page 4 → 32-token multiple), ≥ cap, memory within budget; CPU stays 76.
- `test/registered/unit/disaggregation/test_prefill_abort_result_cleanup.py`: `test_finished_prefill_frees_swa_beyond_window_on_raiden`: a finished, non-aborted prefill calls `_release_swa_beyond_window` with the raiden backend and not with mooncake, and enters the transfer queue. `test_prefill_swa_window_release.py` already covers what that release frees.

```
pytest test/registered/unit/model_executor/test_pool_configurator.py test/registered/unit/disaggregation/test_prefill_abort_result_cleanup.py test/registered/unit/disaggregation/test_prefill_swa_window_release.py
76 passed, 34 subtests passed
```

## Live validation (TPU v7x, 1 prefill + 1 decode, gpt-oss-120b)

Combined fix branch deployed on both hosts.

| Case | Before the fix | With the fix |
|---|---|---|
| Trigger 2: decode pool 262,144 tokens, 4 × 131k prompts, 1,024 output tokens, concurrency 4 (`gpt-oss_A_128k_smallpool.json`) | 1 of 4 completed; 3 hung to the 1,200 s client timeout; prefill spinning at SWA usage 0.99, chunks 2,112 → 640 → none | **4 of 4 in 33 s**, 0 errors, needles 3/3 on every answer; chunk trail stays at the healthy 8,192 / 2,880 alternation (SWA usage 0.58 / 0.84), no stall |
| Trigger 1, **monolithic control** (same engine, radix off, 1 slot, 1M shape, 256k prompt; no P/D so no handshake effects) | main `09d1244c26`: SWA pool 16,384; chunks 8,192 → 8,000 at SWA usage 0.99, then no batch; client read timed out after 1,800 s | fix: SWA pool 16,896; **1,024 tokens returned**, TTFT 19.2 s, e2e 25.4 s, chunk trail 320 / 8,064 alternation |
| Trigger 1: single slot, 1M shape, 256k / 512k / 1M prompts (`gpt-oss_A_1m.json`) | prefill never finished any prompt: third chunk budget 0 at SWA pool 16,384 (4 of 4) | SWA pool **16,896** (cap 16,832 aligned up); **prefill completed all three prompts** (850 chunks, 8,192 / 320 alternation, `#pending-token` reaches 0 each time, no stall). The responses still did not reach the client: the decode ran 0 batches and its handshake ended only on the client abort, which is the single-slot handshake hold we report separately (F-12), not this bug |

| Scenario battery on the PR head (upstream `2cec1cf194` + this branch), both hosts | | |
| S1 Config A 128k, 8 slots (SWA pool 19,968) | – | byte-identical to the c=1 monolithic golden at 8k; 131k diverges numerically as before the fix (chunk schedule, F-4 class), needles 3/3; 8 × 131k at c=8 in 59 s; 6 client aborts (2 concurrent) handled, 0 exceptions |
| S2 decode pool 262k, heavier | 1 of 4 before | 8 × 131k at c=8: 8/8 in 59 s; 16 × 32k at c=16: 16/16 in 31 s |
| S3 1M shape, 4 slots (pool 18,432 = cap 17,984 aligned up) | – | 4 × 256k at c=4: 4/4 in 81 s; 1M prompts served |
| S4 radix on (B, C): fix is a no-op | – | 131k served, reuse `cached_tokens` 122,880, 0 exceptions |
| S5 Qwen3-VL Config A (no SWA) | – | 8k text byte-identical to the pre-fix pair; image cases hit the pre-existing F-10 recompile crash (also on upstream main), unrelated to this change |

Evidence: `results/fix_validate/smallpool.log`, `smallpool_out.jsonl`, prefill log of run `A_128k_smallpool_fixval`, `results/phase_fix_validate3.log`.
