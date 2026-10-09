# PR: [TPU/PD] Keep the chunked-prefill SWA continuation budget above zero

**Branch:** `pd-swa-chunk-cap-deadlock` (1 commit on top of `09d1244c26`) · **Fixes:** the chunked-prefill deadlock with the radix cache off (our F-11 report, both triggers)

## Problem

gpt-oss-120b with `--disable-radix-cache` uses `SWAChunkCapPoolConfigurator`, which sizes the sliding-window (SWA) pool as two chunks plus a 384-token hold per slot. A chunked request's next chunk is capped at `remaining_swa − (window + page)`. Pages outside the window are only evicted in `maybe_evict_swa`, which runs when a batch containing the request runs. When the cap reaches 0 first, no batch forms, no eviction runs, and the request waits forever with both engines reporting healthy. Two triggers observed on a raiden pair:

1. `--max-running-requests 1`, 1M-context shape: `kv_cache_configurator` rounds the SWA pool **down** to a multiple of 8 pages for the compiled TPU decode (19,520 → 19,456 tokens at 8 slots; 16,832 → 16,384 at 1 slot). The removed tokens are exactly the third chunk's headroom. 256k, 512k and 1M prompts never finish (4 of 4).
2. Default 8 slots, decode under back-pressure (full decode pool, slow decode, held handshake): a request whose prefill finished but whose KV the decode has not pulled has no next chunk, so nothing evicts its last chunk's SWA pages. It holds ~8k tokens instead of 384; one such request plus the chunk in flight fills the pool, the next request's budget hits 0, and prefill and decode wait on each other (py-spy: prefill spins in `process_prefill_chunk`, decode in `_start_chunked_pull`). 1 of 4 requests completed; the rest hung until the client timeout.

## Change (4 files, +26/−7 in `srt`)

- `hardware_backend/tpu/kv_pool.py`: one constant, `SWA_POOL_PAGE_MULTIPLE = 8`, the sublane tiling factor the compiled decode needs the SWA page count to be a multiple of.
- `model_executor/pool_configurator.py` (`SWAChunkCapPoolConfigurator`): on TPU, align the chunk cap **up** to that multiple before the full pool is sized, so the later round-down is a no-op and the cap is honoured after alignment (at most 7 extra pages, accounted in the memory budget). Other devices unchanged.
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

See "Validation results" below (filled from `results/fix_validate/`).
