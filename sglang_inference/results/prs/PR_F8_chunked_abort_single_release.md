# PR: [TPU/PD] Release an aborted chunked prefill's kv row exactly once when the raiden sender must drain

**Branch:** `pd-chunked-abort-single-release` (1 commit on top of `09d1244c26`) · **Fixes:** the prefill-engine crash on client disconnect during chunked prefill (our F-8 report)

## Problem

A client that disconnects while its prompt is being chunk-prefilled on a raiden prefill engine with the radix cache off kills the engine:

```
_drain_deferred_chunked_sender_aborts -> release_kv_cache
AssertionError: Only a mamba-capable tree cache allows freeing before alloc
```

`process_pending_chunked_abort` parks the aborted request on `_deferred_chunked_sender_aborts` because the raiden transfer group still has native I/O in flight (`group_failure_requires_drain`), so its pages must stay allocated until the group drains. The in-flight chunk's result then reaches `_retire_aborted_prefill_result`, which released the kv row anyway. On the next step the drain released it again; `ChunkCache` has no mamba state, so the second release trips the precondition and every rank exits. Reproduced 3 of 3 times (two 8k prompts aborted mid-chunk, one 131k prompt aborted in its last chunks); not reproducible with the radix cache on, where the drain path is not taken.

## Change (2 files, +9/−3 in `srt`)

- `disaggregation/prefill.py` `_retire_aborted_prefill_result`: skip `release_kv_cache` while the request is parked on the deferred list. The drain is the single owner of the release, so pages are also no longer freed while raiden may still read them (the early free was the latent hazard behind the assertion).
- `managers/scheduler.py` `_drain_deferred_chunked_sender_aborts`: guard the release with `holds_kv or holds_mamba`, as the other three abort sites already do, so a parked request whose row is already gone cannot assert.

No new module, no behaviour change for senders that do not require a drain (release stays immediate), no change to the decode.

## Tests

Added to the existing files, CPU-only, run with the repo's `register_cpu_ci` suite:

- `test/registered/unit/managers/test_scheduler_chunked_abort_race.py`: `TestChunkedAbortDeferredRelease` drives `process_pending_chunked_abort` → `_retire_aborted_prefill_result` → drain with a raiden-like sender that drains after N polls. Release happens exactly once, in the drain, for the overlap (result a step later) and non-overlap (result same step) orderings; immediate release when no drain is required; an already-drained sender releases at abort time; a parked request whose row is gone does not assert; two parked requests release independently. The fake release asserts `holds_kv`, so the old double release fails the test on `main` (4 of 6 cases fail without the fix).
- `test/registered/unit/disaggregation/test_prefill_abort_result_cleanup.py`: `test_aborted_result_parked_for_sender_drain_keeps_kv_row` drives `process_batch_result_disagg_prefill` with the request parked; the request is finished and streamed, metadata freed, kv row kept.

```
pytest test/registered/unit/managers/test_scheduler_chunked_abort_race.py test/registered/unit/disaggregation/test_prefill_abort_result_cleanup.py
20 passed
```

## Live validation (TPU v7x, 1 prefill + 1 decode, gpt-oss-120b, Config A: radix off, raiden)

See "Validation results" below (filled from `results/fix_validate/`).
