## Motivation

In a P/D pair the prefill engine's HBM and host DRAM tiers are private and small. When a session returns after other traffic evicted its prefix, the prefill recomputes it: about 6 s at 128k tokens and 17 s at 256k on a tpu7x pair with gpt-oss-120b. The decode engine already holds that KV. Every prefilled request lands in decode HBM, and with `--disaggregation-decode-enable-radix-cache --enable-hierarchical-cache` the decode keeps finished requests in its radix tree and writes them through to its host pool.

This PR lets a prefill rank read such a prefix back from its decode peer. HiCache on each engine stays the only directory. Raiden is used for D2H, H2D and the device-to-device transfer only. There is no store process, no registry and no host-to-host DMA.

## How it works

1. A request enters the prefill queue. Its local match has no host hit and misses at least `SGLANG_TPU_PEER_PREFIX_MIN_TOKENS` tokens. The prefill rank locks the matched node and sends `PREFIX_REQ` with the tokens to the decode rank it serves pulls for.
2. The decode rank matches the tokens in its radix tree on its scheduler thread, locks the nodes, and, for a model without a sliding window, restores host pages to HBM through the existing `init_load_back` path and waits for the load event. It calls `register_read` on the device pages beyond the prefill's local match, plus the sliding-window pages of the last window on the SWA engine, and replies `PREFIX_BLOCKS`.
3. `pop_bootstrapped` holds the request. When every rank has the reply, each rank allocates the pages with `alloc_extend` and calls `start_read`. When every rank's read settled, each rank inserts the pages into its `UnifiedRadixCache` with `prev_prefix_len` at the local match and the SWA evicted length at the window start, unlocks, sends `PREFIX_RELEASE`, and lets the request be scheduled. It prefills only the remaining tokens.
4. Stage transitions use MIN/MAX all-reduces over the attention-TP group, so every rank inserts the same key in the same step. A failure or timeout on any rank falls back to recompute. Pages with a read still in flight drain until Raiden settles them.

## Changes

- `sglang_tpu/disaggregation/raiden/serde.py`: `PREFIX_REQ`, `PREFIX_BLOCKS`, `PREFIX_RELEASE`.
- `sglang_tpu/disaggregation/raiden/peer_prefix.py`: `PeerPrefixMixin` on `RaidenKVManager`, both roles. The listener thread only enqueues.
- `sglang_tpu/pd_peer_prefix.py`: scheduler integration behind `SGLANG_TPU_PEER_PREFIX=1`; wraps `process_decode_queue`, `PrefillBootstrapQueue.pop_bootstrapped` and `Scheduler._add_request_to_queue`.
- `sglang_tpu/patches.py`: install.

Flags: decode engine `--disaggregation-decode-enable-radix-cache --enable-hierarchical-cache --hicache-io-backend raiden --hicache-ratio 2 --hicache-write-policy write_through`; both engines `SGLANG_TPU_PEER_PREFIX=1`. The prefill keeps its usual HiCache flags; no storage backend.

## Limits

- A hybrid sliding-window model gets a hit only while the decode still holds the prefix and its last window in HBM. The decode's host copy is full-attention only (`pd_decode_swa_hicache.py`), and a restored prefix without window pages is unusable to the prefill. Offloading the window pages on the decode is the follow-up that lifts this.
- A prefill request with a partial host-tier hit does not use the peer; the local tier serves it.
- One decode engine per prefill rank, learned from its pull requests.

## Testing

CPU, registered in `base-a-test-cpu`: `test/registered/unit/hardware_backend/tpu/test_peer_prefix.py`, 8 tests with fake engines and a fake tree: serve and lock, miss below threshold, restore wait, hybrid window pages, lease release on Raiden's report, start_read failure and timeout, hook pull-insert-release, abandon and drain.

```
python -m pytest -q test/registered/unit/hardware_backend/tpu/test_peer_prefix.py
```

Live (tpu7x 1P/1D, TP8 each, gpt-oss-120b FP8, page 64): Turn-2 TTFT after the prefill's caches were flushed and the decode kept its cache (`live_test` run `planc_3`, 2026-10-22). Recompute and the Plan B store numbers are from the same pair.

| Prefix + new | Prefix local | Recompute | Store-backed L3 (Plan B) | Decode HiCache pull (this PR) |
|---|---|---|---|---|
| 128k + 8k | 0.68 s | 6.33 s | 3.54 s | **1.28 s** |
| 128k + 16k | 1.23 s | 6.88 s | 2.84 s | **1.81 s** |
| 256k + 8k | 1.09 s | 17.51 s | 5.15 s | **2.18 s** |
| 256k + 16k | 1.94 s | 18.35 s | 5.82 s | **3.01 s** |

Every recall served the full prefix (`cached_tokens_details.device` equals the prefix length, 8 commits per request, one per rank). Decode-side HiCache on its own costs the prefill nothing: with it enabled and the pull off, the 4/12/16-session ladder gave 103.7/94.3/44.5 tok/s against the baseline's 100.8/94.2/45.6.

Capacity ladder (4 to 20 concurrent 128k sessions, 4 turns each, prefill at the baseline HiCache configuration, pull on; run `planc_7`):

| Sessions | Baseline tok/s | Plan C tok/s | Baseline TTFT p50 | Plan C TTFT p50 | Pulls |
|---|---|---|---|---|---|
| 4 | 100.8 | 107.2 | 1.5 s | 0.9 s | 0, all local |
| 8 | 121.0 | 124.7 | 1.7 s | 1.0 s | 0, all local |
| 12 | 94.2 | 98.3 | 22.7 s | 19.7 s | 0, local host tier |
| 16 | 45.6 | 44.5 | 90.6 s | 90.5 s | 0: every request answered "short hit" |
| 20 | 43.7 | 43.3 | 114.1 s | 114.0 s | 0, same |

At 16 and 20 sessions the decode answered every request with a miss: the prefix was no longer in decode HBM, and the decode's host copy is full-attention only (`pd_decode_swa_hicache.py`), which a sliding-window model cannot use without the last window's SWA pages. So on gpt-oss the pull's reach is decode HBM, about five 128k contexts on this rig. Offloading the sliding-window pages of cached prefixes on the decode is the follow-up that extends it to decode DRAM; the serve path already restores from the decode host pool for models without a sliding window.

Stress (`spill_stress.py` with a prefill-only flush, health probes every 5 s on both engines, 16 probes, 0 slow, 0 errors): 4 concurrent recalls all served as device hits (TTFT 0.6 to 1.5 s); a client abort 0.5 s into a recall left no hang and the re-issued request hit locally; flushing both engines gave a clean miss and recompute with no hang, and the next recall worked. 48 commits, one per rank per recall; no read needed quarantine or draining.

Five defects showed up on hardware only and are fixed with tests: a wrong import path in the decode hook; the decode built its radix key from a list while the tree compares int64 arrays; the prefill's idle pool-invariant check counted a pull's landing pages as a leak; a rank-local refusal of a pull desynchronized the ranks' collectives; a lock on the local prefix during a pull starved the prefill's own scheduling.