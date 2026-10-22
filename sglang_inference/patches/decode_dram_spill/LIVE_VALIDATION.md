# Live validation on 1P/1D (2026-10-20)

Hosts: prefill `wyzhang-tpu7x-16-a` (10.128.15.217), decode `wyzhang-tpu7x-16-b` (10.128.0.90). TP8 each.
Model: `openai/gpt-oss-120b`, FP8 KV, page 64. Base: sglang-torchtpu `f2588cdf79` + `changes.patch`.
Serving venv: torch 2.11.0+cpu, torch_tpu 0.1.1+release.2026.9.22, tpu_sync_torch 0.0.1.dev20261002213151.
All scripts, recipes and raw results are in `live_test/`. The design page is at
https://claude.ai/artifact/88muTVYCfUV99d8RXiGk3C (private).

## Result

The feature works on hardware. Each flush-and-recall request got its full prefix from the
decode-host store. Four local fixes were necessary before the first spill (see below).

## Fixes applied before the first spill

1. `python/sglang/srt/arg_groups/hicache_hook.py` rejects any storage backend with
   `--hicache-io-backend raiden`. The patch now exempts a `dynamic` backend whose `module_path`
   starts with `sglang_tpu.raiden_dram`. This is a core file. The bundle patch includes it.
2. The installed tpu-sync registry (dev20261002) does not return `kv_pool_group` or `evict_tier`
   in `StoreInfo`. `client.resolve()` rejected each store. The client now accepts a store on its
   identity when both fields are absent, and writes one warning. Use a wheel that matches
   tpu-sync 73499f1 to remove this shim.
3. `unified_radix_cache.check_hicache_events()` sets `prefetch_stats` on the object that
   `get_stats()` returns. `get_stats()` now returns a `StorageMetrics` subclass. It also supports
   `["key"]` reads, so the tests do not change.
4. The HiCache storage hash needs the JIT C++ extension `hicache_hash_cpp`. The hosts had no
   `ninja` and no OpenSSL headers. Install `ninja` in the venv, `libssl-dev` on the host, and put the
   venv `bin` directory on the engine PATH. This is an environment fix, not a code fix.

## Single-request benchmark (`live_test/spill_bench.py`)

Turn 1: prefix only, 1,024 output tokens. Turn 2: prefix + new tokens, 1,024 output tokens.
Greedy, one request at a time. Baseline run `spill_base_1`, store run `spill_dram_4`.

| Prefix | New | Turn-2 TTFT, prefix local | Turn-2 TTFT, flushed, recompute | Turn-2 TTFT, flushed, recall from store |
|---|---|---|---|---|
| 128k | 8k | 0.91 s | 6.33 s | **3.54 s** |
| 128k | 16k | 1.23 s | 6.88 s | **2.84 s** |
| 256k | 8k | 1.28 s | 17.51 s | **5.15 s** |
| 256k | 16k | 1.99 s | 18.35 s | **5.82 s** |

Each recall served the full prefix from storage (`cached_tokens_details.storage` = prefix length).
Decode speed did not change (185 to 203 tok/s).

## Agentic workload (`live_test/spill_agentic.py`)

Variant A: 8 sessions at the same time, 6 turns each, 128k private context, 4k tool tokens and
512 output tokens per turn, think 2 to 8 s between turns. Prefill host tier `hicache_ratio 0.5`
for the baseline.

| Run | Wall, 48 turns | Output tok/s | Later-turn TTFT p50 / p95 | Prefix tokens by tier |
|---|---|---|---|---|
| Baseline, no store (`agentic_base_r05`) | 300.7 s | 81.7 | 40.2 / 46.6 s | 302k HBM |
| Store, `cache` host mode (`agentic_dram_r05`) | 221.0 s | 111.2 | 26.1 / 31.6 s | 2.36M HBM, 0 store |
| **Store, `buffer_only` host mode (`agenticA_dram_bo`)** | **128.2 s** | **191.8** | **5.1 / 18.8 s** | 1.88M HBM, **3.66M store** |

Do not read the `cache` mode row as a recall win. Its prefill counters show 17.2M tokens of
storage hits and 17.2M unfulfilled: the host tier was full of write-through staging pages, so no
recall could land. Use `--hicache-host-memory-mode buffer_only` for this workload. In that mode
the counters show 29.3M tokens of hits, 29.1M prefetched and 0 unfulfilled.

Variant B: 12 sessions at the same time, 10 turns each, a 16k system prefix that all sessions
share, 112k private context, 2k tool tokens and 256 output tokens per turn, think 1 to 4 s.

| Run | Wall, 120 turns | Output tok/s | Later-turn TTFT p50 / p95 | Prefix tokens by tier |
|---|---|---|---|---|
| Baseline, no store (`agenticB_base_r05`) | 703.5 s | 43.7 | 65.5 / 71.5 s | 2.06M HBM (the shared prefix only) |
| **Store, `buffer_only` (`agenticB_dram_bo`)** | **240.4 s** | **127.8** | **16.2 / 31.9 s** | 2.94M HBM, **12.25M store** |

From turn 3 on, all 12 sessions got their private context from the store at each turn. The
prefill counters: 127.3M tokens of storage hits, 127.2M prefetched, 0 unfulfilled.

## Capacity ladder (`live_test/phase_capacity.sh`)

Throughput as a function of the working set. 4, 8, 12, 16 and 20 concurrent sessions, 128k
context each, 4 turns, 4k tool tokens and 256 output tokens per turn. Baseline = HBM (786k
tokens) + the full local host tier (ratio 2, 1.57M tokens, `cache` mode). Store pair = HBM +
`buffer_only` staging + 16 stores x 28 GiB on the decode host. Caches flushed and stores
restarted between points.

| Sessions | Working set | Baseline tok/s | Store tok/s | Gain | Baseline TTFT p50 | Store TTFT p50 | Prefix tokens by tier (baseline -> store) |
|---|---|---|---|---|---|---|---|
| 4 | 0.53M | 100.8 | 103.9 | 1.0x | 1.5 s | 1.4 s | 1.62M HBM -> 1.62M HBM |
| 8 | 1.05M | 121.0 | 112.8 | 0.9x | 1.7 s | 6.0 s | 0.13M HBM + 3.11M host -> 1.87M HBM + 1.38M store |
| 12 | 1.58M | 94.2 | 97.2 | 1.0x | 22.7 s | 19.9 s | 0.13M HBM + 3.95M host -> 0.39M HBM + 4.47M store |
| **16** | 2.11M | 45.6 | **94.4** | **2.1x** | 90.6 s | **26.1 s** | 0.13M HBM + 0.65M host -> 0.25M HBM + 6.24M store |
| **20** | 2.64M | 43.7 | **95.2** | **2.2x** | 114.1 s | **33.2 s** | 0.13M HBM + 0.38M host -> 0.25M HBM + 7.86M store |

The added capacity does not raise peak throughput. It removes the cliff. The local tiers hold
about 12 to 14 concurrent 128k sessions on this rig. Past that, the baseline recomputes almost
every context and its throughput halves; the store pair stays flat at about 95 tok/s with
0 unfulfilled prefetches (159.6M tokens of storage hits over the ladder).

## Race, deadlock and hang checks

CPU (`files/test/.../test_raiden_dram_concurrency.py`, 5 tests): 8 threads and a flusher for 10 s
with 751 spill and recall rounds, 0 errors, 0 quarantines; a stalled `WriteRemote` holds a lookup
for the RPC timeout and not longer; a registry outage returns misses at once; `close()` during IO
does not deadlock; KV and SWA pools from 6 threads at the same time.

Live (`live_test/spill_stress.py`), health probes every 5 s on both engines, 264 probes, 0 slow,
0 errors:

| Phase | Result |
|---|---|
| 4 concurrent recalls | All 4 served from storage, TTFT 2.1 to 3.2 s. |
| Client abort during recall, then re-issue | No hang. The re-issued request hit HBM. |
| One store paused 90 s (> 60 s RPC timeout) | No hang. The cold request during the pause waited 61 s for its first token: its lookup waited behind the stalled write on the backend lock. After resume, recall served 24,576 of 32,768 tokens. |
| Registry stopped, then restarted | Recall degraded to recompute with no hang. After the restart the stores did not register again; the directory is lost until the stores restart. |

## Limits found

- A decode-host store is append-only until restart. tpu-sync 73499f1 allocates landing blocks
  locked and never unlocks them; the host-store node has no sweep. After `budget / page bytes`
  pages (29.8k per rank at 16 GiB) each spill fails with `RESOURCE_EXHAUSTED ... Evictable: 0`
  and sessions fall back to recompute. Size the budget for the session population.
- The backend lock serializes lookups behind writes. A stalled store adds up to `rpc_timeout_s`
  to the TTFT of a request that needs a lookup.
- A registry restart loses the directory until the stores restart.
- `cache` host-memory mode starves recalls under churn. Use `buffer_only`.
