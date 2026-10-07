# Bug reports — sglang-torchtpu 1P/1D on TPU v7x

Generated 2026-10-07 22:46 UTC from FINDINGS.md.

| ID | Class | First seen (UTC) | Severity | Area | Status | File |
|---|---|---|---|---|---|---|
| F-1 | bug | 2026-10-07 19:35 | Low (env/robustness) | launcher + torch_tpu tier-2 compile cache | observed, repro 1/1, workaround applied | [F-1.md](F-1.md) |
| F-2 | bug (missing compile policy / perf) | 2026-10-07 19:59 | High (perf / TC-5 scope) | torch.compile coverage, Qwen3-VL-30B-A3B | observed, repro 1/1 | [F-2.md](F-2.md) |
| F-3 | test-methodology note (not a bug) | 2026-10-07 20:02 | Medium (test-methodology / numerics) | greedy decode reproducibility on TPU (monolithic gpt-oss) | observed 10/10 prompts | [F-3.md](F-3.md) |
| F-4 | bug candidate (correctness, under isolation) | 2026-10-07 20:20 | **High (correctness, under isolation)** | PD prefill vs monolithic for multi-chunk prompts (gpt-oss) | observed 6/6 long prompts, 0/4 short; isolating | [F-4.md](F-4.md) |
| F-5 | bug (crash) | 2026-10-07 20:32 | **High (crash: single request kills the prefill engine)** | TPU compiled backend / request validation | observed 1/1, repro queued | [F-5.md](F-5.md) |
| F-6 | bug candidate (hang, repro pending) | 2026-10-07 20:28 | **High (hang, candidate)** | `n=2` parallel sampling through PD | observed 1/1 (300 s no response), repro queued | [F-6.md](F-6.md) |
| F-7 | observation (recovered; misleading log line) | 2026-10-07 20:33 | Medium (resource leak after peer death, TC-7) | decode engine after prefill crash | observing | [F-7.md](F-7.md) |
| F-8 | bug (crash) | 2026-10-07 20:43 | **High (crash: client abort during chunked prefill kills the prefill engine)** | prefill engine, chunk cache (radix off) | observed 1/1 (first abort of TC-7), repro queued | [F-8.md](F-8.md) |
| F-9 | bug | 2026-10-07 22:10 | Medium (correctness, candidate) | PD prefill vs monolithic for page/chunk-unaligned prompt lengths (both models) | observed 4/4 unaligned prompts; exact check vs c=1 golden queued | [F-9.md](F-9.md) |
| F-10 | bug | 2026-10-07 22:13 | **High (crash: one large image kills the Qwen prefill engine)** | ViT compile buckets × `fail_on_recompile` stance | observed 1/1, repro queued | [F-10.md](F-10.md) |
