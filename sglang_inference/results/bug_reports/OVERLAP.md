# Overlap check against the open issues of sgl-project/sglang-torchtpu

Checked on 2026-10-08 01:30 UTC. 69 open issues (pull requests excluded). Every title was read; the bodies of the 16 closest
candidates were read in full. Source list: `results/bug_reports/open_issues_2026-10-08.json`.

| Our ID | Verdict | Open issue | Why |
|---|---|---|---|
| F-1 tier-2 cache abort | new | none | No issue mentions `/dev/shm/torch_tpu_cache` or the tier-2 `CHECK`. |
| F-2 Qwen3-VL-30B decoder eager | new | #1069 is a request for a nightly Qwen3-VL-30B test | The missing decoder compile policy is not reported anywhere. |
| F-3 greedy not batch-invariant | related | #321, #391 | Those report other non-reproducibility sources (prefix cache hit vs cold; different DP ranks). Batch composition is not covered. |
| F-4 PD vs monolithic for long gpt-oss prompts | new | #321 is the nearest | The PD-vs-monolithic difference and the chunk-schedule cause are not reported. |
| F-5 prompt-logprob request crashes prefill | partial duplicate | **#432** (open since 2026-08-19, "feature") | Same code location. #432 says the feature is missing; it does not say one request kills the engine. Add our crash evidence as a comment, or file the crash as a bug that references #432. |
| F-6 `n=2` no response | new, unconfirmed | none | — |
| F-7 decode holds request after prefill death | duplicate in mechanism | **#1113** (P0, 2026-10-07) | Same mechanism: decode keeps the request until the 120 s raiden deadline. Our case adds a full prefill-engine death. Add as a comment. |
| F-8 client abort during chunked prefill crashes prefill | new | none (#1114 is a different hold) | The `release_kv_cache` assertion is not reported. |
| F-9 unaligned prompt length differs | new, unconfirmed | none | — |
| F-10 large image crashes Qwen prefill | same root cause class | **#488** (2026-08-25) | #488: a constrained-decoding request triggers a recompile and the `fail_on_recompile` stance kills the scheduler. Ours: the vision encoder triggers it for an image above the precompiled patch range. The fix for #488 (reject or allow the recompile instead of dying) also covers F-10. #1068 proposes a recompile test gate. |

## Other open issues that touch our test area (no overlap, but useful context)

- #1113, #1114, #1112: raiden deadline holds and reconnect storm after injected transfer failures (the colleague's fault-injection campaign).
- #1115, #1116: mini-LB stream termination and abort accounting.
- #1076, #829, #718, #1077: KV corruption or geometry errors under asymmetric TP resharding. Our 1P1D runs are symmetric (TP8 → TP8), so they do not apply.
- #1078: decode reshard handler exceptions are dropped and the room waits 300 s. Not hit in our runs.
- #856: decode radix cache plus EAGLE3 is rejected at startup; relevant for our Config E (10/30).
- #361: SWA prefix cache reuses a tombstoned prefix; relevant for Config B/C on gpt-oss.
- #1066, #1067: requests for long-context multi-turn PD tests with prefix cache, KV offload and EAGLE3 (our Configs C to E cover the same ground).
- #363 (closed or not in the open list): KV pool below ~605k tokens hangs the gpt-oss precompile; our recipes stay above it.

## Suggested filing order

1. F-8 (new crash, simple repro).
2. F-10 as a comment on #488, or a new bug that references #488 and #1068.
3. F-5 as a bug that references #432.
4. F-2 (missing compile policy), referencing #1069.
5. F-4 with the chunk-schedule data; F-1; F-9 and F-6 after confirmation.
6. F-7 as a comment on #1113.
