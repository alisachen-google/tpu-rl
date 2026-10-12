# P/D disaggregation nightly: gap summary and tests to add

Short version of [pd_disagg_nightly_regression_plan.md](pd_disagg_nightly_regression_plan.md) (2026-10-12). Scope: `sgl-project/sglang-torchtpu`, P/D over raiden on TPU v7x, gpt-oss-120b and Qwen3-VL.

## What runs today and what it misses

Two nightlies already use real TPU capacity: the in-project cluster (01:00 UTC, Kueue nightly lanes) and the external cluster (10:30 UTC, four 2x2x2 and two 2x2x4 slices). Together they run about 14 P/D jobs: gpt-oss-120b 1P1D and 2P2D with radix, KV offload, EAGLE3 and resharding, one Qwen3-Coder-480B row and one Qwen3-VL-8B row. Every one is happy-path traffic (GSM8K, short prompts, modest concurrency) with accuracy as the pass signal.

They would not have caught most defects filed since September: admission and priority crashes, abort and disconnect hangs, peer death, silent token corruption, request shapes that kill an engine, long-context retraction deadlocks, serving-time recompiles, lying health endpoints, resource leaks. Both nightlies have also been red on every scheduled run for four days, so the P/D signal is buried.

## Tests to add

1P1D on 1+1 hosts unless stated. Probes already exist in `sglang_inference/scripts/` and port with little change.

| # | Test | Covers | Catches | Prio | Est. |
|---|---|---|---|---|---|
| N1 | `pd_singlehost_admission` | `--max-queued-requests` with and without priority, limits 0/1/4: exact admit and refuse counts, 503 through the router in under 2 s, no prefill work for a refused request, idle after | #1188 #1299 #1267 #1140 #1254 | P0 | 25 m |
| N2 | `pd_singlehost_abort_lifecycle` | client disconnect at every stage (bootstrap, 131k chunked prefill, transfer, decode, streaming, 8 at once): KV back to 0 in 60 s, abort propagation under 10 s | #1144 #1145 #1279 #1204 #1115 | P0 | 30 m |
| N3 | `pd_singlehost_peer_fault` | SIGKILL prefill and decode, raiden port blocked 30 s: survivor healthy, in-flight requests error within the timeout, pair recovers after restart | #1148 #1113 #1114 #1271 #1274 #1276 | P0 | 40 m |
| N4 | `pd_vs_monolithic_identity` | token-identical output PD vs monolithic at unaligned lengths (8191, 8193, 16k+7, 127,999), fp8 KV, both models; the only oracle for transfer corruption | #984 #829 #1076 #1077 F-4 | P0 | 45 m |
| N7 | `pd_small_pool_retraction` | decode pool at 262k tokens, 7 holders forcing retraction, SWA continuation budget at 0: all finish, identical output, no deadlock | #1142 #1181 #1270 #434 | P0 | 40 m |
| N13 | `pd_qwen3_vl_30b_mm_matrix` | production VL model with compile: image counts and sizes incl. ViT bucket edges, chunk-straddling image, cache-key hit/miss, n=2 and logprobs with an image: never a dead engine, 0 text recompiles | #1146 #1136 F-2 F-10 | P0 | 45 m |
| N14 | `pd_vl_identity_mrope` | Qwen3-VL monolithic vs 1P1D on image and text prompts: token identity, identical `prompt_tokens`, MMMU-Pro-50 within 0.02 | #984 class for image KV, #1077 | P0 | 45 m |
| N5 | `pd_request_shape_matrix` | 30 edge requests (n=2/4, logprobs, prompt logprobs, empty input, out-of-vocab ids, max_new_tokens 0, context+1, constrained decoding, odd images): 2xx or 4xx, never 5xx or a dead engine | #1147 #1244 #1266 #1189 #488 | P1 | 15 m |
| N6 | `pd_long_context_multiturn` | 128k/256k/1M recipes, 4 concurrent 127,999 prompts, 8-turn conversations with prefix cache + KV offload, needles each turn; the 128k/256k prefix + 8k/16k suffix + 1k output shape | #1067 #1066 #443 F-11 | P1 | 90 m |
| N8 | `pd_health_metrics_contract` | health codes track engine state within 15 s with prefill stopped or transfers failing; gauges move; watchdog arms | #1274 #1276 #1303 #1278 | P1 | 15 m |
| N9 | `pd_compile_stance` | every nightly model with `--enable-torch-compile`: decoder policy installed, 0 recompiles after warm-up, decode tok/s above a per-model floor | #1136 #1268 #1068 F-1 F-2 | P1 | 20 m/model |
| N12 | `pd_upgrade_canary` | N1 short + N4 10 prompts on the newest torch_tpu/libtpu wheel, advisory | #668 #1137 #1065 | P1 | 45 m |
| N15 | `pd_gpt_oss_harmony_chat` | `/v1/chat/completions`: reasoning_effort, reasoning/content split, return and call stops, tool call, multi-turn, streaming with usage | TC-9 gap, #1115 #1273 | P1 | 15 m |
| N16 | `pd_gpt_oss_swa_boundaries` | 35 exact lengths at page, window and chunk edges; 1M with 1 and 2 slots; hybrid-SWA-off identity | #1142 #1181 F-11 F-4 | P1 | 60 m |
| N17 | `pd_spec_corner_cases` | EAGLE3 without the recompile escape hatch: first spec request, odd batch sizes, accept length within 20 % of the previous night, VL-8B with images | campaign 7.x, #1066 | P1 | 40 m |
| N18 | `pd_kv_dtype_identity` | bf16, fp8 e4m3 and MXINT8 KV each identical to its own monolithic golden, both models | F-4 #829 #1066 | P1 | 60 m |
| N1v/N2v/N5v | VL variants of N1, N2, N5 | same probes with image prompts on Qwen3-VL-30B-A3B | VL prefill lengths and chunking | P1 | 70 m |
| N19 | `pp_pcp_dcp_matrix` | gpt-oss-20b PP4xPCP2 and PP2xTP2, gpt-oss-120b with DCP on the decode: accuracy within 0.02 of TP-only, no halt | #1137 class | P1 | 45 m |
| N10 | `pd_kv_canary_smoke` | `--kv-canary log` over GSM8K-50 once the TPU canary runtime lands | #1176 #1177 | P2 | 30 m |
| N11 | `pd_router_contract` | Rust PD gateway: priority passthrough, prefill error mapping, decode 503 relay, `/health` with one engine down | #1273 #1274 #1189 | P2 | 15 m |

### 2P/2D additions from the 2026-10-12 four-host run

| # | Test | Covers | Catches | Prio | Est. |
|---|---|---|---|---|---|
| N20 | `pd_2p2d_routing_and_locality` | per-request pair attribution so all four prefill-to-decode paths are asserted, not just sampled; second turn of a long-prefix conversation must land on the prefill holding its KV (needs the cache-aware gateway; the MiniLB gives 50 %) | 2P2D-F6, #1274 | P1 | 20 m |
| N21 | `pd_2p2d_partial_failure` | kill one prefill 3 s into 128k prefills and one decode 6 s into long decodes, rejoin each while traffic flows, orphaned transfer-queue entry freed without a client abort, greedy identity after two cycles | 2P2D-F4, 2P2D-F5 follow-up, #1115 | P1 | 30 m |
| N22 | `pd_2p2d_mixed_config` | one decode with a different page size (must be refused loudly) and one with a different KV layout (#1077 predicts silent garbage with HTTP 200) | #1077 | P1 | 20 m |
| N23 | `pd_2p2d_soak` | 2 h mixed traffic (random, MT-Bench, abort bursts): gauges drain every round, failure counters flat, host RSS flat per engine | #1286 class, leaks | P2 | 120 m |

## Cross-cutting assertion for every P/D test (cheapest first step)

One shared post-run helper, wired into all existing and new P/D files, turns six check classes from accidental into explicit: engine-log exception scan (`Scheduler hit an exception`, `AssertionError`, unexpected `Traceback`); serving-time recompile count 0 (`compile_audit.sh`); all queue gauges and `num_used_tokens` back to baseline; all ranks alive and health 200; `num_transfer_failed_reqs_total` and `num_bootstrap_failed_reqs_total` 0 outside injected faults; start-up and precompile duration recorded against a per-recipe budget. Per-recipe performance floors (0.8x of the 7-day dashboard median) turn chart-only regressions into red runs.

## Cost

P0 set: about 4.5 h of 1+1-host time per night, fits one external 2x2x2 slice, which today is idle about 20 h a day. P1 adds about 7 h plus one 2P2D slot. Nothing is replaced: the accuracy gates stay. The in-project Kueue pool is shared with PR gating and already saturated (repeated slice-admission timeouts), so new P/D load belongs on the external cluster.
