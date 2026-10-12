# P/D disaggregation on TPU: nightly regression plan

Date: 2026-10-12. Scope: `sgl-project/sglang-torchtpu`, prefill/decode disaggregation over raiden on TPU (v7x), single-host 1P1D and multi-host 2P2D. Sources: the nightly workflows and registered tests at main 2e14706a62, the 91 PD/raiden issues in the tracker, and the October 2026 live campaign (findings F-1 to F-16, PRs #1181, #1191, #1192/#1217, #1193, #1237, #1245, #1299).

## 1. What runs nightly today

Two schedules. The in-project nightly (`nightly-test-tpu.yml`, 01:00 UTC) builds once and fans out feature validation, perf and accuracy stages; P/D jobs are hand-wired in `feature-validation-tpu.yml` (`pd-mh-qwen3-480b-kvoff-reshard`, `pd-mh-qwen3-vl-8b-spec`). The external-cluster nightly (`nightly-test-external-tpu.yml`, 10:30 UTC) runs the P/D jobs listed in `.github/tpu-ci/external/external-topology.yaml` on tpu7x slices (1P1D on 1+1 hosts, 2P2D on 2+2 hosts) plus the 2x2x4 serving-benchmark matrix. Results land on the TPU nightly dashboard (`register_tpu_ci(suite=..., nightly=True, labels=[...])`, metrics via `dump_metric`).

| Test (test/registered/tpu/e2e) | Topology | Features | Signal | Est. | Where |
|---|---|---|---|---|---|
| gpt_oss_120b_pd_singlehost | 1P1D | baseline | GSM8K accuracy | 45 min | external |
| gpt_oss_120b_pd_singlehost_decode_radix | 1P1D | decode radix cache, multi-turn | GSM8K | 50 min | external |
| gpt_oss_120b_pd_singlehost_kv_offload | 1P1D | radix + RAM KV offload both engines | GSM8K, greedy | 60 min | external |
| gpt_oss_120b_pd_singlehost_kv_offload_spec | 1P1D | + EAGLE3 | GSM8K, greedy | 90 min | external |
| gpt_oss_120b_pd_singlehost_spec | 1P1D | EAGLE3 | GSM8K | 120 min | external |
| gpt_oss_120b_pd_multihost | 2P2D (TP16) | baseline | GSM8K, needle | 120 min | external + in-project |
| gpt_oss_120b_pd_multihost_spec_gsm8k | 2P2D | EAGLE3 + prefix cache | GSM8K, needle | 120 min | external + in-project |
| gpt_oss_120b_pd_multihost_prefix_cache_kv_offload | 2P2D | prefix cache + KV offload | GSM8K, needle, greedy identity | 180 min | external |
| ..._prefix_cache_kv_offload_spec | 2P2D | + EAGLE3 | GSM8K, needle, greedy | 180 min | external + in-project |
| ..._prefix_cache_kv_offload_spec_reshard | 2P2D, heterogeneous attention TP | push-reshard | GSM8K, needle, greedy identity | 180 min | external + in-project |
| gpt_oss_120b_pd_multihost_kv_offload_resharding | 2P2D | TP reshard + KV offload | GSM8K | 180 min | external |
| qwen3_coder_480b_pd_multihost_kv_offload_resharding | 2P2D | TP reshard + KV offload | GSM8K | 180 min | in-project (`pd-mh-qwen3-480b-kvoff-reshard`) |
| qwen3_vl_8b_pd_multihost_eagle3 | 2P2D | vision + EAGLE3 | MMMU-Pro, needle | 180 min | in-project (`pd-mh-qwen3-vl-8b-spec`) |
| bench: gpt_oss_120b / qwen3_coder_480b pd_2p2d_pc_spec_kvoff (8k1k, agentic) | 2P2D behind mini-LB | prefix cache + EAGLE3 + KV offload | TTFT / TPOT / throughput (perf dashboard) | 120 min each | external 2x2x4 |

What these have in common: one model family (gpt-oss-120b, plus one 480B and one VL row), happy-path traffic (GSM8K-style short prompts, modest concurrency), and accuracy as the pass signal. They would catch a model-quality regression and a transfer that corrupts tokens on the main path. They would not have caught most of the defects filed in the last six weeks.

## 2. Failure modes the current nightly cannot see

Grouped from the issue tracker and the live campaign. Numbers are issues; F-n are campaign findings.

| Class | Examples | Why today's tests miss it |
|---|---|---|
| Overload and admission | #1188 queue limit ignored in PD; #1140/#1254/#1272 burst deadlock prefill vs decode; #1267 capped transfer queue deadlock; #1303 metrics freeze with queued requests; #1299 door admission | Tests send bounded, polite concurrency; no burst above `max_running_requests`, no queue cap set. |
| Abort and cancellation | #1144/#1145 abort during chunked prefill crashes prefill (F-8); #1279 abort before dispatch leaks KV 300 s; #1204 early ABORT tombstoned; #431 prefill can't take decode abort; #1115/#1116 streaming abort accounting | No client ever disconnects mid-request in the suite. |
| Handshake, timeouts, peer failure | #1148 handshake never completes (F-12); #1230/#1114/#1113/#1271 one side waits the full timeout after the other gave up; #1078/#1232 handler exceptions dropped; #1285 ZMQ thread wedge; #435/#656 killed peer | Both engines are healthy for the whole run; no fault injection. |
| Numerical and layout correctness | #984 prompts >8k got other requests' KV; #829 fp8 KV + token-major regroup corrupts; #1076 SWA resharded with wrong head count; #1077 pool layout not compared; #718 1024-token bucket corrupts under reshard; F-4/F-9 PD vs monolithic divergence on unaligned lengths | GSM8K prompts are short and aligned; accuracy thresholds tolerate small token drift; no PD-vs-monolithic identity check except in two tests. |
| Request-shape edge cases | #1147 n>1 hangs (F-6); #1244 prompt logprobs kill the engine (F-5); #1266 validation gaps; #1189 empty input 500; #488 constrained decoding recompile; F-10 odd image size recompile | Only plain text, n=1, no logprobs, fixed image sizes. |
| Memory sizing and retraction | #1142/#1181 SWA chunk-cap deadlock (F-11); #1270 pool sizing ignores head_dim padding; #1269 host pool budget unchecked; #434/#503/#436 SWA retraction and resume | Pools are generously sized; nothing is retracted. |
| Health and observability | #1274 LB /health 200 with a dead engine; #1276 decode healthy while every transfer fails; #1303 gauges freeze; #1278 failed transfers carry no reason; #1177 kv checksum CUDA-only | Tests assert outputs, never the health or metrics contract. |
| Compile stance | #1136 MoE decoder runs eager (F-2); #1268 dense Qwen3 crashes at TP>1; #1068 recompile during serving; F-1 compile-cache SIGABRT; F-10 | No throughput floor, no "compiled" assertion, no recompile counter. |
| Long context and multi-turn | #1067 long-context multi-turn stress; #1066 MXINT8 + offload + EAGLE3 multi-turn; #443 1M context PD; F-11 1M recipe hang | Prompts stay short; multi-turn appears in one decode-radix test only. |
| Dependency upgrades | #668 torch_tpu upgrade broke compile and PD; #1137/#1065 libtpu >= 0.0.46 kernel halts; #520/#410 pins | Nightly pins the known-good binaries. |

## 3. Proposed nightly tests

Each is one registered file under `test/registered/tpu/e2e/` (`register_tpu_ci(suite="tpu-integration", nightly=True, labels=["pd", ...])`, wired as a `pd:` command in `external-topology.yaml`). Pass criteria are explicit so a red run names the bug class. 1P1D on 1+1 hosts unless stated. The probes exist already in the campaign repo (`sglang_inference/scripts/`: `door_probe.py`, `queue_limit_probe.py`, `handshake_timeout_probe.py`, `golden_diff.py`, recipes under `recipes/`) and port with little change.

| # | Test | What it does | Pass criteria | Catches | Est. |
|---|---|---|---|---|---|
| N1 | `pd_singlehost_admission` | gpt-oss-120b 1P1D with `--max-queued-requests 4` (priority on and off, limit 0 and 1): idle burst of 20, full queue newcomers, 8k refusals, streaming refusal, 16-request client-abort storm, 5 soak cycles, greedy identity under pressure | exact admit/refuse counts; refusals 503 through the router in < 2 s; no prefill batch for a refused request; engines idle and bootstrap queues empty after each block; zero scheduler exceptions | #1188, #1299, #1267, #1140, #1254, #1272, #1303 | 25 min |
| N2 | `pd_singlehost_abort_lifecycle` | client disconnects at each stage: during bootstrap, during a 131k chunked prefill, during transfer, during decode, streaming and non-streaming, 8 at once; then normal traffic | both engines alive; KV used tokens return to 0 within 60 s; prefill bootstrap queue 0; no 300 s waits (abort propagation < 10 s); follow-up GSM8K-20 passes | #1144, #1145, #1279, #1204, #431, #1115, #1116, F-8 | 30 min |
| N3 | `pd_singlehost_peer_fault` | fault injection: SIGKILL the prefill mid-stream, restart it, send traffic; SIGKILL the decode; block the raiden port for 30 s | surviving engine stays healthy; in-flight requests end with an error within the configured timeout, never hang; pair recovers after restart without redeploy; `/health` and `/health_generate` reflect reality | #1148, #1113, #1114, #1271, #1274, #1276, #435, #656, #1285 | 40 min |
| N4 | `pd_vs_monolithic_identity` | same weights served monolithic and 1P1D; 40 greedy prompts at unaligned lengths (1k, 8191, 8193, 16k+7, 32k, 127,999), fp8 KV, gpt-oss (hybrid SWA) and Qwen3-VL; 64 tokens each | token-identical output PD vs monolithic for every prompt; first-token logprob identical | #984, #829, #1076, #1077, #718, F-4, F-9, #321 | 45 min |
| N5 | `pd_request_shape_matrix` | one PD pair, 30 cheap requests: n=2 and n=4, logprobs on/off, prompt logprobs with the flag off and on, empty input, input_ids out of vocab, max_new_tokens 0, context-length+1 prompt, constrained decoding (regex/json), odd image sizes for VL | each gets a 2xx or 4xx within 30 s; no 5xx; both engines healthy after; `/metrics` scrape succeeds | #1147, #1244, #1266, #1189, #488, F-5, F-6, F-10 | 15 min |
| N6 | `pd_long_context_multiturn` | 128k, 256k and 1M recipes: 4 concurrent 127,999-token prompts; 8-turn conversations with prefix cache + KV offload; needle retrieval each turn | all turns answer; needles found; TTFT per turn under a recipe-specific bound; no hang (watchdog 15 min); identity of turn-1 output vs monolithic | #1067, #1066, #443, F-11, #354 | 90 min |
| N7 | `pd_small_pool_retraction` | decode pool sized to 262k tokens; 7 holders with 32k prompts generating 16k each (forces retraction); hybrid-SWA model; chunked prefill with the SWA continuation budget at 0 | at least one retraction observed; all holders finish; newcomers answered; outputs identical to an unretracted run; no deadlock; idle after | #1142, #1181, #1270, #434, #503, #436 | 40 min |
| N8 | `pd_health_metrics_contract` | with the pair healthy, then with the prefill stopped, then with transfers failing: poll LB and engine health endpoints and `/metrics` every 5 s | health codes match engine state within 15 s; gauges change while requests are queued; failed transfers carry a reason in logs; watchdog arms when queued-but-no-batch | #1274, #1276, #1303, #1278, #1177 | 15 min |
| N9 | `pd_compile_stance` | every nightly model, 1P1D with `--enable-torch-compile`: assert the decoder policy installed and no "decode steps run eager" line; count recompiles after warm-up; measure decode tokens/s vs a per-model floor | 0 recompiles after warm-up; throughput >= floor (e.g. 150 tok/s for Qwen3-VL-30B-A3B at TP8); `fail_on_recompile` never trips | #1136, #1268, #1068, F-1, F-2, F-10 | 20 min per model |
| N10 | `pd_kv_canary_smoke` | gpt-oss 1P1D with `--kv-canary log` once the TPU canary runtime lands (#1193 attachers) | engines start; canary reports 0 corruptions over GSM8K-50 | #1176, #1177 | 30 min |
| N11 | `pd_router_contract` | sglang-router PD mode: priority passthrough on `/v1/chat/completions`, prefill error mapping, decode 503 relay, `/health` with one engine down | priority reaches the scheduler (observed via eviction order); client status equals the engine's status; no hung client | #1273, #1274, #1189 | 15 min |
| N12 | `pd_upgrade_canary` | the full 1P1D smoke (N1 short + N4 10 prompts) against the newest torch_tpu / libtpu nightly wheel, separate from the pinned run | same criteria as N1/N4; a red here is advisory until the pin moves | #668, #1137, #1065, #520 | 45 min |

Cross-cutting rule for all of them: assert the absence of `Scheduler hit an exception`, `AssertionError` and `Traceback` in both engine logs (allowing the launcher's known `freeze_gc` connection-refused lines), and record decode `num_used_tokens`, `num_queue_reqs` and the prefill bootstrap queue at the end as `kind: info` metrics so a leak shows in the dashboard before it becomes a hang.

## 4. Priority and impact

| Priority | Tests | Impact if the class regresses | Rationale |
|---|---|---|---|
| P0 | N1 admission, N2 abort lifecycle, N3 peer fault, N4 identity, N7 retraction | whole-engine crash or hang (all ranks exit, or requests held for 120-300 s), or silently wrong tokens; each class has produced at least two P0 issues since September | These are the classes behind #1140, #1144, #1148, #984, #1142: production-ending and invisible to accuracy tests. N4 is the only one that catches silent corruption. |
| P1 | N5 request shapes, N6 long context, N8 health contract, N9 compile stance, N12 upgrade canary | single-request crashes that any client can trigger (N5), slow or stalled long sessions (N6), monitoring that lies (N8), 10-30x slower serving with a clean log (N9), surprise breakage at upgrade time (N12) | High frequency in the tracker, bounded blast radius, or advisory by nature. |
| P2 | N10 kv canary, N11 router contract | feature not yet shipped on TPU (N10); gateway behaviour outside the engine (N11) | Add once the underlying features land or when the router is pinned for production. |

Cost: P0 set is about 3 hours of 1P1D time (2 hosts) per night and fits one external 1+1 slot alongside the existing singlehost jobs; P1 adds about 3.5 hours plus one 2P2D slot for N6. Running P0 and P1 doubles PD coverage time but replaces nothing: the current accuracy gates stay.

## 5. Rollout

1. Week 1 (P0, cheapest first): N1 and N5 port directly from `door_probe.py` and the campaign recipes; N4 from `golden_diff.py`. Register under `tpu-integration`, labels `["singlehost", "gpt-oss-120b", "pd", ...]`, one `pd: {prefill_hosts: 1, decode_hosts: 1}` entry each in `external-topology.yaml`; run the dashboard-contract lint (`test/registered/unit/test_tpu_dashboard_contract.py`).
2. Week 2: N2 and N7 (abort and retraction probes exist in `handshake_timeout_probe.py` and `queue_limit_probe.py --mode retract`); N8 as a thin wrapper around the health endpoints.
3. Week 3: N3 fault injection (needs a JobSet hook to kill one pod and restart it), N6 long context on a 2P2D slot, N9 per model.
4. Later: N10 after the canary runtime, N11 when the router version is pinned, N12 as a separate advisory workflow with its own schedule so it never blocks the pinned nightly.

Owner: PD/TPU on-call rotation; one file per test, one job per file (dashboard rule); keep `id` stable once registered.
