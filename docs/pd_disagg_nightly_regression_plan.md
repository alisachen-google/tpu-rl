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

## 6. Model coverage review (added 2026-10-12)

Question asked: does the plan cover both gpt-oss-120b and Qwen3-VL, exercise each model's own features, and reach the corner cases the live campaign found? Answer: gpt-oss is covered broadly but misses its Harmony chat path and its sliding-window corner cases; Qwen3-VL is barely covered, and the model that production will serve (Qwen3-VL-30B-A3B, MoE) has no nightly test at all. The matrix below is against the tests of section 1 plus N1-N12.

### 6.1 Model-specific features and where they are tested

| Feature | Model | Nightly today | Plan N1-N12 | Gap |
|---|---|---|---|---|
| Hybrid sliding-window KV pool (window 128), chunk-cap budget, `--disable-hybrid-swa-memory` variant | gpt-oss | implicit in every run, never at the boundaries | N7 (continuation budget at 0), N4 (unaligned lengths) | no window/page/chunk-edge sweep (TC-2: 35 lengths at the 64/128/8192 edges), no 1M single-slot trigger (F-11), no ratio-pool guard case |
| MXFP4 MoE weights, fp8 e4m3 KV vs bf16 KV | gpt-oss | fp8 only in the perf bench rows | N4 (fp8) | no fp8-vs-bf16 identity against the monolithic golden (F-4 class, #829), no MXINT8 KV (#1066) |
| Harmony chat: `reasoning_effort`, reasoning/content split, `<\|return\|>`/`<\|call\|>` stops, tool calls, multi-turn with dropped reasoning | gpt-oss | none (all tests use `/generate` or GSM8K) | none | TC-9 found the tool-call parser missing from the recipe; nothing nightly would catch a Harmony regression |
| EAGLE3 and DFlash speculation | gpt-oss | spec and dflash rows, GSM8K accept-length | N9 throughput floor | no verify-graph recompile check without `SGLANG_TPU_ALLOW_SPEC_RECOMPILE` (campaign 7.5), no odd `bs x draft` batch (7.3), no first-spec-request-after-warm-up (7.2) |
| DPA / hybrid DPA, PP x PCP, push-reshard | gpt-oss | dpa rows, reshard rows | N3 (peer fault) | fine as is |
| 1M context recipe, long multi-turn | gpt-oss | none | N6 | N6 is gpt-oss only and text only |
| ViT compile buckets: resolution outside the precompiled range (F-10, #1146), image straddling the 8192 chunk, `--mm-process-config` pixel caps | Qwen3-VL | 8B tests pin `max_pixels` so no bucket is ever missed | N5 "odd image sizes" (one line) | no image-count sweep (1/4/16/64), no 448^2 vs 1024^2, no straddle, no corrupt image -> 400, no 1920x1080 kill check |
| MoE language model compiled (F-2, #1136, #1237) | Qwen3-VL-30B-A3B | no 30B test exists; 8B is dense | N9 floor 150 tok/s | the model with the defect is not in the nightly (#1069 asks for it) |
| MRoPE after the P/D hand-off (decode re-processes images; no rope state crosses the wire) | Qwen3-VL | 8B PD test scores MMMU-Pro report-only, accept length gated | N4 names Qwen3-VL but not images | no 1,024-token image-prompt identity vs monolithic, which is the detector (campaign 5.4) |
| Multimodal prefix-cache keys: same image + different question hits, different image + same text misses; HiCache with image KV | Qwen3-VL | none | none | TC-10 and config C Qwen covered these live; nothing nightly |
| VL on the request-shape edges: n>1 with images, logprobs with images, image-only, text-only, empty image list | Qwen3-VL | none | N5 partly (text cases) | add the VL rows to N5 |
| VL overload, abort and peer-fault behaviour | Qwen3-VL | none | N1-N3 are gpt-oss only | image prompts change prefill length and chunking; at least N2 and N5 need a VL run |
| Router and mini-LB error propagation (4xx stays 4xx, decode death ends the stream with an error) | both | none | N11 (router only) | mini-LB is what the CI recipes launch; run the contract against both (campaign 6.1/6.2) |

### 6.2 Tests to add or parametrize

| # | Test | What it does | Pass criteria | Catches | Est. |
|---|---|---|---|---|---|
| N13 | `pd_qwen3_vl_30b_mm_matrix` | Qwen3-VL-30B-A3B 1P1D, `--enable-torch-compile`: images 1/4/16/64 at 448^2 and 1024^2; one image straddling the 8192 chunk; image-only; text-only; corrupt image; 1920x1080 and 896x504 (outside and inside the ViT buckets); same image + different question, different image + same text; n=2 with an image; logprobs with an image | every request 2xx or 4xx, never a dead engine (F-10 becomes a 4xx or a handled recompile); cache-key hits and misses as specified; decoder policy installed; 0 serving-time recompiles for text; both engines idle after | #1146, #1136, #1069, F-10, F-2, TC-10 | 40 min |
| N14 | `pd_vl_identity_mrope` | Qwen3-VL-30B-A3B monolithic vs 1P1D: 10 image prompts and 10 text prompts, greedy, 1,024 tokens, plus a 32k text prompt | token-identical outputs and first-token logprob; `prompt_tokens` identical between engines; MMMU-Pro-50 within 0.02 of monolithic | campaign 5.4, #984 class for image KV, #1077 | 45 min |
| N15 | `pd_gpt_oss_harmony_chat` | `/v1/chat/completions` through the mini-LB and the router: `reasoning_effort` low/medium/high, reasoning vs content split, `<\|return\|>` and `<\|call\|>` stop positions with `ignore_eos=false`, a tool call with the parser configured, 3-turn multi-turn with dropped reasoning, streaming with `usage`, 6k system prompt | each case returns the expected shape; tool call parsed; reasoning never leaks into content; stream ends with `[DONE]` and usage; identical answers via both LBs | TC-9 gap, #1115, #1273, campaign 5.1/5.2/6.1/6.2 | 15 min |
| N16 | `pd_gpt_oss_swa_boundaries` | 35 exact prompt lengths at the 64/128 page, 128 window and 8192 chunk edges (1 to 16,385 tokens), 160 tokens each; prefix-cache hit ending mid-window; 1M recipe with 1 slot and with 2 slots on 32k/131k/256k prompts; `--disable-hybrid-swa-memory` identity against the default pool | 35/35 complete with exact token counts; needles found; 1-slot 1M completes (F-11 trigger); hybrid-off outputs byte-identical to monolithic | #1142, #1181, F-11, F-4, TC-2 | 60 min |
| N17 | `pd_spec_corner_cases` | gpt-oss EAGLE3 1P1D without `SGLANG_TPU_ALLOW_SPEC_RECOMPILE`: first spec request after warm-up, decode batch sizes 3/5/7 with `tpu_compile_bs` [1,2,4,8], a chat workload vs a random-text workload for accept length; Qwen3-VL-8B EAGLE3 the same with image prompts | no verify-graph recompile crash; accept length within 20% of the previous nightly; outputs vs the no-spec twin NUM-class, needles 3/3 | campaign 7.2/7.3/7.5, #1066 | 40 min |
| N18 | `pd_kv_dtype_identity` | both models, 1P1D vs monolithic at bf16 KV, fp8 e4m3 KV and MXINT8 KV: 20 greedy prompts incl. unaligned 127,999 | token identity per dtype vs its own monolithic golden; GSM8K-50 within 0.02 | F-4, #829, #1066 | 60 min |
| N1v, N2v, N5v | VL variants of admission, abort lifecycle and request shapes | same probes as N1/N2/N5 with image prompts on Qwen3-VL-30B-A3B | same criteria | VL-specific prefill lengths and chunking under the same classes | 25 + 30 + 15 min |

### 6.3 Priority changes

- N13 and N14 are P0: they put the production VL model into the nightly for the first time and they are the only tests that would have caught F-2 and F-10, both of which were silent (slow) or fatal (engine dead) with a clean accuracy score.
- N15 and N16 are P1: Harmony is the production chat path for gpt-oss, and the SWA edges are where every gpt-oss hang of the campaign lived.
- N17 and N18 are P1, N1v/N2v/N5v P1 once N13 is green.
- Cost: about 4.5 hours of 1P1D time on top of section 4, all on 1+1 hosts; N14 and N18 also need a monolithic slot for the goldens, which can be produced once per weights version and cached.

### 6.4 Corner cases carried over from the campaign that no proposed test covers yet

Decode engine death mid-stream through the mini-LB (#1115) beyond N3; load-balancer restart with requests in flight; the abort storm with the radix cache off (F-8 class, now fixed); the exact ViT bucket edge (4,900 vs 4,901 patches); ISL = context + 1 at the 1M shape; both engines on one host sharing RAM (#1269, #1286); the Rust router in `--pd-disaggregation` mode with `--worker-startup-timeout-secs` and no rediscovery. These stay in `sglang_inference/results/TEST_CASES.md` section 5 as the backlog.

## 7. Feature coverage audit (added 2026-10-12)

Scope: every serving feature the TPU backend exposes (server flags, `sglang_tpu/` modules, env knobs) against the registered end-to-end tests at main 2e14706a62. "Nightly" = suites `external-nightly-*`, `nightly-tpu-*` and `tpu-integration` tests registered with `nightly=True`; "PR gate" = `stage-b-test-tpu-4-v7` (runs on PRs, not nightly); "campaign" = covered only by the October live campaign; "none" = no end-to-end coverage anywhere. Unit tests are not counted: they prove the code path, not the served behaviour.

| Feature area | Nightly | PR gate only | Campaign only | None | Proposed |
|---|---|---|---|---|---|
| P/D 1P1D and 2P2D over raiden, bootstrap, transfer | gpt-oss-120b singlehost x5, multihost x6, 480B reshard, VL-8B eagle3 | | | | covered; add failure paths (N1-N3) |
| P/D TP reshard (push-reshard, heterogeneous attention TP) | gpt-oss and 480B reshard rows | | | | covered |
| P/D decode radix cache, KV offload on the decode | singlehost decode_radix, kv_offload rows | | | | covered |
| P/D optimistic prefill, decode allocation policy `prefill_complete`, host receive threshold, extra decode slots | | | | yes | add a policy matrix row to N1 (one run per policy) |
| P/D retraction backup (`cpu_tensor`), rebootstrap | | | TC-8 retraction, boundary (3) | | N7 |
| P/D role switch, pdmux, encoder disaggregation (`encoder_only`/`language_only`), colocated PD (`SGLANG_TPU_PD_COLOCATED`) | | | | yes | not production features yet; one smoke each when they are |
| Prefix cache (radix) on prefill, multi-turn | prefix_cache rows, decode_radix | | O-3 multi-turn depth | | add an 8-turn conversation to N6 (done) |
| HiCache RAM offload over raiden (`hicache_io_backend raiden`, `page_first_fused`, write-through) | kv_offload rows | | eviction + reload bit-exact (TC-3b), decode HiCache (config E) | | add evict-then-reload to N6; write-back policy and `hicache_storage_backend` (lmcache/flexkv) have no test and no TPU use yet |
| HiSparse, session radix cache, unified cache linker, unified memory | | | | yes | not enabled in any recipe; smoke only if adopted |
| EAGLE3 speculation (gpt-oss, 480B, VL-8B) | spec rows, VL-8B eagle3 | | acceptance by workload (TC-4, TC-16), verify-graph recompile stance (7.5) | | N17 |
| DFlash speculation | multihost_dflash, singlehost_dflash | dflash_smoke | | | covered |
| ngram, dspark, domino, decoupled spec, adaptive spec | | | | yes | dspark has unit tests only; add a smoke when any ships on TPU |
| DP attention (dpa), hybrid dpa | dpa rows (gpt-oss, 480B) | | | | covered |
| Pipeline parallel, PP x PCP | | gpt-oss-20b pp4_pcp2 | | | promote to nightly on a 1-host slice (N19) |
| Prefill context parallel (PCP), decode context parallel (DCP, `pallas_tpu_rpa_dcp`) | | qwen3-30b-a3b pcp | | DCP: none | N19 adds a DCP row (#1137 class halts live in this path) |
| Attention backends: `pallas_tpu_rpa_v3` (default), `rpa_batched`, `flash`, `mla`, GDN/KDA | v3 everywhere; MLA/KDA via kimi-k3 compile (nightly oot) | kimi_k3_tiny compile | | `rpa_batched` and `flash` as served backends: none | one GSM8K-20 row per backend in N9 |
| MoE kernels: MXFP4 (gpt-oss), fp8 MoE (480B), fp4 MoE, w4a16 MoE, SparseCore gather, EP local expert reads | gpt-oss and 480B rows | | | fp4 and w4a16: none | add the quantized-MoE models that production will serve |
| KV cache dtypes: bf16, fp8 e4m3, fp8 e5m2, MXINT8 | fp8 e4m3 in bench rows only | | fp8 vs bf16 identity (F-4) | MXINT8 and e5m2: none | N18 |
| Hybrid SWA pool, `swa_full_tokens_ratio`, `--disable-hybrid-swa-memory`, SWA prefix tails | implicit in every gpt-oss run | | TC-2 edges, F-11, F-4 | | N16 |
| Chunked prefill, dynamic chunking, mixed chunk, prefill delayer | chunked prefill implicit | | unaligned lengths | dynamic chunking, mixed chunk, prefill delayer: none | add flags as variants of N16 |
| Scheduling: priority scheduling, preemption threshold, low-values-first, `max_queued_requests`, schedule policies | | | TC-18, queue-limit batteries | | N1 (PD) + a monolithic admission row |
| Request shapes: n>1, logprobs, prompt logprobs, logit bias, stops, penalties, grammar/constrained decoding | | qwen3-0.6b prompt_logprob | TC-6 (19 cases), TC-9 chat | constrained decoding with a real grammar backend: none on TPU | N5 |
| Chat paths: Harmony (gpt-oss) reasoning and tools, Qwen chat template, streaming with usage | `--reasoning-parser` set but only `/generate` and GSM8K exercised | | TC-9 | tool calls: none | N15 |
| Multimodal: Qwen3-VL images (count, resolution, straddle, cache keys), `mm-process-config`, ViT compile buckets, `SGLANG_TPU_MM_EMBED_ON_DEVICE` | VL-8B MMMU-Pro (fixed max_pixels) | | TC-10 on VL-30B | VL-30B-A3B: none | N13, N14 |
| Compile: `--enable-torch-compile`, compile buckets, compile cache (tier-2/tier-3), compile bundle save/load, recompile check, compile tracer | kimi-k3 compile (oot) | qwen3-0.6b compile_tracer, torch_compile_bundle | F-1, F-2, F-10, 0-recompile audits | per-model compiled assertion: none | N9 |
| Overlap schedule, async copy worker, result copy worker, continuous decode steps | implicit defaults | | | disabled-overlap variant: none | one variant row in N9 |
| KV free fence, host heap trim, host metadata cache, req_to_token on host | unit tests only | | fence hold observed (1 of 4 runs) | | N7 records fence pages at the end; N8 records host RSS |
| KV integrity: `--kv-canary`, `--disaggregation-enable-kv-checksum` | | | both unusable on TPU (F-13, F-14) | | N10 after the runtime lands |
| Health, metrics, Perfetto PD tracing, profiler | metrics scraped by the perf bench only | | dashboard panels (37) | health contract: none | N8 |
| Load balancers: mini-LB (CI), Rust PD gateway | bench rows behind the mini-LB | | mini-LB error paths (F-15) | Rust router in PD mode: none | N11 |
| LoRA | | | | yes | not supported on TPU; keep out of scope |
| Diffusion (Wan2.2 TI2V, multihost collectives) | suite="external-nightly-1host" / suite="tpu-integration" | | | | outside P/D; owned by the diffusion track |
| Long context: 128k, 256k, 1M recipes | 128k only | | 1M (TC-12) | 256k/1M nightly: none | N6 |
| Upgrades: torch_tpu, libtpu, JAX wheels | pinned | | | drift: none | N12 |

Summary: of 32 feature areas, 12 are covered nightly on their happy path, 4 only by the PR gate, 7 only by the live campaign, and 9 have no end-to-end coverage at all. Everything in the "campaign only" and "none" columns that production will enable is already assigned to a proposed test above; the areas with no proposed test are features not yet used in any production recipe (pdmux, encoder disaggregation, HiSparse, LoRA, ngram/dspark speculation), which should get a one-row smoke the day a recipe turns them on.

| # | Test | What it does | Pass criteria | Est. |
|---|---|---|---|---|
| N19 | `pp_pcp_dcp_matrix` | gpt-oss-20b at PP4 x PCP2 and PP2 x TP2; gpt-oss-120b with DCP on the decode; GSM8K-50 and a 32k needle each | accuracy within 0.02 of TP-only; needles 3/3; no halt (#1137 class) within the watchdog | 45 min |
