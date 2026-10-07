# Findings log — sglang-torchtpu 1P/1D break-test

Build: sglang-torchtpu main `09d1244c26` (2026-10-07), sglang 0.5.13.dev7249, torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151. Hosts: wyzhang-tpu7x-16 a/b (tpu7x, 4 chips / 8 cores each, single-host isolation).

| # | Date (UTC) | Severity | Area | Status |
|---|---|---|---|---|
| F-1 | 2026-10-07 19:35 | Low (env/robustness) | launcher + torch_tpu tier-2 compile cache | observed, repro 1/1, workaround applied |

## F-1 — Hard abort (SIGABRT) when the shared tier-2 compile-cache root is not writable

**Symptom.** Monolithic Qwen3-VL-30B launch (`benchmark/tpu/launch_server.py`, recipe `qwen3-vl_G_128k_fp8_e4m3.json`) died during scheduler init:
`RuntimeError: Rank 0 scheduler died during initialization (exit code: -6)`. The real cause is 8× (one per rank):
```
F0000 ... tier2_compilation_cache.cc:241] Check failed: status.ok() Failed to create tier-2 cache directory:
/dev/shm/torch_tpu_cache/pd_golden_qwen3-vl_G_128k_fp8_e4m3/ee8e2de63f954d59 with error: ...
```
`/dev/shm/torch_tpu_cache` already existed (mode 755, owned by another user) on this shared host.

**Why it matters.** `benchmark/tpu/launch_server.py:prepare_runtime_env` pre-creates the tier-2 directory inside `try/except OSError: pass` ("the tier-2 cache is optional"), but torch_tpu treats the failure as fatal (`CHECK`), so the engine aborts with a misleading "scheduler died / maybe OOM" message. On a multi-user host every second user's launch fails this way.

**Repro.** `sudo chmod 755 /dev/shm/torch_tpu_cache` (owned by someone else) → launch any recipe with `enable_torch_compile` → abort within ~10 s of weight loading start.

**Workaround.** `sudo chmod 1777 /dev/shm/torch_tpu_cache` on both hosts (done 19:38 UTC). Alternative: `--runtime-envs '{"TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": null}'`.

**Suggested fix.** Launcher: if the tier-2 directory cannot be created, either unset `TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE` or fail fast with a clear message. torch_tpu: fall back to no tier-2 cache instead of `CHECK`.

Log: `results/runs/G_qwen_128k_fp8/golden/server.attempt1_tier2cache_abort.log`.

| F-2 | 2026-10-07 19:59 | High (perf / TC-5 scope) | torch.compile coverage, Qwen3-VL-30B-A3B | observed, repro 1/1 |

## F-2 — Qwen3-VL-30B-A3B language model runs eager on TPU (no decoder compile policy)

**Symptom.** Monolithic Qwen3-VL-30B-A3B golden (TP8/EP8, `--enable-torch-compile`, recipe `qwen3-vl_G_128k_fp8_e4m3.json`): decode runs at
`gen throughput 12.7 tok/s` with 2 running requests (≈160 ms per step), 1,024-token outputs take 171–194 s. gpt-oss-120b on the identical stack
decodes at 176 tok/s for one request (5.7 ms/step). The Qwen log shows only `VisionTpuCompilePolicy: installed qwen3_vl_vit_encoder ...`
(29 patch-count buckets) and no decoder policy; `TpuGraphRunner precompile done` fires after the vision warm-up.

**Cause (code).** `python/sglang/srt/models/qwen3_vl.py:1380-1388` attaches `Qwen3VLDecoderTpuCompilePolicy` only to the dense `Qwen3LLMModel`;
the MoE variant `Qwen3MoeLLMModel` in `models/qwen3_vl_moe.py` carries no policy. Because the vision policy installs, `TpuGraphRunner`
(`model_executor/tpu_graph_runner.py:216-295`) skips the whole-model `DecoderTpuCompilePolicy` fallback, so the MoE decoder never compiles.

**Impact.** For this model TC-5 ("zero decoder recompiles") is vacuous and any TPOT/TTFT gate (TC-8) is meaningless; 1M-token PD runs will be
decode-bound at ~6 tok/s per request. Not a correctness bug, but it hides compile-related bugs that the gpt-oss path would surface.

**Suggested fix.** Register a decoder compile policy for `Qwen3MoeLLMModel` (mirroring `qwen3_vl.py:1380`) or let the graph runner apply the
whole-model fallback when only a vision policy is installed. Logs: `results/runs/G_qwen_128k_fp8/golden/server.log`, `results/golden/qwen3-vl_fp8/run.log`.

| F-3 | 2026-10-07 20:02 | Medium (test-methodology / numerics) | greedy decode reproducibility on TPU (monolithic gpt-oss) | observed 10/10 prompts |

## F-3 — Greedy outputs are not batch-invariant: golden vs golden diverges after 3–59 tokens

**Symptom.** Same monolithic gpt-oss-120b server, same prompts, `temperature=0`, `ignore_eos`, 1,024 new tokens. Run 1 at concurrency 2, run 2
at concurrency 1. All 10 prompts (ISL 1k, 8k, 32k, 128k, 127,999) produce different text; token-exact common prefix 3–59 tokens; first-token
logprob identical in 4/5 sets (|Δ| = 0.0247 at 128k). Report files: `results/golden/gpt-oss_fp8_rep2_c1/determinism_*.json`.

**Interpretation.** The prefill numerics are stable (first token identical); divergence happens in decode when the batch composition differs
(bs=2 vs bs=1 → different padding/kernels; MoE top-4 routing flips on tiny logit differences, then free-form text diverges chaotically).
Whether the stack is reproducible at a *fixed* batch size is still to be measured (c=1 vs c=1 repeat queued on the next golden launch and on PD).

**Impact on the plan.** The "token-exact first 256 tokens / ≥95 % agreement" differential gate is only meaningful when golden and PD runs use
identical batch composition (concurrency 1) — and even then only if the c1-vs-c1 repeat is identical. For concurrent runs the gates must be:
needle answers identical (they are the first ~30 tokens), finish_reason/completion_tokens exact, structural checks, and GSM8K accuracy
(golden: 0.99 on 100 examples). `--enable-deterministic-inference` (batch-invariant ops) exists as a flag; test whether TPU honours it.

| F-4 | 2026-10-07 20:20 | **High (correctness, under isolation)** | PD prefill vs monolithic for multi-chunk prompts (gpt-oss) | observed 6/6 long prompts, 0/4 short; isolating |

## F-4 — PD output diverges from monolithic only for prompts longer than one prefill chunk (gpt-oss, Config A)

**Observation.** Config A (1P1D raiden, fp8 KV, page 64, chunk 8192, radix off) vs monolithic golden, both at concurrency 1, greedy, 1,024 new tokens
(`results/A/tc1/gpt-oss/golden_diff_c1_*.json`):

| ISL | chunks | identical over 1,024 tokens | token prefix | first-token logprob Δ |
|---|---|---|---|---|
| 1,024 (×2) | 1 | **yes / yes** | 1024 | 0.0000 |
| 8,192 (×2) | 1 | **yes / yes** | 1024 | 0.0000 |
| 32,768 (×2) | 4 | no / no | 3 / 3 | 0.104 / 0.156 |
| 131,072 (×2) | 16 | no / no | 5 / 15 | 0.032 / 0.031 |
| 127,999 (×2) | 16 | no / no | 19 / 3 | 0.0004 / 0.005 |

Monolithic golden is itself reproducible for the first token across concurrency 1 vs 2 (Δ = 0.0000 at 32k, 0.025 at 128k, F-3), so a 0.10–0.16
first-token logprob shift at 32k is outside the monolithic noise. The first token is computed on the **prefill** engine, so the difference is in the
PD prefill forward, and it appears only when the prompt spans ≥ 2 chunks. All needles are retrieved on both sides (semantic gate passes).

**Hypotheses to isolate (queued).** (a) sliding-window KV across chunk boundaries in PD prefill (SWA pool / "window rides on the closing read");
(b) chunked-prefill path differences in prefill mode (chunk cache vs radix-off monolithic); (c) attention kernel block sizes chosen from a
different `max_kv`. Isolation steps: PD self-repeat at c=1 (is PD deterministic → systematic?), Qwen3-VL (no SWA) first-token logprob vs golden,
`--disable-hybrid-swa-memory` variant, 2-chunk prompts (8193, 16384) with tiny goldens, GSM8K-100 through PD for accuracy impact.

**Update 20:27 UTC.** PD self-repeat at c=1 (`results/A/tc1/gpt-oss/rep2/pd_self_*.json`): 32k and 128k outputs are token-identical over all
1,024 tokens on repeat → PD is deterministic at fixed batch; the PD-vs-monolithic difference for multi-chunk prompts is **systematic**.

| F-5 | 2026-10-07 20:32 | **High (crash: single request kills the prefill engine)** | TPU compiled backend / request validation | observed 1/1, repro queued |
| F-6 | 2026-10-07 20:28 | **High (hang, candidate)** | `n=2` parallel sampling through PD | observed 1/1 (300 s no response), repro queued |
| F-7 | 2026-10-07 20:33 | Medium (resource leak after peer death, TC-7) | decode engine after prefill crash | observing |

## F-5 — A request with input/prompt logprobs crashes the whole prefill engine (all 8 TP ranks)

**Symptom.** `POST /generate` through the LB with `return_logprob=true, logprob_start_len=0, top_logprobs_num=5` (params_matrix case
`return_logprob_top5`). All 8 prefill schedulers log `Scheduler hit an exception` and exit; the launcher then stops the mini-LB; every later
request gets HTTP 500. Decode engine survives.
```
File ".../sglang/srt/hardware_backend/tpu/compilation/compiled_forward_shim.py", line 967, in _forward
    raise NotImplementedError(rejection)
NotImplementedError: input/prompt logprobs (echo) need SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1 on the TPU compiled backend;
only output (next-token) logprobs are served by default.
```
(path: prefill.py:773 event_loop_overlap_disagg_prefill → run_batch → eager_runner._execute_extend → compiled_forward_shim._forward)

**Why it matters.** An unsupported *per-request* option is rejected in the model forward, i.e. after scheduling, where the only outcome is a
scheduler crash. It must be rejected at request validation (HTTP 400) or the server flag must be forced on/off at startup. Any client can take
down a 500P deployment's prefill engines with one request. Expected per plan Section 0: clean "unsupported" error, never a crash.

**Repro.** Config A gpt-oss; `curl LB/generate -d '{"text":"hi","sampling_params":{"max_new_tokens":4},"return_logprob":true,"logprob_start_len":0}'`.
Also to test: same request against the monolithic server (likely identical crash → not PD-specific) and with `SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1`.
Log: `results/runs/A_gptoss_128k_run1/prefill/server.attempt1_crash_prompt_logprob.log`.

## F-6 — `n=2` request through PD: no response for 300 s (candidate hang)

params_matrix case `n_2` (`sampling_params.n=2`, temperature 0, 128 tokens) returned nothing within the 300 s timeout while every other case
before it answered in < 2 s. Whether the request was still running when F-5 killed the prefill is being reconstructed; dedicated repro queued
(with the watchdog recording `num_running_reqs`/PD queue gauges, against PD and against monolithic).

## F-7 — After the prefill engine died, the decode engine keeps one request in its transfer queue

Decode `/metrics` 5 min after the prefill crash: `num_running_reqs=1, num_decode_transfer_queue_reqs=1, num_transfer_failed_reqs_total=1,
kv_transfer_oldest_inflight_age_seconds=300 (growing), num_used_tokens=16512`; log: `Lost connection with prefill instance (bootstrap_addr:
10.128.15.217:8998), 0 requests affected` on every rank — yet one request stays in-flight. Our config sets
`SGLANG_DISAGGREGATION_WAITING_TIMEOUT=7200`, `SGLANG_RAIDEN_TIMEOUT_S=600`. Observing whether it clears at 600 s / on prefill reconnect
(`results/A/tc7_prefill_death_decode_state.log`).

**F-7 resolution (20:34–20:48 UTC poll, `results/A/tc7_prefill_death_decode_state.log`).** The parked request cleared within ~2 min of the prefill
death (`num_transfer_failed_reqs_total` 1→2, running/transfer queues → 0, used tokens → 0) — i.e. decode recovered without the 600 s raiden or
7200 s waiting timeout. Downgraded to **observation**: only the log line "Lost connection with prefill instance … 0 requests affected" is misleading
(one request was affected). Not a bug report.

**F-6 status.** Dedicated repro could not run (LB was down after F-8). Still a candidate; repro re-queued.

| F-8 | 2026-10-07 20:43 | **High (crash: client abort during chunked prefill kills the prefill engine)** | prefill engine, chunk cache (radix off) | observed 1/1 (first abort of TC-7), repro queued |

## F-8 — Client abort during a chunked prefill crashes all prefill schedulers (`AssertionError: Only a mamba-capable tree cache allows freeing before alloc`)

**Symptom.** Config A gpt-oss (radix cache disabled on prefill → ChunkCache). The first TC-7 client-abort case (`abort_test.py --phase prefill
--abort-after 2` on a 131,072-token prompt: the HTTP stream is closed 2 s into prefill) made all 8 prefill schedulers exit at 20:43:28 with
`AssertionError: Only a mamba-capable tree cache allows freeing before alloc`; the launcher then killed the group (`exited with -9`), the LB went
down, every later request failed (aborts, GSM8K, F-6/F-5 repros of that chain are therefore void). Decode engine survived.
Preceding load: the 8×128k retraction test had just completed successfully (0 errors, all needles found).

**Why it matters.** A client disconnect is a routine event; here it takes down the prefill engine. Plan Section 0 "Crash" definition and TC-7
"Client abort" expectation ("both servers release resources within 60 s") are violated. Likely the abort path frees a request's KV pages before
the chunked-prefill allocation completed, and the non-radix ChunkCache asserts.

**Repro (queued, run LAST in a session).** `abort_test.py --url LB --prompt-file prompts/gpt-oss/131072.jsonl --phase prefill --abort-after 2 --repeat 1`
on Config A. Also to test: Config B (radix on) — does the same abort crash the radix tree cache? and the monolithic server.
Log: `results/runs/A_gptoss_128k_run1/prefill_server.attempt2_abort_crash.log` (traceback below once extracted).

## Config A gpt-oss edge-chain summary (20:37–20:43 UTC, before F-8)
- params matrix 17/17 PASS (json_schema/regex → clean 400; max_new_tokens 0 and 1 fine; penalties/stop/logit_bias OK). `n=2` and prompt-logprobs excluded (F-6/F-5).
- Concurrency sweep ISL 131072/OSL 256: c=1 TTFT 7.0 s, ITL 5.4 ms; c=2 TTFT 11.2 s (p99 13.8), ITL 6.0; c=4 TTFT 21.2 s (p99 27.0), ITL 6.1 ms, p99 10.6. 0 failures.
- Retraction test 8×131072 @ c=8, 1,024 new tokens: 8/8 complete, needles 3/3 each, 0 errors; `num_retracted_reqs` gauge stayed 0 (pool 786k vs 8×132k — the decode admitted ≤ 5 at a time instead of retracting; outputs differ from the c=1 run as per F-3).
- Compile audit decode: 0 serving-time recompiles across the whole session (24 startup).
- chat_suite: tool bug (KeyError) — fixed, rerun queued.

**F-8 traceback (TP0, identical on all ranks):**
```
prefill.py:757 event_loop_overlap_disagg_prefill → prefill.py:672 get_next_disagg_prefill_batch_to_run
→ scheduler.py:3734 process_pending_chunked_abort → scheduler.py:3795 _drain_deferred_chunked_sender_aborts
→ release_kv_cache(req, self.tree_cache, is_insert=False)   (mem_cache/common.py:298)
AssertionError: Only a mamba-capable tree cache allows freeing before alloc
```
Context lines: the aborted request (`rid=abort-0-…`) was mid-chunk (`#pending-token: 78976`, chunks of 8064/2880 tokens alternating — note the
odd chunking of a 131,072-token prompt into 8064+2880 pieces); TokenizerManager logged "Received output for rid=… but the state was deleted".
Root cause: the deferred chunked-sender abort path frees the request's KV before its allocation is complete; `ChunkCache` (radix disabled) asserts.
Expected to also affect Config B/C (radix tree) unless `supports_mamba()` differs — to verify.

**F-4 isolation update (Qwen3-VL Config A vs monolithic golden, first-token logprob = prefill-engine output; `results/phase_A_qwen.log`).**

| ISL | chunks | aligned to page 128 / chunk 8192 | first-token logprob Δ (k0 / k1) |
|---|---|---|---|
| 1,024 | 1 | yes | 0.0 / 0.0 |
| 8,192 | 1 | yes | 0.0 / 0.0 |
| 32,768 | 4 | yes | 0.0 / 0.0 |
| 131,072 | 16 | yes | 0.0 / 0.0 |
| 127,999 | 16 | **no** | 0.0114 / **0.2278** |

Qwen3-VL has no sliding-window layers. Its PD prefill matches monolithic **bit-exactly** for every page/chunk-aligned prompt, including 16-chunk
prompts → chunked prefill in PD mode is numerically identical to monolithic per se. Therefore the gpt-oss divergence at aligned 32k/128k (F-4
table) is attributable to the **hybrid sliding-window KV path** in PD prefill (hypothesis (a)); the `--disable-hybrid-swa-memory` run will confirm.
A second, separate effect exists for **unaligned prompt lengths** (127,999 on both models): PD prefill differs from monolithic when the last
chunk/page is partial — candidate F-9 (to confirm against the c=1 Qwen golden for 127,999, queued).

| F-9 | 2026-10-07 22:10 | Medium (correctness, candidate) | PD prefill vs monolithic for page/chunk-unaligned prompt lengths (both models) | observed 4/4 unaligned prompts; exact check vs c=1 golden queued |
| F-10 | 2026-10-07 22:13 | **High (crash: one large image kills the Qwen prefill engine)** | ViT compile buckets × `fail_on_recompile` stance | observed 1/1, repro queued |

## F-9 — Unaligned prompt lengths: PD prefill differs from monolithic (candidate)

For prompt lengths that are not multiples of the page/chunk size (127,999 on both models) the PD first-token logprob differs from the monolithic
golden (Qwen: Δ 0.011 / 0.228; gpt-oss: Δ 0.0004 / 0.005 — see F-4 tables), whereas every aligned Qwen prompt (1k…128k, up to 16 chunks) is
bit-identical. Suspects: handling of the partial last page/chunk in PD prefill (padding of the final extend chunk, or transfer/pre-allocation of a
partially filled page). Exact confirmation against the concurrency-1 Qwen golden for 127,999 and the tiny set (8191/8193/16383/16385) is queued
(`_phase_goldens_1m_and_F4.sh`).

## F-10 — A single 1920×1080 image crashes the Qwen3-VL prefill engine (ViT recompile vs `fail_on_recompile`)

**Symptom.** Config A Qwen3-VL-30B (default `--mm-process-config`, i.e. no `max_pixels` cap). One `/v1/chat/completions` request with one
1920×1080 PNG (mm_request.py) → HTTP 500 after 65 s; all 8 prefill schedulers exit; the launcher kills the group; LB down.
```
[TP0] ViTTpuCompileRunner: patch count 8160 exceeds the largest precompiled bucket 4900; this image triggers a one-time ViT recompile.
      Raise SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES, or lower max_pixels via --mm-process-config.
[TP0] Scheduler hit an exception: ... qwen3_vl.py:1801 forward → mm_utils.py:766 general_mm_embed_routine → mm_schedule.py:354 _encode_vision_items
      → self.visual(pixel_values, grid_thw) → RuntimeError: Detected recompile when torch.compile stance is 'fail_on_recompile'.
```
The stance was armed at 21:25:32 ("Warmup complete; froze torch.compile via set_stance('fail_on_recompile')"). The ViT runner announces an
intended one-time recompile, but the frozen stance turns it into a scheduler crash. 1×448², 4×448², 16×448² (3,198 tokens) and 1×1024² worked.

**Why it matters.** Any client image larger than the precompiled ViT bucket range (default max 4900 patches ≈ 1960×1960 px total area, but the
non-square 1920×1080 maps to 8160 patches) takes down the prefill engine. Plan TC-5 expects "vision-encoder compiles allowed on first sight of a
new resolution"; TC-1 expects a clean 400/413 for oversized payloads, never a crash.

**Repro.** `mm_request.py --url LB --model qwen3-vl-30b --num-images 1 --resolution 1920x1080 --max-tokens 48` on Config A Qwen (queued LAST in
`_phase_A_qwen_redo.sh`, with a second non-square 896×504 case). Mitigations to test: `--mm-process-config '{"image":{"max_pixels":...}}'` (the 8B
recipe sets 313600) → request should be downscaled; `SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES` higher; or exempt the ViT runner from the stance.
Log: `results/runs/A_qwen_128k_run1/prefill_server.log` (line ~7648 ff).

## Config A Qwen3-VL summary (21:25–22:16 UTC)
- Cold start 452 s (prefill). TC-1 c=1: 10/10 outputs 1,024 tokens, needles 3/3 on all; text diverges from the c=2 golden after 48–208 tokens (F-3 noise; c=1 golden queued).
- First-token logprob PD vs golden: bit-identical for all aligned ISLs (1k…128k); differs only at 127,999 (F-9).
- Tiny boundary sweep (page 128): 35/35 prompts 160 tokens, 0 errors, needle found in all 32 that have one (contrast: gpt-oss 7 misses → model/prompt-format effect, not PD).
- Multimodal: 1/4/16 × 448², 1 × 1024² OK (TTFT 15–35 s eager); 16 images: needle not echoed in 192 tokens (semantic, not an error); 1920×1080 → F-10 crash; remaining mm/chat/abort items voided → redo queued.
- Decode compile audit: 0 serving-time decoder recompiles (vacuous, F-2); 8 ViT compile events on decode during serving (new resolutions on the decode side as well — decode re-encodes images for MRoPE).
- verify_pd_disagg needle probes with raw text returned empty (model emits EOS at once for the terse raw prompt; `ignore_eos` runs retrieve fine) → use `--api chat` for instruct VL models.
