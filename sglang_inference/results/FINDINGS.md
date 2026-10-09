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

**F-4 / F-9 isolation update (23:20 UTC).**
- Server shape is not a factor: gpt-oss goldens from the 128k-shape and the 1m-shape servers (both c=1) are token-identical at 32k.
- Qwen3-VL PD (Config A, c=1) vs Qwen monolithic golden (c=1): **token-identical over 1,024 tokens for 8/8 prompts at 1k, 8k, 32k, 128k**.
- Tiny boundary set (1…16,385 tokens, 160 new tokens): PD identical to golden for **35/35 lengths on both models**, including gpt-oss 8,193 and 16,385 (2–3 chunks).
- Prefill chunk schedule differs between modes for gpt-oss only. Histogram of `#new-token` per prefill step for the same 131k prompts:
  PD prefill engine: 8192 ×239, 2752 ×185, 2880 ×92, 8064 ×83; monolithic (128k shape): 8064 ×85, 1344 ×81, 8192 ×40. Qwen uses 8192 everywhere.
  The chunk size is capped by free slots in the small hybrid-SWA pool (`token_capacity_swa` = 6,144 tokens at the 128k shape), and the two modes
  consume that pool differently (the PD prefill holds SWA pages for the transfer). Different chunk boundaries → different accumulation order in
  bf16/fp8 → different argmax after a few tokens. This explains why only gpt-oss (hybrid SWA) and only long prompts diverge, and why short
  multi-chunk prompts (8,193 / 16,385) do not.
- Working classification of F-4: a **numerical** difference caused by a mode-dependent chunk schedule, not a functional KV-transfer bug. The
  `--disable-hybrid-swa-memory` run (queued) should make both schedules 8192-regular and the outputs identical; GSM8K through PD will show
  whether accuracy is affected. Severity stays "high" until both checks are in.
- F-9: the c=1 Qwen golden for 127,999 was not generated (edit landed after the phase script had started); re-queued as a separate step.

**Observation O-1 (2026-10-07 23:14 UTC, PASS for the plan's "clean start-up failure" case).** Config A gpt-oss with the 1M shape
(`context_length 1056768`) and the default `SGLANG_RAIDEN_HOST_POOL_GB=128` refuses to start on both engines with a clear message:
`ValueError: raiden host staging: one transfer slot needs 9.07 GiB but this engine's share of SGLANG_RAIDEN_HOST_POOL_GB is 8.00 GiB; raise
SGLANG_RAIDEN_HOST_POOL_GB or lower the context length` (128 GiB / 8 ranks / 2 engines = 8 GiB per slot). Capacity rule for the plan: the
raiden host pool must be ≥ 16 × (context_length × KV bytes/token/rank) → for 1M gpt-oss ≈ 9.1 GiB per slot → set 320 GiB. Recipes for 512k/1M
now set 256/320 GiB. The driver lost ~90 min in its health wait; not a product bug.

**Observation O-2 (2026-10-08 01:00 UTC, operational).** `~/.cache/sglang/inductor_dp1_dprank0_tp8` (the TorchInductor cache SGLang sets per
TP layout) grew to 23 GB on host a and 17 GB on host b after ~10 server launches (≈2–4 GB per launch, 272–436 files, never pruned). Together with
the tier-3 compile cache (12 GB) this filled the 96 GB root disk of host a to 100 % and would have broken the next launch and the 60-minute soak
(TC-6). Mitigation in the harness: `TORCHINDUCTOR_CACHE_DIR` moved under `compile_cache/`, and `launch_role.sh` prunes it when < 20 GB is free.
Worth a note upstream: a size cap or LRU for the inductor cache on TPU.

| F-11 | 2026-10-08 01:35 | **High (hang: prefill stalls forever on prompts ≥ ~16k tokens in the 1M-context config)** | PD prefill, hybrid-SWA KV pool sizing | observed 4/4 requests ≥ 22k tokens; recovered only via client abort |

## F-11 — PD prefill engine hangs on any prompt longer than its sliding-window pool (gpt-oss, 1M-context recipe)

**Symptom.** Config A gpt-oss with the 1M shape (`recipes/gpt-oss_A_1m.json`: context 1,056,768, pool 1,114,112 tokens, `max_running_requests 1`,
raiden host pool 320 GiB). Requests of 1k and 8k tokens work. Every request of ~22k tokens or more never returns: the prefill logs two
chunks (`#new-token: 8192`, then `8000`), then no further `Prefill batch` line for 30 minutes, until the client's 1,800 s timeout aborts the
request (HTTP 500 via the mini-LB). During the stall: `num_running_reqs=1`, `full_token_usage=0.0145`, **`swa_token_usage=0.988`**,
`num_used_tokens=16192`, `max_total_num_tokens_swa=16384`. Both engines stay healthy; after the abort the prefill returns to 0 usage.
The monolithic server with the same 1M shape served the same 256k/512k/1M prompts without problems (`results/golden/gpt-oss_fp8_1m_c1`).

**Mechanism (high confidence).** The hybrid sliding-window (SWA) KV pool of this configuration holds only 16,384 tokens. In PD prefill mode the
SWA pages of the prompt are not released as the window slides (they are kept for the P→D transfer of the window), so a single request consumes
SWA slots in proportion to its length. After 16,192 tokens the next chunk cannot be allocated; with one running request nothing can be evicted,
and the scheduler waits forever instead of failing the request. In the 128k-shape recipe the SWA pool was 6,144 tokens and 128k prompts
worked, with `swa token usage` cycling 0.58→0.84, so page release does happen there; the difference between the two shapes
(`max_running_requests` 8 vs 1, context 140k vs 1M) is the next isolation target.

**Why it matters.** This is the plan's hang definition (a): a running request with no progress for > 300 s. It makes the 1M-context target
unreachable with this recipe, and a stalled prefill blocks the only running slot indefinitely (no watchdog fires because no forward step is running).

**Repro.** `bash scripts/pd.sh start recipes/gpt-oss_A_1m.json f11` → `python scripts/verify_pd_disagg.py --url http://10.128.15.217:8080 --router --lengths 32768 --timeout 600`
→ no answer in 600 s; `curl $PREFILL/metrics | grep swa_token_usage` shows 0.99. Evidence: `results/F11/`, `results/runs/A_gptoss_1m_run2/decode/server.log`,
prefill log on host a `runs/A_gptoss_1m_run2/prefill/server.log`.

**Expected.** Either size the SWA pool from the context length (a 1M context needs more than 16k SWA slots, or page release during chunked
prefill), or reject the request with a clear error, or let the watchdog fail a request that makes no progress.

**Next isolation (queued):** same 1M shape with `max_running_requests 4` (does the SWA pool scale with the running-request count?), and the
128k-shape recipe with `max_running_requests 1`.

**Repro results, gpt-oss Config A redo (2026-10-08 02:52–03:07 UTC, `results/phase_A_gptoss_redo.log`).**
- **F-5 confirmed (2/2):** one prompt-logprob request → HTTP 500 after 65 s, prefill engine dead (`/health` 000). Relaunched.
- **F-8 confirmed (2/2):** one client abort 2 s into a 131k prefill → prefill engine dead. Decode-phase aborts (6, two concurrent) **PASS**: both engines back to baseline, follow-up requests OK.
- **F-6 confirmed (3/3):** `n=2` with temperature 0 and with 0.7 → no response in 120 s each; during the stall the decode engine shows `num_running_reqs=1`, `num_used_tokens=128`
  (one of the two samples finished its 32+... tokens and sits there), prefill idle. Both engines stay healthy; the stuck request remains until the client disconnects.
  Reclassified from candidate to **confirmed hang**.
- **GSM8K-100 through PD: 0.98** (monolithic golden 0.99; CI gate 0.80). → F-4 has no measurable accuracy impact; F-4 downgraded to a numerical
  difference (P2) with the chunk-schedule cause documented.
- Chat suite 10/11: Harmony reasoning low/medium/high, multi-turn, stop strings, streaming with usage, 6k system prompt, oversized max_tokens → 400 all PASS.
  `tool_call` returned no `tool_calls` (finish_reason stop, 61 reasoning tokens, empty content): the recipe sets `reasoning_parser gpt-oss` but no
  `--tool-call-parser`, so this is a configuration gap of the test recipe, not a PD defect; re-test with `--tool-call-parser gpt-oss` when the recipe is next restarted.
- KV transfer bootstrap: 114 transfers, mean bootstrap ≈ 1 ms.

**F-6 code-level suspects (not yet proven).** For `n>1` the tokenizer manager expands one HTTP request into `n` sub-requests with new rids
(`tokenizer_manager.py` ~L2037-2050). Bootstrap parameters are normalised in `io_struct.py::_normalize_bootstrap_params` (L886-910): an integer
room becomes `[room+i for i in range(num)]`, but a **list** room is repeated with `list * parallel_sample_num`, i.e. every parallel sample of the
same prompt gets the **same bootstrap room**. The mini-LB sends a list of rooms for batched requests (`mini_lb.py` L495-505) and a single int
otherwise (L507-513). Decode-side state during the hang (1 running request, 128 used tokens = 2 pages, no throughput) is consistent with the two
samples colliding in the bootstrap/transfer bookkeeping or with the LB waiting for a single response object. To pin it: send `n=2` directly to the
prefill and decode engines with explicit distinct rooms (bypassing the LB) and with an explicit shared room.

**F-4 resolved (2026-10-08 03:27 UTC): PD prefill is numerically identical to monolithic once the hybrid-SWA pool is out of the picture.**
Paired comparison with `--disable-hybrid-swa-memory` on both the PD pair (`runs/A_gptoss_128k_noswamem`) and the monolithic golden
(`golden/gpt-oss_fp8_noswamem_c1`), both at concurrency 1, 1,024 new tokens: **identical for 6 of 6 prompts at 8k, 32k and 128k**
(`results/A_noswamem/golden_diff_noswamem_c1_*.json`). Control: the monolithic server with SWA memory on vs off differs at 32k
(token prefix 24 / 3), so the SWA-memory mode itself changes the numerics. Together with the chunk-size histograms (PD 8192/2752/2880/8064
vs monolithic 8064/1344/8192 with SWA memory on; uniform 8192 with it off) and GSM8K through PD = 0.98 (monolithic 0.99), F-4 is a
**configuration-dependent numerical difference, not a PD defect**: the small hybrid-SWA pool makes the prefill chunk schedule depend on the serving
mode, and different chunk boundaries change bf16/FP8 accumulation order. Severity: low / informational. Worth an upstream note so that
PD-vs-monolithic comparisons for hybrid-SWA models are done with `--disable-hybrid-swa-memory` or with a pool large enough for full chunks.

**F-1 extension (2026-10-08 03:09 UTC).** Launching with `ENABLE_COMPILE_CACHE=0` (the launcher then sets no `TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE`)
does **not** disable the tier-2 cache: torch_tpu falls back to the name `default`, i.e. `/dev/shm/torch_tpu_cache/default/<hash>/`, which on a
shared host belongs to whoever launched first. Our server aborted on all ranks with
`tier2_compilation_cache.cc:266 Check failed: lock_fd_.valid() Failed to create tier-2 cache entry lock file: /dev/shm/torch_tpu_cache/default/.../*.lock with error: Permission denied`.
So two users with default settings on one host cannot both run servers, and "compile cache off" is not a work-around. Added to the F-1 report.

**F-2 patch, first real run (2026-10-08 03:29–03:37 UTC).** With `patches/F-2_qwen3_vl_moe_decoder_compile_policy.patch` applied, the server logs
`[TPU] [decoder compile policy] Installed Qwen3VLDecoderTpuCompilePolicy on Qwen3MoeLLMModel.forward (t-bucketer=TokenBucketer, b-bucketer=BatchSizeBucketer)`
right after the 29 vision buckets, and starts the decoder precompile. The run then died of disk exhaustion (O-2 extension below), so speed and
token-identity numbers are still pending; rerun in progress with caches on tmpfs.

**O-2 extension.** Compiling the Qwen3-VL-30B MoE decoder wrote ≈20 GB of TorchInductor cache, 6.1 GB of tier-3 cache and 7.8 GB of tier-2
(/dev/shm) in ~8 minutes before `No space left on device` killed the schedulers (`AOTAutograd cache unable to serialize compiled graph: [Errno 28]`,
`compilation_cache.cc: RESOURCE_EXHAUSTED: failed to write to file`). A 96 GB root disk cannot host more than two or three such launches. The
harness now puts both caches under `/dev/shm/alisachen_cache/` (RAM-backed, 473 GiB). Upstream: document the cache footprint or cap it.

**F-2 patch validated on hardware (2026-10-08 04:02–04:23 UTC, `results/F2_patch/`, run `G_qwen_128k_fp8_F2patch`).**
- Policy installs: `[decoder compile policy] Installed Qwen3VLDecoderTpuCompilePolicy on Qwen3MoeLLMModel.forward`; `Capture target decode TPU graph end. elapsed=1094 s`.
- Cold start with the patch ≈ 19 min (04:02:30 → 04:21:48), of which 18 min is the decoder graph capture (gpt-oss: 4 min). Compile artefacts: 65 GB inductor + 51 GB tier-3 + tier-2.
- **Decode speed 178–183 tokens/s for one request (eager: 6–13); 1,024-token answers in 5.9–7.1 s (eager: 160–195 s).** TTFT 48 ms–1.5 s (1k–32k).
- Serving-time recompiles across the 35-length boundary sweep: **0** (startup recompiles 616 during bucket capture).
- Outputs are **not token-identical** to the eager server: divergence after 9–83 tokens, first-token log-prob Δ 0.03–0.25; tiny set 21/35 identical.
  Compiled and eager kernels differ numerically (expected), so an accuracy check is the right gate: GSM8K-100 compiled vs eager is queued.
- Patch reverted; tree clean. Recommendation: upstream the registration; also budget compile time and cache space for this model.

**F-11 isolation (2026-10-08 05:06–05:28 UTC, `results/F11/`) and root cause from code.**
- 1M shape with `max_running_requests 4` (SWA pool 17,920 tokens): 8k, 32k and 131k prompts all complete (7–13 s). The pool cycles 0.54→0.91→0.54.
  So neither the context length nor the pool size per se is the trigger; `max_running_requests 1` is.
- SWA pool size comes from `SWAChunkCapPoolConfigurator` (`pool_configurator.py`, used when radix cache is off and `max_running_requests` is explicit):
  `cap = 2 × chunked_prefill_size + (window 128 + eviction_interval 128 + page 64 + decode page 64) × max_running_requests + page 64`
  → 16,832 (mrr 1), 17,984 (mrr 4), 19,520 (mrr 8). Observed logs match exactly.
- TPU-only rounding (`kv_cache_configurator.py`: "SWA page count must be a multiple of 8") rounds DOWN to a multiple of 512 tokens: 16,832 → **16,384 = exactly two chunks**.
  The 448 tokens of headroom the cap formula reserved are removed.
- Scheduling of a chunked request (`prefill_budget.py` `_continuation_chunk_cap`): the next chunk may use `remaining_swa − (window + page) = remaining_swa − 192`, floored to a page.
  Chunk 1 = 8192, chunk 2 = 8192 − 192 = **8000** (matches the log), leaving 192 free → chunk 3 cap = 0 → `fit_chunk` returns None → no chunk scheduled.
- Out-of-window SWA pages of a chunked request are only freed inside `maybe_evict_swa` when the request's NEXT chunk is scheduled (and, with overlap scheduling, only from the third chunk on).
  Deadlock: the eviction that would free ~16k slots runs only when a chunk is scheduled, and no chunk can be scheduled until slots are freed. No forward runs, so no watchdog fires.
- With mrr ≥ 2 the rounding leaves ≥ 512 tokens of headroom → a small third chunk (e.g. 1,344 tokens in the mr4 run) gets scheduled, triggers the eviction, and the pool cycles. Only mrr 1 deadlocks.
- Fix candidates: round the TPU alignment UP (`ceil`) or add the alignment slack to the cap; or evict the chunked request's out-of-window SWA pages before computing its continuation budget; or fail the request with an error when `fit_chunk` returns None and nothing else is running.
- Open: in the 128k shape with mrr 1 (pool also 16,384) the FIRST 8k prompt (one chunk) also stalled: prefill holds it in the inflight queue (swa 0.5), the decode engine (mrr 1, SWA pool 768→512) never ran a batch and shows empty queues. Different symptom, not yet explained; the harness aborts at 900 s and continues with 32k.

**F-9 resolved (2026-10-08 04:40–05:05 UTC).** Qwen3-VL, two prompts of 127,999 tokens (not page- or chunk-aligned), PD vs monolithic, both at
concurrency 1, 1,024 new tokens: **identical 2 of 2**, first-token log-prob delta 0.0000 (`results/A/tc1/qwen3-vl/golden_diff_c1_127999.json`).
The earlier deltas (0.011, 0.228) were against a concurrency-8 golden, i.e. the F-3 batch-composition effect. F-9 is closed as not a bug.

**Eager Qwen3-VL GSM8K-100 = 0.97** (`results/golden/qwen3-vl_fp8_1m_c1/gsm8k_eager.log`); reference for the F-2 compiled-server check (running).

**F-11 isolation summary.** (a) 1M shape, `max_running_requests 4`: no hang up to 131k (see root-cause entry above). (b) 128k shape,
`max_running_requests 1`: not testable — the pair was blocked by a different stall (F-12 below) from its first request; stopped at 06:02 UTC.
The chunk-deadlock prediction for (b) therefore rests on the code analysis and the exact match of the 8192/8000 chunk sizes and pool numbers.

## F-12 (candidate) — P/D handshake never completes; request held on both engines with no deadline (gpt-oss, Config A, fresh pair)

**Symptom (2026-10-08 05:16–06:02 UTC, run `F11_gpt-oss_A_128k_mr1`, `results/F11/handshake_stall/`, logs in `results/logs/runs/F11_gpt-oss_A_128k_mr1/`).**
The first request to a freshly started pair (8,192-token prompt, one chunk) prefilled normally at 05:16:56 and then never produced a token.
Prefill: `num_running_reqs 1`, `num_prefill_inflight_queue_reqs 1`, SWA usage 0.5, scheduler idle in `recv_requests` (py-spy). Decode: no batch
ever ran; the TP0 scheduler sat in `pop_preallocated → _update_handshake_waiters → poll_and_all_reduce` (py-spy), i.e. the room stayed in
`KVPoll.Bootstrapping`; decode metrics showed empty queues the whole time. No error on either side for 30 min. When the client's 900 s timeout
(plus the queued second prompt's 900 s) aborted the request at 05:46:56 both sides logged it at once: prefill `Prefill transfer failed … Aborted by AbortReq`,
decode `Decode handshake failed … Aborted by AbortReq`. The next request then stalled the same way. The 1M-shape pair started 10 min earlier the
same way (`F11_gpt-oss_A_1m_mr4`) served six requests normally, so the trigger is not deterministic.

**What the code says.** Raiden is consumer-pull: the prefill sender moves its room straight to `WaitingForInput` without waiting for the decode
(`raiden/sender.py` ~line 1000) and bounds the handshake with `fail_decode_silence(room, init_time, bootstrap_timeout)` (300 s) — which did not
fire here, so the prefill believed the decode had engaged. The raiden receiver's only deadline (`_check_waiting_timeout`, 300 s) starts in
`send_metadata`, i.e. after the room leaves `Bootstrapping`; `poll()` has no deadline while the room is still bootstrapping
(`raiden/receiver.py` ~lines 1140–1170), and the decode prealloc queue treats `Bootstrapping` as "keep waiting" (`decode.py:1160`). So once the
two sides disagree about the handshake state, nothing times out. Related: #1113 / #1078 (decode waits out transfer/reshard deadlines) — those at
least expire; this one does not. Needs a repro with raiden debug logging to find why the decode never saw the prefill's per-request reply.

**F-2 accuracy gate passed (2026-10-08 06:03–06:11 UTC, `results/F2_patch/gsm8k_compiled.log`).** Patched (compiled) Qwen3-VL-30B, GSM8K-100,
greedy, 1,024 max tokens: **0.99** (eager reference 0.97, CI gate 0.80). Wall time 29 s for 100 problems vs ~17 min eager. Warm start with the
caches from run 3: 421 s (cold: ~19 min). Serving-time recompiles: 0. The patch is ready to propose upstream; token non-identity vs eager is
numerical only.

**F-12 reproduced on a second fresh pair (2026-10-08 06:36–06:48 UTC, `results/F12/`, run `F12_gptoss_A_128k_mr1_try1`).** Same recipe
(`gpt-oss_A_128k_mr1`), first 8k request: prefill ran the chunk at 06:38:30 and holds it (`num_prefill_inflight_queue_reqs 1`, SWA 0.5);
decode never ran a batch, scheduler idle in `pop_preallocated` (py-spy `F12/F12_gptoss_A_128k_mr1_try1_8192_decode_pyspy.txt`), decode's
HTTP layer still holds the rid ten minutes later (`Duplicate request ID detected` when the id is reused). **2 of 2 pairs with this recipe.**
Correlation so far: fails on `gpt-oss_A_128k_mr1` (mrr 1, compile bs [1], ctx 140k) 2/2; works on `gpt-oss_A_1m` (same args except ctx 1M,
pool 1.1M, host pool 320 GiB) and on `gpt-oss_A_128k` (mrr 8). The recipes differ only in those fields. Mechanism unchanged: decode receiver
never leaves `Bootstrapping`, no deadline covers that state. Note for the harness: the engines see the abort only when the client *process*
exits (sockets close), not at the per-request timeout, so stalled requests outlive the client timeout.

**F-12: 3 of 3 fresh pairs (2026-10-08 06:59–07:06 UTC, `F12_gptoss_A_128k_mr1_try2`).** Third pair, first 8k request stalled again within
45 s; decode in `_update_handshake_waiters`. Deterministic for recipe `gpt-oss_A_128k_mr1`. Repro loop stopped; a control pair with
`max_running_requests 2` (`recipes/gpt-oss_A_128k_mr2.json`, `_phase_F11_control_mr2.sh`) runs next: it is both the F-11 control (predicted
pool 17,216 → 16,896, no deadlock) and the F-12 isolation (does the handshake stall need one slot?).

**F-8 mechanism confirmed from code (2026-10-08 07:20 UTC).** Double release of the aborted chunked request's KV row: `process_pending_chunked_abort`
defers the release (raiden `group_failure_requires_drain`, sender not drained) and parks the request in `_deferred_chunked_sender_aborts`
(scheduler.py 3720–3777); one step later the in-flight chunk's result reaches `_retire_aborted_prefill_result` (prefill.py 1198–1222), which
sees `holds_kv` and frees the row; the later drain (scheduler.py 3779–3796) calls `release_kv_cache` without a `holds_kv` guard → the
assertion in `common.py` 292–300. The chunk cache's `release_aborted_request` is a no-op, so it is not the first releaser. Secondary: the
result-path free happens before the raiden group has drained, which the deferral was meant to prevent. Issue draft: `bug_reports/F-8_github_issue.md`.

**F-10 mechanism located in code (2026-10-08 08:20 UTC).** `ViTTpuCompileRunner.bucket_for_patches` (vit_tpu_compile_runner.py 616–634)
returns the raw patch count for an image above the top bucket and warns about a "one-time recompile"; `VisionTpuCompilePolicy.run` (348–364)
then always calls `compiled_fn`, which recompiles → `RuntimeError` under the stance armed in `_maybe_arm_fail_on_recompile` (scheduler.py 5187–5208).
The runner saves the uncompiled forward as `eager_fn` (line 186) but never calls it, whereas `MoonVitTpuCompilePolicy._run_one` (moonvit 189–214)
and `Gemma4VisionTpuCompilePolicy` (gemma4 176–194) fall back to `eager_fn` for exactly this case. Image is not downscaled because
`SGLANG_IMAGE_MAX_PIXELS` defaults to 16384×28×28. 1920×1080 → 120×68 = 8,160 patches vs 4,900 (70×70) top bucket; 1024² fits, 896×504 (1,512) fits.
Issue draft: `bug_reports/F-10_github_issue.md`.

**F-6 mechanism located in code (2026-10-08 08:30 UTC).** `tokenizer_manager._handle_batch_request` (L2003–2060) expands `n=2` into a
prefix request (`max_new_tokens 0`) and two samples, all copied from `obj[i]` with i = prompt index, so all three carry `bootstrap_room[0]`;
`_normalize_bootstrap_params` (io_struct L886–910) had produced `[room, room+1]` but the second room is never used; the mini-LB attaches one
int room per HTTP request (mini_lb.py L490–513). Three requests, one room → collision in the per-room transfer state; decode holds one with
128 used tokens (prompt pages), prefill idle; no deadline (same gap as F-12). Drafts written: F-1, F-5, F-6, F-12 (`bug_reports/*_github_issue.md`);
triage file `bug_reports/PRIORITIES.md`.

**Correction to F-12/F-6 (2026-10-08 08:40 UTC).** All our recipes set `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=7200` and
`SGLANG_DISAGGREGATION_WAITING_TIMEOUT=7200` (to survive 1M-token prefills). So "no deadline fired in 30 min" is expected for every deadline
that reads those variables (sender `fail_decode_silence`, receiver `_check_waiting_timeout`). The code gap that stands: the raiden receiver has
no deadline at all while a room is in `Bootstrapping`. Drafts and PRIORITIES.md corrected.

**F-11 control confirmed; F-12 narrowed (2026-10-08 08:32 UTC, `F11_gpt-oss_A_128k_mr2`).** 140k recipe with `max_running_requests 2`:
pool `17216 -> 16896`; the 32k prompt's chunks are 8192, 8192, **320**, 7680 … with SWA usage 0.97 → 0.51 around the 320-token chunk
(exactly the predicted budget); 8k and 32k complete in ~6 s. No handshake stall on this pair → F-12 needs the single slot (necessary, not
sufficient: the single-slot 1M recipe works).

## Config B (gpt-oss, prefill radix cache on) — 2026-10-08 08:44–08:54 UTC, `results/B/gpt-oss/`, run `B_gpt-oss_128k`
- TC-1 vs c=1 golden: 8k identical 2/2; 32k and 128k diverge after 3–24 tokens (F-4 class; identical numbers across two runs → PD deterministic); needles 3/3 at every length.
- Prefix cache works through PD: after one pass 24,576 of 32,768 tokens cached (last chunk not inserted), second pass 32,704 cached, prefill computes 64 tokens; text identical cold vs cached, first-token logprob delta 0.034 / 0.001 (#321 class).
- Params matrix (n_2, return_logprob_top5 excluded) all PASS; chat 10/11 (tool_call: known parser config); GSM8K 0.99; concurrency c=1/2/4 at 128k all OK (ITL 5.5–8.9 ms); decode aborts 6/6 PASS.
- **Prefill-phase abort does NOT crash with the radix cache on** (F-8 needs `--disable-radix-cache`). Compile audit: 0 serving-time recompiles on both engines.
- Tiny sweep: 7 of 35 prompts flagged (see next entry).
- First attempt of B died in the params matrix: `return_logprob_top5` sets `logprob_start_len 0` → F-5 (same in A); case excluded, harness now relaunches after crashes.
- Tiny sweep flags: the same 7 prompts as in Config A (100–1,025 tokens), same output text ("the text appears to be a random collection…"):
  gpt-oss reasons for all 160 tokens and never states the needle. Test-design limit of the tiny set (160 tokens), not a serving defect; 28/35
  needles found, all 35 produce exactly 160 tokens with no errors.

## Config C (gpt-oss, radix on + HiCache raiden host offload) — 2026-10-08 08:54– UTC, `results/C/gpt-oss/`
- Start OK (154 s warm). TC-1, reuse, params, chat, concurrency: identical numbers to Config B (HiCache adds no numerical change).
- Write-through works: after TC-1 `hicache_backup_tokens_total` 454,912; after the concurrency sweep (14 × 131k random prompts) 1.50M tokens
  backed up to the 1.57M-token host pool, `hicache_dropped_tokens_total 0`, device cache evicted (`cache_hit_rate` 0.0 right after the sweep).
- **Host reload works and is bit-exact:** re-sent 32k prompts served with `#hbm-hit-token: 0, #host-hit-token: 32704`; first-token logprobs
  identical to the HBM-hit run (−0.79959, −0.75351); needles 3/3. Text diverged later only because GSM8K ran concurrently (F-3 batch effect).
- Reload timing under concurrent GSM8K load: e2e 16.0 s / 8.9 s for 1,024 tokens (≈10 s / 3.3 s of prefill-from-host for 32k tokens, vs 1.3 s
  cold prefill). Confounded by load; clean idle measurement queued (`_phase_hicache_reload.sh`).
- Config C completed 09:05 UTC with 0 relaunches: GSM8K 0.98; decode aborts 6/6 PASS; prefill-phase abort handled (no crash, as in B);
  compile audit 0 serving-time recompiles on both engines.

**Config D start failure (09:12 UTC) was TPU contention, not a bug.** Both engines aborted at init with `Failed to acquire a TPU device node
'/dev/vfio/N' because it is already opened by another process`; another user's servers started on both hosts at 09:11:50, between our
holder check (09:12:00, still empty) and device open. Chain stopped; D, E and Qwen B/C are requeued behind `tpu_free`.

## Config D (gpt-oss, C + EAGLE3 on both engines, bf16 KV) — 2026-10-08 11:50–12:10 UTC, `results/D/gpt-oss/`
- First launch (09:12) lost the TPUs to another user's job; relaunch 11:50 OK, cold start 587 s (target-verify graphs for bs 1–8, 4 draft tokens).
- TC-1 vs bf16 c=1 golden: first-token logprob delta 0.0000 at 8k (prefill identical), text diverges after 11–151 tokens (NUM: the verify
  path's numerics differ from plain decode), needles 3/3 at all lengths, 1,024 tokens each.
- **EAGLE3 acceptance is workload-dependent, as designed, not a defect:** synthetic random-word prompts (TC-1, bench_serving random): accept
  length ≈1.0–1.15, accept rate 1–5 % → pure overhead (c=1 output 32.6 tok/s vs 50 in B; ITL 7.5 vs 5.5 ms; 1,024 tokens in 9–13 s vs 6 s).
  GSM8K-100 (natural text, 4 threads): accept length p50 1.48, max 3.58, cumulative avg 2.13; **770 tok/s vs 561 (A) / 578 (B) / 491 (C)**,
  score **0.99**. Repo's own PD-EAGLE3 e2e gate is accept_length > 1.0.
- Monolithic EAGLE3 control (same flags, GSM8K acceptance) queued: `_phase_eagle3_control.sh`.

## Config E (gpt-oss, D + decode radix cache + decode HiCache) — 2026-10-08 12:08–12:22 UTC, `results/E/gpt-oss/`
- Pair up in 271 s (prefill) / decode ready 60 s later. TC-1 vs bf16 golden: same spec-path NUM as D, needles 3/3, 1,024 tokens each.
- Reuse: both passes fully cached (32,704), identical text, first-token delta 0.0000; prefill hit rate 0.998. Decode-side tiers active:
  462,912 tokens written through to the decode host pool, 252,736 cached tokens on the decode radix tree.
- Params all PASS; chat 10/11 (tool_call parser config); concurrency c=1/2/4 all successful (ITL 8.1–12.8 ms with spec); **GSM8K 0.98 at
  809 tok/s** (fastest config); decode aborts 6/6 PASS; prefill-phase abort handled (no crash); **decode scheduler 0 exceptions after
  multi-turn + concurrency traffic → #856 (decode radix + EAGLE3 dies after multi-turn) does not reproduce in this build with this traffic.**
- Compile audit note for D/E: `compile_audit.sh` reports 8 "serving-time" recompile lines on the decode, but they are the EAGLE3
  `_fused_draft_steps` graph for batch size 1 compiling during warm-up, between the `precompile done` marker (the audit's "ready" line) and
  `Warmup complete; froze torch.compile`. Recompiles after the freeze: **0** in both D and E. Audit artefact; for spec configs the ready marker
  should be the freeze line.

## Config B (Qwen3-VL, prefill radix cache on) — 2026-10-08 12:22–13:37 UTC, `results/B/qwen3-vl/`
- Pair up 415 s (eager decoder, F-2 unpatched). TC-1 vs c=1 golden: **identical 6/6** at 8k/32k/128k (as A); tiny sweep 35/35 clean.
- Prefix cache: 32,640 of 32,768 cached (page 128), identical text, delta 0.0000. Params 17/17; chat 8/8; 1×448 and 4×448 images OK.
- Concurrency c=1/2/4 all successful, ITL 155–172 ms (eager decode, F-2). GSM8K 0.96 (golden 0.97). Decode aborts PASS; prefill-phase abort
  handled (no crash, radix on). 0 serving-time recompiles.

## Config C (Qwen3-VL, radix on + HiCache raiden host offload) — 2026-10-08 13:37–14:48 UTC, `results/C/qwen3-vl/`
- Pair up 196 s. TC-1 vs c=1 golden: **identical 6/6**; tiny 35/35; reuse 32,640 cached, identical, delta 0.0000.
- HiCache write-through: 448,768 tokens backed up, 0 dropped. Params 17/17; chat 8/8; 1×448 / 4×448 images OK; concurrency all successful
  (eager ITL); GSM8K 0.96; decode aborts PASS; prefill-phase abort handled; 0 serving-time recompiles. No new defects.

**HiCache timing on an idle Config C pair (17:25 UTC, `results/C/gpt-oss/hicache_reload/`).** 32k prompts, streaming, c=1: cold TTFT
1,254 / 1,232 ms; HBM hit (24,576 cached) 425 / 439 ms; outputs and first-token logprobs identical across cold, HBM-hit and recompute.
The host-hit step did not measure a host hit: the eviction load (12 × 131k = 1.57M tokens) filled the 1.57M-token host pool and evicted the
32k prefixes from the host tier too (`#host-hit-token: 0`, TTFT 1,242 ms = cold). Rerun queued with 8 prompts (`_phase_hicache_reload2.sh`).
The earlier host hit in Config C (14 × 131k < pool) remains the evidence that reload works and is bit-exact.

## F-13 — `--kv-canary` kills every engine at start-up on TPU (gpt-oss: `AttributeError: 'TpuFusedMHATokenToKVPool' object has no attribute 'k_buffer'`)

**Symptom (2026-10-08 17:29 UTC, run `A_gpt-oss_128k_kvcheck`).** Config A gpt-oss with `kv_canary: log`, `kv_canary_real_data: partial`,
`kv_canary_sweep_interval: 50` (+ `disaggregation_enable_kv_checksum`) on both engines: 6 s after `Load weight end`, all 8 ranks on both hosts
raise in `kv_canary/pool_patcher/adapters/swa.py:65` (`_build_subpool_group`: `sub_pool.k_buffer[0]`), SIGQUIT → launcher kills the group
(`exited with -9`). The TPU fused pool keeps a single fused 5-D `kv_buffer` (`hardware_backend/tpu/kv_pool.py:71`,
`TpuFusedMHATokenToKVPool(MHATokenToKVPool)`), so neither `k_buffer` nor `v_buffer` exists. The attacher registry
(`kv_canary/pool_patcher/api.py:26-30`) maps only `MHATokenToKVPool`, `MHATokenToKVPoolFP4`, `SWAKVPool`, `DeepSeekV4TokenToKVPool`; no TPU
module registers the fused pool, and `attach_canary_buffers` looks up the exact type. gpt-oss hits the SWA adapter via `SWAKVPool` whose
sub-pools are TPU fused pools; a non-SWA TPU model should hit `NotImplementedError: no attacher registered` (Qwen run pending).
**Why it matters.** The canary is the build's own use-after-free / page-overlap detector; on TPU it cannot be turned on, and the failure is a
crash at start-up rather than a validation error. **Fix:** register a TPU attacher that reads the fused layout (or make the SWA/MHA adapters use
`get_fused_kv_buffer` when `k_buffer` is absent); until then reject `--kv-canary` on `--device tpu` at argument validation with a message.
Not in the open-issue list (no issue mentions kv-canary).
- F-13 confirmed on Qwen3-VL (18:00 UTC, `A_qwen3-vl_128k_kvcheck`): `NotImplementedError: kv-canary: no attacher registered for pool class
  TpuFusedMHATokenToKVPool`. Both TPU pool shapes fail → `--kv-canary` unusable on TPU. Draft: `bug_reports/F-13_github_issue.md`.
  Checksum-only TC-15 runs queued (`_phase_kv_checksum_gptoss.sh`, then Qwen).

**EAGLE3 monolithic control (18:03–18:13 UTC, `results/D/gpt-oss/eagle3_control/`).** Same spec flags, bf16 KV, host b only.
- TC-1 8k vs plain bf16 golden: diverges after 22 / 16 tokens, agreement 0.031 / 0.028 — **identical to PD Config D's numbers** → the
  divergence is the speculative verify path itself; PD adds nothing.
- Acceptance: random-text TC-1 1.38; GSM8K window gauge 2.64 (rate 0.55); decode-batch accept len over the run p50 2.66, min 1.20, max 3.16.
  GSM8K 0.96, 745 tok/s, 48.7 s (PD-D: 0.99, 770 tok/s, 47.4 s). PD-D's decode-batch p50 during its GSM8K window was 1.48 (with 4
  concurrent threads and different batch composition), yet end-to-end throughput was equal → the per-batch numbers are not like-for-like.
  **Inconclusive**; controlled comparison queued (`_phase_spec_accept_compare.sh`: same 20 natural prompts, c=1, per-request
  `completion_tokens / spec_verify_ct` from `meta_info`, PD-D vs monolithic).

**HiCache host-reload timing, clean (18:17 UTC, `results/C/gpt-oss/hicache_reload2/`, eviction 8 × 131k = 1.05M tokens).** 32k prompts,
c=1 streaming: cold TTFT 1,244 ms (8192-token chunks); HBM hit 422 ms (24,576 cached, 8,192 computed); **host hit 257 ms (32,704 served from
the host tier, 64 computed)** → host→device reload of 32k tokens well under 0.25 s. First-token logprob of the host hit (−0.79959) equals the
Config C host hit with the same cache split; differs from the HBM-hit run only because that run computed a full 8k chunk (F-4-class split
effect). HiCache offload + reload: correct and fast.

## F-14 — `--disaggregation-enable-kv-checksum` is CUDA-only: TPU prefill passes health, then dies on the first real request

**Symptom (2026-10-08 18:21–18:25 UTC, `A_gpt-oss_128k_kvcheck_nocanary`).** Pair up normally (prefill ready 136 s, LB health_generate OK);
first 8k TC-1 request → all 8 prefill ranks: `prefill.py:1418 send_kv_chunk → checksum.py:105 compute: assert kv_page_indices_gpu.is_cuda`
→ engine dead, every later request a connection error. Health-check requests skip the checksum (`is_health_check_req → value 0`), which is why
the engine looked healthy. `KvChecksumComputer` is built with `torch.device("cuda:…")` (prefill.py:230-240) and uses the CUDA Adler-32 kernels
(`sglang/kernels/ops/memory/adler32`); nothing TPU-side implements or rejects it. **Fix:** reject on `--device tpu` at validation; implement a
host-side or Pallas checksum. No open issue mentions the checksum. Draft: `bug_reports/F-14_github_issue.md`. Qwen checksum-only run in
progress (same code path expected).
**TC-15 outcome:** neither built-in KV integrity check (canary F-13, checksum F-14) can run on TPU in this build; the P/D transfer is verified
only by the differential tests (TC-1: Qwen bit-identical at all lengths; gpt-oss identical with SWA memory off) and GSM8K.
- F-14 confirmed on Qwen3-VL (18:31 UTC, `A_qwen3-vl_128k_kvcheck_nocanary`): pair healthy, first TC-1 request → all prefill ranks die at
  `checksum.py:105` (`is_cuda` assertion). 2/2 models.

**EAGLE3 acceptance, controlled comparison (18:33–18:40 UTC, `results/D/gpt-oss/spec_accept_compare/`).** Same 20 natural-text prompts,
greedy, 256 tokens, c=1, per-request `completion_tokens / spec_verify_ct`: **PD Config D mean 2.229 (p50 2.265, 1.84–2.46), 236.8 tok/s;
monolithic EAGLE3 mean 2.229 (p50 2.265, 1.84–2.46), 232.5 tok/s; verify counts identical per request.** Speculative decoding through the
P/D pair is exactly as effective as on one server; the earlier per-batch "1.5 vs 2.6" gap was a measurement artefact (different windows and
batch composition). EAGLE3 question closed: no PD defect, no new issue. Single-stream decode with EAGLE3: ~235 tok/s vs 176 without.

**Environment note (18:46 UTC).** First 1M Config B launch: the decode engine died at start-up with `torch.distributed.DistNetworkError:
... port: 34587 ... EADDRINUSE`, two minutes after the previous server on host b was stopped. Transient port collision in the launcher's
dist-store port choice; relaunched. Not counted as a finding (single occurrence).

**F-12 at the 1M shape with the radix cache on (19:03 UTC, `B_gpt-oss_1m`, single slot).** First 32k request: prefill ran all 4 chunks in
1 s and holds the request (`num_prefill_inflight_queue_reqs 1`, `num_used_tokens 32768`); decode never ran a batch, all queues 0, for 13+
minutes. Same stall as F-12. New trigger-table row: ctx 1M / pool 1.11M / host pool 320 GiB / radix ON / 1 slot → **stall**. Single slot is
present in every stalling pair (4 pairs, 2 recipes); the only single-slot pair that worked is `A_1m` (radix off). Decode metrics in `results/F12/B_1m_stall/` (the py-spy dump failed: run without sudo). The 1M B/C runs continue with 2-slot recipes (`gpt-oss_[BC]_1m_mr2.json`).

## 1M shape, Config B with 2 slots (`B_gpt-oss_1m_mr2`, 19:17 UTC–) — `results/B/gpt-oss_1m_mr2/`
- Pair up 245 s; SWA pool ratio-sized 890,880 (radix on) → no F-11; 2 slots → no F-12 stall. 32k probe: same F-4 divergence vs golden as at
  128k (prefix 24 / 3), deterministic across shapes.
- 256k: TTFT 18.2 s, 1,024 tokens in 24.8 s, both prompts. Needles 0/3 — **and the monolithic golden also finds 0/3 at 256k**: beyond
  gpt-oss's native 128k context neither server retrieves the planted facts (both spend the budget reasoning). Needle gate is uninformative
  above 128k; the checks that matter there are: served without error, healthy engines, same behaviour as golden.
- 255,123 (unaligned): both prompts served (TTFT 17.4 s). PD output flagged `degenerate_4gram_x64/x61`, but the **golden is equally
  degenerate** (max 4-gram repeats 33 / 67 vs PD 64 / 61; both tails are loops). gpt-oss beyond its native 128k context loops regardless of
  PD; structural flags above 128k apply to both sides and are not a PD defect.
- 512k / 511,777: TTFT 57 s / 55 s; 1M (1,047,552): **TTFT 196 s, 1,024 tokens in 205 s**, both prompts; 999,999: TTFT 180 s. All served,
  engines healthy. Prefill rate ≈ 5.3k tokens/s at 1M.
- Boundary step false alarm: `verify_pd_disagg.py --boundary` without `--tokenizer-json` sizes prompts at 3.6 chars/token; the "ctx−1024"
  prompt was 704,702 server tokens and "ctx+1" was also under the limit, so the server's 200 was correct (and fast: cached prefix of the
  previous prompt). Rerun with the exact tokenizer in `B/gpt-oss_1m_mr2/boundary_exact.txt`. The earlier "Boundary 4xx: PASS" in the
  catalogue was unverified and has been corrected.

**F-12 with 2 slots at the 1M shape (`B_gpt-oss_1m_mr2`, 19:43–20:40 UTC).** After the long TC-1 sweep (32k…1M, 14 prompts), the boundary
step (a ~705k and a ~1M-token prompt) and the 256k reuse pass, GSM8K's first requests never returned: prefill `num_running_reqs 1`,
`num_prefill_inflight_queue_reqs 2`, `num_queue_reqs 2`; decode idle in `pop_preallocated → _update_handshake_waiters`. When the clients were
killed at 20:13:54 both sides logged the paired aborts (`Decode handshake failed … Aborted by AbortReq`, `Prefill transfer failed … Aborted by
decode-side abort notification`) and the next requests stalled again. A concurrent exact-tokenizer boundary request (1,055,744 tokens) was
in flight when the stall began. **So the handshake stall is not confined to one request slot**; it has now occurred with 2 slots after a
burst of very long requests. Evidence: `results/F12/B_1m_mr2_stall/` (decode stack, both engines' metrics), `results/logs/runs/B_gpt-oss_1m_mr2/`.
Config B at 1M otherwise served every length (TC-1 complete); Config C at 1M skipped (same family), MT-Bench phase released.

## O-3 — Multi-turn prefix caching in PD only covers the prompt, not the previous answer (MT-Bench, Config A vs B, 20:40–20:58 UTC)
MT-Bench 80 questions × 2 turns, c=1, gpt-oss. Config A (radix off): turn-2 cached 0, TTFT 336 ms. Config B (radix on): turn-2 cached
**102 of 572 tokens (18 %)**, TTFT 350 ms, output 191 tok/s (A: 195). The cached 102 tokens are the system prompt + turn-1 question; the
turn-1 **answer (~414 tokens) was generated on the decode engine, so its KV never existed on the prefill** and turn 2 recomputes it. With
short chat turns the prefill radix cache therefore gives no measurable TTFT or throughput benefit in PD mode. The repo's multi-turn PD tests
rely on the decode-side cache / host restore for this; Config E (decode radix + decode HiCache) queued on MT-Bench to measure it.

## TC-16 MT-Bench (80 questions × 2 turns, gpt-oss, greedy, ≤512 tokens) — 20:38–21:25 UTC, `results/mtbench/<cfg>/summary.txt`

| Config | c | out tok/s | TTFT t1 / t2 (ms) | e2e per turn t1 / t2 (s) | turn-2 cached | accept len |
|---|---|---|---|---|---|---|
| A (radix off) | 1 | 195.5 | 333 / 336 | 2.12 / 2.25 | 0 / 572 | – |
| A | 4 | 598.7 | 393 / 406 | 2.71 / 2.89 | 0 | – |
| B (radix on) | 1 | 191.3 | 327 / 350 | 2.15 / 2.30 | 102 / 572 (18 %) | – |
| B | 4 | 599.4 | 380 / 406 | 2.71 / 2.86 | 109 (19 %) | – |
| C_bf16 (= D without spec) | 1 | 187.1 | 339 / 350 | 2.17 / 2.36 | 102 (18 %) | – |
| C_bf16 | 4 | 590.3 | 431 / 410 | 2.76 / 2.91 | 119 (21 %) | – |
| **D (EAGLE3)** | 1 | **268.1 (+43 %)** | 454 / 567 | **1.50 / 1.69 (−31 %)** | 102 (18 %) | **2.46 mean, 2.41 p50, 1.65–3.79** |
| **D** | 4 | **740.1 (+25 %)** | 633 / 653 | 2.15 / 2.35 | 124 (22 %) | 2.46 mean |

- **Speculation (D vs C_bf16):** +43 % single-stream and +25 % at c=4 on real chat traffic, 160/160 turns answered; cost ≈ +115–240 ms TTFT.
- **Prefix caching (B vs A):** no measurable gain on short multi-turn chat in PD mode — only the prompt (18–22 %) is cached because the
  previous answer's KV lives on the decode (O-3). Config E (decode radix) and C (host tier) queued on the same set.
- All 4 configs: 160/160 turns OK at both concurrencies; prefill hit rate 0.10 with the cache on.
- MT-Bench E and C (21:24–21:46 UTC): **E** (decode radix + decode HiCache + EAGLE3) c=1 236 tok/s, TTFT 521/608 ms, turn-2 cached 18 %,
  accept 2.46; c=4 727 tok/s, turn-2 cached 32 %. **C** (host tier) c=1 192 tok/s, turn-2 cached 18 %; c=4 595 tok/s, 19 %. Neither the
  decode-side radix cache nor the host tier raises the prefill's cached fraction for the previous answer in this build: O-3 holds for every
  configuration. E is ~12 % slower than D single-stream (decode-side cache bookkeeping overhead) with identical acceptance.

**Boundary with exact token counts (22:05 UTC, Config A, 128k shape, `results/A/boundary/boundary_exact_128k.txt`).** ctx−1024 (139,264
tokens): 200, 7.7 s. ctx+1 (140,289): **400** with `The input (140292 tokens) is longer than the model's context length (140288 tokens)`.
Boundary row closed (PASS). The tool's two 1k-needle "FAILs" are gpt-oss reasoning before answering within an 8-token budget (same artefact as
the tiny sweep), not retrieval failures.

**F-12 with 300 s deadlines (23:27–23:52 UTC, `A_128k_mr1_t300`, `results/boundary2/f12_t300.*`).** `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`
and `..._WAITING_TIMEOUT` = 300 (instead of our 7200). Stall reproduced on the first 8k request. **Prefill:** at t+300 s all ranks log
`Prefill transfer failed … Request <room> timed out after 300.0s waiting for the decode to ask for its KV` (`fail_decode_silence`) → the
decode never sent its pull request; prefill frees the slot. **Decode:** no deadline fires; the room is released only when the client
disconnects (`Decode handshake failed … Aborted by AbortReq` at the client's 720 s read timeout). **Client:** `Read timed out` after 720 s for
each of the two requests (the mini-LB waits for the decode's response). So with default deadlines F-12 is still a user-visible hang of
client-timeout length per request plus a decode-side hold until disconnect; only the prefill side recovers. Mechanism sharpened: the decode's
bootstrap for the room never completes, so it never asks the prefill for the KV.

**Boundary phase 2, Config A 128k (23:54–23:56 UTC, `results/boundary2/`).** (2a) ISL 131,072 + `max_new_tokens` 20,000 (sum > context
140,288): **400 in 0.19 s**, rejected up front. (2b) ISL 131,072 + 9,000 (fits): 200 in 55 s; both engines back to 0 used tokens. (2c) empty
`text`: 400; whitespace `text`: 200; empty `input_ids` list: **500 in 1 ms** (validation gap, engines unaffected; body captured below).
(2d) same `rid` twice concurrently: one 400, one 200 (correct). (2e) abort of a 131k prompt at 6.4 s: the abort landed in the **last two
chunks** of prefill (log: `#pending-token: 2368`, token usage drops to 0.00, then `Received output for rid=... but the state was deleted`) and
the prefill died with `AssertionError: Only a mamba-capable tree cache allows freeing before alloc` from `_drain_deferred_chunked_sender_aborts`
→ `release_kv_cache`: **third reproduction of F-8** (now 3/3 with radix off), same traceback as the draft. The ~0.2 s window between the last
chunk and the decode pull was not hit; a transfer-window abort proper remains partial coverage. (2f) abort storm skipped (prefill down).

**F-15 (00:05 UTC): the empty-`input_ids` 500 is the mini load balancer, not the engine.** Same body sent to the prefill and decode `/generate`
directly → `{"error":{"message":"input_ids cannot be empty."}}` HTTP 400. Via the mini-LB → `Internal Server Error` HTTP 500:
`mini_lb.py:576 _get_request_batch_size` evaluates `input_ids[0]` on the empty list (`IndexError`). Not in the upstream open-issue snapshot.
Draft `bug_reports/F-15_github_issue.md`; S3/P3.

**Boundary phase 3: decode KV pool exhaustion deadlocks the whole pair through F-11 (00:00–00:20 UTC, `A_128k_smallpool`: decode pool
262,144 tokens, prefill unchanged at 8 slots / 786k, radix off; 4 × 131,072-token prompts, 1,024 output tokens, c=4; `results/boundary2/`).**
Timeline from the logs: #1 prefilled and transferred, decode runs it (132,096 of 262,144 used, so #2 cannot be pre-allocated); the prefill
prefills #2 (8,192/2,752 alternating chunks, SWA usage 0.57/0.85, normal) and #2 goes **inflight** waiting for the decode. The moment #2 leaves
chunking, the prefill's SWA usage jumps to **0.96–0.99** and #3's chunks collapse: 2,112 → 8,192 → **640 → none** (`#pending-token: 253312`
never moves again). #1 finished on the decode at 00:00:20, the same second the last prefill batch ran; nothing ran afterwards on either engine.
A fresh 8-token request through the LB got no response in 55 s. `py-spy` dumps (`results/boundary2/stacks/`): prefill ranks spin in
`get_next_disagg_prefill_batch_to_run` → `process_prefill_chunk` → `cache_unfinished_disagg_prefill` / `resolve_waiting_queue_bootstrap`
(the F-11 zero-budget loop); decode ranks spin in `pop_transferred` → `maybe_start_pull` / `_start_chunked_pull` and
`pop_preallocated` → `_update_handshake_waiters` (waiting for KV the prefill never sends). Both `/health` return 200; no watchdog fires.
**Mechanism.** The SWA pool is 19,456 tokens = 2 × 8,192 + 8 × 384 + 64 rounded down to 512, i.e. two chunks plus a 384-token hold per slot.
A request whose prefill has finished but whose KV the decode has not pulled keeps its *last chunk's* SWA pages (nothing evicts them: eviction
only runs when that request's next chunk runs, and there is none), so an inflight hold is ~8k tokens, not 384. One such hold plus the chunk
in flight exhausts the pool, the continuation budget `remaining_swa − (window + page)` hits 0, no batch forms, eviction never runs: F-11, now
at 8 slots. Because the decode can only pull once the prefill loop makes progress, and the prefill only makes progress once the decode
pulls, the pair is wedged. Any decode back-pressure (full decode pool, slow decode, F-12 hold) reproduces it; the small pool only made the
window easy to hit. F-11 severity raised to P0 for PD. **Client outcome:** 1 of 4 completed (13.9 s); the other three got `Read timed out` at the client's 1,200 s limit; `num_retracted_reqs` stayed 0
(the retraction path was never reached because nothing was ever admitted to retract). After the clients disconnected the prefill's queues
drained (`inflight 0, queue 0`), i.e. the wedge is released only by client aborts, exactly as F-12's hold.

**Fix validation (00:41–03:12 UTC, branches `pd-chunked-abort-single-release` + `pd-swa-chunk-cap-deadlock` deployed on both hosts,
`results/fix_validate/`, `phase_fix_validate*.log`).** F-8: 6 client aborts during chunked prefill of a 131k prompt (3 at 6.4 s, 3 at 2.0 s):
PASS, 0 scheduler exceptions, engines back to baseline, follow-ups served (before: engine dead on the first abort). F-11 trigger 2 (decode pool
262k, 4 × 131k, c=4): 4 of 4 in 33 s, needles 3/3, healthy chunk trail (before: 1 of 4, pair wedged). F-11 trigger 1 (1 slot, 1M shape): SWA
pool 16,896 (cap aligned up from 16,832); prefill completed 256k, 512k and 1M (850 chunks, no stall; before: never past chunk 3). E2E
responses still time out on this pair because of F-12 (decode ran 0 batches; handshake ended only on client abort), unrelated to the fix.
Note: the first validation round left the pool at 19,456/16,384 because the `--max-total-tokens` sizing path bypassed the alignment; fixed
in the branch (`_swa_cap_tokens` used by both paths) and re-validated.

**Upstream (04:15 UTC).** PR for F-11 / #1142 opened: https://github.com/sgl-project/sglang-torchtpu/pull/1181 (branch
`pd-swa-chunk-cap-deadlock`, rebased on 2cec1cf194). F-8 / #1145 was fixed upstream on 10-07/10-08 (bab49bcec2 + 7366de9460 +
2645aa14af, same single-owner approach as our branch); our branch is superseded; verification of upstream main against our abort reproduction
queued (`_phase_f8_upstream_check.sh`); comment draft in `prs/COMMENT_1145_fixed_upstream.md`. #1161 (priority scheduling validation) test
plan written: `TEST_PLAN_1161_priority_scheduling.md`.

**Monolithic before/after for F-11 trigger 1 (03:33–04:14 UTC, `_phase_fix_validate_mono.sh`, recipe `gpt-oss_G_1m_mr1_noradix.json`).**
Same engine on host b, radix off, 1 slot, 1M shape, 256k prompt, no P/D. main: SWA pool 16,384, chunks 8,192 → 8,000 at SWA 0.99, then no
batch; client timed out at 1,800 s. Fix branch: pool 16,896, 1,024 tokens returned, TTFT 19.2 s, e2e 25.4 s. Added to PR #1181.

**#1145 (F-8) verified fixed on upstream main 2cec1cf194 (04:18–04:22 UTC, `_phase_f8_upstream_check.sh`, `results/fix_validate/f8_upstream_abort_*.txt`).**
Config A pair built from origin/main on both hosts: 3 aborts at 2.0 s and 3 at 6.4 s during chunked prefill of a 131k prompt, all PASS,
follow-ups identical, baselines restored, both engines healthy. Closing comment ready in `prs/COMMENT_1145_fixed_upstream.md` (not posted).

**F-16 (04:35 UTC, #1161 battery, `1161_ps_q`).** `--max-queued-requests 4` with priority scheduling on a PD pair: newcomers of priority 3, 5 and 9
into a full queue of priority-5 requests all got 200; nothing evicted. Cause: `_add_request_to_queue` calls `_abort_on_queued_limit` only
when `disaggregation_mode == NULL` (scheduler.py ~L3458), so the limit and the priority eviction are inert on prefill and decode engines.
Draft `bug_reports/F-16_github_issue.md`; S2/P2. Monolithic control queued (`_phase_1161d.sh`).

**#1161 battery, ps_dis (04:33 UTC).** With priority scheduling disabled and `--abort-on-priority-when-disabled`, a request carrying a priority is
rejected with HTTP **503** and the message "Using priority is disabled for this server. Please send a new request without priority";
plain requests are served. Behaviour correct; observation O-4: the status is 503 (retryable) for what is a client error (400 would stop
clients from retrying forever). Not filed.

**PR #1181 scenario battery S1 (04:38 UTC, Config A 128k, PR head on both hosts, SWA pool 19,968).** Correctness: vs the c=1 monolithic golden
(`golden/gpt-oss_fp8_rep2_c1`) the pair is **byte-identical at 8k** (both prompts, 1,024 tokens; also identical to the pre-fix pair's TC-1
outputs). At 131k the output diverges after 11 chars (pre-fix pair: 21 / 77 chars) with needles 3/3 in both: the F-4 effect (chunk schedule
changes numerics, no accuracy impact), expected because the pool size changes the chunk split. Note: the battery script compared against the
c=8 golden first (`golden/gpt-oss_fp8`), which is the F-3/F-9 batch-invariance artefact, not a regression. 8 × 131k at c=8: 8/8 in 59 s.

**PR #1181 scenario battery S2–S5 (04:41–05:20 UTC, PR head = upstream 2cec1cf194 + F-11 fix, `results/fix_scenarios/`).**
S2 small decode pool (262k): 8 × 131k at c=8 → 8/8 in 59 s; 16 × 32k at c=16 → 16/16 in 31 s (before the fix 1 of 4 completed). S3 1M shape, 4
slots (SWA pool 18,432 = cap 17,984 aligned up): 4 × 256k at c=4 → 4/4 in 81 s; 2 × 1M → 2/2 in 390 s. S4 radix-on regressions: B and C serve
131k, reuse pass 12–13 s with `cached=122880`, no exceptions (fix is a no-op with the radix cache on). S5 Qwen3-VL Config A: 8k text outputs
**byte-identical** to the pre-fix pair (TC-1); image cases: the 1920×1080 image killed the prefill with the **F-10** recompile
(`RuntimeError: Detected recompile when torch.compile stance is 'fail_on_recompile'`), so F-10 is still present on upstream main 2cec1cf194;
the 4×448 and 16×448 cases then failed only because the engine was down. No exception anywhere else in S1–S5; all pairs healthy at the end.

**#1161 preemption cases on the PD pair (05:24–05:32 UTC, `1161_ps_pre` threshold 10, `1161_ps_pre_t0` threshold 0).** C1 (priority diff =
threshold): newcomer waits for a slot, no retraction (correct, strict >). C2 (diff = threshold+1, all 8 slots refilled with priority-0
decodes): **no preemption either**, newcomer waited 11.1 s for a slot (`num_retracted_reqs` 0, holders all completed). Cause, by code reading:
`preempt_to_schedule` is invoked only from `get_new_batch_prefill` when `running_batch.batch_is_full` (scheduler.py ~L4244), i.e. the
prefill-admission path of a monolithic server or a prefill engine. A PD decode engine admits requests through the prealloc/transfer queues, so
its running decodes are never candidates for priority preemption, whatever the flag says. Consequence for the 10/15 launch: the requirement
"no preemption of a running decode" holds by construction in PD (observation O-6); `--disable-priority-preemption` is redundant there but
harmless. Ordering (B2/B3) and no-preemption (B1) PASS in every PD configuration with the corrected probe. Monolithic preemption control
queued (`_phase_1161g.sh`) to show the mechanism where it applies.

**#1161 monolithic queue-limit control (05:38 UTC, `1161_mono_ps_q`).** D1 lower newcomer → 503 "The request queue is full."; D3 equal → 503;
D2 higher newcomer → the newest priority-5 waiting request evicted ("The request is aborted by a higher priority request."), newcomer served.
Confirms F-16 as a PD-only gap.

**#1161 monolithic preemption control (05:45 UTC, `1161_mono_ps_pre`: 4 slots, threshold 10).** C1 diff = threshold → waits 6.6 s for a slot
(no preemption). C2 diff = threshold+1 at full occupancy → admitted in 0.1 s (a running priority-0 decode was preempted); the preempted holder
resumed and all 4 holders completed with full output. Ordering 7→0 and FIFO ties PASS. So the mechanism works on a single engine and is
simply unreachable on a PD decode (O-6). Low-values-first rerun on the pair (`1161_ps_low_b`): order 0→7, ties FIFO, no preemption, PASS.

**PR #1181 ready for review (06:40 UTC).** All 28 CI checks succeeded (CPU shards, build, TPU unit tests, stage-B e2e on v7x); local unit
tests, live validation and scenario battery green → draft converted to ready for review.

**#1161 second battery, block 1 (06:38 UTC, `1161h_B_np`, Config B, priority scheduling, no preemption).** B5 starvation: under a 60 s
stream of 10 priority-10 clients (170 requests) a priority-0 request was served in 1.9 s: no starvation; high-priority arrivals spend their
wait in prefill/transfer, so the decode queue is rarely contended when a slot frees. B6: with one free slot, a priority-50 8k request sent
0.5 s after a priority-0 131k request got its first token at the same time (6.9 s): the running chunked prefill is not interrupted, so a long
low-priority prompt causes **head-of-line blocking on the prefill engine** (observation O-7; only preemption on the prefill could avoid it).
B7: prefix reuse across priorities works (cached_tokens 32,704, TTFT 1.38 → 0.57 s) but the cached run's greedy output **differed from the
cold run**; cause (cache hit vs priority) isolated by `_phase_1161i.sh`. F3: `priority` label present on prefill and decode gauges (our
dashboard selects no priority label, so unaffected). F4: 20 mixed-priority streaming aborts (10 concurrent): 0 client errors, used tokens
back to 0, follow-up served.

**#1161 F1 soak (06:42–06:57 UTC, Config B, priority scheduling, no preemption).** 15 min, 12 closed-loop clients, prompts 1k/8k/32k, outputs
64–256, priorities 0/5/10/50 uniformly: 5,035 requests, 0 errors, 8 running with 2–4 waiting throughout, used tokens and running back to 0 at
the end, 0 exceptions. Mean TTFT by priority: 0 → 3.3 s, 5 → 1.9 s, 10 → 1.8 s, 50 → 1.7 s: lower priority waits longer, none starved.

**#1161 B8 HiCache across priorities (07:10 UTC, `1161h_C_np`, Config C).** 8 distinct 131k documents warmed at priority 0 (exceeds the 786k
device pool, so the earliest are evicted to host); document 0 re-sent at priority 50: `cached_tokens=131072`, prefill log
`#hbm-hit-token: 0, #host-hit-token: 131072`, TTFT 0.91 s. Mechanism PASS. The cached run's greedy text differed from the cold run, the same
symptom as B7 (HBM hit); whether this is a priority effect or the cached-prefix numerics is isolated by `_phase_1161i.sh` (cold vs cached,
with and without priority scheduling).

**#1161 E3/E5 Qwen3-VL multimodal with priorities (07:14 UTC, `1161h_qwen_B_np`, Config B).** Three concurrent 4-image requests at
priorities 0/10/50: all 200, 48 tokens each (needle not reached within the 48-token budget; output-length matter). E5 same images, priority 0
then 50: `cached_tokens` 768 of 818 on both (the image span, shared with the E3 requests): image prefix reuse across priorities works. 0
exceptions.

**#1161 monolithic preemption aborts (07:19 UTC, `1161h2_mono_pre`, 4 slots, threshold 10).** C9 abort the preemptor 2 s after it preempted
(first token 0.35 s): all 4 victims resumed and completed, used tokens 0. C10 close all victims' streams right after a preemption: newcomer
served (TTFT 2.9 s), used 0, running 0. C15 three successive newcomers (11, 12, 13) each admitted in ≤0.12 s, all 4 holders completed. 0
exceptions.

**Cold vs cached identity (07:23–07:27 UTC, `_phase_1161i.sh`, Config B with and without priority scheduling, c=1, 256 tokens).** 8k and 32k
prompts: cached run byte-identical to the cold run in all four cases (full-prompt hits, TTFT 0.68 → 0.32 s and 1.52 → 0.45 s with priority,
0.59 → 0.34 s and 1.49 → 0.47 s without); a second hit identical again. The B7/B8 "output differed" lines were a probe artefact: the
re-sent request omitted `ignore_eos` while the cold one had it, so the texts diverged at the end-of-sequence point. Corrected: B7 and B8
PASS (prefix reuse and HiCache host reload across priorities, identical output).

**#1161 E4 Qwen3-VL 256k with priorities (07:42 UTC, `1161j_qwen_B_256k`, Config B 256k shape, 4 slots, no preemption).** A priority-0
255k text prompt (1,024 tokens: TTFT 40.6 s, e2e 218 s) and, 20 s later, a priority-50 4-image request: both served (image request TTFT 33 s,
waiting behind the in-flight 255k prefill, O-7), 0 exceptions, used tokens back to 0, both engines healthy.

**Filed upstream (08:20 UTC).** F-16 → https://github.com/sgl-project/sglang-torchtpu/issues/1188 ; F-15 → https://github.com/sgl-project/sglang-torchtpu/issues/1189 (both authored by alisachen-google from the drafts).

**#1145 closed (08:30 UTC).** Closing comment posted (fixed by upstream PR #1138, verified with our six-abort reproduction on main 2cec1cf194); issue closed as completed.

**PR #1181 test revision (08:45 UTC).** The gate test in `test_prefill_abort_result_cleanup.py` no longer uses `@patch`; it stubs the fake
scheduler's `cache_unfinished_disagg_prefill` and `_release_swa_beyond_window` and asserts `req.kv.holds_kv` (reviewer feedback). Head fd82eab2f3; CI re-running.

**Fix PRs opened as drafts (08:55–09:10 UTC), CI requested with `/tag-run-ci-label`.**
- #1191 for #1188 (F-16): prefill applies `--max-queued-requests` over bootstrap + waiting queues; priority eviction from the waiting queue;
  decode untouched. Tests in `test_priority_scheduling_disaggregation.py` (50 passed with related files).
- #1192 for #1147 (F-6): one bootstrap room per parallel-sampling sub-request (`r_i + k * batch_size`), prefix request skipped in PD mode.
  Tests in `test_tokenizer_manager_rid_cleanup.py` (91 passed with io_struct tests).
- #1193 for #1176 (F-13): `hardware_backend/tpu/kv_canary_tpu.py` fused-pool attacher (MLA-style single canary pair per slot), lazy TPU
  registration in the patcher, SWA adapter delegates fused sub-pools. Tests in `test_kv_canary_headroom.py` (7 passed).
- #1181 (F-11): imports moved to module top per review; gate test without `@patch`; head 309bc399af, CI re-running.
Live validation of all three on the pair: `_phase_fix_validate_v2.sh` → `results/fix_validate_v2/`.

**Live validation of the fix PRs (`_phase_fix_validate_v2.sh`, 09:53–14:03 UTC, `results/fix_validate_v2/`).**
- #1192 (n>1 rooms): **PASS** on Config A: `n=2` → 2 outputs in 1.2 s, `n=3` → 3 outputs in 1.1 s, chat `n=2` → 2 choices; 0 exceptions, both
  engines idle afterwards (before: no response until the client timed out).
- #1191 (queue limit on the prefill): probe still FAIL (D1/D3 200, nothing evicted). With short prompts the prefill is never contended; the
  backlog sits in the decode's waiting queue after transfer, which the prefill-side limit does not see. The decode's waiting queue is the
  analogue of the single-engine queue → the limit (with KV release for newcomer and victim) must also apply there. Rework in progress.
- #1193 (canary attacher): attaching now succeeds on both pools (`attach_tpu_fused` for Qwen's fused pool, `attach_swa` with fused sub-pools
  for gpt-oss), but the gpt-oss prefill never reached `/health` (5,400 s) and the Qwen pair hung on `/health_generate` → the canary's torch
  reference path or the spliced transfer buffer list stalls on TPU; diagnosing.

**PR #1192 ready for review (19:45 UTC):** CI 25/25 green, unit tests in the existing file, live validation on the pair recorded in the PR.

**#1176 / PR #1193 live results (09:53–14:03 and 19:34–19:50 UTC).** With the attacher, `--kv-canary log` attaches on both pools (logged
`attach_tpu_fused` for Qwen's fused pool, `attach_swa` with fused sub-pools for gpt-oss). Then: with the compiled decode graph the gpt-oss
prefill stalls at "Capture target decode TPU graph begin" (never healthy in 5,400 s) and the Qwen pair hangs on the first generation; with
`enable_torch_compile=false` the Qwen prefill scheduler dies during initialization with exit code **-11 (SIGSEGV)**. The canary's torch
reference launchers are not functional on TPU in either mode, so the attacher alone cannot make the feature work. PR #1193 re-scoped: keep
the attacher (correct, tested, prerequisite for the runtime port) and make `install_canary` refuse on TPU with a clear message instead of
dying mid-start, which is the issue's documented alternative.

**#1188 / PR #1191 live re-run (fixv3, 19:55–20:15 UTC) — prefill-side rejection is harmful.** With the limit at the prefill's intake,
3 of the 8 holder requests got 503 from the prefill (its bootstrap backlog hit 4). The mini-LB returns the *decode's* response and discards
the prefill's, so the client never saw the 503; the 3 decode twins sat in the decode transfer queue waiting for KV forever (gauges frozen at
running 5 / transfer 3 / used 16768 after the other 5 finished). The decode then had free slots, so the probe's fillers and newcomers all ran
(D1/D3/D2 FAIL for the wrong reason). Conclusion: only the decode can shed load on a pair; PR re-scoped to decode-only (prefill intake
unchanged). Re-validation queued as fixv4. Side note: this is the F-12 hold triggered from the prefill side — any prefill-side abort before
the KV handshake leaves the decode twin hanging with no timeout.

**PR #1193 live check (canary_refuse_qwen, 20:17 UTC): PASS.** With the re-scoped branch, `--kv-canary log` on the Qwen3-VL pair makes
both engines exit 28 s after launch with `kv-canary: the canary runtime is not available on TPU yet (--kv-canary log); start with
--kv-canary none.` — no NotImplementedError, no stall, no segfault. Remaining: CI on the re-authored head, then mark ready.

**#1188 / PR #1191 fixv4 (decode-only, 20:34–20:40 UTC): mechanism works, counting wrong.** The decode rejected 2 of the 8 holders with
503 at 20:36:54 although only 6 were running: transferred requests pop from the transfer queue in a burst and pass through `waiting_queue`
for one scheduling step even when running slots are free, so a plain `len(waiting_queue)` over-counts. Fix: count the backlog net of free
running slots (`len(waiting_queue) - (min(pool.size, max_running) - running)`), apply the limit only to the part that must wait.
Everything else was clean: 0 exceptions, end-idle used 0 (no more hung twins). Re-validation queued as fixv5.

**#1188 / PR #1191 fixv5 (decode-only, backlog net of free slots, 20:42 UTC): PASS.** D1 priority 3 → 503 "queue is full"; D3 equal → 503;
D2 priority 9 → evicted the newest priority-5 filler (3) with "aborted by a higher priority request", newcomer 200; all 8 holders admitted
(no burst rejection); 0 exceptions; end-idle used 0. Matches the monolithic reference. PR body updated; mark ready once CI is green.
