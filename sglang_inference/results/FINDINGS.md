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
