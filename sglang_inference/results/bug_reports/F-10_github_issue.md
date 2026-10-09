# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [TPU] One image above the precompiled ViT range kills the Qwen3-VL prefill engine: `ViTTpuCompileRunner` recompiles under `fail_on_recompile` instead of running the image eagerly like the MoonViT and Gemma4 policies do

**Labels:** bug, tpu, multimodal

---

## Summary

A single `/v1/chat/completions` request with one 1920×1080 image stops the Qwen3-VL-30B-A3B prefill engine. The image resizes to 120 × 68
patches of 16 px, 8,160 patches. The vision encoder was precompiled for square-image buckets up to 4,900 patches (the default
`SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES`). The generic ViT runner logs that the image "triggers a one-time ViT recompile" and then calls the
compiled encoder with the new shape. But the scheduler froze `torch.compile` with `set_stance("fail_on_recompile")` at the end of warmup, so
the recompile raises `RuntimeError` inside the model forward. The exception is not caught, all eight prefill scheduler ranks exit, the launcher
kills the process group, and the load balancer goes down with it. Reproduced 2 of 2 on fresh pairs. Images of 448², 1024² and 896×504 work.

Two other vision policies in the same directory already handle this case correctly. `MoonVitTpuCompilePolicy._run_one` and
`Gemma4VisionTpuCompilePolicy` check whether a bucket fits and, if not, run the saved uncompiled encoder (`self.eager_fn`) with the comment
"recompiling is not allowed once warmup has finished (fail_on_recompile), so run the uncompiled encoder for it". The base runner used by
Qwen3-VL saves `eager_fn` too, but never uses it: `bucket_for_patches` returns the raw patch count for an oversized image and `run` always
calls `compiled_fn`. The warning text ("one-time recompile") describes behavior the stance no longer allows.

The default image processor does not downscale the image either: `SGLANG_IMAGE_MAX_PIXELS` defaults to 16384 × 28 × 28 ≈ 12.8 M pixels, so a
2.07 M-pixel photo passes through at full resolution. The default server configuration therefore accepts an image that the default compiled
range cannot serve, and the result is an engine outage rather than a slow answer or an HTTP 4xx.

**Expected behavior.** One of:

1. The image runs through the uncompiled encoder, slower but correct, as MoonViT and Gemma4 already do (the warning should then say "running
   this image eagerly").
2. Or the request is rejected at validation with HTTP 400 or 413 stating the maximum supported image size.
3. In every case the scheduler survives; a per-request failure never escapes the event loop.

## Environment

Still reproduces on `main` @ `2cec1cf194` (2026-10-09, 1920×1080 image through a PD pair: `RuntimeError: Detected recompile when
torch.compile stance is 'fail_on_recompile'`, prefill engine exits).

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips / 8 TensorCores each). Host a: prefill engine. Host b: decode engine. TP 8, EP 8 on each. Also applies to a single monolithic server |
| Model | `Qwen/Qwen3-VL-30B-A3B-Instruct` (BF16 weights), KV cache fp8_e4m3, page size 128, `--enable-torch-compile`, default `--mm-process-config` |
| Relevant defaults | `SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES=4900`, `SGLANG_IMAGE_MAX_PIXELS=12845056`, no `max_pixels` in `--mm-process-config` |

## Code trace

All links at commit 09d1244c26.

**1. Precompiled bucket ladder.** The Qwen3-VL vision policy computes square-image patch-count buckets up to the env limit (default 4,900, i.e. 70 × 70 patches):
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/graph_runner/vit_tpu_compile_runner.py#L486-L499
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/environ.py#L1895

The server log at startup lists the 29 buckets it warms (`VisionTpuCompilePolicy.precompile: [29/29] patch_count=4900`).

**2. Warmup ends and recompiles are forbidden.** After warmup the scheduler arms the global stance:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/scheduler.py#L5187-L5208

Log: `[tpu] Warmup complete; froze torch.compile via set_stance('fail_on_recompile').`

**3. An oversized image is not bucketed and is sent to the compiled function anyway.** `bucket_for_patches` returns the raw count with a
warning that promises a "one-time recompile"; `run` then calls `compiled_fn` unconditionally:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/graph_runner/vit_tpu_compile_runner.py#L616-L634
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/graph_runner/vit_tpu_compile_runner.py#L348-L364

The uncompiled forward is saved at install time but has no caller in this class:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/graph_runner/vit_tpu_compile_runner.py#L155-L158
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/graph_runner/vit_tpu_compile_runner.py#L186

**4. The sibling policies do it right.** MoonViT and Gemma4 fall back to `eager_fn` when no bucket fits:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/graph_runner/moonvit_tpu_compile_runner.py#L189-L214
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/graph_runner/gemma4_vision_tpu_compile_runner.py#L176-L194

The Qwen3-VL policy's run adapter has no such branch:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/graph_runner/vit_tpu_compile_runner.py#L450-L484

**5. The image is never downscaled.** `smart_resize` only shrinks an image above `max_pixels`, whose default is 16384 × 28 × 28:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/multimodal/processors/qwen_vl.py#L59-L61
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/multimodal/processors/qwen_vl.py#L153-L180
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/environ.py#L1416

The in-tree Qwen3-VL-8B recipe works around this with `--mm-process-config '{"image":{"min_pixels":65536,"max_pixels":313600}}'`; the 30B
recipe path we used has no such cap:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/benchmark/tpu/recipe/qwen3_vl_8b/server.sh#L29-L31

**6. The failure propagates out of the scheduler.** Traceback (identical on all ranks):

```
[TP0 EP0] ViTTpuCompileRunner: patch count 8160 exceeds the largest precompiled bucket 4900; this image triggers a one-time ViT recompile.
          Raise SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES, or lower max_pixels via --mm-process-config.
[TP0 EP0] Scheduler hit an exception: Traceback (most recent call last):
  File ".../disaggregation/prefill.py", line 773, in event_loop_overlap_disagg_prefill
  File ".../managers/scheduler.py", line 4670, in run_batch
  File ".../managers/tp_worker.py", line 646, in forward_batch_generation
  File ".../model_executor/model_runner.py", line 1783, in forward
  File ".../model_executor/runner/eager_runner.py", line 385, in _execute_extend
  File ".../models/qwen3_vl.py", line 1801, in forward
  File ".../managers/mm_utils.py", line 766, in general_mm_embed_routine
  File ".../managers/mm_schedule.py", line 354, in _encode_vision_items
  File ".../hardware_backend/tpu/mm_schedule_tpu.py", line 79, in encode_vision_items
  File ".../models/qwen3_vl.py", line 1546, in get_image_feature
  File ".../models/qwen3_vl.py", line 1589, in _get_visual_feature
    visual_features = self.visual(pixel_values, grid_thw=grid_thw)
  ...
RuntimeError: Detected recompile when torch.compile stance is 'fail_on_recompile'.
```

followed by `Rank 0 scheduler died` and the launcher's `exited with -9`.

## How we detected it

Test case "multimodal matrix" of our 1-prefill + 1-decode validation campaign, run with `scripts/mm_request.py`. The tool draws a synthetic
PNG at a requested resolution, embeds it as a `data:` URL in a chat request, and records status, token counts and latency; the harness checks
both engines' `/health` after each case.

- 1 × 448², 4 × 448², 16 × 448² (3,198 image tokens) and 1 × 1024² answered normally.
- 1 × 1920 × 1080 returned HTTP 500 after 65 s; the prefill `/health` check then returned no HTTP status. The prefill log showed the warning
  and the traceback above, 10 s after the `Warmup complete; froze torch.compile` line of that session.
- On a fresh pair the next day the same request crashed the engine again; 896 × 504 (1,512 patches) on the relaunched pair worked.
- The patch count explains the apparent contradiction with the 1024² success: the bucket ladder is built from square images, so a 4,900-patch
  cap allows about 1120 × 1120 but not 1920 × 1080, whose 8,160 patches exceed it although its pixel area is similar.

## How to reproduce

1. Start Qwen3-VL-30B-A3B-Instruct on TPU with `--enable-torch-compile` and no `max_pixels` cap (one monolithic server is enough).
2. Wait for `Warmup complete; froze torch.compile via set_stance('fail_on_recompile')` in the log.
3. Send one chat request with one 1920 × 1080 image, for example:
   `python scripts/mm_request.py --url $SERVER --model qwen3-vl-30b --num-images 1 --resolution 1920x1080 --max-tokens 48 --temperature 0`
   (any PNG of that size as a `data:` URL in `/v1/chat/completions` behaves the same).
4. `curl -s -o /dev/null -w '%{http_code}\n' $SERVER/health`.

**Observed.** Step 3 returns HTTP 500 after about 65 s. Step 4 returns `000`. The log shows the `ViTTpuCompileRunner: patch count 8160 exceeds
the largest precompiled bucket 4900` warning followed by `RuntimeError: Detected recompile when torch.compile stance is 'fail_on_recompile'`
on every rank.
**Expected.** HTTP 200 with a slower first token (eager encoder), or HTTP 400/413; the server stays healthy either way.

Controls: the same request with `--mm-process-config '{"image":{"max_pixels":313600}}'` should downscale the image and succeed; with
`SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES=8192` the image should land in a precompiled bucket and succeed.

## Fix proposal

1. **Port the eager fallback to the generic ViT runner.** In `VisionTpuCompilePolicy.run` (or in `Qwen3VLVisionTpuCompilePolicy._install_run_adapter`),
   when `bucket_for_patches(total)` returns an unbucketed count and `self.eager_fn` is set, call `eager_fn` under `torch.no_grad()` exactly as
   `MoonVitTpuCompilePolicy._run_one` and `Gemma4VisionTpuCompilePolicy` do, and change the warning to "running this image eagerly". This is the
   minimal fix and it matches the behavior the other two policies already ship.
2. **Reject what cannot be served.** If eager execution is not wanted for a deployment, validate at request time: compute the patch count from
   `grid_thw` in the processor and return HTTP 413 with the maximum supported pixels when it exceeds the top bucket. Also consider shipping a
   `max_pixels` default for Qwen3-VL on TPU that matches the compiled range (the 8B recipe's 313,600 or a value derived from
   `SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES`), so the default configuration is self-consistent.
3. **Contain the failure.** A `RuntimeError` from the multimodal encoder for one request should fail that request, not the scheduler. Catching
   encoder exceptions in `_encode_vision_items` (or at the `run_batch` boundary for multimodal items) and aborting the affected requests with a
   500 would keep the engine up for this and any similar shape surprise. #1068 proposes failing CI on serving-time recompiles, which would have
   caught the gap between the warning text and the stance.

## Related

- #488: a constrained-decoding request triggers a recompile and the `fail_on_recompile` stance kills the scheduler. Same failure class, different
  trigger; the fix proposed there (recompile allowed, or request rejected, never a crash) also covers this report.
- #1068: fail tests on serving-time recompiles.
- #1069: request for a nightly Qwen3-VL-30B test; a 1920 × 1080 image should be part of it.

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/A/chat/qwen3-vl/` and `results/phase_A_qwen_redo.log` (section "(4) F-10 repro",
both reproductions with status and health), prefill log of `A_qwen_128k_run1` on host a (lines 4987 and 7647 ff.), `results/bug_reports/F-10.md`.
