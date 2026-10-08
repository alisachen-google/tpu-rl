# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [TPU] Qwen3-VL-30B-A3B language model runs eager with `--enable-torch-compile`: no decoder compile policy on `Qwen3MoeLLMModel`

**Labels:** bug, tpu

---

## Summary

With `--enable-torch-compile`, only the vision encoder of `Qwen3-VL-30B-A3B-Instruct` is compiled. The MoE language model runs in eager mode
for every decode step. Decode speed is about 6 tokens/s for one request (about 160 ms per step). On the same hosts and software stack,
gpt-oss-120b decodes at 176 tokens/s (5.7 ms per step), although it activates more parameters per token.

The cause is a missing registration. `qwen3_vl.py` attaches the decoder compile policy only to the dense language model class. The MoE class
used by the 30B-A3B model has no policy. Because the vision policy was installed, the graph runner treats the model as handled and skips its
whole-model fallback, so the decoder never compiles. Nothing in the log warns about this.

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | TPU v7x, one host, 4 chips / 8 TensorCores, TP 8, EP 8 |
| Model | `Qwen/Qwen3-VL-30B-A3B-Instruct` (BF16 weights), KV cache fp8_e4m3, page size 128 |

## Code

- Dense Qwen3-VL language model gets the decoder policy:
  https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/models/qwen3_vl.py#L1380-L1388
- The MoE variant defines `Qwen3MoeLLMModel` and selects it as `language_model_cls`, with no `support_torch_compile` call anywhere in the file:
  https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/models/qwen3_vl_moe.py#L43
  https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/models/qwen3_vl_moe.py#L204-L210
- The graph runner applies the whole-model fallback only when **no** policy was installed. The vision policy installs, so the fallback is skipped:
  https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/model_executor/tpu_graph_runner.py#L274-L295
- The same `Qwen3VLDecoderTpuCompilePolicy` already serves a MoE decoder (Qwen3.5), so it is suitable for `Qwen3MoeLLMModel`:
  https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/models/qwen3_5.py#L2377-L2380
  https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/compilation/decoder_compile_policy.py#L1473

## How we detected it

1. We ran monolithic servers for gpt-oss-120b and Qwen3-VL-30B-A3B with identical flags (`--enable-torch-compile`, TP 8, EP 8) as golden
   references for a prefill/decode disaggregation test campaign.
2. Text-only requests with 1,024 new tokens took 7 to 13 s on gpt-oss and 160 to 195 s on Qwen3-VL. For a model that activates about 3B
   parameters per token (Qwen) against about 5B (gpt-oss), the Qwen result is about 25 times too slow for the hardware.
3. The Qwen server log contains only one compile policy line, for the vision tower, and no decoder policy:
   ```
   [TP0 EP0] VisionTpuCompilePolicy: installed qwen3_vl_vit_encoder on ViTTpuCompileRunner.forward.
   [TP0 EP0] VisionTpuCompilePolicy.precompile: warming 29 patch counts [196, 256, ..., 3364]
   [TP0 EP0] [TPU] TpuGraphRunner precompile done
   ```
   The gpt-oss log shows the whole-model fallback instead (`declares no TPU compile policies; falling back to DecoderTpuCompilePolicy`).
4. Every `Decode batch` line for Qwen reports `gen throughput (token/s): 6` to `13` (two requests) and the step time matches eager dispatch
   overhead, not compute. Outputs are correct (token-identical to a second server, needles retrieved), so this is a performance defect.

## How to reproduce

1. Start the server (one TPU v7x host):
   ```
   python -m sglang.launch_server --device tpu --dtype bfloat16 --host 0.0.0.0 --port 30000 \
     --model-path Qwen/Qwen3-VL-30B-A3B-Instruct --served-model-name qwen3-vl-30b \
     --tp-size 8 --ep-size 8 --attention-backend pallas_tpu_rpa_v3 --page-size 128 --kv-cache-dtype fp8_e4m3 \
     --context-length 140288 --max-total-tokens 786432 --max-running-requests 4 \
     --chunked-prefill-size 8192 --max-prefill-tokens 8192 --tpu-compile-tokens 512 1024 2048 4096 8192 --tpu-compile-bs 1 2 4 \
     --enable-torch-compile --torch-compile-max-bs 4 --mem-fraction-static 0.5 --sampling-backend pytorch \
     --skip-server-warmup --enable-metrics --grammar-backend none
   ```
2. Wait for `The server is fired up and ready to roll!`.
3. Send one text request:
   ```
   curl -s http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
     -d '{"text":"Write a long story about a lighthouse keeper.","sampling_params":{"temperature":0,"max_new_tokens":1024,"ignore_eos":true}}'
   ```
4. Read the server log.

**Observed:** the request takes about 160 s. `Decode batch` lines show `gen throughput (token/s): 6`. The startup log has no decoder compile policy.
**Expected:** a decoder policy is installed (or the whole-model fallback runs), and decode speed is in the range of other compiled MoE models
on this hardware (about 5 to 10 ms per step).

## Proposed fix

Register the existing policy for the MoE language model, mirroring `qwen3_vl.py`:

```diff
--- a/python/sglang/srt/models/qwen3_vl_moe.py
+++ b/python/sglang/srt/models/qwen3_vl_moe.py
@@
 from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
 from sglang.srt.runtime_context import get_exec
+from sglang.srt.compilation.support_torch_compile import support_torch_compile
+from sglang.srt.platforms import current_platform
 from sglang.srt.utils.hf_transformers_utils import get_processor
@@
+if current_platform.supports_torch_compile():
+    from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
+        Qwen3VLDecoderTpuCompilePolicy,
+    )
+
+    support_torch_compile(
+        Qwen3MoeLLMModel,
+        policy=Qwen3VLDecoderTpuCompilePolicy,
+    )
+
+
 class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
```

Alternative or additional hardening in `TpuGraphRunner`: apply the whole-model fallback (or log a warning) when policies were installed only on
non-decoder modules, so a missing decoder registration cannot pass silently.

## Validation of the patch on hardware (2026-10-08)

| Measure | Eager (main @ 09d1244c26) | With patch |
|---|---|---|
| Decoder policy in log | none | `Installed Qwen3VLDecoderTpuCompilePolicy on Qwen3MoeLLMModel.forward` |
| Decode speed, one request | 6 to 13 tokens/s | 178 to 183 tokens/s |
| 1,024-token answer | 160 to 195 s | 5.9 to 7.1 s |
| Cold start | 474 s | about 1,160 s; decoder graph capture alone 1,094 s |
| Compile artefacts | small | 65 GB TorchInductor cache + 51 GB tier-3 cache |
| Serving-time recompiles over 35 prompt lengths (1 to 16,385 tokens) | 0 (nothing compiled) | 0 |
| Outputs vs eager | reference | not token-identical (divergence after 9 to 83 tokens); needles retrieved correctly in both |

The numerical difference between compiled and eager kernels is expected and does not affect accuracy: GSM8K (100 examples, greedy, 1,024
max tokens) scores 0.99 with the patch against 0.97 eager, and the 100 problems take 29 s instead of about 17 minutes. A warm start with
existing compile caches takes 421 s. Two side observations for reviewers: the decoder capture for this model takes 18 minutes, and the compile caches need over 100 GB of
space, which exhausted a 96 GB root disk on the first attempt.

## Related

- #1069 asks for a nightly Qwen3-VL-30B-A3B test; such a test with a latency gate would have caught this.
