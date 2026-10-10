## Problem

With `--enable-torch-compile`, only the vision encoder of `Qwen3-VL-30B-A3B-Instruct` is compiled; the MoE language model runs eager on every decode step (#1136): about 6 tokens/s for one request (160 ms per step) on a TPU v7x host where gpt-oss-120b decodes at 176 tokens/s. Outputs are correct and the log is clean, so it looks like a slow model rather than a defect.

## Cause

`qwen3_vl.py` attaches `Qwen3VLDecoderTpuCompilePolicy` to `Qwen3LLMModel`, the dense language model. The MoE model serves its decoder through `Qwen3MoeLLMModel` (`qwen3_vl_moe.py`), which carried no policy. `TpuGraphRunner` applies its whole-model fallback only when *no* policy installed; the vision policy did install, so the fallback was skipped and the decoder never compiled.

## Change (2 files)

- `models/qwen3_vl_moe.py`: annotate `Qwen3MoeLLMModel` with `Qwen3VLDecoderTpuCompilePolicy` (the policy already serves the Qwen3.5 MoE decoder; it only diverges from the plain decoder policy when the owner exposes the VL input_embeds path, as this class does). The TPU policy import is platform-guarded at the top of the file, mirroring `qwen3_vl.py` / `qwen3_5.py`.
- `model_executor/tpu_graph_runner.py` (TPU-only file): warn when compile policies installed but none covers the decoder, so a vision-only annotation shows up in the log as "decode steps run eager" instead of as a slow server.

## Tests

Added to the existing files: `test/registered/unit/models/test_qwen3_vl_tpu_compile_annotation.py` checks that the MoE language model class carries the decoder policy and that the multimodal wrapper stays unannotated; `test/registered/unit/hardware_backend/tpu/test_tpu_graph_runner.py` checks that the eager-decoder warning fires for a vision-only annotation and not for an annotated decoder.

```
pytest test/registered/unit/hardware_backend/tpu/test_tpu_graph_runner.py test/registered/unit/models/test_qwen3_vl_tpu_compile_annotation.py
34 passed, 4 subtests passed
```

## Live validation

Running on a TPU v7x host (monolithic Qwen3-VL-30B-A3B, bf16, TP 8 / EP 8, the server flags from #1136): this branch against upstream main, same prompts. Decode throughput, compile-policy log lines and output identity will be added here before the PR leaves draft.

Fixes #1136
