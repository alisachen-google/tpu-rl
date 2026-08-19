# Rollout-side tuning from aysu's gemma4-e2b serving run (ubench `aysu-ubench-8terxbx7`)

Source: gs://ubench-logs/aysu-ubench-8terxbx7 (2026-07-30, bodaborg-tpu7x-nap, 2x2x1,
image `vllm/vllm-tpu:nightly-20260730-30dcf7a-48aa8d8`, InferenceX client
`89ce6098ef2bc4576a735c43f39c7d972b091cfc`, 16k in / 4k out, 1024 prompts, concurrency 218).
Copies of its resolved config and helm values are alongside this file.

Server command (from the recipe):
```
USE_BATCHED_RPA_KERNEL=1 VLLM_ENGINE_READY_TIMEOUT_S=60000 vllm serve google/gemma-4-E2B-it \
  --tensor-parallel-size=4 --data-parallel-size=2 --max-model-len=20480 --max-num-batched-tokens=4096 \
  --block-size=256 --no-enable-prefix-caching --kv-cache-dtype=fp8 --no-async-scheduling \
  --disable-chunked-mm-input --enable-auto-tool-choice --tool-call-parser=gemma4
```
Engine log confirms: "Using experimental batched RPA kernel", kv fp8, block 256, prefix caching off,
GPU KV cache 4.94M tokens per DP rank.

## Mapping onto our GRPO rollout (MaxText train_rl → tunix VllmSampler → vLLM EngineArgs)

| Knob | Verdict | Mechanism in our stack |
|---|---|---|
| `USE_BATCHED_RPA_KERNEL=1` | ADOPT (rung) | env var; our image's tpu_inference fork (be9cdf04/7aca5d42) has it in `envs.py` |
| `block_size=256` | ADOPT (rung) | not in rl.yml; `patches/patch_engine_kwargs.py` forwards `RL_VLLM_ENGINE_KWARGS` json into `rollout_vllm_kwargs` |
| `kv_cache_dtype=fp8` | OPTIONAL, separate table | same mechanism; breaks the STRICT-BF16 headline (GEMMA_BENCH_PLAN R5) |
| `async_scheduling=False` | already on | tianyu stack |
| `enable_prefix_caching=False` | REJECT | serving used random prompts; GRPO shares each prompt across 8 generations — keep True |
| tp=4 (dp=2 per 4-chip host) | ADOPT as knob | `R1_TP=4` (→ dp16 on the 64-device sampler half at 4x4x4) vs `R1_TP=8` |
| `max_num_batched_tokens=4096` | keep ours (8448) | probe later; ours = max_target_length + kv_cache_buffer |
| `VLLM_ENGINE_READY_TIMEOUT_S=60000` | adopt (harmless) | launcher default |
| `disable_chunked_mm_input`, tool parser, newer vllm-tpu nightly | N/A | serving-only / image pinned to jax 0.10.2 for Pathways |

Launcher switches (both r1 launchers): `USE_BATCHED_RPA_KERNEL`, `RL_VLLM_ENGINE_KWARGS`, `R1_TP`,
`R1_RPA` (RPA_V3_DECODE_BLOCK_SIZES). Chart overlay `values-r1-smoke-tuned.yaml` carries the tuned smoke.
