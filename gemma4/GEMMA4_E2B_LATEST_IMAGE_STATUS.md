# Gemma4-E2B RL recipe on the latest image (`2026-09-22-head`): status 2026-10-29

**Request:** organize the 35 s recipe with the latest image for every library and verify there is no
performance regression with a 10-update rerun (at `xla_tpu_dvfs_p_state=6`).

**Result: not runnable on this image without code changes.** No valid step time could be measured.
The pinned image (`maxtext-rl-ironwood@sha256:f7c957d0…`) remains the recipe image for both the
35 s parity run and the 24 s optimized run.

## Image under test

`gcr.io/supercomputer-testing/maxtext-rl-ironwood@sha256:4dc6ca720017bf93b6bae9524519cf5b7a8b16706ca6aab2a355af26d75a6bf3`
(tag `2026-09-22-head`): jax 0.11.0, libtpu 0.0.44, orbax 0.12.4, flax 0.12.8, vllm 0.28.1rc1,
transformers 5.12.1; Pathways `20261028-jax_0.11.0`.

## Compatibility mapping done (all in `parity/build_g4pl10.py`)

* 13 of the 30 perf patches no longer apply; most of their levers are native now
  (`DP_SCHED_ROUTING=round_robin`, `free_kv_cache_during_weight_sync=false`, `vllm_block_size=128`,
  tunix `overlap_postprocessing`, jitted unroll, `eos_tokens`). The sharded-sampling patch must be
  skipped (`sample() got an unexpected keyword argument 'allow_distributed_sampling'`: distributed
  sampling is native in tpu_inference).
* The parity patcher was re-based (`parity/patches/patch_sampler_is_newimg.py`, applies cleanly).
* The RL config gained `use_weight_converter`, `use_standalone_converter`, `use_hf_mapping`; the
  vLLM gemma4 model changed its QKV layout and `maxtext_vllm_rollout.py` was rewritten (674 lines).

## Four runs, four distinct blockers

| run | variant | outcome | evidence |
|---|---|---|---|
| g4pl10 | default (`use_weight_converter=True`) | rollout model emits gibberish: step-0 eval 0.00 % (100 % no-box, e.g. `trucks trucks trucks …`); exit 1 at the first training step, 0 train steps | `parity/logs/latest_image/g4pl10_default_sync_key_lines.log` |
| g4pl10a | legacy weight sync (`use_weight_converter=false`) | identical gibberish; first training step crashes in `attention_op.create_sa_config`: `'<' not supported between instances of 'int' and 'NoneType'` | `g4pl10a_legacy_sync_key_lines.log` |
| g4pl10b | checkpoint re-converted with this image's `to_maxtext` (`gemma4-e2b-it-newimg`, 5 min) | identical gibberish, so the cause is not checkpoint-format drift | `g4pl10b_reconverted_ckpt_key_lines.log` |
| g4pl10d | native vLLM gemma4 model + `use_standalone_converter=true` (no `maxtext_config`), flash patch skipped, local splash fields set | `ValueError: gemma4-e2b vLLM weight mapping not found` (`integration/tunix/weight_mapping/` ships deepseek3, gemma3, gpt_oss, llama3, qwen2, qwen3 only) | `g4pl10d_native_vllm_standalone_key_lines.log` |

### Diagnosis

1. **Rollout weights are wrong on this image for gemma4 via the MaxText-in-vLLM path.** The
   gibberish is identical across both sync paths and across two checkpoint conversions, so it is
   in the rewritten `maxtext_vllm_rollout.py` / vLLM gemma4 model pairing itself (the trainer's
   checkpoint restore logs are clean). The alternative path (native vLLM gemma4 model) is missing
   its weight mapping entirely.
2. **Trainer side:** the new `attention_op.py` reads the local-attention splash fields
   (`local_sa_block_q`, …) directly; the RL config leaves them `None`, so the first training step
   crashes. Setting them explicitly (`local_sa_*=512`, layouts `HEAD_DIM_MINOR`,
   `eval_local_sa_*`) is the fix prepared in `build_g4pl10.py` (`NOFLASH=1`), but it could not be
   validated because the rollout blocker comes first in the step.

Both are upstream MaxText/tunix issues on the gemma4 integration, not configuration. The 10-step
perf regression check therefore has no valid number; the step-0 times seen (149 s, 51 s) are not
comparable to the 34.6 / 36.2 s pinned-image baselines because the model was producing garbage.

## Manifests (all p-state 6, 10 updates)

`parity/manifests/g4pl10-latest-image-10step.yaml`, `g4pl10a-latest-image-legacy-sync-10step.yaml`,
`g4pl10b-latest-image-reconverted-ckpt-10step.yaml`, `g4pl10d-latest-image-native-vllm-10step.yaml`.
Builder knobs: `WCONV=0` (legacy sync), `NEWCKPT=1` (re-converted checkpoint), `NOFLASH=1`
(native splash fields + explicit local_sa settings), `NATIVE=1` (native vLLM model).

## What a working port needs

* A gemma4 entry in `maxtext/integration/tunix/weight_mapping/` (standalone path), or a fix to the
  MaxText-in-vLLM gemma4 weight sync in `maxtext_vllm_rollout.py`.
* RL config defaults for the `local_sa_*` / `eval_local_sa_*` fields (or the explicit settings above).
* Re-run `NOFLASH=1 python3 build_g4pl10.py` once the rollout path is fixed; the parity patcher and
  the remaining 17 perf patches already apply.
