# Gemma4-E2B RL recipe on the latest image (`2026-09-22-head`): status 2026-10-30

**Request:** organize the 35 s recipe with the latest image for every library and verify there is no
performance regression with a 10-update rerun (at `xla_tpu_dvfs_p_state=6`).

**Result (2026-10-30): the 24 s configuration runs correctly on the latest image and is faster.** The gibberish was
a KV-cache aliasing bug in the new MaxText-in-vLLM adapter (fixed by `parity/patches/patch_kv_alias_dedupe.py`,
one line); the trainer's `None` local-attention splash fields are set explicitly. Validation run `g4l24f`
(24 s configuration, latest image, p-state 6, 20 updates):

| item | latest image (`2026-09-22-head`) | pinned image (same configuration, p-state 6) |
|---|---|---|
| step-0 accuracy (scorer-v2, 1319 Qs) | **80.89 %** | 81.27 % (GPU band 80.14–80.59) |
| step-20 accuracy | **91.66 %** (periodic) / **91.96 %** (post-training) | 91.66 % (GPU step-20 mean 91.03 ± 0.53) |
| sampler↔trainer k3 KL / ESS | 3.5–3.7e-4 / 0.9993 | 3.8e-4–2.0e-3 / 0.9991 |
| step time, 18 timed updates | **median 28.9 s** (min 27.8, mean 30.8 with two transient outliers 51.5 / 39.1 s) | median 32.1 s |
| generation phase / training phase | ≈ 18.5 s / ≈ 7.5 s | ≈ 26 s / ≈ 4.5 s |
| rollout length / truncation | 460–490 tokens, 0.05–0.9 % at the 2048 cap | same |

**p-state comparison on the latest image** (`g4l24f7`: identical run at `xla_tpu_dvfs_p_state=7`, 10 updates,
user-requested; the recipe default stays p-state 6):

| | p-state 6 (`g4l24f`) | p-state 7 (`g4l24f7`) |
|---|---|---|
| step-0 accuracy | 80.89 % | 80.89 % (bit-identical eval) |
| accuracy after training | 91.66 % @20 (91.96 % post) | 91.21 % @10 |
| timed updates (step ≥ 2) | 18 | 8 |
| median / min | **28.9 s** / 27.8 s | **29.7 s** / 28.3 s |
| median excluding the two deterministic outliers | 28.7 s | 28.6 s |
| outliers (same batches in both runs) | step 5: 51.5 s (42 s generation, straggler rollout); step 6: 39.1 s (training-side recompile) | step 5: 52.0 s; step 6: 43.6 s |

p-state 7 gives no measurable gain on this image (differences are within step-to-step noise), unlike the
+7.5 % seen on the pinned image (23.5 → 25.3 s). The generation phase (~18.5–19 s) dominates and is not
clock-bound at this batch size; p-state 6 remains the recipe setting.
Deploy: `parity/manifests/g4l24f-latest-image-24s-config-kvfix-20step.yaml` via `parity/tools/apply-jobset.sh`;
rebuild with `FIX=patches/patch_kv_alias_dedupe.py STEPS=<n> python3 parity/build_g4l24.py` (`PSTATE=7` for the
comparison variant). Logs: `parity/logs/latest_image/g4l24f_pstate6_key_lines.log`, `g4l24f7_pstate7_key_lines.log`, step times
`g4l24f_pstate6_steptimes.txt`, `g4l24f7_pstate7_steptimes.txt`.

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

## Root cause of the gibberish (2026-10-30)

tpu_inference allocates one KV cache per *non-shared* layer and maps every KV-shared layer to its donor's
cache (`runner/kv_cache_manager.py`, `layer_name_to_kvcache_index`). gemma4-E2B has 35 layers, 20 of them
KV-shared (donors: layer 13 for the sliding layers 15–33, layer 14 for the global layers 19/24/29/34), so there
are 15 physical caches. MaxText's gemma4 decoder indexes `kv_caches` by compact slot 0..14
(`gemma4_small.kv_cache_slot_map`, `nnx_decoders._apply_gemma4_small_layers`).

The OLD adapter passed the 15-entry list straight through. The NEW adapter (`integration/vllm/hybrid_cache_utils.py`,
`resolve_layer_kv_cache_indices` / `gather_layer_kv_caches` / `scatter_layer_kv_caches`) expands the list to one
entry **per layer** (35 entries, 20 of them aliases of caches 13/14), hands it to MaxText, which updates only
entries 0..14, and then scatters all 35 entries back; the untouched alias entries 15..34 (the stale pre-forward
donor caches) overwrite the donors' updates ("highest-numbered layer wins"). Layers 13/14 and the 20 layers sharing
them therefore never see any past K/V, and greedy decoding degenerates into repeated tokens. Weights are correct,
which is why every weight-sync variant produced byte-identical gibberish.

**Fix** (`patch_kv_alias_dedupe.py`, fail-loud, idempotent): present each physical cache once, in first-use layer
order, which is exactly MaxText's slot order:

```python
physical_indices = [by_layer[lyr] for lyr in range(num_layers)]
physical_indices = list(dict.fromkeys(physical_indices))   # _KV_ALIAS_DEDUPE
```

Models without KV sharing have unique indices already, so they are unaffected. The other blocker (trainer crash on
`None` local-attention splash fields) is handled by the explicit `local_sa_*` settings in `parity/build_g4l24.py`.
Diagnosis evidence: CPU simulation inside the new container with the real helpers reports physical caches whose
update is lost = [13, 14] unpatched, [] patched.

**Recipe on the latest image:** `parity/build_g4l24.py` (24 s configuration: cap 2048, async reward, round-robin DP
routing native, `free_kv_cache_during_weight_sync=false`, `vllm_block_size=128`, p-state 6; 33 of the 36 perf
patches still apply, sharded-sampling and flash-sa are native/skipped), applied with `parity/tools/apply-jobset.sh`.
