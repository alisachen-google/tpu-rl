# Description

`scatter_layer_kv_caches` wrote every decoder layer's entry back to its physical KV cache, with the highest-numbered layer winning. For models with **KV sharing** the decoder only updates the donor layers; the sharing layers' entries are still the gathered pre-forward cache objects. Writing those back overwrote the donors' freshly written K/V on every forward pass.

On **gemma4-e2b** (35 layers; layers 15–34 share the caches of layers 13 and 14, so tpu-inference allocates 15 physical caches), the shared layers therefore attended over empty caches and greedy decoding degenerated into repeated tokens ("Here is the trucks trucks trucks …"). GRPO rollouts through the MaxText vLLM adapter were unusable: GSM8K accuracy at step 0 was **0.00 %** (100 % of answers without `\boxed{}`), identically across weight-sync paths and checkpoint conversions, because the weights were fine.

**Fix:** skip entries that are still the gathered input object (`layer_kv_caches[lyr] is kv_caches[idx]`): the decoder did not touch that cache through this layer, so there is nothing to write back. Among entries the decoder *did* replace, the highest-numbered layer still wins, so behaviour for models without KV sharing — and the existing `test_shared_physical_cache_is_visible_to_every_sharing_layer` semantics — are unchanged. The check is a Python identity test on the traced list, so it costs nothing at runtime.

Why here rather than in the gemma4 decoder: the adapter's contract is a layer-ordered view (`kv_caches[lyr]`), and the write-back is the only place that knows several layers alias one physical cache. Fixing it there also protects any future model with KV sharing.

# Tests

Unit (CPU), `tests/post_training/unit/vllm_hybrid_cache_test.py`:
- `test_unchanged_sharing_layers_do_not_clobber_the_donor_update` — a sharing layer that only reads keeps the donor's update.
- `test_gemma4_e2b_kv_sharing_round_trip_keeps_every_donor_update` — gemma4-e2b shaped 35-layer / 15-cache resolve → gather → donor update → scatter keeps all 15 updates.

Both new tests fail without the fix and pass with it; the existing 7 tests in `LayerOrderedKvCachesTest` still pass (9 passed).

```
PYTHONPATH=src:. python3 -m pytest -q tests/post_training/unit/vllm_hybrid_cache_test.py -k LayerOrderedKvCachesTest
9 passed
```

End to end (TPU v7x, `tpu7x` 4x4x4, 64 chips, Pathways; `train_rl` GRPO on gemma4-e2b-it / GSM8K through `MaxTextForCausalLM` on vLLM-TPU; image `maxtext-rl-ironwood:2026-09-22-head` with only this change applied on top):

| | GSM8K test accuracy at step 0 (1319 questions, greedy) | sample output |
|---|---|---|
| without the fix | 0.00 % (100 % no `\boxed{}`) | `Here is the trucks trucks trucks trucks …` |
| **with this change** | **80.82 %** | `Here is the step-by-step calculation: **Step 1: …` |
| reference (same model on GPU / verl) | 80.31 ± 0.24 % | |

Training then proceeds normally (sampler↔trainer k3 KL ≈ 4e-4, ESS ≈ 0.999); a 250-update run with the equivalent fix reached 91.81 % vs 92.42 ± 0.42 % on GPU.

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have run end-to-end tests and provided workload links above if applicable.
- [ ] I have made or will make corresponding changes to the doc if needed (not needed: internal helper behaviour).
