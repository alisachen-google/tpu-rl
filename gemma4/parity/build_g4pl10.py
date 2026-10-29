"""Build g4pl10.yaml: the 35 s GSM8K parity recipe (seed 42) on the LATEST image, 10 updates, to check for a
performance regression against the pinned image (34.6 s / 36.2 s median per update, seeds 42 / 43).

Latest libs (image 2026-09-22-head @sha256:4dc6ca72...): jax/jaxlib 0.11.0, libtpu 0.0.44, orbax 0.12.4,
flax 0.12.8, vLLM 0.28.1rc1 (tpu_inference), transformers 5.12.1. Pathways server/proxy 20261028-jax_0.11.0.

Patches that no longer apply on the new image are replaced by the knobs the image now has natively:
  patch_dp_rr_routing  -> env DP_SCHED_ROUTING=round_robin (tpu_inference DPScheduler.RoutingPolicy)
  patch_sync_keepkv    -> free_kv_cache_during_weight_sync=false (MaxText config)
  patch_block_size     -> vllm_block_size=128 (MaxText config; the env patch still applies too)
  patch_async_detok / patch_async_logprobs -> tunix VllmSampler overlap_postprocessing=True (default)
  patch_jit_unroll     -> tunix generate/utils._unroll_scanned_layers is jax.jit'ed upstream
  patch_stop_tokens    -> tunix eos_tokens handling upstream (default = chat-turn terminator)
  patch_sharded_sampling -> tpu_inference distributed sampling (native; the patch is DROPPED, see below)
  patch_cd_proper / host_extract / textual_stop / padding_bucket / rpa_blocks -> tpu_inference decode_loop
                          rewrite upstream (verified by phase times in the run, not by code)
  patch_async_reward   -> inert for the parity harness (custom reward); native reward_num_workers left 0
  patch_sampler_is     -> re-based anchors (patch_sampler_is_newimg.py); all other parity edits unchanged
Usage: python3 build_g4pl10.py   (from g4pt1.yaml built with CKPT=1)
"""
import json, os, re, yaml
SP = os.environ.get("PARITY_DIR", os.path.dirname(os.path.abspath(__file__)))
raw = open(f"{SP}/g4pt1.yaml", encoding="utf-8").read()
NEW_IMG = "gcr.io/supercomputer-testing/maxtext-rl-ironwood@sha256:4dc6ca720017bf93b6bae9524519cf5b7a8b16706ca6aab2a355af26d75a6bf3"
OLD_IMG = "gcr.io/supercomputer-testing/maxtext-rl-ironwood@sha256:f7c957d0b286e03966fdd6f9b15e7609fae8460a933a9da0b9fbab3e04723c94"

def repl(old, new, n=1):
    global raw
    c = raw.count(old); assert c == n, f"{old!r}: count {c} != {n}"
    raw = raw.replace(old, new)

repl(OLD_IMG, NEW_IMG)
repl("20260730-jax_0.10.2", "20261028-jax_0.11.0", 3)
# WCONV=0: legacy trainer->rollout weight-sync path (the new image defaults use_weight_converter=True, a
# generic "direct" WeightConverter; the pinned fork image used the gemma4-specific converter). With the
# default, the rollout model produced gibberish at step 0 (0 % accuracy, 100 % no-box) on 2026-10-29.
NAME = "g4pl10"
if os.environ.get("WCONV") == "0":
    repl("        num_batches=250 \\", "        num_batches=250 \\\n        use_weight_converter=false \\")  # before the 250->10 rewrite below
    NAME = "g4pl10a"
raw = raw.replace("g4pt1", NAME)
# ported parity patcher (embedded after the rename so the ConfigMap value equals the file byte for byte)
patch = open(f"{SP}/patches/patch_sampler_is_newimg.py", encoding="utf-8").read()
m = re.search(r'^(\s*)patch_sampler_is\.py: ".*"$', raw, re.M); assert m
raw = raw[:m.start()] + f"{m.group(1)}patch_sampler_is.py: {json.dumps(patch)}" + raw[m.end():]
# drop patches that the new image has upstreamed AND whose stale replacement breaks the native call:
#   patch_sharded_sampling -> tpu_inference distributed sampling (sampling.py `allow_distributed_sampling`,
#   `_distributed_sampling_*`); its appended `sample(rng, mesh, logits, md)` lacks the new kwarg ->
#   "TypeError: sample() got an unexpected keyword argument 'allow_distributed_sampling'" (g4pl10 08:15Z).
# Removing the ConfigMap key makes the launcher's `if [ -f ... ]` skip it.
# the launcher applies it under `if [ -f ... ]`; neutralise the apply line instead of deleting the (block-scalar) key
repl("  python3 /workload/patches/patch_sharded_sampling.py",
     "  echo \"SHARDED-SAMPLING PATCH: SKIPPED (native distributed sampling in tpu_inference on this image)\"")
# native knobs
repl("    export ASYNC_REWARD=off", "    export ASYNC_REWARD=off\n    export DP_SCHED_ROUTING=round_robin   # native round-robin DP routing (replaces patch_dp_rr_routing on this image)")
repl("        num_batches=250 \\", "        num_batches=10 \\\n        free_kv_cache_during_weight_sync=false \\\n        vllm_block_size=128 \\")
repl("    maxRestarts: 10", "    maxRestarts: 3")
open(f"{SP}/{NAME}.yaml", "w", encoding="utf-8").write(raw)
docs = [d for d in yaml.safe_load_all(raw) if d]
cm = [d for d in docs if d["kind"] == "ConfigMap" and "patch_sampler_is.py" in d["data"]][0]
assert cm["data"]["patch_sampler_is.py"] == patch
l = [d for d in docs if d["kind"] == "ConfigMap" and "launch-workload.sh" in d["data"]][0]["data"]["launch-workload.sh"]
for k in ("DP_SCHED_ROUTING=round_robin", "num_batches=10", "free_kv_cache_during_weight_sync=false", "vllm_block_size=128", "data_shuffle_seed=42", "max_target_length=3072"):
    assert k in l, k
js = [d for d in docs if d["kind"] == "JobSet"][0]
imgs = sorted({c["image"] for rj in js["spec"]["replicatedJobs"] for c in rj["template"]["spec"]["template"]["spec"].get("containers", []) + rj["template"]["spec"]["template"]["spec"].get("initContainers", [])})
print("images:", imgs)
print(f"BUILT {SP}/{NAME}.yaml")
