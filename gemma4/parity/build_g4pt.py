"""Build g4pt{N}.yaml: the TIS-parity run (standard offline path + token TIS), 5120 cap.

From g4pf.yaml (faithful -it ckpt, native template, full 250-step/1319-Q parity recipe,
scorer-v2, cap 2560). g4pf plateaued ~87.6% @ step20 vs GPU 91.03% because TIS was INERT:
the old patch_sampler_is.py set algo_config.sampler_is but the STANDARD GrpoLearner never
reads it (only the agentic learner does) and its anchor didn't exist in tunix 0.1.7.

This builder:
  1. Replaces the g4pf-patches ConfigMap value `patch_sampler_is.py` with the validated
     self-discovering patcher (patch_sampler_is_new.py) that COMPUTES sampler_is_weights in
     grpo_learner.py (via get_actor_per_token_logps) and enables return_logprobs in train_rl.
     The launcher already runs patch_sampler_is.py, so no apply-wiring change is needed.
  2. Raises the response cap 2560 -> 5120 (max_target_length / kv_cache_buffer 3072 -> 5632,
     prefill stays 512) to cut truncation toward the GPU 8K contract while fitting 64 chips
     (8704 OOM'd; 5632 is 1.83x the 3072 that ran clean).
  3. Adds data_shuffle_seed / init_weights_seed for the 3-run gate.
  4. Renames g4pf -> g4pt{N} everywhere (separate ConfigMaps / JobSet / output dir).

Usage: python3 build_g4pt.py <seed_index 1|2|3>
"""
import json
import os
import re
import sys

SP = os.environ.get("PARITY_DIR", os.path.dirname(os.path.abspath(__file__)))
SRC = f"{SP}/g4pf.yaml"
PATCH = f"{SP}/patches/patch_sampler_is.py"
IDX = sys.argv[1] if len(sys.argv) > 1 else "1"
assert IDX in ("1", "2", "3"), IDX
SEED = {"1": 42, "2": 43, "3": 44}[IDX]
# Response cap (tokens). Default 2560 = g4pf's cap (guaranteed to fit on 64 chips; isolates the
# TIS variable cleanly). 5120 adds the middle-ground cap bump (target 5632). 8192 OOM'd.
CAP = sys.argv[2] if len(sys.argv) > 2 else "2560"
assert CAP in ("2048", "2560", "4096", "5120"), CAP
TARGET = {"2048": "2560", "2560": "3072", "4096": "4608", "5120": "5632"}[CAP]  # +512 prefill
NAME = (f"g4pt{IDX}" if CAP == "2560" else f"g4pt{IDX}c{CAP}") + ("p" if os.environ.get("PERF") == "1" else "")
OUT = f"{SP}/{NAME}.yaml"

raw = open(SRC, encoding="utf-8").read()
new_patch = open(PATCH, encoding="utf-8").read()
assert raw.count("g4pf") > 0

# ---- 1) swap ConfigMap patch values (single-line JSON-escaped scalars) ----
#   patch_sampler_is.py -> the TIS config patcher (sampler_is=token + return_logprobs + eval 1319)
#   patch_flash_sa.py   -> sa_block_kv 2048->1024 so splash divides the UNPACKED TIS recompute (3072)
_flash = open(f"{SP}/patches/patch_flash_sa.py", encoding="utf-8").read()
if CAP == "2048":
    # target 2560 = 2^9*5: the unpacked TIS recompute runs at q_seq_len=2560, which 1024 does not divide
    # ("q_block_size=1024 should divide q_seq_len=2560", g4pt1c2048p 2026-10-29). 512 divides 2560 and the
    # packed 8192. Tiling only; no numeric effect.
    assert _flash.count('return "1024"') == 2, _flash.count('return "1024"')
    _flash = _flash.replace('return "1024"', 'return "512"')
SWAPS = {
    "patch_sampler_is.py": new_patch,
    "patch_flash_sa.py": _flash,
    # 2026-10-27: ndarray/bytes gold handling + REWARD-DEBUG line (training rewards were 0.1-capped:
    # tunix passes the batched `answer` column as a numpy array, the old helper replicated the whole
    # array as every completion's gold). Eval path was unaffected.
    "reward_parity_boxed.py": open(f"{SP}/patches/reward_parity_boxed.py", encoding="utf-8").read(),
}


def swap_cm_value(text, key, content):
    val = json.dumps(content)  # JSON string == valid YAML double-quoted scalar
    out, n = [], 0
    for l in text.split("\n"):
        m = re.match(r'^(\s*)' + re.escape(key) + r': "', l)
        if m:
            assert l.rstrip().endswith('"'), f"{key} value is not single-line"
            out.append(f"{m.group(1)}{key}: {val}")
            n += 1
        else:
            out.append(l)
    assert n == 1, f"{key} found {n} times (expected 1)"
    return "\n".join(out)


for _k, _c in SWAPS.items():
    raw = swap_cm_value(raw, _k, _c)

# ---- 2) optional cap change (default 2560 keeps g4pf's 3072 target; isolate TIS) ----
def repl(old, new):
    global raw
    assert raw.count(old) == 1, f"cap anchor {old!r} count {raw.count(old)}"
    raw = raw.replace(old, new, 1)

if CAP != "2560":
    repl("        max_target_length=3072 \\", f"        max_target_length={TARGET} \\")
    repl("        kv_cache_buffer=3072 \\",   f"        kv_cache_buffer={TARGET} \\")

# ---- fail-fast on a persistent crash (e.g. cap OOM) instead of churning to deactivation.
# g4ps1 (8192 cap) hit 23 JobSet restarts under maxRestarts=50 before Kueue deactivated it;
# a low cap makes an OOM surface loudly within one window instead of silently churning.
repl("    maxRestarts: 50", "    maxRestarts: 10")  # 3 was exhausted by a transient head-pod proxy crashloop (2026-10-27 04:12); restarts resume from the last checkpoint now

# ---- NO mid-run checkpoints. g4pt1 (checkpoint_period=20) proved two things on this container:
# (1) resume is BROKEN: every restart died in tunix maybe_restore ("Failed to restore from step 20"
#     <- orbax v1 NoEntryError "incompatible handler" for model_params), so a saved checkpoint turns
#     any restart into a fatal one (3 restarts -> JobSet Failed);
# (2) the step-40 save coincided with the eval and the tensorboardX GCS event-file writer raced
#     itself into a 412 PreconditionFailed that killed the main thread.
# The accuracy curve needs no checkpoint; the initial -it load (load_parameters_path) is separate.
# NOTE: the inherited command already carries `enable_checkpointing=true` AFTER checkpoint_period;
# pyconfig takes the LAST occurrence, so flip that line in place (an inserted duplicate loses).
if os.environ.get("CKPT") == "1":
    # resumable run: patcher edit I makes restore work (RNG-free model_params); save every 20 steps so a
    # Kueue preemption costs <= 20 steps. checkpoint_period=100 in the inherited command -> 20.
    repl("        checkpoint_period=100 \\", "        checkpoint_period=20 \\")
else:
    repl("        enable_checkpointing=true \\", "        enable_checkpointing=False \\")

# ---- precision parity (guide sec 4/47): GPU uses FP32 master + FP32 Adam, BF16 compute.
# rl.yml defaults weight_dtype=bfloat16 (bf16 master) and mu_dtype="" inherits weight_dtype
# (bf16 Adam) -> small per-step updates (||dw||/||w||~1e-6) are lost to bf16 rounding -> undertrain.
# Override the TRAINER to fp32 master + fp32 Adam; compute stays bf16 via `dtype`; the vLLM
# rollout keeps bf16 via its own vllm_additional_config. Anchored on the cosine LR line.
repl("        lr_schedule_type='cosine' \\",
     "        lr_schedule_type='cosine' \\\n"
     "        weight_dtype=float32 \\\n"
     "        mu_dtype=float32 \\\n"
     "        num_epoch=5 \\")
# ---- num_epoch=5: train_rl.py does `train_dataset[:num_batches*batch_size]` (7473 GSM8K rows < 32000)
# then `.repeat(num_epoch)`; the default num_epoch=1 yields 7473/128 = 58 batches and the run exits
# cleanly at step 58 (g4pt1 2026-10-26 23:25, JobSet Completed). GPU = 250 steps ~ 4.3 epochs. 5x58=290
# >= 250; the trainer stops at train_steps=250. ----

# ---- 2a) PERF=1: the two levers of the 24 s recipe the parity harness does not use by default
# (round-robin routing, approx-top-k and EOS-check interval 32 are already in the launcher):
# async reward scoring during the rollout (ASYNC_REWARD=on). Async logprobs stays off (TIS needs the
# sampler logprobs returned synchronously). Combine with CAP=2048 for the 24 s response cap. ----
if os.environ.get("PERF") == "1":
    repl("export ASYNC_REWARD=off", "export ASYNC_REWARD=on")

# ---- 2b) SPOT=1: pin the TPU workers to the spot flavor. The ubench ClusterQueue has tpu7x-flavor
# nominal 0 since 2026-10-26 22:14Z, so every admission is a cohort BORROW that any tenant with nominal
# quota reclaims (g4pt1 was preempted 3x in 2h, once 3 min into training). The queue may also borrow
# up to 160 chips of `tpu7x-spot` from the spot-pool CQ (0 used, reclaimWithinCohort=Never). Adding
# gke-spot=true to the worker nodeSelector makes the reserved flavor's TAS fit fail (no reserved node
# carries that label) so Kueue falls through to tpu7x-spot; Kueue injects that flavor's tolerations.
# Risk moves from Kueue reclamation to GCE spot preemption. Same chips, same recipe. ----
if os.environ.get("SPOT") == "1":
    repl("            nodeSelector:\n              cloud.google.com/gke-tpu-topology: \"4x4x4\"\n              cloud.google.com/gke-tpu-accelerator: tpu7x\n",
         "            nodeSelector:\n              cloud.google.com/gke-tpu-topology: \"4x4x4\"\n              cloud.google.com/gke-tpu-accelerator: tpu7x\n              cloud.google.com/gke-spot: \"true\"\n")

# ---- 2c) NON-DEFAULT PATHWAYS PORTS. The head pod is hostNetwork with fixed ports; other tenants'
# Pathways heads (e.g. default/lb-xprof-c4-pathways-head-0-0, a non-JobSet pod our preferred
# anti-affinity cannot see) bind 29000/29001 on the same cpu-np node -> our proxy "Address already in
# use" and our RM aborts on a strict-compat pipe to THEIR RM; 10 JobSet restarts in 7 min, JobSet
# Failed (2026-10-27 04:12 and 04:33). Proxy 29000->29140, RM 29001->29141 (worker ports stay). ----
PORTS = {
    "export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000": "export JAX_BACKEND_TARGET=grpc://127.0.0.1:29140",
    "value: grpc://g4pf-pathways-head-0-0.g4pf:29000": "value: grpc://g4pf-pathways-head-0-0.g4pf:29140",
    "- --server_port=29001": "- --server_port=29141",
    "- --server_port=29000": "- --server_port=29140",
    "- --resource_manager_address=g4pf-pathways-head-0-0.g4pf:29001": "- --resource_manager_address=g4pf-pathways-head-0-0.g4pf:29141",
}
for _o, _n in PORTS.items():
    _c = raw.count(_o)
    assert _c >= 1, f"port anchor {_o!r} not found"
    raw = raw.replace(_o, _n)
assert "29000" not in raw and "29001" not in raw, "default pathways ports still present"

# ---- DVFS: the cluster does not allow xla_tpu_dvfs_p_state=7 (user, 2026-10-29); use 6 everywhere ----
assert raw.count("--xla_tpu_dvfs_p_state=7") >= 1, "p_state anchor"
raw = raw.replace("--xla_tpu_dvfs_p_state=7", "--xla_tpu_dvfs_p_state=6")

# ---- 3) add seed knobs after rollout_micro_batch_size=128 ----
ANCHOR = "        rollout_micro_batch_size=128 \\"
assert raw.count(ANCHOR) == 1, f"seed anchor count {raw.count(ANCHOR)}"
raw = raw.replace(
    ANCHOR,
    ANCHOR + "\n"
    f"        data_shuffle_seed={SEED} \\\n"
    f"        init_weights_seed={SEED} \\",
    1,
)

# ---- 4) rename g4pf -> g4ptN everywhere ----
raw = raw.replace("g4pf", NAME)

open(OUT, "w", encoding="utf-8").write(raw)

# ---- verify ----
v = open(OUT, encoding="utf-8").read()
checks = {
    f"renamed to {NAME}": v.count("g4pf") == 0 and v.count(NAME) > 0,
    "new TIS patcher embedded": "_TIS_PARITY" in v and "sampler_is=" in v,  # quotes get JSON-escaped in YAML
    "old inert env-marker gone": "_SAMPLER_IS_PARITY" not in v,  # old patch's injected marker
    "return_logprobs enabled by patch": "return_logprobs=True,  # _TIS_PARITY" in v,
    "eval full-coverage (1319)": "drop_remainder=False)  # _TIS_PARITY" in v,
    "splash kv block divides unpacked recompute": ('startswith(\\"sa_block_kv\\"): return \\"%s\\"' % ("512" if CAP == "2048" else "1024")) in v,
    f"cap target {TARGET}": f"max_target_length={TARGET}" in v,
    f"cap kv {TARGET}": f"kv_cache_buffer={TARGET}" in v,
    "prefill 512 kept": "max_prefill_predict_length=512" in v,
    f"data_shuffle_seed={SEED}": f"data_shuffle_seed={SEED}" in v,
    f"init_weights_seed={SEED}": f"init_weights_seed={SEED}" in v,
    "it-fresh checkpoint": "gemma4-e2b-it-fresh/0/items" in v,
    "native processor": "gemma_data_parity_native.py" in v,
    "250 updates": "num_batches=250" in v,
    "scorer-v2": "reward_functions='compute_score'" in v,
    "conversion guarded": "skipping conversion" in v,
    "patch still applied by launcher": "python3 /workload/patches/patch_sampler_is.py" in v,
}
for k, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {k}")
bad = [k for k, ok in checks.items() if not ok]
assert not bad, f"VERIFICATION FAILED: {bad}"

# the swapped ConfigMap value must itself be valid YAML (parse the whole file)
import yaml  # noqa: E402
docs = list(yaml.safe_load_all(v))
cms = {d["metadata"]["name"]: d for d in docs if isinstance(d, dict) and d.get("kind") == "ConfigMap"}
pcm = next((d for d in cms.values() if "patch_sampler_is.py" in d.get("data", {})), None)
assert pcm is not None, "patch_sampler_is.py not found after build"
for _k, _c in SWAPS.items():
    assert pcm["data"][_k] == _c, f"{_k} content mismatch after YAML round-trip"
print("  [OK] YAML parses; both embedded patches match byte-for-byte")

print(f"\nBUILT {NAME} (seed {SEED}, cap {CAP}/target {TARGET}, TIS active): {OUT}  lines={v.count(chr(10))} bytes={len(v)}")
