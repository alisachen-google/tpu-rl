"""Build the 24 s configuration (g4pt1c2048p: GSM8K parity recipe + cap 2048 + async reward + 24 s levers, p-state 6)
on the LATEST image (2026-09-22-head), N updates, for the perf/regression validation the user asked for.
Variants (env):
  NOPERF=1  -> apply ONLY the parity patcher (bisect: does the bare recipe work on the new image?)      name g4l24n
  (default) -> parity patcher + the perf patches that still apply + native knobs                        name g4l24
  STEPS=<n> -> number of updates (default 10)
  EXTRA="k=v k=v" -> extra train_rl knobs appended (e.g. a weight-sync fix)
  FIX=<file> -> an extra patcher embedded as patch_newimg_fix.py and run after the parity patcher
Always: new image + pathways tags, re-based parity patcher, sharded-sampling patch skipped, flash_sa patch skipped +
explicit local_sa_* settings, maxRestarts 3. Apply with tools/apply-jobset.sh (ownerReference workaround). Fix for the gibberish: FIX=patches/patch_kv_alias_dedupe.py
"""
import json, os, re, yaml
SP = os.path.dirname(os.path.abspath(__file__))
raw = open(f"{SP}/manifests/g4pt1c2048p-seed42-24s-config.yaml", encoding="utf-8").read()
NEW_IMG = "gcr.io/supercomputer-testing/maxtext-rl-ironwood@sha256:4dc6ca720017bf93b6bae9524519cf5b7a8b16706ca6aab2a355af26d75a6bf3"
OLD_IMG = "gcr.io/supercomputer-testing/maxtext-rl-ironwood@sha256:f7c957d0b286e03966fdd6f9b15e7609fae8460a933a9da0b9fbab3e04723c94"
STEPS = int(os.environ.get("STEPS", "10")); NOPERF = os.environ.get("NOPERF") == "1"
NAME = "g4l24n" if NOPERF else "g4l24"
NAME = os.environ.get("NAME", NAME)

def repl(old, new, n=1):
    global raw
    c = raw.count(old); assert c == n, f"{old!r}: count {c} != {n}"
    raw = raw.replace(old, new)

repl(OLD_IMG, NEW_IMG)
repl("20260730-jax_0.10.2", "20261028-jax_0.11.0", 3)
assert "dvfs_p_state=6" in raw and "dvfs_p_state=7" not in raw
LOCAL_SA = ("local_sa_block_q=512 local_sa_block_kv=512 local_sa_block_kv_compute=512 local_sa_block_q_dkv=512 "
            "local_sa_block_kv_dkv=512 local_sa_block_kv_dkv_compute=512 local_sa_block_q_dq=512 local_sa_block_kv_dq=512 "
            "local_sa_use_fused_bwd_kernel=false local_sa_q_layout=HEAD_DIM_MINOR local_sa_k_layout=HEAD_DIM_MINOR "
            "local_sa_v_layout=HEAD_DIM_MINOR local_sa_fuse_reciprocal=true local_sa_use_base2_exp=true "
            "eval_local_sa_block_q=512 eval_local_sa_block_kv=512 eval_local_sa_block_kv_compute=512 "
            "eval_local_sa_q_layout=HEAD_DIM_MINOR eval_local_sa_k_layout=HEAD_DIM_MINOR eval_local_sa_v_layout=HEAD_DIM_MINOR")
extra = LOCAL_SA.split() + [f"num_batches={STEPS}"] + os.environ.get("EXTRA", "").split()
if not NOPERF:
    extra += ["free_kv_cache_during_weight_sync=false", "vllm_block_size=128"]
repl("        num_batches=250 \\", "".join(f"        {kv} \\\n" for kv in extra).rstrip("\n"))
# launcher patch lines
lines = raw.split("\n"); out = []
for ln in lines:
    m = re.match(r"^(\s*)python3 /workload/patches/(patch_[a-z0-9_]+)\.py(.*)$", ln)
    if m:
        ind, pname, rest = m.groups()
        if pname == "patch_sampler_is":
            out.append(ln); continue
        if pname in ("patch_sharded_sampling", "patch_flash_sa") or NOPERF:
            out.append(f'{ind}echo "{pname.upper()}: SKIPPED on the latest image ({"bisect: perf patches off" if NOPERF else "native"})"'); continue
    out.append(ln)
raw = "\n".join(out)
n_ar = raw.count("export ASYNC_REWARD=on"); assert n_ar >= 1
if not NOPERF:
    raw = raw.replace("export ASYNC_REWARD=on", "export ASYNC_REWARD=on; export DP_SCHED_ROUTING=round_robin")  # native round-robin DP routing on this image
else:
    raw = raw.replace("export ASYNC_REWARD=on", "export ASYNC_REWARD=off")  # bisect: perf levers off
repl("    maxRestarts: 10", "    maxRestarts: 3")
raw = raw.replace("g4pt1c2048p", NAME)
patch = open(f"{SP}/patches/patch_sampler_is_newimg.py", encoding="utf-8").read()
m = re.search(r'^(\s*)patch_sampler_is\.py: ".*"$', raw, re.M); assert m
raw = raw[:m.start()] + f"{m.group(1)}patch_sampler_is.py: {json.dumps(patch)}" + raw[m.end():]
fix = os.environ.get("FIX")
if fix:
    fsrc = open(fix, encoding="utf-8").read()
    m = re.search(r'^(\s*)patch_sampler_is\.py: ".*"$', raw, re.M)
    raw = raw[:m.end()] + f"\n{m.group(1)}patch_newimg_fix.py: {json.dumps(fsrc)}" + raw[m.end():]
    m2 = re.search(r"^(\s*)python3 /workload/patches/patch_sampler_is\.py$", raw, re.M); assert m2
    raw = raw[:m2.end()] + f"\n{m2.group(1)}python3 /workload/patches/patch_newimg_fix.py" + raw[m2.end():]
open(f"{SP}/{NAME}.yaml", "w", encoding="utf-8").write(raw)
docs = [d for d in yaml.safe_load_all(raw) if d]
l = [d for d in docs if d["kind"] == "ConfigMap" and "launch-workload.sh" in d["data"]][0]["data"]["launch-workload.sh"]
applied = re.findall(r"^\s*python3 /workload/patches/(patch_[a-z0-9_]+)\.py", l, re.M)
for k in (f"num_batches={STEPS}", "max_target_length=2560", "local_sa_block_q=512", "data_shuffle_seed=42"):
    assert k in l, k
js = [d for d in docs if d["kind"] == "JobSet"][0]
imgs = sorted({c["image"] for rj in js["spec"]["replicatedJobs"] for c in rj["template"]["spec"]["template"]["spec"].get("containers", []) + rj["template"]["spec"]["template"]["spec"].get("initContainers", [])})
print(f"BUILT {SP}/{NAME}.yaml steps={STEPS} noperf={NOPERF} patches_applied={len(applied)} {applied[:3]}... coordinator={'coordinator' in js['spec']} images={[i.split('/')[-1][:40] for i in imgs]}")
