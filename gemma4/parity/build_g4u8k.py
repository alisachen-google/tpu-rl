"""Build g4u8k.yaml: the optimized OpenMathInstruct-2 recipe (g4ucq3: 256 prompts x 8 samples, prefill 4096, p-state 6)
on the LATEST image with the KV-alias fix and response cap 8192 (max_target_length 12288 = 4096 + 8192), STEPS updates,
no-op patches removed. Usage: [STEPS=20] [CAP=8192] python3 build_g4u8k.py
"""
import json, os, re, yaml
SP = os.path.dirname(os.path.abspath(__file__))
raw = open(f"{SP}/manifests/g4ucq3-optimized-24s-pstate6.yaml", encoding="utf-8").read()
NEW_IMG = "gcr.io/supercomputer-testing/maxtext-rl-ironwood@sha256:4dc6ca720017bf93b6bae9524519cf5b7a8b16706ca6aab2a355af26d75a6bf3"
OLD_IMG = "gcr.io/supercomputer-testing/maxtext-rl-ironwood@sha256:f7c957d0b286e03966fdd6f9b15e7609fae8460a933a9da0b9fbab3e04723c94"
STEPS = int(os.environ.get("STEPS", "20")); CAP = int(os.environ.get("CAP", "8192")); PREFILL = 4096; T = PREFILL + CAP
NAME = os.environ.get("NAME", "g4u8k" if CAP == 8192 else f"g4u{CAP}")
def repl(old, new, n=1):
    global raw
    c = raw.count(old); assert c == n, f"{old!r}: count {c} != {n}"
    raw = raw.replace(old, new)
repl(OLD_IMG, NEW_IMG); repl("20260730-jax_0.10.2", "20261028-jax_0.11.0", 3)
assert "dvfs_p_state=6" in raw and "max_prefill_predict_length=4096" in raw
repl("max_target_length=6144", f"max_target_length={T}"); repl("kv_cache_buffer=3840", f"kv_cache_buffer={T}")
assert T % 512 == 0
LOCAL_SA = ("local_sa_block_q=512 local_sa_block_kv=512 local_sa_block_kv_compute=512 local_sa_block_q_dkv=512 "
            "local_sa_block_kv_dkv=512 local_sa_block_kv_dkv_compute=512 local_sa_block_q_dq=512 local_sa_block_kv_dq=512 "
            "local_sa_use_fused_bwd_kernel=false local_sa_q_layout=HEAD_DIM_MINOR local_sa_k_layout=HEAD_DIM_MINOR "
            "local_sa_v_layout=HEAD_DIM_MINOR local_sa_fuse_reciprocal=true local_sa_use_base2_exp=true "
            "eval_local_sa_block_q=512 eval_local_sa_block_kv=512 eval_local_sa_block_kv_compute=512 "
            "eval_local_sa_q_layout=HEAD_DIM_MINOR eval_local_sa_k_layout=HEAD_DIM_MINOR eval_local_sa_v_layout=HEAD_DIM_MINOR")
extra = LOCAL_SA.split() + [f"num_batches={STEPS}", "free_kv_cache_during_weight_sync=false", "vllm_block_size=128"]
m = re.search(r"^(\s*)num_batches=100 \\$", raw, re.M); assert m, "num_batches anchor"
ind = m.group(1); raw = raw[:m.start()] + "\n".join(f"{ind}{kv} \\" for kv in extra) + raw[m.end():]
raw = raw.replace("export ASYNC_REWARD=on", "export ASYNC_REWARD=on; export DP_SCHED_ROUTING=round_robin")
NOOP = {"patch_cd_proper", "patch_rpa_blocks", "patch_async_detok", "patch_async_logprobs", "patch_dp_rr_routing",
        "patch_stop_tokens", "patch_async_reward", "patch_jit_unroll", "patch_sync_keepkv", "patch_host_extract",
        "patch_padding_bucket", "patch_textual_stop", "patch_packer_fast", "patch_packing_knob", "patch_profiler_fix",
        "patch_flash_sa", "patch_sharded_sampling", "patch_perf_v2", "patch_fast_detok", "patch_gemma4_prs",
        "patch_length_shape", "patch_reward_dump"}
lines = []
for ln in raw.split("\n"):
    mm = re.match(r"^\s*python3 /workload/patches/(patch_[a-z0-9_]+)\.py.*$", ln)
    if mm and mm.group(1) in NOOP: continue
    lines.append(ln)
raw = "\n".join(lines)
m = re.search(r"^(\s*)python3 /workload/patches/patch_lendist\.py.*$", raw, re.M); assert m
raw = raw[:m.start()] + f"{m.group(1)}python3 /workload/patches/patch_newimg_fix.py\n" + raw[m.start():]
# KV-alias fix: add the patcher to the patches ConfigMap (re-dumped last, since it also holds the launcher)
fix = open(f"{SP}/patches/patch_kv_alias_dedupe.py", encoding="utf-8").read()
parts = re.split(r"(?m)^---\s*$", raw); out = []
for part in parts:
    if "kind: ConfigMap" in part and "patch_lendist.py" in part:
        d = yaml.safe_load(part)
        for k in list(d["data"].keys()):
            if k.replace(".py", "").replace(".diff", "") in NOOP or k == "gemma4_prs.diff": d["data"].pop(k)
        d["data"]["patch_newimg_fix.py"] = fix
        part = "\n" + yaml.safe_dump(d, sort_keys=False, width=10**6)
    out.append(part)
raw = "---".join(out)
repl("    maxRestarts: 3", "    maxRestarts: 3") if "    maxRestarts: 3" in raw else None
raw = raw.replace("g4ucq3", NAME)
open(f"{SP}/{NAME}.yaml", "w", encoding="utf-8").write(raw)
docs = [d for d in yaml.safe_load_all(raw) if d]
l = [d for d in docs if d["kind"] == "ConfigMap" and "launch-workload.sh" in d["data"]][0]["data"]["launch-workload.sh"]
cm = [d for d in docs if d["kind"] == "ConfigMap" and "patch_lendist.py" in d["data"]][0]
inv = re.findall(r"python3 /workload/patches/(patch_[a-z0-9_]+)\.py", l)
assert all(p + ".py" in cm["data"] for p in inv), [p for p in inv if p + ".py" not in cm["data"]]
for k in (f"num_batches={STEPS}", f"max_target_length={T}", "local_sa_block_q=512", "DP_SCHED_ROUTING=round_robin", "nvidia/OpenMathInstruct-2"):
    assert k in l, k
print(f"BUILT {SP}/{NAME}.yaml steps={STEPS} cap={CAP} target={T} patches={len(inv)} {inv}")
