#!/bin/bash
# Image probe: print package versions + feature greps, then exit 0. No TPU work.
echo "PROBE: image contents"
python3 - <<'PY'
import importlib.metadata as md, importlib.util, os, re, subprocess
for p in ["tunix","tpu-inference","tpu_inference","vllm","jax","jaxlib","libtpu","maxtext","pathwaysutils","flax","orbax-checkpoint","transformers"]:
    try: print(f"PKG {p}=={md.version(p)}")
    except Exception as e: pass
def loc(m):
    s=importlib.util.find_spec(m); return list(s.submodule_search_locations)[0] if s and s.submodule_search_locations else None
for m in ["tunix","tpu_inference","vllm","maxtext"]:
    print("LOC",m,loc(m))
t=loc("tunix")
if t:
    for f,pat in [("rl/rl_cluster.py","compute_logps_chunk_size|chunk_size|offload_to_cpu"),("rl/common.py","chunk_size|skip_lm_head|compute_final_logits|def compute_per_token_logps"),("rl/rl_learner.py","chunk"),("generate/vllm_sampler.py","engine_kwargs|Direct Weight|additional_config"),("rl/rollout/base_rollout.py","rollout_vllm_kwargs|compute_logps_chunk_size")]:
        p=os.path.join(t,f)
        if os.path.exists(p):
            hits=[l.strip()[:140] for l in open(p) if re.search(pat,l)]
            print(f"GREP {f}: {len(hits)} hits"); [print("   ",h) for h in hits[:12]]
        else: print("GREP",f,"MISSING")
    try:
        d=os.path.join(t,"..", "tunix-"+md.version("tunix")+".dist-info"); print("DISTINFO", os.path.exists(d))
    except Exception: pass
    print("TUNIX git:", subprocess.run(["git","-C",t,"log","-1","--format=%H %cd"],capture_output=True,text=True).stdout.strip() or "n/a")
for b in ["patch","git"]:
    print("BIN",b, subprocess.run(["which",b],capture_output=True,text=True).stdout.strip() or "missing")
PY
echo "PROBE: done"
exit 0
