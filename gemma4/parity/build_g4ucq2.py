"""Build g4ucq2.yaml: a faithful RERUN of g4ucq, the optimized ~24-25 s/step recipe
(OpenMathInstruct-2, 256x8 rollouts, response cap 2048 = max_target 6144 - prefill 4096,
round_robin DP routing, approx_max_k sampling, EOS-check interval 32, async reward/logprobs,
100 steps, 64 chips 4x4x4), to re-check its reward parity (check_numbers in the GPU band
0.50-0.57, ~0.55 target; score/mean 0.54-0.65) with tb_parity.py on the GCS tensorboard file.

Changes vs g4ucq.yaml (none touch the recipe):
  * rename g4ucq -> g4ucq2 (own ConfigMaps / JobSet / gs://ubench-logs/g4ucq2 output dir)
  * maxRestarts 50 -> 3 (fail loudly instead of churning; resume is broken on this image)
  * SPOT=1: add gke-spot=true to the worker nodeSelector (borrow tpu7x-spot; see build_g4pt.py)
  * TIS=1 (optional, NOT default): also run patch_sampler_is.py (token TIS, threshold 3.0) by
    adding the patch to the ConfigMap and the launcher. Off by default = exact rerun.
Usage: [SPOT=1] [TIS=1] python3 build_g4ucq2.py
"""
import os, re, yaml

SP = os.environ.get("PARITY_DIR", os.path.dirname(os.path.abspath(__file__)))  # needs g4ucq.yaml (source manifest) here
raw = open(f"{SP}/g4ucq.yaml", encoding="utf-8").read()
NAME = "g4ucq2"


def repl(old, new):
    global raw
    assert raw.count(old) == 1, f"anchor {old!r} count {raw.count(old)}"
    raw = raw.replace(old, new, 1)


repl("    maxRestarts: 50", "    maxRestarts: 3")
# non-default Pathways ports (other tenants' heads own 29000/29001 on the shared cpu-np nodes; see build_g4pt.py)
for _o, _n in {
    "export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000": "export JAX_BACKEND_TARGET=grpc://127.0.0.1:29140",
    "value: grpc://g4ucq-pathways-head-0-0.g4ucq:29000": "value: grpc://g4ucq-pathways-head-0-0.g4ucq:29140",
    "- --server_port=29001": "- --server_port=29141",
    "- --server_port=29000": "- --server_port=29140",
    "- --resource_manager_address=g4ucq-pathways-head-0-0.g4ucq:29001": "- --resource_manager_address=g4ucq-pathways-head-0-0.g4ucq:29141",
}.items():
    assert raw.count(_o) >= 1, f"port anchor {_o!r} not found"
    raw = raw.replace(_o, _n)
assert "29000" not in raw and "29001" not in raw
if os.environ.get("SPOT") == "1":
    repl("            nodeSelector:\n              cloud.google.com/gke-tpu-topology: \"4x4x4\"\n              cloud.google.com/gke-tpu-accelerator: tpu7x\n",
         "            nodeSelector:\n              cloud.google.com/gke-tpu-topology: \"4x4x4\"\n              cloud.google.com/gke-tpu-accelerator: tpu7x\n              cloud.google.com/gke-spot: \"true\"\n")
if os.environ.get("TIS") == "1":
    import json
    patch = open(f"{SP}/patch_sampler_is_new.py", encoding="utf-8").read()
    # add the patch file to the patches ConfigMap (after patch_flash_sa.py) and run it in the launcher
    m = re.search(r'^(\s*)patch_flash_sa\.py: ".*"$', raw, re.M)
    assert m, "patch_flash_sa.py CM entry not found"
    raw = raw[:m.end()] + f"\n{m.group(1)}patch_sampler_is.py: {json.dumps(patch)}" + raw[m.end():]
    m2 = re.search(r'^(\s*)if \[ -f /workload/patches/patch_profiler_fix\.py \]; then$', raw, re.M)
    assert m2, "launcher anchor not found"
    raw = raw[:m2.start()] + f"{m2.group(1)}python3 /workload/patches/patch_sampler_is.py\n" + raw[m2.start():]
    NAME = "g4ucq2t"
raw = raw.replace("g4ucq", NAME)
out = f"{SP}/{NAME}.yaml"
open(out, "w", encoding="utf-8").write(raw)
docs = list(yaml.safe_load_all(raw))
assert raw.count("g4ucq\n") == 0 and NAME in raw
assert "max_target_length=6144" in raw and "nvidia/OpenMathInstruct-2" in raw and "num_batches=100" in raw
print(f"BUILT {out}: spot={os.environ.get('SPOT')=='1'} tis={os.environ.get('TIS')=='1'} docs={len(docs)}")
