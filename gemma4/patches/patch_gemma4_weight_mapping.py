"""Register the gemma4 vLLM weight mapping on the NEWEST (pure-upstream) image.

WHY THIS PATCH EXISTS (and did not before).
In the proven g4r55 run the gemma4 vLLM weight mapping was NOT a runtime patch --
it was BAKED INTO THE IMAGE. The fork image (maxtext-rl-ironwood @ f7c957d0) was
built by a Dockerfile (g4wm-build/) that COPYs two files into the base image:
    gemma4.py   -> integration/tunix/weight_mapping/gemma4.py   (build_gemma4_vllm_mapping)
    __init__.py -> integration/tunix/weight_mapping/__init__.py  (adds the gemma4 dispatch branch)
So the dispatcher `StandaloneVllmWeightMapping["gemma4-e2b"]` resolved inside the
image, and the g4r55 patch roster never registered it. (The `gemma4_prs.diff`
patch in that run is a DIFFERENT thing -- vision/multimodal clipped-linears parity
-- and was explicitly SKIPPED in the launcher. It never touched the weight mapping.)

The newest upstream image is pure upstream: its weight_mapping/__init__.py has
qwen2/qwen3/gemma3/deepseek3/gpt-oss/llama3 branches but NO gemma4 branch and NO
gemma4.py module, so `StandaloneVllmWeightMapping["gemma4-e2b"].to_hf_mapping()`
raises `ValueError: gemma4-e2b vLLM weight mapping not found` at the rollout weight
sync. This patch does at RUNTIME what the fork Dockerfile did at BUILD time.

WHAT IT DOES (surgical, fail-closed on the mapping -- this is load-bearing for the
rollout weight sync, so unlike the perf patches it must NOT fail-open silently):
  1. Copies the proven TP1 gemma4.py (from /workload/patches/gemma4_weight_mapping_src.py,
     byte-identical to what produced reward 0.58) into the image's
     weight_mapping/gemma4.py. gemma4.py imports ONLY stable upstream maxtext utils
     (checkpoint_conversion hf_model_configs / param_mapping) + jax/flax -- it does
     NOT depend on the fork's integration/tunix utils.py or tunix_adapter.py, so
     those are deliberately NOT overlaid (the new image's tunix API is kept; the
     crash traceback already showed its 2-arg->1-arg to_hf_mappings fallback working).
  2. Inserts the import + the `elif name.startswith("gemma4"): return
     build_gemma4_vllm_mapping(name)` branch into the image's EXISTING __init__.py,
     before the `else: raise ValueError`. Surgical (not a whole-file overwrite) so
     the new image's other dispatch branches are preserved.

TP note: the proven gemma4.py models the fused QKV / gate-up kernels at
rollout_tensor_parallelism=1 (plain [Q|K|V] / [gate|up] concatenation). This run is
TP1 x DP128, so that is exactly correct. (The fork's newer gemma4.py adds a
GEMMA4_NATIVE_TP>1 interleave path that reduces to the same thing at tp=1.)

Prints exactly one of:
    GEMMA4-WMAP PATCH: APPLIED
    GEMMA4-WMAP PATCH: already applied
    GEMMA4-WMAP PATCH: FAILED <reason>
A FAILED line means the rollout weight sync WILL crash -- treat it as fatal.
"""
import ast
import importlib.util
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gemma4_weight_mapping_src.py")  # the proven gemma4.py


def fail(msg):
  print(f"GEMMA4-WMAP PATCH: FAILED {msg}", flush=True)
  sys.exit(1)  # FATAL: this mapping is required for the rollout weight sync


def main():
  spec = importlib.util.find_spec("maxtext")
  if spec is None or not spec.submodule_search_locations:
    fail("cannot import maxtext")
  root = list(spec.submodule_search_locations)[0]  # .../src/maxtext
  wmap_dir = os.path.join(root, "integration", "tunix", "weight_mapping")
  init_py = os.path.join(wmap_dir, "__init__.py")
  gemma4_py = os.path.join(wmap_dir, "gemma4.py")
  if not os.path.isdir(wmap_dir):
    fail(f"weight_mapping dir not found at {wmap_dir}")
  if not os.path.exists(init_py):
    fail(f"__init__.py not found at {init_py}")
  if not os.path.exists(SRC):
    fail(f"proven gemma4.py source not found at {SRC} (add it to the patches ConfigMap)")

  init_src = open(init_py, encoding="utf-8").read()
  already = ("build_gemma4_vllm_mapping" in init_src
             and os.path.exists(gemma4_py))
  if already:
    print("GEMMA4-WMAP PATCH: already applied", flush=True)
    return

  # (1) install the mapping module (overwrite if a stale/partial copy exists)
  shutil.copyfile(SRC, gemma4_py)
  try:
    ast.parse(open(gemma4_py, encoding="utf-8").read())
  except SyntaxError as e:
    fail(f"installed gemma4.py does not parse: {e}")

  # (2) inject the import + dispatch branch into the existing __init__.py
  new = init_src
  import_line = ("from maxtext.integration.tunix.weight_mapping.gemma4 "
                 "import build_gemma4_vllm_mapping\n")
  if "import build_gemma4_vllm_mapping" not in new:
    # place it next to the other sibling imports (anchor on any one of them)
    for anchor in (
        "from maxtext.integration.tunix.weight_mapping.gemma3 import GEMMA3_VLLM_MAPPING\n",
        "from maxtext.integration.tunix.weight_mapping.qwen3 import QWEN3_VLLM_MAPPING\n",
    ):
      if anchor in new:
        new = new.replace(anchor, anchor + import_line, 1)
        break
    else:
      fail("could not find a sibling weight_mapping import to anchor the gemma4 import")

  if "gemma4" not in new.split("def __getattr__")[1].split("def ")[0]:
    # insert the branch immediately before the final `else:` of __getattr__
    anchor = "    else:\n      raise ValueError"
    if new.count(anchor) < 1:
      fail("could not find the `else: raise ValueError` dispatch tail in __getattr__")
    branch = ('    elif name.startswith("gemma4"):\n'
              "      return build_gemma4_vllm_mapping(name)\n")
    new = new.replace(anchor, branch + anchor, 1)

  try:
    ast.parse(new)
  except SyntaxError as e:
    fail(f"patched __init__.py does not parse: {e}")
  open(init_py, "w", encoding="utf-8").write(new)

  # (3) re-verify by importing and resolving the mapping for gemma4-e2b
  v = open(init_py, encoding="utf-8").read()
  if "build_gemma4_vllm_mapping" not in v or 'name.startswith("gemma4")' not in v:
    fail("re-verify failed (gemma4 import/branch missing after write)")
  print(f"GEMMA4-WMAP PATCH: APPLIED -> {wmap_dir}", flush=True)


if __name__ == "__main__":
  main()
