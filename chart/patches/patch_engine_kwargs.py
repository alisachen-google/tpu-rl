"""Forward extra vLLM engine kwargs into the GRPO rollout engine via env.

WHY. maxtext train_rl.py hardcodes the vLLM `rollout_vllm_kwargs` dict (hf_overrides,
enable_expert_parallel, enable_prefix_caching, dtype). Rollout-side serving tuning
validated by aysu on gemma4-e2b/tpu7x (ubench aysu-ubench-8terxbx7: block_size=256,
kv_cache_dtype=fp8, ...) needs engine args that rl.yml does not expose. tunix passes
`rollout_vllm_kwargs` straight through to vllm EngineArgs, so one extra `**dict` there
is enough.

WHAT. Inserts, as the LAST entry of the rollout_vllm_kwargs literal (so it overrides
the hardcoded keys):
    **json.loads(os.environ.get("RL_VLLM_ENGINE_KWARGS", "{}")),
Usage:  RL_VLLM_ENGINE_KWARGS='{"block_size": 256}'   (any vllm EngineArgs field)
Reserved tunix keys (tensor/data/expert_parallel_size) are rejected by tunix itself.

FAIL-OPEN. Anchor mismatch prints FAILED and exits 0 without touching the file.
Prints exactly one of:
    ENGINE-KWARGS PATCH: APPLIED
    ENGINE-KWARGS PATCH: already patched
    ENGINE-KWARGS PATCH: FAILED <reason>
"""
import importlib.util
import os
import sys

MARK = '**json.loads(os.environ.get("RL_VLLM_ENGINE_KWARGS", "{}")),'
ANCHOR = '"dtype": trainer_config.weight_dtype.value,\n'


def fail(msg):
  print(f"ENGINE-KWARGS PATCH: FAILED {msg}")
  sys.exit(0)


def main():
  spec = importlib.util.find_spec("maxtext")  # top-level: does not import the package
  if spec is None or not spec.submodule_search_locations:
    fail("cannot locate maxtext package")
  path = os.path.join(list(spec.submodule_search_locations)[0], "trainers", "post_train", "rl", "train_rl.py")
  if not os.path.exists(path):
    fail(f"{path} missing")
  src = open(path).read()
  if MARK in src:
    print("ENGINE-KWARGS PATCH: already patched")
    return
  n = src.count(ANCHOR)
  if n != 1:
    fail(f"anchor count {n} != 1 in {path}")
  if "import json" not in src or "import os" not in src:
    fail("train_rl.py lacks json/os import")
  # keep the anchor's indentation for the inserted line
  idx = src.index(ANCHOR)
  line_start = src.rfind("\n", 0, idx) + 1
  indent = src[line_start:idx]
  new = src.replace(ANCHOR, ANCHOR + indent + MARK + "\n", 1)
  open(path, "w").write(new)
  if MARK not in open(path).read():
    fail("re-verify failed")
  extra = os.environ.get("RL_VLLM_ENGINE_KWARGS", "{}")
  print(f"ENGINE-KWARGS PATCH: APPLIED (RL_VLLM_ENGINE_KWARGS={extra}) -> {path}")


if __name__ == "__main__":
  main()
