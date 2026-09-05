"""Apply maxtext PR #4409 (OOM-safe chunked log-probs for the RL actor) at runtime.

WHY. Fix #21: gemma4's 262k-vocab logits force train_micro_batch_size=1 (256 grad-acc
microsteps per update, ~131 s train segment at 4x4x4). PR #4409 (lokic233, stale-closed
2026-08-18 for CLA, never rejected) threads tunix's in-tree `compute_logps_chunk_size`
through MaxText: the actor projects the LM head in sequence chunks (scan+remat), capping
live logits at [B, chunk, V]. DEFAULT-OFF: byte-identical unless
`rl_compute_logps_chunk_size>0` is passed.

WHAT. The diff `pr4409_chunked_logps.diff` (PR rebased onto the image's tree = maxtext fork
`ironwood-rl-gemma4` @ 840ffd3; one conflict in tunix_adapter.py resolved by keeping both the
fork's JAX compat shims and the PR's NAVI_MEM_LOG helper) is applied with `patch -p3` (or
`git apply`) under the importable maxtext package root. Touches: configs/post_train/rl.yml,
configs/types.py, integration/tunix/tunix_adapter.py, layers/decoders.py,
layers/nnx_decoders.py, models/models.py, trainers/post_train/rl/train_rl.py.
Run it BEFORE patch_engine_kwargs (independent hunks, but keep the order deterministic).

FAIL-OPEN. Any failure prints FAILED and exits 0 with the tree untouched (dry-run first).
Prints exactly one of:
    PR4409 PATCH: APPLIED
    PR4409 PATCH: already patched
    PR4409 PATCH: FAILED <reason>
"""
import importlib.util
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIFF = os.path.join(HERE, "pr4409_chunked_logps.diff")
MARK = "rl_compute_logps_chunk_size"


def fail(msg):
  print(f"PR4409 PATCH: FAILED {msg}")
  sys.exit(0)


def main():
  spec = importlib.util.find_spec("maxtext")
  if spec is None or not spec.submodule_search_locations:
    fail("cannot import maxtext")
  root = list(spec.submodule_search_locations)[0]  # .../src/maxtext
  if not os.path.exists(DIFF):
    fail(f"diff not found at {DIFF}")
  rl_yml = os.path.join(root, "configs", "post_train", "rl.yml")
  if os.path.exists(rl_yml) and MARK in open(rl_yml).read():
    print("PR4409 PATCH: already patched")
    return
  if shutil.which("patch"):
    base = ["patch", "-p3", "-d", root, "--forward", "--batch", "-i", DIFF]
    dry = subprocess.run(base + ["--dry-run"], capture_output=True, text=True)
    if dry.returncode != 0:
      fail(f"patch --dry-run failed under {root}: {dry.stdout[-600:]} {dry.stderr[-300:]}")
    r = subprocess.run(base, capture_output=True, text=True)
    if r.returncode != 0:
      fail(f"patch apply failed: {r.stdout[-600:]} {r.stderr[-300:]}")
  elif shutil.which("git"):
    base = ["git", "apply", "--unsafe-paths", "-p3", f"--directory={root}", DIFF]
    chk = subprocess.run(base + ["--check"], capture_output=True, text=True)
    if chk.returncode != 0:
      fail(f"git apply --check failed under {root}: {chk.stderr[-600:]}")
    r = subprocess.run(base, capture_output=True, text=True)
    if r.returncode != 0:
      fail(f"git apply failed: {r.stderr[-600:]}")
  else:
    fail("neither `patch` nor `git` available in the image")
  if MARK not in open(rl_yml).read():
    fail("re-verify failed (rl.yml lacks new knob)")
  # Follow-up fix for trees whose decoder is a native NNXDecoder (no `rngs` kwarg on
  # apply_output_head): only forward rngs when the signature accepts it. The PR's static-key
  # trick targets the ToNNX-wrapped Linen decoder; on NNXDecoder it raised
  # TypeError: apply_output_head() got an unexpected keyword argument 'rngs' (run g4n9s).
  models_py = os.path.join(root, "models", "models.py")
  ms = open(models_py).read()
  old_line = '    extra = {"rngs": rngs} if rngs is not None else {}\n'
  new_line = ('    extra = {"rngs": rngs} if (rngs is not None and "rngs" in __import__("inspect")'
              '.signature(self.decoder.apply_output_head).parameters) else {}\n')
  if old_line in ms:
    open(models_py, "w").write(ms.replace(old_line, new_line, 1))
  elif new_line not in ms:
    fail("models.py rngs-forwarding anchor not found after apply")
  print(f"PR4409 PATCH: APPLIED -> {root}")


if __name__ == "__main__":
  main()
