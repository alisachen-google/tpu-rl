"""Apply the gemma4 correctness/parity PR stack at runtime (maxtext #4807, #4860, #4790).

WHY. Three open PRs fix gemma4 model/checkpoint fidelity:
  #4807  vision std_bias/std_scale classification per HF `standardize` (config + types + gemma4_vision)
  #4860  emit vision-tower shapes in the HF export map (hf_shape.py) — used by offline mapping checks
  #4790  Gemma-4 E2B image (multimodal) E2E parity: clipped-linears, padded-patch masking, PLE
         substitution; also adds the E2B image-contract config fields and their param mappings.

TEXT-PATH IMPACT (verified before adopting): every param_mapping change is gated behind
`use_multimodal` / `use_clipped_linears_for_vit`, both False on the RL path, so the pushed
text parameter set is unchanged. The stack is carried to (a) keep our tree aligned with the
fidelity fixes, (b) let a multimodal-config run be attempted later, and (c) catch RLConfig
schema drift early (Fix #9/#10/#20 class) via the smoke gate.

WHAT. Applies `gemma4_prs.diff` (the three PRs squashed and rebased onto the image's tree =
maxtext fork `ironwood-rl-gemma4` @ 35b7903) with `patch -p3` under the importable maxtext
package root. Touches only `src/maxtext/**` (tests are excluded from the diff). Coexists with
patch_pr4409 (verified: 4409 applies cleanly on top of this stack) — run this one FIRST.

FAIL-OPEN. Dry-run first; any failure prints FAILED and exits 0 with the tree untouched.
Prints exactly one of:
    GEMMA4-PRS PATCH: APPLIED
    GEMMA4-PRS PATCH: already patched
    GEMMA4-PRS PATCH: FAILED <reason>
"""
import importlib.util
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIFF = os.path.join(HERE, "gemma4_prs.diff")
MARK = "use_clipped_linears_for_vit"


def fail(msg):
  print(f"GEMMA4-PRS PATCH: FAILED {msg}")
  sys.exit(0)


def main():
  spec = importlib.util.find_spec("maxtext")
  if spec is None or not spec.submodule_search_locations:
    fail("cannot import maxtext")
  root = list(spec.submodule_search_locations)[0]  # .../src/maxtext
  if not os.path.exists(DIFF):
    fail(f"diff not found at {DIFF}")
  mark_file = os.path.join(root, "configs", "models", "gemma4-e2b.yml")
  if os.path.exists(mark_file) and MARK in open(mark_file).read():
    print("GEMMA4-PRS PATCH: already patched")
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
  if MARK not in open(mark_file).read():
    fail("re-verify failed (gemma4-e2b.yml lacks the new image-contract fields)")
  print(f"GEMMA4-PRS PATCH: APPLIED -> {root}")


if __name__ == "__main__":
  main()
