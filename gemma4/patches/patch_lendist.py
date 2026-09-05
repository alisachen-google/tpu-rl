"""Record per-step ISL / OSL / truncation distributions from the GRPO rollout.

WHY. Tunix logs only completions/{mean,max,min}_length -- three scalars. We have never
seen the length DISTRIBUTION, and `max_length` is 8192 (the OSL cap) in every single
step, so we know something is truncated but not whether it is 1 sequence or 40%. That
number gates the tuning plan: holding the OSL cap fixed is only safe if truncation is
rare.

WHAT IT COSTS. Nothing measurable. Every array read here is ALREADY host-side numpy at
this point in the function:
    rollout_output.tokens                 list[np.ndarray], UNPADDED  -> OSL = len()
    rollout_output.left_padded_prompt_tokens  np.ndarray              -> ISL
    pad_value / eos_value                 plain ints (rollout.pad_id()/eos_id())
    rollout_config.max_tokens_to_generate  the cap
No device->host sync is introduced, so step timing is unaffected. ~30 KB/step.

Using rollout_output.tokens directly (rather than inferring from completion_mask) makes
OSL exact: len() cannot be fooled by a pad token appearing mid-sequence.

FAIL-OPEN. This patches a file on the live training path. Any anchor mismatch prints
FAILED and exits 0 without modifying anything -- it must never crash a run. The injected
code is itself wrapped in try/except so a logging bug cannot take down training.

Prints exactly one of:
    LENDIST PATCH: APPLIED
    LENDIST PATCH: FAILED <reason>
"""

import ast
import os
import sys

SITE = os.environ.get("PATCH_ROOT", "/usr/local/lib/python3.12/site-packages")
ROOT = os.path.join(SITE, "tunix")
GL = os.path.join(ROOT, "rl/grpo/grpo_learner.py")


def fail(msg):
    print(f"LENDIST PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(GL):
    fail(f"{GL} not found")
src = open(GL, encoding="utf-8").read()

if "_LENDIST_DUMP" in src:
    print("LENDIST PATCH: already patched")
    sys.exit(0)

ANCHOR = "    # Log completion lengths.\n    agg_completion_mask = completion_mask.sum(axis=-1)\n"
if src.count(ANCHOR) != 1:
    fail(f"anchor found {src.count(ANCHOR)} times, expected 1")

# Verify the names we rely on are actually in scope in this function.
for name in ("rollout_output", "rollout_config", "pad_value", "eos_value"):
    if name not in src:
        fail(f"expected local '{name}' not present in grpo_learner.py")

INJECT = '''    # --- _LENDIST_DUMP: per-step ISL/OSL/truncation distributions -------------
    # All inputs are already host-side numpy here; no device sync is added.
    try:
      import json as _json, os as _os, numpy as _np
      _cap = int(getattr(rollout_config, "max_tokens_to_generate", 0) or 0)
      _osl, _trunc, _eos_end = [], 0, 0
      for _t in rollout_output.tokens:
        _n = int(len(_t))
        _osl.append(_n)
        if _cap and _n >= _cap:
          # Distinguish "ran into the wall" from "happened to finish exactly there".
          if _n > 0 and int(_t[-1]) == int(eos_value):
            _eos_end += 1
          else:
            _trunc += 1
      # _OSL_EFF: OSL to end of FIRST textual '<end_of_turn>' marker per sequence
      _eot = [236820, 643, 236779, 1340, 236779, 887, 236813]
      _ne = len(_eot)
      _osl_eff, _n_eot = [], 0
      for _t in rollout_output.tokens:
        _tl = list(_np.asarray(_t).tolist()); _pos = -1
        for _j in range(len(_tl) - _ne + 1):
          if _tl[_j] == 236820 and _tl[_j:_j + _ne] == _eot:
            _pos = _j + _ne; break
        if _pos >= 0:
          _osl_eff.append(_pos); _n_eot += 1
        else:
          _osl_eff.append(int(len(_tl)))
      _p = _np.asarray(rollout_output.left_padded_prompt_tokens)
      _isl = (_p != pad_value).sum(axis=-1).astype(int).tolist()
      _rec = {
          "step": int(self.rl_cluster.global_steps),
          "mode": str(mode),
          "cap": _cap,
          "n": len(_osl),
          "osl": _osl,
          "isl": _isl,
          "truncated_at_cap": _trunc,
          "eos_exactly_at_cap": _eos_end,
          "osl_eff": _osl_eff,
          "n_with_eot": _n_eot,
      }
      _run = _os.environ.get("JOB_IDENTIFIER", "unknown-run")
      _art = _os.environ.get("ARTIFACT_DIR", "")
      _line = _json.dumps(_rec)
      print("LENDIST step=%d n=%d cap=%d truncated=%d (%.2f%%) eos_at_cap=%d "
            "osl_mean=%.1f isl_mean=%.1f osl_eff_mean=%.1f with_eot=%d (%.1f%%)"
            % (_rec["step"], _rec["n"], _cap, _trunc,
               100.0 * _trunc / max(1, len(_osl)), _eos_end,
               sum(_osl) / max(1, len(_osl)), sum(_isl) / max(1, len(_isl)),
               sum(_osl_eff) / max(1, len(_osl_eff)), _n_eot,
               100.0 * _n_eot / max(1, len(_osl_eff))),
            flush=True)
      if _art.startswith("gs://"):
        from google.cloud import storage as _st
        _b, _, _pre = _art[5:].partition("/")
        _key = "%s/%s/lendist/step_%06d.json" % (_pre.rstrip("/"), _run, _rec["step"])
        _st.Client().bucket(_b).blob(_key.lstrip("/")).upload_from_string(
            _line, content_type="application/json")
    except Exception as _e:  # never take down training for a logging bug
      print("LENDIST ERROR: %r" % (_e,), flush=True)
    # --- end _LENDIST_DUMP ----------------------------------------------------

'''

src = src.replace(ANCHOR, INJECT + ANCHOR, 1)

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")

open(GL, "w", encoding="utf-8").write(src)

v = open(GL, encoding="utf-8").read()
problems = []
if "_LENDIST_DUMP" not in v:
    problems.append("marker missing")
if "truncated_at_cap" not in v:
    problems.append("truncation logic missing")
if v.count("agg_completion_mask = completion_mask.sum(axis=-1)") != 1:
    problems.append("original metric line disturbed")
try:
    ast.parse(v)
except SyntaxError as e:
    problems.append(f"reparse failed: {e}")
if problems:
    fail("; ".join(problems))
print("LENDIST PATCH: APPLIED")