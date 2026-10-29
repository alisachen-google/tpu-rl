"""LENGTH-SHAPE: add a conciseness penalty to the GRPO total reward that targets the
STRAGGLER tail (long, usually-WRONG rambling completions) WITHOUT touching correct answers.

WHY (measured, g4r38 reward dump, n=44): correct completions are short (median 1776,
p95 2982 chars, max 6283); WRONG completions are long (median 4292, p90 8155, max 20346).
The rollout straggler (longest-of-2048 per step => makespan = longest * per-step) is
dominated by these long wrong rambles. So a penalty that ramps in ABOVE the correct-p95
steers the model off the long-wrong tail while leaving correct answers (all below target)
essentially unpenalized.

DESIGN (parity-preserving by construction, not just by gate):
  penalty_i = -LAMBDA * clip((n_chars_i - TARGET) / SCALE, 0, 1)
  total_reward_i += penalty_i
- Correct answers (n_chars < TARGET) get ~0 penalty => check_numbers parity structurally held.
- The penalty magnitude (<= LAMBDA ~ 0.25) stays below the correctness reward (~1.0), so in
  mixed groups correct answers keep positive GRPO advantage; it only re-ranks WITHIN a group
  toward shorter (among wrong: shorter wrong > longer wrong; among correct: ~unchanged).
- check_numbers and every other reward_fn are UNTOUCHED => the reported parity metric
  (REWARD-DUMP score=) stays pure and gated.

WIRING: wraps SequenceRewardManager._compute_rewards. MUST be applied AFTER
patch_reward_parallel.py (which installs _rp_parallel_compute); this wrapper calls whatever
_compute_rewards is currently bound, then adds the penalty to result["rewards"] and records
diagnostics in result["log_metrics"] (len_mean / len_p90 / shape_pen_mean / shape_frac_hit).

Env knobs (all optional):
  LENGTH_SHAPE=1            enable (default off => exact no-op passthrough)
  LENGTH_SHAPE_LAMBDA=0.25  max penalty magnitude
  LENGTH_SHAPE_TARGET=3500  n_chars where penalty starts (set > correct p95 ~2982)
  LENGTH_SHAPE_SCALE=9000   n_chars over which it ramps from 0 to LAMBDA

FAIL-OPEN. Prints exactly one of: LENGTH-SHAPE PATCH: APPLIED | FAILED <reason>.
"""
import ast
import os
import sys

ROOT = "/usr/local/lib/python3.12/site-packages/tunix"
RM = os.path.join(ROOT, "rl/reward_manager.py")


def fail(msg):
    print(f"LENGTH-SHAPE PATCH: FAILED {msg}", flush=True)
    sys.exit(0)  # non-fatal: let the run start so the failure is visible in logs


if not os.path.exists(RM):
    fail(f"{RM} not found")
src = open(RM, encoding="utf-8").read()

if "_LENGTH_SHAPE" in src:
    print("LENGTH-SHAPE PATCH: already patched", flush=True)
    sys.exit(0)

if "class SequenceRewardManager" not in src or "def _compute_rewards" not in src:
    fail("SequenceRewardManager/_compute_rewards not present")

APPEND = '''

# --- _LENGTH_SHAPE: straggler-tail conciseness penalty on the total reward ---------
# Appended by patch_length_shape.py (see its docstring). Wraps whatever
# _compute_rewards is currently bound (serial OR the reward-parallel version) so it
# MUST be invoked after patch_reward_parallel.py. No-op unless LENGTH_SHAPE=1.
try:
  import os as _ls_os
  import numpy as _ls_np
  import logging as _ls_logging

  _ls_inner = SequenceRewardManager._compute_rewards

  def _ls_compute(self, prompts, completions, **kwargs):
    out = _ls_inner(self, prompts, completions, **kwargs)
    if _ls_os.environ.get("LENGTH_SHAPE", "0") != "1":
      return out
    try:
      lam = float(_ls_os.environ.get("LENGTH_SHAPE_LAMBDA", "0.25"))
      tgt = float(_ls_os.environ.get("LENGTH_SHAPE_TARGET", "3500"))
      scl = float(_ls_os.environ.get("LENGTH_SHAPE_SCALE", "9000"))
      lens = _ls_np.array([len(c) for c in completions], dtype=_ls_np.float64)
      ramp = _ls_np.clip((lens - tgt) / max(scl, 1.0), 0.0, 1.0)
      pen = -lam * ramp                      # <= 0, zero for lens <= tgt
      r = _ls_np.asarray(out["rewards"], dtype=_ls_np.float64)
      if r.shape[0] != pen.shape[0]:
        raise RuntimeError("length mismatch reward %d vs pen %d" % (r.shape[0], pen.shape[0]))
      out["rewards"] = r + pen
      # NOTE: do NOT inject into out["log_metrics"] — rl_cluster.buffer_metrics expects
      # each value to be a (value, op) tuple; a plain float raises "cannot unpack
      # non-iterable float object" and crash-loops the run (observed in g4ucl). The
      # print below is the monitoring channel instead.
      print("LENGTH-SHAPE step: len_mean=%.0f len_p90=%.0f pen_mean=%.4f frac_hit=%.3f "
            "(lam=%.3f tgt=%.0f scl=%.0f)" % (
            float(lens.mean()), float(_ls_np.percentile(lens, 90)), float(pen.mean()),
            float((ramp > 0).mean()), lam, tgt, scl), flush=True)
    except Exception as _e:
      _ls_logging.warning("LENGTH-SHAPE: skipped this call (%r)", _e)
    return out

  SequenceRewardManager._compute_rewards = _ls_compute
except Exception as _e:
  print("LENGTH-SHAPE ERROR (install): %r" % (_e,), flush=True)
# --- end _LENGTH_SHAPE ------------------------------------------------------------
'''

new = src + APPEND
try:
    ast.parse(new)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")

open(RM, "w", encoding="utf-8").write(new)

v = open(RM, encoding="utf-8").read()
if "_ls_compute" not in v or v.count("def _compute_rewards") != 1:
    fail("verification failed (wrapper missing or serial def disturbed)")
try:
    ast.parse(v)
except SyntaxError as e:
    fail(f"reparse failed: {e}")
print("LENGTH-SHAPE PATCH: APPLIED (enable with LENGTH_SHAPE=1)", flush=True)
