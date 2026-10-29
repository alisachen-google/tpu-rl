"""OVERLONG-ADV: post-normalization advantage bias that pushes the GRPO policy off the
long-WRONG straggler tail at the SOURCE, without lowering the OSL cap and without touching
the check_numbers reward metric.

WHY THIS, AND WHY IT IS DIFFERENT FROM patch_length_shape.py (which was proven INERT):
  patch_length_shape added a length penalty to the *reward*. Under GRPO the advantage is
  A_i = (r_i - mean_g(r)) / (std_g(r) + eps) computed WITHIN each group of num_generations.
  An all-wrong straggler group is uniformly long, so a near-constant reward penalty is
  removed by the "- mean_g" and cancelled by the "/ std_g" => zero net gradient. Measured
  inert at lambda=0.25 AND lambda=1.0 (g4ucn/g4uco/g4ucp).

  This patch instead biases the advantage AFTER normalization:
      A_i := A_i - LAMBDA * clip((len_tok_i - TARGET)/SCALE, 0, 1)        (len in TOKENS)
  Subtracting post-norm is NOT cancelled by the group mean/std -- it is a direct, additive
  push-down on the policy-gradient weight of over-long completions. That is the mechanism
  reward-shaping could not reach.

PARITY-SAFE BY CONSTRUCTION (not just by gate):
  Default OVERLONG_WRONG_ONLY=1 => the bias is applied ONLY where reward == 0 (wrong). Every
  CORRECT completion keeps its exact GRPO advantage, so the reinforcement of correct answers
  -- hence check_numbers -- is structurally untouched. Among the WRONG set (advantage already
  negative, below group mean) the extra push makes long-wrong MORE negative than short-wrong,
  so the gradient prefers "stop sooner when you're going to be wrong". The OSL cap stays high
  (2560) so genuinely-long CORRECT answers are never truncated -- this shortens the tail at the
  source rather than clipping it. Still parity-GATED empirically (tb_parity.py over 100 steps).

WIRING: wraps GrpoLearner._generate_and_compute_advantage by inserting one call right after
the advantage_estimator() line. The helper sees `advantages` (post-norm, per-completion),
`completion_mask` (=> per-completion token length) and `rewards` (per-completion). No-op unless
OVERLONG_ADV=1.

Env knobs (all optional):
  OVERLONG_ADV=1            enable (default off => exact no-op passthrough)
  OVERLONG_ADV_LAMBDA=0.5   max advantage push (advantages are ~z-scores, O(1))
  OVERLONG_ADV_TARGET=1024  token length where the push starts (> correct typical length)
  OVERLONG_ADV_SCALE=1024   tokens over which it ramps from 0 to LAMBDA (1024->2048)
  OVERLONG_WRONG_ONLY=1     apply only to reward==0 completions (default on; parity-safe)

FAIL-OPEN. Prints exactly one of: OVERLONG-ADV PATCH: APPLIED | FAILED <reason>.
"""
import ast
import os
import sys

ROOT = "/usr/local/lib/python3.12/site-packages/tunix"
GL = os.path.join(ROOT, "rl/grpo/grpo_learner.py")


def fail(msg):
    print(f"OVERLONG-ADV PATCH: FAILED {msg}", flush=True)
    sys.exit(0)  # non-fatal: let the run start so the failure is visible in logs


if not os.path.exists(GL):
    fail(f"{GL} not found")
src = open(GL, encoding="utf-8").read()

if "_OVERLONG_ADV" in src:
    print("OVERLONG-ADV PATCH: already patched", flush=True)
    sys.exit(0)

# Anchor: the advantage_estimator(...) call. Must match exactly once.
ANCHOR = """      advantages = advantage_estimator(
          rewards=rewards, num_generations=self.algo_config.num_generations
      )
"""
if src.count(ANCHOR) != 1:
    fail(f"advantage_estimator anchor not found exactly once (count={src.count(ANCHOR)})")

INJECT = ANCHOR + """      advantages = _OVERLONG_ADV_bias(
          advantages, completion_mask, rewards)
"""

HELPER = '''

# --- _OVERLONG_ADV: post-normalization length bias on the GRPO advantage ----------
# Appended by patch_overlong_adv.py (see its docstring). Pushes the policy off the
# long-WRONG straggler tail at the source. No-op unless OVERLONG_ADV=1. Fail-open:
# on ANY error it returns the advantages unchanged so a run never crashes on this.
def _OVERLONG_ADV_bias(advantages, completion_mask, rewards):
  import os as _oa_os
  if _oa_os.environ.get("OVERLONG_ADV", "0") != "1":
    return advantages
  try:
    import numpy as _oa_np
    lam = float(_oa_os.environ.get("OVERLONG_ADV_LAMBDA", "0.5"))
    tgt = float(_oa_os.environ.get("OVERLONG_ADV_TARGET", "1024"))
    scl = float(_oa_os.environ.get("OVERLONG_ADV_SCALE", "1024"))
    wrong_only = _oa_os.environ.get("OVERLONG_WRONG_ONLY", "1") == "1"
    # per-completion TOKEN length from the mask (straggler = long token sequence)
    cm = _oa_np.asarray(completion_mask)
    lens = cm.reshape(cm.shape[0], -1).sum(axis=-1).astype(_oa_np.float64)
    ramp = _oa_np.clip((lens - tgt) / max(scl, 1.0), 0.0, 1.0)
    pen = lam * ramp                                  # >= 0, grows with length
    if wrong_only:
      r = _oa_np.asarray(rewards, dtype=_oa_np.float64).reshape(-1)
      if r.shape[0] == pen.shape[0]:
        pen = _oa_np.where(r <= 0.0, pen, 0.0)        # untouched where correct
    # broadcast pen against advantages' leading (per-completion) axis
    import numpy as _np2
    adv_arr = advantages
    nd = getattr(adv_arr, "ndim", 1)
    if getattr(adv_arr, "shape", (pen.shape[0],))[0] != pen.shape[0]:
      raise RuntimeError("adv/pen leading-dim mismatch %r vs %d" % (
          getattr(adv_arr, "shape", None), pen.shape[0]))
    pen_b = pen.reshape((-1,) + (1,) * (nd - 1))
    out = adv_arr - pen_b                             # SUBTRACT post-norm => not cancelled
    frac = float((ramp > 0).mean())
    hit = float(((pen > 0)).mean())
    print("OVERLONG-ADV step: len_mean=%.0f len_p90=%.0f pen_mean=%.4f "
          "frac_over=%.3f frac_pushed=%.3f (lam=%.2f tgt=%.0f scl=%.0f wrong_only=%d)" % (
          float(lens.mean()), float(_oa_np.percentile(lens, 90)), float(pen.mean()),
          frac, hit, lam, tgt, scl, int(wrong_only)), flush=True)
    return out
  except Exception as _e:
    print("OVERLONG-ADV: skipped this call (%r)" % (_e,), flush=True)
    return advantages
# --- end _OVERLONG_ADV ------------------------------------------------------------
'''

new = src.replace(ANCHOR, INJECT, 1) + HELPER
try:
    ast.parse(new)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")

open(GL, "w", encoding="utf-8").write(new)

v = open(GL, encoding="utf-8").read()
if "_OVERLONG_ADV_bias" not in v or v.count("def _OVERLONG_ADV_bias") != 1:
    fail("verification failed (helper missing or duplicated)")
if v.count("advantages = _OVERLONG_ADV_bias(") != 1:
    fail("verification failed (injection call missing or duplicated)")
try:
    ast.parse(v)
except SyntaxError as e:
    fail(f"reparse failed: {e}")
print("OVERLONG-ADV PATCH: APPLIED (enable with OVERLONG_ADV=1)", flush=True)
