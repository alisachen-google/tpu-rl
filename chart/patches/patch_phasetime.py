"""Phasetime v2 -- split the `other` residual INTO its components.

V1 (runs 10/10b) plus TensorBoard arithmetic already proved the ~30 s `other`
residual is exactly `generate_and_compute_advantage - vllm_rollout`, i.e. the
reward + advantage + data-marshaling stage between the vLLM generate call and
the train loop (verified to +-0.5 s on three matched steps). V2 answers the
next question: WHICH of those dominates. It keeps all four v1 timers and adds:

    engine_generate    RLCluster.generate wall -- the vLLM call INCLUDING
                       tunix-side detokenization; against TensorBoard's
                       rollout_time this locates where detokenize lands
    compute_rewards    RLLearner._compute_rewards wall -- the Python reward
                       functions (match_format_*, check_numbers) over 2048
                       completions

Derived offline, no extra timers needed:
    marshal+advantage  = generate_and_advantage - engine_generate
                         - compute_rewards
    (contains the 2048 x 8192 pad-and-stack ~67 MB numpy build, jnp
    conversions, advantage estimator, device_put)

Same mechanics as v1: append-and-monkeypatch on the class objects, six
perf_counter pairs per global step, no device sync, JSON per step to
<ARTIFACT_DIR>/<run>/phasetime/step_NNNNNN.json plus a PHASETIME stdout line.

CIRCULAR-IMPORT NOTE. rl_learner imports rl_cluster at module load, so the
block appended to rl_cluster.py must import rl_learner LAZILY inside the
wrapper body, never at module level.

FAIL-OPEN. Any missing class/method prints FAILED and exits 0 without
writing. Prints exactly one of:
    PHASETIME2 PATCH: APPLIED
    PHASETIME2 PATCH: FAILED <reason>
"""

import ast
import os
import sys

SITE = os.environ.get("PATCH_ROOT", "/usr/local/lib/python3.12/site-packages")
ROOT = os.path.join(SITE, "tunix")
RL = os.path.join(ROOT, "rl/rl_learner.py")
GL = os.path.join(ROOT, "rl/grpo/grpo_learner.py")
RC = os.path.join(ROOT, "rl/rl_cluster.py")


def fail(msg):
    print(f"PHASETIME2 PATCH: FAILED {msg}")
    sys.exit(0)


for p in (RL, GL, RC):
    if not os.path.exists(p):
        fail(f"{p} not found")

rl_src = open(RL, encoding="utf-8").read()
gl_src = open(GL, encoding="utf-8").read()
rc_src = open(RC, encoding="utf-8").read()

if any("_PHASETIME" in s for s in (rl_src, gl_src, rc_src)):
    print("PHASETIME2 PATCH: already patched")
    sys.exit(0)

for needle, where in (("class RLLearner", "rl_learner.py"),
                      ("def _run_global_step", "rl_learner.py"),
                      ("def _prepare_data", "rl_learner.py"),
                      ("def _run_all_micro_batch_steps", "rl_learner.py"),
                      ("def _compute_rewards", "rl_learner.py")):
    if needle not in rl_src:
        fail(f"'{needle}' not present in {where}")
for needle in ("class GRPOLearner", "def _generate_and_compute_advantage"):
    if needle not in gl_src:
        fail(f"'{needle}' not present in grpo_learner.py")
for needle in ("class RLCluster", "def generate"):
    if needle not in rc_src:
        fail(f"'{needle}' not present in rl_cluster.py")

RL_APPEND = '''

# --- _PHASETIME v2: break out the `other` residual --------------------------------
# Appended by patch_phasetime2.py. Six perf_counter pairs per global step; no
# device sync. Wrappers are installed on the class objects only.
try:
  import time as _pt_time
  import json as _pt_json
  import os as _pt_os

  _PHASETIME = {}

  def _pt_add(_name, _dt):
    _PHASETIME[_name] = _PHASETIME.get(_name, 0.0) + float(_dt)

  def _pt_wrap(_cls, _meth, _label):
    _orig = getattr(_cls, _meth)

    def _w(self, *a, **k):
      _t0 = _pt_time.perf_counter()
      try:
        return _orig(self, *a, **k)
      finally:
        _pt_add(_label, _pt_time.perf_counter() - _t0)

    _w.__name__ = getattr(_orig, "__name__", _meth)
    setattr(_cls, _meth, _w)

  def _pt_flush(self):
    try:
      _step = -1
      try:
        _step = int(self.rl_cluster.global_steps)
      except Exception:
        pass
      _rec = dict(_PHASETIME)
      _rec["step"] = _step
      print("PHASETIME step=%d " % _step
            + " ".join("%s=%.2f" % (_k, _v) for _k, _v in sorted(_rec.items())
                       if _k != "step"), flush=True)
      _run = _pt_os.environ.get("JOB_IDENTIFIER", "unknown-run")
      _art = _pt_os.environ.get("ARTIFACT_DIR", "")
      if _art.startswith("gs://"):
        from google.cloud import storage as _st
        _b, _, _pre = _art[5:].partition("/")
        _key = "%s/%s/phasetime/step_%06d.json" % (_pre.rstrip("/"), _run, _step)
        _st.Client().bucket(_b).blob(_key.lstrip("/")).upload_from_string(
            _pt_json.dumps(_rec), content_type="application/json")
    except Exception as _e:
      print("PHASETIME ERROR: %r" % (_e,), flush=True)
    finally:
      _PHASETIME.clear()

  _pt_wrap(RLLearner, "_prepare_data", "prepare_data")
  _pt_wrap(RLLearner, "_run_all_micro_batch_steps", "micro_batch_steps")
  _pt_wrap(RLLearner, "_compute_rewards", "compute_rewards")

  _pt_orig_gs = RLLearner._run_global_step

  def _pt_global_step(self, *a, **k):
    _t0 = _pt_time.perf_counter()
    try:
      return _pt_orig_gs(self, *a, **k)
    finally:
      _pt_add("run_global_step", _pt_time.perf_counter() - _t0)
      _pt_flush(self)

  RLLearner._run_global_step = _pt_global_step
except Exception as _e:
  print("PHASETIME ERROR (install): %r" % (_e,), flush=True)
# --- end _PHASETIME ---------------------------------------------------------------
'''

GL_APPEND = '''

# --- _PHASETIME v2: time the GRPO payload inside prepare_data ---------------------
try:
  import time as _pt2_time
  from tunix.rl import rl_learner as _pt2_rl

  _pt2_orig = GRPOLearner._generate_and_compute_advantage

  def _pt2_w(self, *a, **k):
    _t0 = _pt2_time.perf_counter()
    try:
      return _pt2_orig(self, *a, **k)
    finally:
      try:
        _pt2_rl._pt_add("generate_and_advantage", _pt2_time.perf_counter() - _t0)
      except Exception:
        pass

  GRPOLearner._generate_and_compute_advantage = _pt2_w
except Exception as _e:
  print("PHASETIME ERROR (grpo install): %r" % (_e,), flush=True)
# --- end _PHASETIME ---------------------------------------------------------------
'''

RC_APPEND = '''

# --- _PHASETIME v2: time the engine generate call (vLLM + detokenize) -------------
# NOTE: rl_learner imports THIS module at load time, so rl_learner must be
# imported lazily inside the wrapper to avoid a circular import.
try:
  import time as _pt3_time

  _pt3_orig = RLCluster.generate

  def _pt3_w(self, *a, **k):
    _t0 = _pt3_time.perf_counter()
    try:
      return _pt3_orig(self, *a, **k)
    finally:
      try:
        from tunix.rl import rl_learner as _pt3_rl
        _pt3_rl._pt_add("engine_generate", _pt3_time.perf_counter() - _t0)
      except Exception:
        pass

  _pt3_w.__name__ = "generate"
  RLCluster.generate = _pt3_w
except Exception as _e:
  print("PHASETIME ERROR (cluster install): %r" % (_e,), flush=True)
# --- end _PHASETIME ---------------------------------------------------------------
'''

new_rl = rl_src + RL_APPEND
new_gl = gl_src + GL_APPEND
new_rc = rc_src + RC_APPEND
for name, s in (("rl_learner.py", new_rl), ("grpo_learner.py", new_gl),
                ("rl_cluster.py", new_rc)):
    try:
        ast.parse(s)
    except SyntaxError as e:
        fail(f"patched {name} does not parse: {e}")

open(RL, "w", encoding="utf-8").write(new_rl)
open(GL, "w", encoding="utf-8").write(new_gl)
open(RC, "w", encoding="utf-8").write(new_rc)

problems = []
v_rl = open(RL, encoding="utf-8").read()
v_gl = open(GL, encoding="utf-8").read()
v_rc = open(RC, encoding="utf-8").read()
if "_pt_flush" not in v_rl or 'compute_rewards"' not in v_rl:
    problems.append("rl_learner wrappers missing")
if "_pt2_w" not in v_gl:
    problems.append("grpo_learner wrapper missing")
if "_pt3_w" not in v_rc:
    problems.append("rl_cluster wrapper missing")
if v_rl.count("def _run_global_step") != 1:
    problems.append("rl_learner._run_global_step disturbed")
for name, s in (("rl_learner.py", v_rl), ("grpo_learner.py", v_gl),
                ("rl_cluster.py", v_rc)):
    try:
        ast.parse(s)
    except SyntaxError as e:
        problems.append(f"reparse {name}: {e}")
if problems:
    fail("; ".join(problems))
print("PHASETIME2 PATCH: APPLIED")