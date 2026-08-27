"""Parallelize the GRPO reward functions across processes.

WHY. Phasetime v2 (run 11) measured the ~30 s `other` residual as almost
entirely `compute_rewards` (29.77 s at step 1; marshal+advantage is 0.3 s).
The reward functions (match_format_*, check_numbers) are pure-Python
regex/parsing over 2048 independent (prompt, completion) pairs, run serially
on one core while 28-44 head-node cores idle. Expected: ~30 s -> 2-4 s,
i.e. step ~157 s -> ~132 s.

HOW. Append-and-monkeypatch on tunix/rl/reward_manager.py:
`SequenceRewardManager._compute_rewards` is replaced by a version that keeps
the original signature-inspection logic but calls each reward_fn on CHUNKS of
the batch in a fork-based process pool, concatenating results in order.
Correctness properties:
  - per-pair independence: chunking is exact for sequence-level rewards
  - kwargs that are per-example sequences (len == num_prompts, e.g. ground
    truth answer columns) are chunked in lockstep; scalars pass through
  - results are concatenated in chunk order -> byte-identical to serial
  - ANY error in the parallel path (pickling, fn signature, worker crash)
    logs one warning and falls back to the ORIGINAL serial implementation
VALIDITY GATE for the run carrying this patch: rewards/train/mean must equal
the unpatched run step-for-step (deterministic seeds make rewards exactly
reproducible).

Tunables via env:
  TUNIX_REWARD_WORKERS   pool size (default min(32, cpu_count))
  TUNIX_REWARD_PARALLEL  set to "0" to disable (keeps serial path)

FAIL-OPEN. Any missing anchor prints FAILED and exits 0.
Prints exactly one of:
    REWARD-PAR PATCH: APPLIED
    REWARD-PAR PATCH: FAILED <reason>
"""

import ast
import os
import sys

ROOT = "/usr/local/lib/python3.12/site-packages/tunix"
RM = os.path.join(ROOT, "rl/reward_manager.py")


def fail(msg):
    print(f"REWARD-PAR PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(RM):
    fail(f"{RM} not found")
src = open(RM, encoding="utf-8").read()

if "_REWARD_PAR" in src:
    print("REWARD-PAR PATCH: already patched")
    sys.exit(0)

for needle in ("class SequenceRewardManager", "def _compute_rewards",
               "def _prepare_log_metrics"):
    if needle not in src:
        fail(f"'{needle}' not present in reward_manager.py")

APPEND = '''

# --- _REWARD_PAR: process-pool reward computation ---------------------------------
# Appended by patch_reward_parallel.py (see its docstring). The serial
# implementation above is kept intact as the fallback.
try:
  import os as _rp_os
  import numpy as _rp_np
  import multiprocessing as _rp_mp
  from dataclasses import asdict as _rp_asdict
  import inspect as _rp_inspect
  import logging as _rp_logging

  _rp_serial_compute = SequenceRewardManager._compute_rewards
  _rp_pool = [None]
  _rp_warned = [False]
  # Reward fns that must run serially in the parent (learned at runtime):
  # a fn that spawns its own subprocesses fails inside Pool's daemonic
  # workers ("daemonic processes are not allowed to have children") -- true
  # for check_numbers, whose math_verify grading dispatches to its own
  # persistent spawn pool. In the parent that internal pool works normally,
  # and its width is a MaxText config knob (math_verify_num_procs).
  _rp_serial_fns = set()

  def _rp_get_pool():
    if _rp_pool[0] is None:
      # The tunix trainer runs as a DAEMONIC child of the launcher process,
      # and multiprocessing forbids daemons from having children (run 12
      # failed here with "daemonic processes are not allowed to have
      # children"). The flag is inherited launcher plumbing, not a property
      # this code relies on: clear it so the pool can fork. The only
      # daemon-flag semantics lost are auto-termination of OUR workers if
      # the trainer dies mid-step; the pod teardown reaps them anyway.
      try:
        _cur = _rp_mp.current_process()
        if _cur._config.get("daemon"):
          _cur._config["daemon"] = False
      except Exception:
        pass
      n = int(_rp_os.environ.get("TUNIX_REWARD_WORKERS",
                                 min(32, _rp_os.cpu_count() or 8)))
      _rp_pool[0] = _rp_mp.get_context("fork").Pool(processes=n)
    return _rp_pool[0]

  def _rp_parallel_compute(self, prompts, completions, **kwargs):
    if _rp_os.environ.get("TUNIX_REWARD_PARALLEL", "1") == "0":
      return _rp_serial_compute(self, prompts, completions, **kwargs)
    try:
      algo_config_params = _rp_asdict(self.algo_config)
      base_kwargs = kwargs.copy()
      num_prompts = len(prompts)
      num_reward_fns = len(self.reward_fns)
      rewards = _rp_np.zeros((num_prompts, num_reward_fns))

      pool = _rp_get_pool()
      n_chunks = max(1, min(pool._processes * 2, num_prompts))
      bounds = [(num_prompts * c) // n_chunks for c in range(n_chunks + 1)]

      for i, reward_fn in enumerate(self.reward_fns):
        signature = _rp_inspect.signature(reward_fn)
        reward_fn_config_params = {}
        for name, _ in signature.parameters.items():
          if name in ["self", "prompts", "completions", "kwargs"]:
            continue
          if name in algo_config_params and name not in base_kwargs:
            reward_fn_config_params[name] = algo_config_params[name]
        call_kwargs = base_kwargs.copy()
        call_kwargs.update(reward_fn_config_params)

        fn_name = getattr(reward_fn, "__name__", str(reward_fn))
        if fn_name in _rp_serial_fns:
          # This fn previously failed in a worker (e.g. it spawns its own
          # subprocesses, which Pool's daemonic workers forbid -- true for
          # check_numbers' math_verify pool). Run it serially in the parent,
          # where its internal parallelism works as usual.
          full_kwargs = dict(call_kwargs)
          full_kwargs["prompts"] = list(prompts)
          full_kwargs["completions"] = list(completions)
          flat = list(reward_fn(**full_kwargs))
        else:
          try:
            jobs = []
            for c in range(n_chunks):
              lo, hi = bounds[c], bounds[c + 1]
              if lo == hi:
                continue
              ck = {}
              for k, v in call_kwargs.items():
                if isinstance(v, (list, tuple)) and len(v) == num_prompts:
                  ck[k] = list(v[lo:hi])
                elif isinstance(v, _rp_np.ndarray) and len(v) == num_prompts:
                  ck[k] = v[lo:hi]
                else:
                  ck[k] = v
              ck["prompts"] = list(prompts[lo:hi])
              ck["completions"] = list(completions[lo:hi])
              jobs.append(pool.apply_async(reward_fn, kwds=ck))
            parts = [j.get(timeout=180) for j in jobs]
            flat = [x for part in parts for x in list(part)]
          except Exception as _fe:
            _rp_serial_fns.add(fn_name)
            _rp_logging.warning(
                "REWARD-PAR: fn %s failed in workers (%r); running it "
                "serially in the parent from now on.", fn_name, _fe)
            full_kwargs = dict(call_kwargs)
            full_kwargs["prompts"] = list(prompts)
            full_kwargs["completions"] = list(completions)
            flat = list(reward_fn(**full_kwargs))
        if len(flat) != num_prompts:
          raise RuntimeError(
              "parallel reward length mismatch for %s: %d vs %d"
              % (fn_name, len(flat), num_prompts))
        rewards[:, i] = _rp_np.array(flat)

      log_metrics = self._prepare_log_metrics(prompts, completions, rewards)
      return {
          "rewards": _rp_np.nansum(rewards, axis=1),
          "log_metrics": log_metrics,
      }
    except Exception as _e:
      if not _rp_warned[0]:
        _rp_warned[0] = True
        _rp_logging.warning(
            "REWARD-PAR: parallel path failed (%r); falling back to the "
            "serial implementation for the rest of the run.", _e)
      _rp_os.environ["TUNIX_REWARD_PARALLEL"] = "0"
      return _rp_serial_compute(self, prompts, completions, **kwargs)

  SequenceRewardManager._compute_rewards = _rp_parallel_compute
except Exception as _e:
  print("REWARD-PAR ERROR (install): %r" % (_e,), flush=True)
# --- end _REWARD_PAR --------------------------------------------------------------
'''

new = src + APPEND
try:
    ast.parse(new)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")

open(RM, "w", encoding="utf-8").write(new)

v = open(RM, encoding="utf-8").read()
problems = []
if "_rp_parallel_compute" not in v:
    problems.append("wrapper missing")
if v.count("def _compute_rewards") != 1:
    problems.append("serial _compute_rewards definition disturbed")
try:
    ast.parse(v)
except SyntaxError as e:
    problems.append(f"reparse failed: {e}")
if problems:
    fail("; ".join(problems))
print("REWARD-PAR PATCH: APPLIED")
