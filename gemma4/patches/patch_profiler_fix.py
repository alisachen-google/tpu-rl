"""_PROFILER_FIX: disable the tunix trainer profiler when its step window is
invalid. In the RL setup the PeftTrainer is built with initial_step==max_step
(=num_batches*grad_accum, e.g. 160), so first_profile_step = initial_step +
skip_first_n_steps (161) > last_profile_step (160) -> ValueError crash at
tunix/sft/profiler.py __init__. This trainer profiler yields no HBM/roofline
for us (we use the PERF-V2 phased profiler for the rollout); disable it cleanly
instead of crashing. Fail-open, idempotent."""
def _patch():
    try:
        import tunix.sft.profiler as P
        fp = P.__file__
        src = open(fp).read()
        if "_PROFILER_FIX" in src:
            print("PROFILER-FIX: already applied", flush=True); return
        anchor = ('    if self._first_profile_step >= self._last_profile_step:\n'
                  '      raise ValueError(\n'
                  '          f"First profile step {self._first_profile_step} cannot be greater"\n'
                  '          f" than the last profile step {self._last_profile_step}."\n'
                  '      )\n')
        repl = ('    if self._first_profile_step >= self._last_profile_step:\n'
                '      # _PROFILER_FIX invalid window -> disable trainer profiler, do not crash\n'
                '      self._first_profile_step = 1000000000\n'
                '      self._last_profile_step = 1000000001\n')
        n = src.count(anchor)
        if n == 1:
            src = src.replace(anchor, repl, 1)
            compile(src, fp, "exec"); open(fp, "w").write(src)
            print("PROFILER-FIX: disabled invalid-window trainer profiler", flush=True)
        else:
            print("PROFILER-FIX: anchor count %d (skipped)" % n, flush=True)
    except Exception as e:
        print("PROFILER-FIX: failed %r" % (e,), flush=True)
_patch()
