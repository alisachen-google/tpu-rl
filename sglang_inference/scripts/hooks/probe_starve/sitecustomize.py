"""Test hook: starve the /health_generate probe while real traffic flows (sglang-torchtpu #1276, fix 2 validation).

Inert unless SGLANG_HOOK_STARVE_PROBE=1. Loaded through PYTHONPATH as sitecustomize in every interpreter the
launcher starts; patches sglang.srt.managers.scheduler.Scheduler after the module executes:
  * is_fully_idle(for_health_check=True) -> True, so the scheduler never takes the busy shortcut for a probe;
  * handle_generate_request drops any request whose rid starts with HEALTH_CHECK, so the probe is admitted and
    then silently never scheduled: it produces no output of its own, ever.
Real requests are untouched and keep producing scheduler/detokenizer messages. On the old handler those messages
satisfy the probe (200); on the fixed handler the probe times out (503).
"""
import os, sys
if os.environ.get("SGLANG_HOOK_STARVE_PROBE") == "1":
    import importlib.abc, importlib.util
    TARGET = "sglang.srt.managers.scheduler"
    def _say(msg):
        print(f"HOOK-STARVE pid={os.getpid()}: {msg}", file=sys.stderr, flush=True)
    def _patch(mod):
        cls = mod.Scheduler
        orig_idle = cls.is_fully_idle
        def is_fully_idle(self, for_health_check=False, ignore_waiting=False):
            if for_health_check:
                return True
            return orig_idle(self, for_health_check=for_health_check, ignore_waiting=ignore_waiting)
        orig_handle = cls.handle_generate_request
        dropped = [0]
        def handle_generate_request(self, recv_req, *a, **kw):
            rid = getattr(recv_req, "rid", "") or ""
            if rid.startswith("HEALTH_CHECK"):
                dropped[0] += 1
                if dropped[0] <= 3 or dropped[0] % 20 == 0:
                    _say(f"dropped probe {rid} (total {dropped[0]}); it will never be scheduled")
                return
            return orig_handle(self, recv_req, *a, **kw)
        cls.is_fully_idle = is_fully_idle
        cls.handle_generate_request = handle_generate_request
        _say(f"installed on {mod.__name__} ({mod.__file__})")
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig = spec.loader
            class _Loader(importlib.abc.Loader):
                def create_module(self, s): return orig.create_module(s)
                def exec_module(self, m): orig.exec_module(m); _patch(m)
            spec.loader = _Loader()
            return spec
    sys.meta_path.insert(0, _Finder())
    _say("finder armed")
