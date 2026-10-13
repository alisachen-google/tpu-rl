"""Test hook for sglang-torchtpu issue #1276 (decode /health_generate stays 200 while every KV transfer fails).

Inert unless SGLANG_HOOK_1276=1. Loaded through PYTHONPATH as sitecustomize in every interpreter the launcher
starts; installs a meta-path finder that patches sglang_tpu.disaggregation.raiden.conn after it is executed:
RaidenKVManager.is_recv_done always reports the receive as unfinished, so the KV lands but completion is never
signalled and the room fails at SGLANG_DISAGGREGATION_WAITING_TIMEOUT (the issue's repro). The pull is not
re-issued: maybe_start_pull returns True for an accepted room without a new read (pipeline) and the chunked/whole
paths only issue announced chunks once.
"""
import os, sys
if os.environ.get("SGLANG_HOOK_1276") == "1":
    import importlib.abc, importlib.util
    TARGET = "sglang_tpu.disaggregation.raiden.conn"
    def _say(msg):
        print(f"HOOK-1276 pid={os.getpid()}: {msg}", file=sys.stderr, flush=True)
    def _patch(mod):
        cls = mod.RaidenKVManager
        seen = set()
        def is_recv_done(self, room):
            if room not in seen:
                seen.add(room); _say(f"is_recv_done(room={room}) forced False; completion never signalled")
            return False
        cls.is_recv_done = is_recv_done
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
