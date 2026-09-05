"""_GC_SYNC: gate tunix's per-step gc.collect() calls behind env TUNIX_GC_COLLECT (default off).
tunix e4fc1dbf added gc.collect() after rollout.update_params (weight sync) and after the
actor-logps pass in rl_cluster.py -- unconditional, every step. On our colocated (no CPU
offload) setup this is a pure host stall: measured +1.2 s/step in perf/train/weight_sync_time
(run 42 vs run 40). Gating (not deleting) keeps the upstream behaviour one env var away.
Fail-open, idempotent."""
def _patch():
    try:
        import tunix.rl.rl_cluster as RC
        fp = RC.__file__; src = open(fp).read()
        if "_GC_SYNC" in src:
            print("GC-SYNC: already applied", flush=True); return
        n = src.count("gc.collect()")
        if n == 0:
            print("GC-SYNC: no gc.collect() found (skipped)", flush=True); return
        src = src.replace("gc.collect()", "(gc.collect() if __import__('os').environ.get('TUNIX_GC_COLLECT','0')=='1' else None)  # _GC_SYNC")
        compile(src, fp, "exec"); open(fp, "w").write(src)
        print("GC-SYNC: gated %d gc.collect() call(s) in rl_cluster.py (TUNIX_GC_COLLECT=1 restores)" % n, flush=True)
    except Exception as e:
        print("GC-SYNC: failed %r" % (e,), flush=True)
_patch()
