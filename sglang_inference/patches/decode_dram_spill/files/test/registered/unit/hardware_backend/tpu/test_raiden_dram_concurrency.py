"""Concurrency, deadlock and hang checks for RaidenDramStorage (CPU, real gRPC, DMA double).

Reuses the StoreFixture from the bundle's unit test module. Every test runs under a watchdog
that dumps all thread stacks and fails instead of hanging the session.
"""
from __future__ import annotations

import faulthandler
import random
import sys
import threading
import time
from pathlib import Path

import grpc
import pytest

from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang_tpu.raiden_dram import wire_pb2 as w

ROOT = Path(__file__).resolve()
for candidate in (Path.home() / "sglang-spill",):
    sys.path.insert(0, str(candidate / "test/registered/unit/hardware_backend/tpu"))
from test_raiden_dram_storage import StoreFixture, config, make_pool  # noqa: E402


class Watchdog:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self._fire)
        self.timer.daemon = True
        self.timer.start()
        return self

    def _fire(self):
        faulthandler.dump_all_threads(sys.stderr)
        raise SystemExit("WATCHDOG: possible deadlock/hang")

    def __exit__(self, *exc):
        self.timer.cancel()


class SlowStore(StoreFixture):
    """Adds server-side latency and optional stalls to the protocol double."""

    def __init__(self, delay=0.0):
        super().__init__()
        self.delay = delay
        self.stall_writes = threading.Event()  # set -> WriteRemote never finishes
        self.lock = threading.Lock()

    def write(self, request, context):
        if self.stall_writes.is_set():
            yield w.WriteRemoteEvent(ack=w.WriteRemoteAck(operation_id=7, granted_deadline_ms=1))
            while self.stall_writes.is_set():
                time.sleep(0.05)
            context.abort(grpc.StatusCode.UNAVAILABLE, "stalled write released")
        time.sleep(random.random() * self.delay)
        with self.lock:
            yield from super().write(request, context)

    def fetch(self, request, context):
        time.sleep(random.random() * self.delay)
        with self.lock:
            return super().fetch(request, context)

    def poll(self, request, context):
        if self.stall_writes.is_set():
            return w.PollWriteRemoteResponse(state=w.PollWriteRemoteResponse.PENDING)
        return super().poll(request, context)


def build(delay=0.0, page_num=256, **extra):
    peer = SlowStore(delay)
    pool = make_pool(page_num=page_num)
    peer.pools["cpu-fixture:1"] = pool
    cfg = config(peer.address)
    cfg.extra_config.update(extra)  # e.g. rpc_timeout_s
    storage = StorageBackendFactory.create_backend("dynamic", cfg, pool)
    storage.register_mem_pool_host(pool)
    return storage, pool, peer


def test_threaded_spill_recall_flush_no_deadlock():
    """8 writer/reader threads + a flusher for 10 s: no exception, no hang, no pin leak."""
    storage, pool, peer = build(delay=0.01)
    stop = threading.Event()
    errors = []
    counters = {"set": 0, "get": 0, "exists": 0, "flush": 0}
    clock = threading.Lock()
    # /flush_cache only runs with no in-flight requests; model that with a quiesce gate so a
    # flush cannot hand a worker's live slot to another worker (a test artifact, not a bug).
    quiesce = threading.Condition()
    active = [0]
    flushing = [False]

    def bump(k):
        with clock:
            counters[k] += 1

    def worker(tid):
        rng = random.Random(tid)
        n = 0
        try:
            while not stop.is_set():
                with quiesce:
                    while flushing[0]:
                        quiesce.wait(1)
                    active[0] += 1
                try:
                    step(tid, rng, n)
                finally:
                    with quiesce:
                        active[0] -= 1
                        quiesce.notify_all()
                n += 1
        except Exception as exc:  # noqa: BLE001
            errors.append((tid, repr(exc)))

    def step(tid, rng, n):
        if True:
            if True:
                keys = [f"t{tid}-k{n}-{i}" for i in range(rng.randint(1, 6))]
                with pool.lock:
                    idx = pool.alloc(len(keys) * pool.page_size)
                if idx is None:
                    time.sleep(0.005)
                    return
                for key, page in zip(keys, pool._to_page_ids(idx)):
                    pool.payload[page] = key.encode()
                res = storage.batch_set_v1(keys, idx)
                bump("set")
                assert len(res) == len(keys)
                hits = storage.batch_exists(keys)
                bump("exists")
                assert 0 <= hits <= len(keys)
                with pool.lock:
                    dst = pool.alloc(len(keys) * pool.page_size)
                if dst is not None:
                    got = storage.batch_get_v1(keys, dst)
                    bump("get")
                    for ok, key, page in zip(got, keys, pool._to_page_ids(dst)):
                        if ok:
                            assert pool.payload[page] == key.encode()
                    pool.free(dst)
                pool.free(idx)

    def flusher():
        while not stop.is_set():
            time.sleep(0.2)
            with quiesce:
                flushing[0] = True
                while active[0]:
                    quiesce.wait(1)
                with pool.lock:
                    pool.clear()
                flushing[0] = False
                quiesce.notify_all()
            bump("flush")

    with Watchdog(60):
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(8)]
        threads.append(threading.Thread(target=flusher, daemon=True))
        for t in threads:
            t.start()
        time.sleep(10)
        stop.set()
        for t in threads:
            t.join(timeout=20)
            assert not t.is_alive(), "thread did not finish: hang"
    storage.close()
    peer.server.stop(0).wait()
    assert not errors, errors
    stats = storage.get_stats()
    assert counters["set"] > 50 and counters["get"] > 50, counters
    assert stats["quarantined_pages"] == 0, stats
    print("counters", counters, "stats", stats)


def test_lookup_blocked_behind_inflight_spill_is_bounded():
    """The backend serializes IO on one lock: a lookup waits for an in-flight WriteRemote.

    This documents the stall bound (rpc_timeout_s) rather than a deadlock: with the write stalled
    forever, batch_exists must still return within the timeout budget and the slots get quarantined.
    """
    storage, pool, peer = build(rpc_timeout_s=1)
    peer.stall_writes.set()
    idx = pool.alloc(2 * pool.page_size)
    for key, page in zip(["x", "y"], pool._to_page_ids(idx)):
        pool.payload[page] = key.encode()
    result = {}

    def spill():
        result["set"] = storage.batch_set_v1(["x", "y"], idx)

    with Watchdog(30):
        t = threading.Thread(target=spill, daemon=True)
        t.start()
        time.sleep(0.2)
        t0 = time.perf_counter()
        hits = storage.batch_exists(["x", "y"])
        waited = time.perf_counter() - t0
        t.join(10)
        assert not t.is_alive()
    peer.stall_writes.clear()
    assert hits == 0
    assert result["set"] == [False, False]
    assert waited < 5, f"lookup blocked {waited:.1f}s behind a stalled spill"
    # Slots whose remote pull may still be running are retired for the process lifetime.
    assert set(pool._to_page_ids(idx)) <= pool._quarantined_pages
    pool.clear()
    assert not (set(pool._to_page_ids(idx)) & set(pool.free_pages))
    print(f"lookup waited {waited:.2f}s behind a stalled WriteRemote (rpc_timeout_s=1)")
    storage.close()
    peer.server.stop(0).wait()


def test_registry_outage_mid_traffic_is_a_miss_not_a_hang():
    storage, pool, peer = build(rpc_timeout_s=1)
    idx = pool.alloc(pool.page_size)
    pool.payload[pool._to_page_ids(idx)[0]] = b"p"
    assert storage.batch_set_v1(["k"], idx) == [True]
    peer.server.stop(0).wait()  # registry + store gone
    with Watchdog(30):
        t0 = time.perf_counter()
        assert storage.batch_exists(["k"]) == 0
        dst = pool.alloc(pool.page_size)
        assert storage.batch_get_v1(["k"], dst) == [False]
        more = pool.alloc(pool.page_size)
        pool.payload[pool._to_page_ids(more)[0]] = b"q"
        assert storage.batch_set_v1(["k2"], more) == [False]
        elapsed = time.perf_counter() - t0
    assert elapsed < 10, elapsed
    # No DMA was ever admitted (connection refused), so nothing is quarantined.
    assert storage.get_stats()["quarantined_pages"] == 0
    print(f"registry outage: three ops returned in {elapsed:.2f}s")
    storage.close()


def test_close_during_inflight_io_does_not_deadlock():
    storage, pool, peer = build(rpc_timeout_s=1)
    peer.stall_writes.set()
    idx = pool.alloc(pool.page_size)
    pool.payload[pool._to_page_ids(idx)[0]] = b"p"
    done = threading.Event()

    def spill():
        try:
            storage.batch_set_v1(["k"], idx)
        finally:
            done.set()

    with Watchdog(30):
        threading.Thread(target=spill, daemon=True).start()
        time.sleep(0.2)
        storage.close()  # must wait for the lock, not deadlock
        assert done.wait(10)
        with pytest.raises(RuntimeError):
            storage.batch_exists(["k"])
    peer.stall_writes.clear()
    peer.server.stop(0).wait()


def test_hybrid_v2_concurrent_pools():
    """KV and SWA pools used from two threads at once share one client; no cross-pool mixups."""
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

    storage, pool, peer = build(delay=0.005)
    swa = make_pool(endpoint="cpu-fixture:2", page_num=128)
    peer.pools["cpu-fixture:2"] = swa
    storage.register_mem_host_pool_v2(swa, PoolName.SWA)
    errors = []

    def run(name, p, tid):
        try:
            for n in range(40):
                keys = [f"{name}-{tid}-{n}-{i}" for i in range(3)]
                with p.lock:
                    idx = p.alloc(3 * p.page_size)
                for key, page in zip(keys, p._to_page_ids(idx)):
                    p.payload[page] = key.encode()
                res = storage.batch_set_v2([PoolTransfer(name=name, keys=keys, host_indices=idx)])
                assert res[name] == [True, True, True], res
                with p.lock:
                    dst = p.alloc(3 * p.page_size)
                got = storage.batch_get_v2([PoolTransfer(name=name, keys=keys, host_indices=dst)])
                assert got[name] == [True, True, True]
                for key, page in zip(keys, p._to_page_ids(dst)):
                    assert p.payload[page] == key.encode()
                p.free(idx)
                p.free(dst)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    with Watchdog(60):
        ts = [threading.Thread(target=run, args=(PoolName.KV, pool, i), daemon=True) for i in range(3)]
        ts += [threading.Thread(target=run, args=(PoolName.SWA, swa, i), daemon=True) for i in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(50)
            assert not t.is_alive()
    assert not errors, errors
    storage.close()
    peer.server.stop(0).wait()
