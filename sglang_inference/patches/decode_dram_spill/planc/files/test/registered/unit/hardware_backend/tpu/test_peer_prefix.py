"""CPU tests for the reverse prefix pull (Plan C): the manager mixin's producer
and consumer state machines with fake engines and a fake radix tree, and the
prefill scheduler hook that holds, pulls, inserts and releases a request."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import pd_peer_prefix as hook
from sglang_tpu.disaggregation.raiden import peer_prefix as pp
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixBlocks,
    PrefixRelease,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4


class FakeEngine:
    def __init__(self):
        self.registered = []
        self.reads = []
        self.fail_start = False

    def register_read(self, req_id, uuid, blocks):
        self.registered.append((req_id, uuid, list(blocks)))
        return bool(blocks)

    def start_read(self, req_id, uuid, endpoint, remote, local, parallelism=1):
        if self.fail_start:
            raise RuntimeError("no staging")
        assert len(remote) == len(local)
        self.reads.append((req_id, uuid, endpoint, list(remote), list(local)))


class FakeLock:
    def __init__(self, node):
        self.node = node

    def to_dec_params(self):
        return ("dec", self.node)


class FakeTree:
    """Holds one prefix as device-resident tokens plus an optional host tail."""

    def __init__(self, device_tokens, host_tokens=0, page=PAGE):
        self.device_tokens = device_tokens
        self.host_tokens = host_tokens
        self.page = page
        self.locks = []
        self.unlocks = []
        self.inserts = []
        self.load_back_slot = True
        self.event_done = True
        self.is_swa_enabled = False
        self.tree_core = SimpleNamespace(is_eagle=False)

    def swa_reprefill_tail_tokens(self):
        return 0

    def match_prefix(self, params):
        n = min(len(params.key.token_ids), self.device_tokens)
        n = (n // self.page) * self.page
        host = min(self.host_tokens, len(params.key.token_ids) - n)
        host = (host // self.page) * self.page
        return SimpleNamespace(
            device_indices=torch.arange(1000, 1000 + n, dtype=torch.int64),
            host_hit_length=host,
            last_device_node=("dev", n),
            best_match_node=("host", n + host),
            swa_branching_seqlen=None,
        )

    def inc_lock_ref(self, node):
        self.locks.append(node)
        return FakeLock(node)

    def dec_lock_ref(self, node, params):
        self.unlocks.append((node, params))

    def has_free_load_back_slot(self):
        return self.load_back_slot

    def init_load_back(self, params):
        n = params.host_hit_length
        return torch.arange(5000, 5000 + n, dtype=torch.int64), ("restored", n)

    def ready_to_load_host_cache(self):
        return 7

    def is_load_back_event_done(self, idx):
        return self.event_done

    def insert(self, params):
        self.inserts.append(params)
        return SimpleNamespace(
            prefix_len=len(params.key.token_ids), rotation_tail_declined=False
        )


class StubManager(pp.PeerPrefixMixin):
    """Just enough of RaidenKVManager for the mixin."""

    def __init__(self, name):
        self.name = name
        self._message_handlers = {}
        self._state_lock = threading.Lock()
        self._completed_recv = set()
        self._completed_send = set()
        self._send_done_at = {}
        self._failed = set()
        self._native_failed = set()
        self.local_ip = "10.0.0.1"
        self.rank_port = 31000
        self.raiden_endpoint = f"{name}-raiden"
        self.swa_raiden_endpoint = ""
        self._transfer_parallelism = 1
        self.engine = FakeEngine()
        self._swa_engine = None
        self.sent = []
        self.peer = None
        self.drained = 0
        self._peer_init()

    def _ensure_engine(self):
        return self.engine

    def _send_multipart_locked(self, endpoint, parts, is_ipv6=False):
        self.sent.append((endpoint, parts))
        if self.peer is not None:
            self.peer._message_handlers[bytes(parts[0])](parts)

    def _drain_engine(self):
        self.drained += 1

    def _on_pull_req(self, msg):  # the real one lives in the sender mixin
        pass


def pair():
    prefill, decode = StubManager("prefill"), StubManager("decode")
    prefill.peer, decode.peer = decode, prefill
    prefill._peer_decode_endpoint = "tcp://10.0.0.2:31000"
    return prefill, decode


def ctx(tree, **kw):
    kwargs = dict(tree_cache=tree, allocator=None, page_size=PAGE, min_hit_tokens=PAGE)
    kwargs.update(kw)
    return pp.ProducerContext(**kwargs)


def test_serve_registers_locks_and_releases_on_notice():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    tokens = list(range(20))
    assert prefill.peer_fetch_begin(5, tokens, 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    fetch = prefill.peer_fetch(5)
    assert fetch.stage == pp.STAGE_REPLIED and fetch.hit_len == 16
    assert fetch.blocks == [250, 251, 252, 253] and fetch.uuid == pp.peer_uuid(5)
    assert decode.engine.registered == [
        ("peer:5", pp.peer_uuid(5), [250, 251, 252, 253])
    ]
    assert tree.locks == [("dev", 16)] and tree.unlocks == []
    assert prefill.peer_fetch_start_read(5, [10, 11, 12, 13], [])
    assert prefill.engine.reads[0][2] == "decode-raiden"
    assert prefill.peer_fetch_poll(5, 10) == pp.STAGE_READING
    prefill._completed_recv.add("peer:5")
    assert prefill.peer_fetch_poll(5, 10) == pp.STAGE_DONE
    prefill.peer_fetch_end(5)
    assert "peer:5" not in prefill._completed_recv
    decode.peer_prefix_serve(ctx(tree))
    assert tree.unlocks == [(("dev", 16), ("dec", ("dev", 16)))]
    assert decode.peer_stats()["serving"] == 0


def test_short_hit_is_a_miss_without_locks():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=4)
    prefill.peer_fetch_begin(1, list(range(20)), 4, None, None)
    decode.peer_prefix_serve(ctx(tree))
    assert prefill.peer_fetch(1).stage == pp.STAGE_REPLIED
    assert prefill.peer_fetch(1).hit_len == 0
    assert tree.locks == [] and decode.engine.registered == []


def test_host_part_is_restored_before_register():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=8, host_tokens=8)
    tree.event_done = False
    prefill.peer_fetch_begin(2, list(range(20)), 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    assert prefill.peer_fetch(2).stage == pp.STAGE_SENT  # waiting for the H2D
    assert tree.locks == [("dev", 8), ("restored", 8)]
    tree.event_done = True
    decode.peer_prefix_serve(ctx(tree))
    fetch = prefill.peer_fetch(2)
    assert fetch.hit_len == 16
    assert fetch.blocks == [250, 251, 1250, 1251]


def test_hybrid_serves_hbm_only_and_window_pages():
    prefill, decode = pair()
    swa = FakeEngine()
    decode._swa_engine = swa
    decode.swa_raiden_endpoint = "decode-swa"
    tree = FakeTree(device_tokens=16, host_tokens=8)
    mapping = {i: 9000 + i for i in range(1000, 1016)}
    allocator = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda idx: torch.tensor(
            [mapping[int(i)] for i in idx]
        )
    )
    prefill.peer_fetch_begin(3, list(range(24)), 0, None, None)
    decode.peer_prefix_serve(
        ctx(
            tree,
            hybrid=True,
            sliding_window_size=8,
            allocator=allocator,
            restore_allowed=False,
        )
    )
    fetch = prefill.peer_fetch(3)
    assert fetch.hit_len == 16  # host part not served for hybrid models
    assert fetch.swa_blocks == [2502, 2503] and fetch.swa_endpoint == "decode-swa"
    assert swa.registered[0][0] == "peerswa:3"


def test_lease_released_when_raiden_reports_the_send():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    prefill.peer_fetch_begin(6, list(range(20)), 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    decode._completed_send.add("peer:6")
    decode.peer_prefix_serve(ctx(tree))
    assert tree.unlocks and "peer:6" not in decode._completed_send


def test_start_read_failure_and_timeout_fail_closed():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    prefill.peer_fetch_begin(7, list(range(20)), 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    prefill.engine.fail_start = True
    assert not prefill.peer_fetch_start_read(7, [1, 2, 3, 4], [])
    assert prefill.peer_fetch(7).stage == pp.STAGE_FAILED
    prefill.peer_fetch_end(7)
    prefill.peer_fetch_begin(8, list(range(20)), 0, None, None)
    prefill.peer_fetch(8).sent_at -= 1000
    assert prefill.peer_fetch_poll(8, 1) == pp.STAGE_FAILED


# --------------------------------------------------------------------------- #
# Scheduler hook (prefill), world size 1.
# --------------------------------------------------------------------------- #
class FakeAllocator:
    def __init__(self):
        self.page_size = PAGE
        self.freed = []
        self.next = 3000

    def alloc_extend(self, pl, plc, sl, slc, last_loc, n):
        out = torch.arange(self.next, self.next + n, dtype=torch.int32)
        self.next += n
        return out

    def free(self, idx):
        self.freed.append(idx.tolist())


def make_queue(tree, manager):
    scheduler = SimpleNamespace(
        tree_cache=tree,
        token_to_kv_pool_allocator=FakeAllocator(),
        attn_tp_cpu_group=None,
        sliding_window_size=None,
    )
    return SimpleNamespace(scheduler=scheduler, kv_manager=manager, __dict__={})


def make_req(room, n=20):
    return SimpleNamespace(
        rid=f"r{room}",
        bootstrap_room=room,
        origin_input_ids=list(range(n)),
        extra_key=None,
        cache_salt=None,
        _peer_fetch=None,
    )


def test_hook_pulls_inserts_and_releases():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    queue = make_queue(tree, prefill)
    req = make_req(11)
    fetch = hook._Fetch(room=11, matched_len=4, want_len=16)
    fetch.prefix_indices = torch.arange(1000, 1004, dtype=torch.int64)
    fetch.lock_node = ("dev", 4)
    fetch.lock_params = ("dec", ("dev", 4))
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(11, req.origin_input_ids[:16], 4, None, None)
    held = [req]
    assert hook._advance_held(queue, held) == [] and held == [req]  # waiting
    decode.peer_prefix_serve(ctx(tree))
    assert hook._advance_held(queue, held) == []  # replied -> reading
    assert prefill.peer_fetch(11).stage == pp.STAGE_READING
    assert prefill.engine.reads[0][4] == [750, 751, 752]
    prefill._completed_recv.add("peer:11")
    out = hook._advance_held(queue, held)
    assert out == [req] and held == []
    params = tree.inserts[0]
    assert list(params.key.token_ids) == list(range(16))
    assert params.value.tolist() == [1000, 1001, 1002, 1003] + list(range(3000, 3012))
    assert params.prev_prefix_len == 4 and params.inserted_len == 4
    assert tree.unlocks == [(("dev", 4), ("dec", ("dev", 4)))]
    assert req._peer_fetch is None and prefill.peer_fetch(11) is None
    assert decode.sent[-1][1][0] == PrefixBlocks.TAG or any(
        p[1][0] == PrefixRelease.TAG for p in prefill.sent
    )


def test_hook_abandons_on_miss_and_frees_on_failure():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=4)
    queue = make_queue(tree, prefill)
    req = make_req(12)
    fetch = hook._Fetch(room=12, matched_len=4, want_len=16)
    fetch.prefix_indices = torch.arange(1000, 1004, dtype=torch.int64)
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(12, req.origin_input_ids[:16], 4, None, None)
    decode.peer_prefix_serve(ctx(tree))  # miss
    held = [req]
    assert hook._advance_held(queue, held) == [req]
    assert req._peer_fetch is None and tree.inserts == []
    # failure after the read started: pages drain until Raiden settles
    tree2 = FakeTree(device_tokens=16)
    queue2 = make_queue(tree2, prefill)
    req2 = make_req(13)
    f2 = hook._Fetch(room=13, matched_len=0, want_len=16)
    f2.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req2._peer_fetch = f2
    prefill.peer_fetch_begin(13, req2.origin_input_ids[:16], 0, None, None)
    decode.peer_prefix_serve(ctx(tree2))
    held = [req2]
    hook._advance_held(queue2, held)  # reading
    prefill._failed.add("peer:13")
    assert hook._advance_held(queue2, held) == [req2]
    assert queue2.scheduler.token_to_kv_pool_allocator.freed == [
        list(range(3000, 3016))
    ]


def test_decode_hook_serves_through_the_scheduler(monkeypatch):
    """The decode wrapper builds its context from the scheduler and calls serve."""
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    scheduler = SimpleNamespace(
        tree_cache=tree,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=PAGE),
        sliding_window_size=None,
        enable_decode_hicache=True,
        disagg_decode_prealloc_queue=SimpleNamespace(kv_manager=decode),
    )
    monkeypatch.setattr(
        hook,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
    )
    monkeypatch.setattr(hook, "_min_tokens", lambda page: PAGE)
    calls = []
    hook._process_decode_queue(lambda s: calls.append(s), scheduler)
    assert calls == [scheduler]
    prefill.peer_fetch_begin(21, list(range(20)), 0, None, None)
    hook._process_decode_queue(lambda s: None, scheduler)
    assert prefill.peer_fetch(21).hit_len == 16
    assert scheduler._peer_prefix_ctx.restore_allowed is True


def test_inflight_pages_count_as_uncached_until_committed():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    queue = make_queue(tree, prefill)
    req = make_req(31)
    fetch = hook._Fetch(room=31, matched_len=0, want_len=16)
    fetch.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(31, req.origin_input_ids[:16], 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    held = [req]
    hook._advance_held(queue, held)  # replied -> reading: pages allocated
    seen = {}

    def upstream(checker, ps, uncached=0):
        seen["uncached"] = uncached
        return False, ""

    hook._check_full_pool(upstream, object(), None, uncached=5)
    assert seen["uncached"] == 5 + 16
    prefill._completed_recv.add("peer:31")
    hook._advance_held(queue, held)  # committed: the tree owns them
    hook._check_full_pool(upstream, object(), None)
    assert seen["uncached"] == 0


def test_mismatched_reply_abandons_and_frees_before_any_read():
    """A decode reply whose page count differs from the agreed hit is refused
    on this rank; the reduced decision then abandons on every rank."""
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    queue = make_queue(tree, prefill)
    req = make_req(41)
    fetch = hook._Fetch(room=41, matched_len=0, want_len=16)
    fetch.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(41, req.origin_input_ids[:16], 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    peer = prefill.peer_fetch(41)
    peer.blocks = peer.blocks[:-1]  # the decode on this rank answered one page short
    held = [req]
    assert hook._advance_held(queue, held) == [req]
    assert prefill.engine.reads == [] and req._peer_fetch is None
    # The mismatch is seen before any page is allocated: nothing to free.
    assert queue.scheduler.token_to_kv_pool_allocator.freed == []
    assert hook._inflight_tokens() == 0


def test_commit_failure_releases_without_raising():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    tree.insert = lambda params: (_ for _ in ()).throw(RuntimeError("tree says no"))
    queue = make_queue(tree, prefill)
    req = make_req(42)
    fetch = hook._Fetch(room=42, matched_len=0, want_len=16)
    fetch.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(42, req.origin_input_ids[:16], 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    held = [req]
    hook._advance_held(queue, held)  # reading
    prefill._completed_recv.add("peer:42")
    assert hook._advance_held(queue, held) == [req]
    assert req._peer_fetch is None and hook._inflight_tokens() == 0


def test_read_budget_holds_a_second_pull_until_the_first_settles():
    prefill, decode = pair()
    prefill._full_engine_max_blocks = 8  # budget = 4 pages: one 16-token pull at a time
    tree = FakeTree(device_tokens=16)
    queue = make_queue(tree, prefill)
    reqs = []
    for room in (51, 52):
        req = make_req(room)
        fetch = hook._Fetch(room=room, matched_len=0, want_len=16)
        fetch.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req._peer_fetch = fetch
        prefill.peer_fetch_begin(room, req.origin_input_ids[:16], 0, None, None)
        reqs.append(req)
    decode.peer_prefix_serve(ctx(tree))
    held = list(reqs)
    assert hook._advance_held(queue, held) == []
    stages = [prefill.peer_fetch(r._peer_fetch.room).stage for r in reqs]
    assert stages == [pp.STAGE_READING, pp.STAGE_REPLIED]  # the second waits
    assert prefill.peer_inflight_pages() == 4
    prefill._completed_recv.add("peer:51")
    out = hook._advance_held(queue, held)
    assert out == [reqs[0]]
    assert (
        prefill.peer_fetch(52).stage == pp.STAGE_READING
    )  # admitted after the first settled
