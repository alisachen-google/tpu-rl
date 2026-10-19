"""CPU integration/unit coverage. RPCs are real gRPC; DMA is a test double.

The separate native end-to-end test is required for actual TPU/DMA validation.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import grpc
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang_tpu.kv_pool_host import TpuRaidenHostKVCache
from sglang_tpu.raiden_dram import wire_pb2 as w
from sglang_tpu.raiden_dram.client import REGISTRY_SERVICE, STORE_SERVICE
from sglang_tpu.raiden_dram.launch_stores import build_commands
from sglang_tpu.raiden_dram.storage import RaidenDramStorage
from sglang_tpu.raiden_transfer import RaidenKVTransfer

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def make_pool(
    endpoint="cpu-fixture:1", *, page_num=16, page_size=2, dtype="torch.bfloat16"
):
    pool = object.__new__(TpuRaidenHostKVCache)
    pool.page_size, pool.page_num = page_size, page_num
    pool.size = page_size * page_num
    pool.layout = "fused_raiden"
    pool.lock = threading.RLock()
    pool.clear()
    pool.payload = {}
    pool.transfer = SimpleNamespace(
        host_store_descriptor=lambda: dict(
            block_bytes=[8],
            node_id=0,
            layout=[dict(shape=[2, 2], dtype=dtype)],
            endpoints=[dict(endpoint=endpoint, shards=[0])],
        )
    )
    return pool


class StoreFixture:
    """Protocol peer only: never presented as a native Raiden server."""

    def __init__(self):
        self.pools = {}
        self.blocks = {}
        self.metadata = {}
        self.specs = {}
        self.group = None
        self.state = w.PollWriteRemoteResponse.COMMITTED
        self.stream_break = False
        self.ack_only = False
        self.fetch_error = None
        self.write_error = None
        self.incomplete_fetch = False
        self.reject_spec = False
        self.oversized_lookup = False
        self.writes = []
        self.fetches = []
        self.commit_gate = None
        self.accepted = threading.Event()
        self.server = grpc.server(ThreadPoolExecutor(max_workers=4))
        self.address = f"127.0.0.1:{self.server.add_insecure_port('127.0.0.1:0')}"

        def unary(fn, request):
            return grpc.unary_unary_rpc_method_handler(
                fn,
                request_deserializer=request.FromString,
                response_serializer=lambda x: x.SerializeToString(),
            )

        self.server.add_generic_rpc_handlers(
            [
                grpc.method_handlers_generic_handler(
                    REGISTRY_SERVICE,
                    {
                        "RegisterKVTransferSpec": unary(
                            self.spec, w.RegisterKVTransferSpecRequest
                        ),
                        "Lookup": unary(self.lookup, w.LookupRequest),
                        "ResolveStore": unary(self.resolve, w.ResolveStoreRequest),
                    },
                ),
                grpc.method_handlers_generic_handler(
                    STORE_SERVICE,
                    {
                        "WriteRemote": grpc.unary_stream_rpc_method_handler(
                            self.write,
                            request_deserializer=w.WriteRemoteRequest.FromString,
                            response_serializer=lambda x: x.SerializeToString(),
                        ),
                        "PollWriteRemote": unary(self.poll, w.PollWriteRemoteRequest),
                        "Fetch": unary(self.fetch, w.FetchRequest),
                    },
                ),
            ]
        )
        self.server.start()

    def spec(self, request, context):
        self.specs[request.kv_pool_group] = request.spec
        return w.RegisterKVTransferSpecResponse(
            success=not self.reject_spec,
            error_message="test mismatch" if self.reject_spec else "",
        )

    def lookup(self, request, context):
        response = w.LookupResponse()
        if self.oversized_lookup:
            for _ in range(len(request.prefix_hashes) + 1):
                response.results.add()
            return response
        for key in request.prefix_hashes:
            if key not in self.metadata:
                break
            response.results.add().CopyFrom(self.metadata[key])
        return response

    def resolve(self, request, context):
        return w.ResolveStoreResponse(
            found=True,
            store=w.StoreInfo(
                raiden_id=request.raiden_id,
                store_server_address=self.address,
                controller_address=self.address,
                kv_pool_group=self.group or request.raiden_id.data_name,
                evict_tier=1,
            ),
        )

    def write(self, request, context):
        self.writes.append(request)
        if self.write_error:
            context.abort(self.write_error, "injected write refusal")
        existing = [key for key in request.block_hashes if key in self.blocks]
        if existing:
            yield w.WriteRemoteEvent(
                ack=w.WriteRemoteAck(
                    exist_state=(
                        w.WRITE_ALL_EXIST
                        if len(existing) == len(request.block_hashes)
                        else w.WRITE_PARTIAL_EXIST
                    ),
                    existing_hashes=existing,
                )
            )
            return
        yield w.WriteRemoteEvent(
            ack=w.WriteRemoteAck(
                operation_id=9, granted_deadline_ms=request.deadline_ms
            )
        )
        self.accepted.set()
        if self.commit_gate:
            assert self.commit_gate.wait(5)
        if self.state == w.PollWriteRemoteResponse.COMMITTED:
            pool = self.pools[request.src_worker_endpoints[0].endpoints[0].endpoint]
            for key, page in zip(request.block_hashes, request.src_host_block_ids):
                self.blocks[key] = pool.payload[page]
                self.metadata[key] = w.KVBlockMetadata(
                    raiden_id=w.RaidenIdProto(
                        job_name="decode-stores",
                        job_replica_id="0",
                        data_name=request.src_raiden_id.data_name,
                    ),
                    block_id=len(self.blocks) - 1,
                )
        if self.stream_break:
            context.abort(grpc.StatusCode.UNAVAILABLE, "lost result stream")
        if not self.ack_only:
            yield w.WriteRemoteEvent(result=w.WriteRemoteResult(state=self.state))

    def poll(self, request, context):
        return w.PollWriteRemoteResponse(state=self.state)

    def fetch(self, request, context):
        self.fetches.append(request)
        if self.fetch_error:
            context.abort(self.fetch_error, "injected fetch failure")
        if not all(key in self.blocks for key in request.block_hashes):
            context.abort(grpc.StatusCode.NOT_FOUND, "evicted between lookup and fetch")
        pool = self.pools[request.client_worker_endpoints[0].endpoints[0].endpoint]
        for key, page in zip(request.block_hashes, request.host_block_ids):
            pool.payload[page] = self.blocks[key]
        return w.FetchResponse(
            done_block_hashes=[] if self.incomplete_fetch else request.block_hashes
        )


def config(address, **overrides):
    result = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=True,
        is_page_first_layout=True,
        model_name="test-model@revision-1",
        extra_config=dict(
            backend_name="raiden_dram",
            module_path="sglang_tpu.raiden_dram.storage",
            class_name="RaidenDramStorage",
            interface_v1=True,
            global_registry_address=address,
            kv_pool_group="test-pool",
            cache_namespace="immutable-model-and-kv-config-1",
            decode_store_job="decode-stores",
            rpc_timeout_s=2,
        ),
    )
    return replace(result, **overrides)


@pytest.fixture
def setup_storage():
    peer = StoreFixture()
    pool = make_pool()
    peer.pools["cpu-fixture:1"] = pool
    storage = StorageBackendFactory.create_backend(
        "dynamic", config(peer.address), pool
    )
    storage.register_mem_pool_host(pool)
    try:
        yield storage, pool, peer
    finally:
        if peer.commit_gate:
            peer.commit_gate.set()
        storage.close()
        peer.server.stop(0).wait()


def save_pages(storage, pool, keys):
    indices = pool.alloc(len(keys) * pool.page_size)
    assert indices is not None
    for key, page in zip(keys, pool._to_page_ids(indices)):
        pool.payload[page] = f"payload-for-{key}".encode()
    return indices, storage.batch_set_v1(keys, indices)


def test_spill_forget_local_recall_into_different_slots(setup_storage):
    storage, pool, peer = setup_storage
    source, result = save_pages(storage, pool, ["a", "b"])
    assert result == [True, True]
    pool.payload.clear()  # No local byte copy can satisfy recall.
    destination = pool.alloc(4)
    assert pool._to_page_ids(source) != pool._to_page_ids(destination)
    assert storage.batch_exists(["a", "b", "missing"]) == 2
    assert storage.batch_get_v1(["a", "b"], destination) == [True, True]
    assert [pool.payload[p] for p in pool._to_page_ids(destination)] == [
        b"payload-for-a",
        b"payload-for-b",
    ]
    assert len(peer.writes) == len(peer.fetches) == 1
    assert storage.get_stats()["recall_pages"] == 2


def test_admission_ack_does_not_finish_backup(setup_storage):
    storage, pool, peer = setup_storage
    peer.commit_gate = threading.Event()
    with ThreadPoolExecutor(1) as executor:
        task = executor.submit(save_pages, storage, pool, ["a"])
        assert peer.accepted.wait(2)
        assert not task.done()
        peer.commit_gate.set()
        assert task.result(timeout=3)[1] == [True]


@pytest.mark.parametrize(
    "state",
    [
        w.PollWriteRemoteResponse.FAILED,
        w.PollWriteRemoteResponse.PENDING,
        w.PollWriteRemoteResponse.UNKNOWN,
    ],
)
def test_uncertain_write_retires_slots_across_reset(setup_storage, state):
    storage, pool, peer = setup_storage
    peer.state = state
    source, result = save_pages(storage, pool, ["a"])
    assert result == [False]
    assert pool.free(source) == 0
    pool.clear()
    assert 0 not in pool.free_pages
    assert pool.available_size() == (pool.page_num - 1) * pool.page_size
    assert storage.get_stats()["quarantined_pages"] == 1


def test_publication_failure_is_not_success(setup_storage):
    storage, pool, peer = setup_storage
    peer.state = w.PollWriteRemoteResponse.STORED_UNREGISTERED
    source, result = save_pages(storage, pool, ["a"])
    assert result == [False]
    assert pool.free(source) == pool.page_size  # Transfer has settled.


def test_lost_stream_recovers_committed_write(setup_storage):
    storage, pool, peer = setup_storage
    peer.stream_break = True
    assert save_pages(storage, pool, ["a"])[1] == [True]
    assert not pool._quarantined_pages


def test_partial_existing_offer_retries_only_missing_keys(setup_storage):
    storage, pool, peer = setup_storage
    assert save_pages(storage, pool, ["a"])[1] == [True]
    assert save_pages(storage, pool, ["a", "b"])[1] == [True, True]
    assert [len(req.block_hashes) for req in peer.writes] == [1, 2, 1]


def test_all_existing_offer_is_success_without_operation(setup_storage):
    storage, pool, peer = setup_storage
    assert save_pages(storage, pool, ["a"])[1] == [True]
    assert save_pages(storage, pool, ["a"])[1] == [True]


@pytest.mark.parametrize(
    "error,quarantined",
    [
        (grpc.StatusCode.NOT_FOUND, False),
        (grpc.StatusCode.DEADLINE_EXCEEDED, True),
        (grpc.StatusCode.UNAVAILABLE, True),
    ],
)
def test_failed_fetch_never_admits_data_and_protects_late_writes(
    setup_storage, error, quarantined
):
    storage, pool, peer = setup_storage
    save_pages(storage, pool, ["a"])
    target = pool.alloc(2)
    peer.fetch_error = error
    assert storage.batch_get_v1(["a"], target) == [False]
    assert pool.free(target) == (0 if quarantined else 2)


def test_eviction_race_recomputes_instead_of_using_stale_data(setup_storage):
    storage, pool, peer = setup_storage
    save_pages(storage, pool, ["a"])
    peer.blocks.clear()  # Registry still has stale metadata.
    assert storage.batch_exists(["a"]) == 1
    target = pool.alloc(2)
    assert storage.batch_get_v1(["a"], target) == [False]
    assert pool.free(target) == 2


def test_wrong_group_never_transfers(setup_storage):
    storage, pool, peer = setup_storage
    peer.group = "incompatible-layout"
    assert save_pages(storage, pool, ["a"])[1] == [False]
    assert not peer.writes


@pytest.mark.parametrize(
    "field,value", [("tp_rank", 1), ("model_name", "another-model")]
)
def test_namespace_prevents_incompatible_hits(setup_storage, field, value):
    storage, pool, peer = setup_storage
    save_pages(storage, pool, ["a"])
    other = RaidenDramStorage(config(peer.address, **{field: value}))
    try:
        other.register_mem_pool_host(pool)
        assert not other.exists("a")
    finally:
        other.close()


def test_layout_dtype_is_part_of_namespace(setup_storage):
    storage, pool, peer = setup_storage
    save_pages(storage, pool, ["a"])
    other = RaidenDramStorage(config(peer.address))
    try:
        other.register_mem_pool_host(make_pool(dtype="torch.float8_e4m3fn"))
        assert not other.exists("a")
    finally:
        other.close()


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([1, 2]),
        torch.tensor([0, 3]),
        torch.tensor([-2, -1]),
        torch.tensor([32, 33]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([[0, 1]]),
    ],
)
def test_invalid_host_indices_are_rejected_before_rpc(setup_storage, indices):
    storage, _, peer = setup_storage
    with pytest.raises(ValueError):
        storage.batch_set_v1(["a"], indices)
    assert not peer.writes


def test_duplicate_slots_are_rejected(setup_storage):
    storage, _, peer = setup_storage
    with pytest.raises(ValueError):
        storage.batch_set_v1(["a", "b"], torch.tensor([0, 1, 0, 1]))
    assert not peer.writes


def test_empty_batches_do_not_allocate_or_issue_rpc(setup_storage):
    storage, _, peer = setup_storage
    assert storage.batch_set_v1([], torch.tensor([], dtype=torch.int64)) == []
    assert storage.batch_get_v1([], torch.tensor([], dtype=torch.int64)) == []
    assert storage.batch_exists([]) == 0
    assert not peer.writes and not peer.fetches


def test_hybrid_swa_recall_and_noncontiguous_restorable_boundaries(setup_storage):
    storage, pool, peer = setup_storage
    swa = make_pool("cpu-fixture:2")
    peer.pools["cpu-fixture:2"] = swa
    storage.register_mem_host_pool_v2(swa, PoolName.SWA)
    save_pages(storage, pool, ["a", "b", "c", "d"])
    source = swa.alloc(4)
    swa.payload.update({0: b"swa-c", 1: b"swa-d"})
    assert storage.batch_set_v2(
        [PoolTransfer(PoolName.SWA, source, keys=["c", "d"])]
    ) == {PoolName.SWA: [True, True]}
    requirement = PoolTransfer(
        PoolName.SWA, keys=["c", "d"], hit_policy=PoolHitPolicy.TRAILING_PAGES
    )
    hits = storage.batch_exists_v2(["a", "b", "c", "d"], [requirement])
    assert hits.kv_hit_pages == 4
    assert hits.restorable_prefix_pages == [4]
    target = swa.alloc(4)
    swa.payload.clear()
    assert storage.batch_get_v2([PoolTransfer(PoolName.SWA, target, keys=["c", "d"])])[
        PoolName.SWA
    ] == [True, True]
    assert [swa.payload[p] for p in swa._to_page_ids(target)] == [b"swa-c", b"swa-d"]


def test_unexpected_io_error_fails_closed(setup_storage, monkeypatch):
    storage, pool, _ = setup_storage

    def broken(*args):
        raise RuntimeError("native client failed unexpectedly")

    monkeypatch.setattr(storage.client, "spill", broken)
    source, result = save_pages(storage, pool, ["a"])
    assert result == [False]
    assert pool.free(source) == 0


def test_descriptor_uses_metadata_without_tensor_indexing():
    transfer = object.__new__(RaidenKVTransfer)

    class Buffer:
        shape = (100, 4, 8)
        dtype = torch.bfloat16

        def element_size(self):
            return 2

        def __getitem__(self, index):
            raise AssertionError("TPU operation during metadata inspection")

    transfer._device_tensors = [[Buffer()]]
    transfer._mgr = SimpleNamespace(
        node_id=0, get_local_endpoints=lambda: [dict(endpoint="host:1", shards=[0])]
    )
    assert transfer.host_store_descriptor()["block_bytes"] == [64]


@pytest.mark.parametrize(
    "overrides", [{"pp_size": 2}, {"attn_cp_size": 2}, {"is_mla_model": True}]
)
def test_unimplemented_topologies_rejected(overrides):
    with pytest.raises(NotImplementedError):
        RaidenDramStorage(config("localhost:1", **overrides))


def test_missing_namespace_rejected():
    cfg = config("localhost:1")
    del cfg.extra_config["cache_namespace"]
    with pytest.raises(ValueError, match="cache_namespace"):
        RaidenDramStorage(cfg)


def test_spec_mismatch_rejects_attach(setup_storage):
    storage, pool, peer = setup_storage
    peer.reject_spec = True
    other = RaidenDramStorage(config(peer.address))
    try:
        with pytest.raises(ValueError, match="KVTransferSpec"):
            other.register_mem_pool_host(pool)
    finally:
        other.close()


def test_new_prefill_instance_recalls_through_registry(setup_storage):
    storage, pool, peer = setup_storage
    save_pages(storage, pool, ["a", "b"])
    storage.close()
    pool.payload.clear()
    other_pool = make_pool("other-prefill:1")
    peer.pools["other-prefill:1"] = other_pool
    other = RaidenDramStorage(config(peer.address))
    try:
        other.register_mem_pool_host(other_pool)
        assert other.batch_exists(["a", "b"]) == 2
        target = other_pool.alloc(4)
        assert other.batch_get_v1(["a", "b"], target) == [True, True]
        assert other_pool.payload == {0: b"payload-for-a", 1: b"payload-for-b"}
    finally:
        other.close()


def test_recall_groups_owners_and_preserves_destination_order(setup_storage):
    storage, pool, peer = setup_storage
    save_pages(storage, pool, ["a", "b", "c"])
    hashes = storage._hashes(storage._pools["kv"], ["a", "b", "c"])
    peer.metadata[hashes[1]].raiden_id.job_replica_id = "another-owner"
    target = pool.alloc(6)
    assert storage.batch_get_v1(["a", "b", "c"], target) == [True, True, True]
    assert [list(req.host_block_ids) for req in peer.fetches] == [[3, 5], [4]]
    assert [pool.payload[p] for p in [3, 4, 5]] == [
        b"payload-for-a",
        b"payload-for-b",
        b"payload-for-c",
    ]


@pytest.mark.parametrize(
    "error,quarantined",
    [
        (grpc.StatusCode.RESOURCE_EXHAUSTED, False),
        (grpc.StatusCode.INVALID_ARGUMENT, False),
        (grpc.StatusCode.UNAVAILABLE, True),
    ],
)
def test_write_refusal_before_ack_has_safe_slot_lifetime(
    setup_storage, error, quarantined
):
    storage, pool, peer = setup_storage
    peer.write_error = error
    source, result = save_pages(storage, pool, ["a"])
    assert result == [False]
    assert pool.free(source) == (0 if quarantined else 2)


def test_incomplete_fetch_response_cannot_admit_cache_hit(setup_storage):
    storage, pool, peer = setup_storage
    save_pages(storage, pool, ["a"])
    peer.incomplete_fetch = True
    target = pool.alloc(2)
    assert storage.batch_get_v1(["a"], target) == [False]
    assert pool.free(target) == 0


def test_invalid_lookup_verdict_is_a_miss_before_any_transfer(setup_storage):
    storage, pool, peer = setup_storage
    peer.oversized_lookup = True
    assert storage.batch_exists(["a"]) == 0
    target = pool.alloc(2)
    assert storage.batch_get_v1(["a"], target) == [False]
    assert not peer.fetches
    assert pool.free(target) == 2


def test_manifest_bootstraps_rank_specific_native_store(setup_storage, tmp_path):
    _, pool, peer = setup_storage
    cfg = config(peer.address)
    cfg.extra_config["manifest_dir"] = str(tmp_path)
    other = RaidenDramStorage(cfg)
    try:
        other.register_mem_pool_host(pool)
        paths = list(tmp_path.glob("*.json"))
        assert len(paths) == 1
        manifest = json.loads(paths[0].read_text())
        assert manifest["group"] == other._pools["kv"]["group"]
        assert manifest["block_bytes"] == [8]
        commands = build_commands(
            paths,
            replica="0",
            server_ip="decode-host",
            budget=1024,
            binary="native-store",
        )
        assert len(commands) == 1
        assert f'--data_name={manifest["group"]}' in commands[0]
        assert "--dram_budget_bytes=1024" in commands[0]
        assert "--store_server_ip=decode-host" in commands[0]
        with pytest.raises(ValueError, match="duplicate"):
            build_commands(
                paths * 2,
                replica="0",
                server_ip="decode-host",
                budget=1024,
                binary="native-store",
            )
        with pytest.raises(ValueError, match="configured target"):
            build_commands(
                paths,
                replica="unknown",
                server_ip="decode-host",
                budget=1024,
                binary="native-store",
            )
    finally:
        other.close()
