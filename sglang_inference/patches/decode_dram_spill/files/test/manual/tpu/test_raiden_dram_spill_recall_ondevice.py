"""Native TPU -> prefill DRAM -> remote store -> prefill DRAM -> TPU test.

Uses real Raiden registry/store processes and real DMA. The store is a separate
process on the same machine; multi-host serving is tested by verify_engine.py.
Requires torch_tpu, tpu_sync and its registry/host-store binaries on PATH.
"""

import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

import pytest


def test_native_spill_forget_and_recall(tmp_path):
    pytest.importorskip(
        "torch_tpu", reason="Native spill/recall needs torch_tpu and a TPU"
    )
    pytest.importorskip("tpu_sync")
    import grpc
    import torch

    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
    from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
    from sglang_tpu.kv_pool_host import TpuRaidenHostKVCache
    from sglang_tpu.raiden_dram.launch_stores import build_commands

    registry_binary = shutil.which("global_registry_server")
    store_binary = shutil.which("kv_cache_host_store_node_main")
    if not registry_binary or not store_binary:
        pytest.skip("tpu_sync native registry and host-store binaries must be on PATH")
    if torch.accelerator.device_count() == 0:
        pytest.skip("No TPU exposed to this process")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"127.0.0.1:{port}"
    processes, logs = [], []
    storage = None

    def start(command, name):
        log = (tmp_path / f"{name}.log").open("w")
        logs.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def poll(event):
        deadline = time.monotonic() + 60
        while not event.query():
            if time.monotonic() > deadline:
                raise TimeoutError("Native DMA did not settle")
            time.sleep(0.001)

    try:
        start([registry_binary, f"--port={port}"], "registry")
        with grpc.insecure_channel(address) as channel:
            grpc.channel_ready_future(channel).result(timeout=20)
        page_size = 64
        pattern = torch.zeros(8, page_size, 2, 2, 128, dtype=torch.bfloat16)
        pattern[1] = 101
        pattern[2] = 102
        # All writes happen BEFORE registration. Eager writes after registration
        # can replace the PJRT buffer and invalidate Raiden's raw pointer.
        data = pattern.to("tpu:0")
        device_pool = SimpleNamespace(
            layer_num=1,
            start_layer=0,
            end_layer=1,
            page_size=page_size,
            size=8 * page_size,
            store_dtype=data.dtype,
            get_fused_kv_buffer=lambda layer: data,
        )
        pool = TpuRaidenHostKVCache(
            device_pool,
            host_to_device_ratio=2,
            host_size=0,
            page_size=page_size,
            layout="fused_raiden",
        )
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=True,
            is_page_first_layout=True,
            model_name="native-roundtrip-fixture",
            extra_config=dict(
                backend_name="raiden_dram",
                module_path="sglang_tpu.raiden_dram.storage",
                class_name="RaidenDramStorage",
                interface_v1=True,
                global_registry_address=address,
                kv_pool_group="native-roundtrip",
                cache_namespace="stable-bf16-pattern-v1",
                decode_store_job="decode-stores",
                manifest_dir=str(tmp_path / "manifests"),
                rpc_timeout_s=30,
            ),
        )
        storage = StorageBackendFactory.create_backend("dynamic", config, pool)
        storage.register_mem_pool_host(pool)
        commands = build_commands(
            sorted((tmp_path / "manifests").glob("*.json")),
            replica="0",
            server_ip="127.0.0.1",
            budget=32 * 1024 * 1024,
            binary=store_binary,
        )
        assert len(commands) == 1
        remote = start(commands[0], "host-store")
        state = storage._pools["kv"]
        deadline = time.monotonic() + 30
        while storage.client.resolve(state["targets"][0], state["group"]) is None:
            assert remote.poll() is None, (tmp_path / "host-store.log").read_text()
            if time.monotonic() > deadline:
                raise TimeoutError("Host store failed to register")
            time.sleep(0.1)

        source = pool.alloc(2 * page_size)
        poll(pool.transfer.backup([1, 2], pool._to_page_ids(source)))
        assert storage.batch_set_v1(["prefix-a", "prefix-b"], source) == [True, True]
        assert storage.batch_exists(["prefix-a", "prefix-b"]) == 2
        # Erase EVERY prefill host slot via DMA from a known-zero device page.
        poll(pool.transfer.backup([0] * pool.page_num, list(range(pool.page_num))))
        destination = pool.alloc(2 * page_size)
        assert pool._to_page_ids(source) != pool._to_page_ids(destination)
        assert storage.batch_get_v1(["prefix-a", "prefix-b"], destination) == [
            True,
            True,
        ]
        poll(pool.transfer.load(pool._to_page_ids(destination), [4, 5]))
        torch.accelerator.synchronize()
        actual = data.cpu()
        torch.testing.assert_close(actual[4], pattern[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[5], pattern[2], rtol=0, atol=0)
        assert storage.get_stats()["recall_pages"] == 2
    finally:
        if storage is not None:
            storage.close()
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
