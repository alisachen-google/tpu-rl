# SPDX-License-Identifier: Apache-2.0
"""Dynamic HiCache storage backend for Raiden stores on decode hosts.

The ordinary HiCache D2H fence and host references protect backups. Recall
lands in HiCache-owned host slots, then ordinary Raiden H2D and its forward
gate restore prefill HBM. Only the remote node runs a KVCacheStore allocator.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import uuid
from pathlib import Path

import grpc
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    PoolHitPolicy,
    PoolName,
    PoolTransferResult,
)

from . import wire_pb2 as w
from .client import IOResult, RaidenDramClient

logger = logging.getLogger(__name__)


class RaidenDramStorage(HiCacheStorage):
    """Load with --hicache-storage-backend dynamic; see README.md.

    Per-rank, per-pool stores keep independent allocation spaces. Their keys
    include physical layout and rank identity; cross-layout reuse fails closed.
    """

    def __init__(self, storage_config, unused_dynamic_kwargs=None):
        self.config = storage_config
        options = dict(storage_config.extra_config or {})
        for field in (
            "global_registry_address",
            "kv_pool_group",
            "cache_namespace",
            "decode_store_job",
        ):
            if not isinstance(options.get(field), str) or not options[field].strip():
                raise ValueError(f"Raiden DRAM storage requires a nonempty {field}")
        if not options.get("interface_v1"):
            raise ValueError("Raiden DRAM storage requires interface_v1=true")
        if not storage_config.model_name:
            raise ValueError("Raiden DRAM storage requires a model identity")
        if (
            storage_config.is_mla_model
            or storage_config.pp_size != 1
            or storage_config.attn_cp_size != 1
        ):
            raise NotImplementedError(
                "Raiden DRAM storage currently supports MHA/GQA, PP=1, attention CP=1"
            )
        timeout = float(options.get("rpc_timeout_s", 30))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("rpc_timeout_s must be positive and finite")
        replicas = options.get("decode_store_replicas", ["0"])
        if (
            not isinstance(replicas, list)
            or not replicas
            or not all(isinstance(x, str) and x for x in replicas)
        ):
            raise ValueError("decode_store_replicas must be a nonempty list of strings")
        self.options = options
        self.replicas = replicas
        self.client = RaidenDramClient(options["global_registry_address"], timeout)
        self._lock = threading.RLock()
        self._pools = {}
        self._closed = False
        self._instance = uuid.uuid4().hex
        self._stats = dict(
            spill_pages=0, recall_pages=0, failed_pages=0, quarantined_pages=0
        )

    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)
        # HybridCacheController supplies its anchor here and registers the
        # extra pools through register_mem_host_pool_v2 afterwards.
        self.register_mem_host_pool_v2(mem_pool_host, PoolName.KV)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        name = str(host_pool_name)
        if name not in ("kv", "swa"):
            raise NotImplementedError(
                f"Raiden remote DRAM does not support the {name} pool yet"
            )
        with self._lock:
            if name in self._pools:
                if self._pools[name]["pool"] is not host_pool:
                    raise ValueError(f"Cannot replace the live Raiden {name} pool")
                return
            if not callable(getattr(host_pool, "quarantine", None)):
                raise TypeError(
                    "Raiden DRAM storage requires TpuRaidenHostKVCache with quarantine support"
                )
            description = host_pool.transfer.host_store_descriptor()
            block_bytes = description["block_bytes"]
            if not block_bytes or any(n <= 0 for n in block_bytes):
                raise ValueError("Invalid Raiden block geometry")
            # Every SGLang TPU rank has one native manager, with one shard.
            if description["node_id"] != 0 or sorted(
                s for ep in description["endpoints"] for s in ep["shards"]
            ) != [0]:
                raise NotImplementedError(
                    "Raiden DRAM expects one node_id=0 worker with one shard per SGLang rank"
                )
            if not description["endpoints"] or any(
                not ep["endpoint"] for ep in description["endpoints"]
            ):
                raise ValueError("Raiden manager has no reachable data-plane endpoint")
            material = {
                "version": 1,
                "namespace": self.options["cache_namespace"],
                "model": self.config.model_name,
                "tp_rank": self.config.tp_rank,
                "tp_size": self.config.tp_size,
                "dp_rank": self.config.dp_rank,
                "pp_rank": self.config.pp_rank,
                "pool": name,
                "page_size": host_pool.page_size,
                "layout": description["layout"],
                "base_group": self.options["kv_pool_group"],
            }
            namespace = hashlib.sha256(
                json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
            ).digest()
            group = f'{self.options["kv_pool_group"]}.{name}.{namespace.hex()[:24]}'
            source = w.RaidenIdProto(
                job_name="sglang-prefill-hicache",
                job_replica_id=self._instance,
                data_name=group,
            )
            endpoints = [
                w.RaidenWorkerEndpointsProto(
                    node_id=0,
                    worker_id=f"sglang-{self.config.tp_rank}-{name}",
                    endpoints=[
                        w.RaidenTransferEndpointProto(**ep)
                        for ep in description["endpoints"]
                    ],
                )
            ]
            targets = [
                w.RaidenIdProto(
                    job_name=self.options["decode_store_job"],
                    job_replica_id=replica,
                    data_name=group,
                )
                for replica in self.replicas
            ]
            # A native host-store node bootstraps from this exact geometry.
            self.client.publish_spec(group, block_bytes)
            self._pools[name] = dict(
                pool=host_pool,
                namespace=namespace,
                group=group,
                source=source,
                endpoints=endpoints,
                targets=targets,
            )
            super().register_mem_host_pool_v2(host_pool, host_pool_name)
            manifest = {
                "version": 1,
                "group": group,
                "pool": name,
                "global_registry_address": self.options["global_registry_address"],
                "job_name": self.options["decode_store_job"],
                "data_name": group,
                "replicas": self.replicas,
                "block_bytes": block_bytes,
                "namespace": material,
            }
            if self.options.get("manifest_dir"):
                directory = Path(self.options["manifest_dir"])
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{namespace.hex()}.json"
                temporary = path.with_suffix(f".{self._instance}.tmp")
                temporary.write_text(json.dumps(manifest, indent=2) + "\n")
                os.replace(temporary, path)
            logger.info(
                "Raiden remote DRAM pool registered: %s",
                json.dumps(manifest, sort_keys=True),
            )

    def _state(self, name):
        if self._closed:
            raise RuntimeError("Raiden DRAM backend is closed")
        return self._pools[str(name)]

    @staticmethod
    def _hashes(state, keys):
        # get_hash_str produces strings; reject ambiguous/untyped identities.
        if not all(isinstance(key, str) and key for key in keys):
            raise ValueError("Storage keys must be nonempty hash strings")
        return [
            hashlib.sha256(state["namespace"] + key.encode()).digest() for key in keys
        ]

    @staticmethod
    def _pages(pool, keys, indices):
        if (
            not isinstance(indices, torch.Tensor)
            or indices.device.type != "cpu"
            or indices.ndim != 1
        ):
            raise ValueError(
                "Raiden storage indices must be a one-dimensional CPU tensor"
            )
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Raiden storage indices must be integers")
        values = indices.tolist()
        size = pool.page_size
        if len(values) != len(keys) * size:
            raise ValueError(
                "Raiden storage requires exactly one complete host page per key"
            )
        pages = []
        for offset in range(0, len(values), size):
            start = values[offset]
            if (
                start < 0
                or start % size
                or start // size >= pool.page_num
                or values[offset : offset + size] != list(range(start, start + size))
            ):
                raise ValueError(
                    "Raiden storage requires aligned contiguous host-page runs"
                )
            pages.append(start // size)
        if len(set(pages)) != len(pages) or len(set(keys)) != len(keys):
            raise ValueError("Duplicate host pages or keys in a Raiden storage batch")
        return pages

    def _io(self, name, keys, indices, *, write):
        keys = list(keys)
        if not keys:
            return []
        with self._lock:
            state = self._state(name)
            pool = state["pool"]
            pages = self._pages(pool, keys, indices)
            hashes = self._hashes(state, keys)
            try:
                if write:
                    result = self.client.spill(
                        state["group"],
                        state["targets"],
                        state["source"],
                        state["endpoints"],
                        hashes,
                        pages,
                    )
                else:
                    result = self.client.recall(
                        state["group"],
                        state["source"],
                        state["endpoints"],
                        hashes,
                        pages,
                    )
            except Exception:
                # Keep the controller's IO worker alive, and fail closed if an
                # unexpected error occurred after a peer borrowed these slots.
                logger.exception("Raiden storage IO failed; retiring its host slots")
                result = IOResult(quarantine=frozenset(range(len(keys))))
            if result.quarantine:
                selected = torch.cat(
                    [
                        indices[i * pool.page_size : (i + 1) * pool.page_size]
                        for i in sorted(result.quarantine)
                    ]
                )
                pool.quarantine(selected)
                self._stats["quarantined_pages"] += len(result.quarantine)
            self._stats["spill_pages" if write else "recall_pages"] += len(result.done)
            self._stats["failed_pages"] += len(keys) - len(result.done)
            return [i in result.done for i in range(len(keys))]

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        return self._io(PoolName.KV, keys, host_indices, write=False)

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        return self._io(PoolName.KV, keys, host_indices, write=True)

    def batch_get_v2(self, transfers, extra_info=None):
        return {
            transfer.name: self._io(
                transfer.name, transfer.keys or [], transfer.host_indices, write=False
            )
            for transfer in transfers
        }

    def batch_set_v2(self, transfers, extra_info=None):
        return {
            transfer.name: self._io(
                transfer.name, transfer.keys or [], transfer.host_indices, write=True
            )
            for transfer in transfers
        }

    def _hit_count(self, name, keys):
        state = self._state(name)
        try:
            return len(self.client.lookup(self._hashes(state, keys), state["source"]))
        except (grpc.RpcError, RuntimeError) as exc:
            logger.warning("Raiden storage lookup failed; treating as a miss: %s", exc)
            return 0

    def batch_exists(self, keys, extra_info=None):
        with self._lock:
            return self._hit_count(PoolName.KV, keys)

    def exists(self, key):
        return self.batch_exists([key]) == 1

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        with self._lock:
            kv_pages = self._hit_count(PoolName.KV, keys)
            candidates = set(range(1, kv_pages + 1))
            extra_hits = {}
            for transfer in pool_transfers or []:
                if not candidates:
                    break
                name = transfer.name
                if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                    count = self._hit_count(name, keys[:kv_pages])
                    allowed = set(range(1, count + 1))
                elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                    # Registry Lookup stops at the first gap. Probe individual
                    # sidecar pages so a missing early SWA page cannot hide a
                    # complete later window. Intersect valid boundaries across
                    # pools (and let HiCache intersect them across ranks).
                    present = [
                        bool(self._hit_count(name, [key])) for key in keys[:kv_pages]
                    ]
                    window = max(1, len(transfer.keys or []))
                    allowed = {
                        end
                        for end in candidates
                        if all(present[max(0, end - window) : end])
                    }
                else:
                    raise ValueError(f"Unknown pool hit policy {transfer.hit_policy}")
                candidates &= allowed
                extra_hits[str(name)] = max(allowed, default=0)
            final = max(candidates, default=0)
            return PoolTransferResult(
                final,
                {k: min(v, final) for k, v in extra_hits.items()},
                restorable_prefix_pages=sorted(candidates),
            )

    # Fail loudly if the generic tensor-serialization path is selected.
    def get(self, *args, **kwargs):
        raise NotImplementedError("Use the Raiden HiCache v1/v2 host-index interface")

    def batch_get(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def set(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def batch_set(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def clear(self):
        raise NotImplementedError(
            "A serving process must not clear a shared remote DRAM pool"
        )

    def get_stats(self):
        with self._lock:
            return dict(self._stats)

    def close(self):
        with self._lock:
            self.client.close()
            self._closed = True
