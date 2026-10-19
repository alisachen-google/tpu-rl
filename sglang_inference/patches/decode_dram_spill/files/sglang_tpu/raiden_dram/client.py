# SPDX-License-Identifier: Apache-2.0
"""Raiden's store RPCs; payload stays on the native block transport.

SGLang owns and pins the caller's host slots. The destination owns its slots.
Never treat WriteRemote admission as completion. An uncertain DMA result must
quarantine the caller's slots before HiCache is allowed to release them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import grpc

from . import wire_pb2 as w

logger = logging.getLogger(__name__)
REGISTRY_SERVICE = "tpu_raiden.kv_cache.global_registry.GlobalRegistryService"
STORE_SERVICE = "tpu_raiden.kv_cache.proto.KVCacheStoreService"


@dataclass(frozen=True)
class IOResult:
    # Indices into the submitted batch, not physical host-page IDs.
    done: frozenset[int] = frozenset()
    quarantine: frozenset[int] = frozenset()


class RaidenDramClient:
    def __init__(self, registry_address: str, timeout_s: float = 30):
        self.timeout_s = timeout_s
        self._channels = {}
        self._registry_address = registry_address

    def _rpc(self, address, service, method, response_cls, *, stream=False):
        channel = self._channels.setdefault(address, None)
        if channel is None:
            channel = self._channels[address] = grpc.insecure_channel(address)
        make = channel.unary_stream if stream else channel.unary_unary
        return make(
            f"/{service}/{method}",
            request_serializer=lambda req: req.SerializeToString(),
            response_deserializer=response_cls.FromString,
        )

    def publish_spec(self, group: str, block_bytes: list[int]):
        response = self._rpc(
            self._registry_address,
            REGISTRY_SERVICE,
            "RegisterKVTransferSpec",
            w.RegisterKVTransferSpecResponse,
        )(
            w.RegisterKVTransferSpecRequest(
                kv_pool_group=group,
                spec=w.KVTransferSpec(
                    block_arrays=[
                        w.KVTransferSpec.BlockArray(block_bytes=n) for n in block_bytes
                    ],
                    num_kv_shards=1,
                    num_workers=1,
                ),
            ),
            timeout=self.timeout_s,
        )
        if not response.success:
            raise ValueError(
                f"Raiden rejected KVTransferSpec for {group}: {response.error_message}"
            )

    def lookup(self, hashes: list[bytes], source_id):
        if not hashes:
            return []
        response = self._rpc(
            self._registry_address,
            REGISTRY_SERVICE,
            "Lookup",
            w.LookupResponse,
        )(
            w.LookupRequest(prefix_hashes=hashes, client_raiden_id=source_id),
            timeout=self.timeout_s,
        )
        if len(response.results) > len(hashes):
            raise RuntimeError("Raiden registry returned more hits than requested")
        return list(response.results)

    def resolve(self, identity, group):
        # Resolve afresh: a restarted host-store may listen on another port.
        response = self._rpc(
            self._registry_address,
            REGISTRY_SERVICE,
            "ResolveStore",
            w.ResolveStoreResponse,
        )(w.ResolveStoreRequest(raiden_id=identity), timeout=self.timeout_s)
        if not response.found:
            return None
        store = response.store
        if store.kv_pool_group != group or store.evict_tier < 1:
            raise ValueError(
                "Raiden destination is not a higher-tier store in this KV pool group"
            )
        if not store.store_server_address or store.raiden_id != identity:
            raise ValueError("Invalid Raiden store registration")
        return store.store_server_address

    @staticmethod
    def _subset(hashes, reported):
        reported = set(reported)
        if not reported.issubset(hashes):
            raise ValueError("Raiden reported hashes outside the submitted batch")
        return frozenset(i for i, key in enumerate(hashes) if key in reported)

    def _write(self, address, source_id, endpoints, hashes, pages):
        """One offer. The caller retains every source slot through this call."""
        all_indices = frozenset(range(len(hashes)))
        operation_id = 0
        acknowledged = False
        call = None
        try:
            call = self._rpc(
                address,
                STORE_SERVICE,
                "WriteRemote",
                w.WriteRemoteEvent,
                stream=True,
            )(
                w.WriteRemoteRequest(
                    src_raiden_id=source_id,
                    block_hashes=hashes,
                    src_host_block_ids=pages,
                    src_worker_endpoints=endpoints,
                    # Leave time for the terminal result before the RPC deadline.
                    deadline_ms=max(1, int(self.timeout_s * 800)),
                ),
                timeout=self.timeout_s,
            )
            for event in call:
                kind = event.WhichOneof("event")
                if kind == "ack" and not acknowledged:
                    acknowledged = True
                    ack = event.ack
                    if ack.exist_state == w.WRITE_ALL_EXIST:
                        return IOResult(all_indices)
                    if ack.exist_state == w.WRITE_PARTIAL_EXIST:
                        return IOResult(self._subset(hashes, ack.existing_hashes))
                    if (
                        ack.exist_state != w.WRITE_EXIST_STATE_UNSPECIFIED
                        or not ack.operation_id
                    ):
                        raise ValueError("Invalid WriteRemote admission")
                    operation_id = ack.operation_id
                elif kind == "result" and operation_id:
                    return self._write_verdict(hashes, event.result)
                else:
                    raise ValueError("WriteRemote events arrived out of order")
        except (grpc.RpcError, ValueError) as exc:
            logger.warning("Raiden WriteRemote did not settle on its stream: %s", exc)
            # These statuses are rejections before any transfer in the pinned
            # native service. No page was borrowed by the receiver.
            if (
                not acknowledged
                and isinstance(exc, grpc.RpcError)
                and exc.code()
                in (
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    grpc.StatusCode.UNIMPLEMENTED,
                )
            ):
                return IOResult()
        finally:
            if call is not None:
                call.cancel()

        # An accepted operation survives stream cancellation. Ask once, as the
        # native client does; never turn an unknown outcome into a success.
        if operation_id:
            try:
                status = self._rpc(
                    address,
                    STORE_SERVICE,
                    "PollWriteRemote",
                    w.PollWriteRemoteResponse,
                )(
                    w.PollWriteRemoteRequest(operation_id=operation_id, wait_ms=0),
                    timeout=self.timeout_s,
                )
                return self._write_verdict(hashes, status)
            except (grpc.RpcError, ValueError) as exc:
                logger.warning("Raiden write recovery failed: %s", exc)
        # The peer might still be pulling. Preserve memory until process exit;
        # a gRPC cancellation is not a native DMA cancellation.
        return IOResult(quarantine=all_indices)

    def _write_verdict(self, hashes, result):
        all_indices = frozenset(range(len(hashes)))
        state = result.state
        if state in (
            w.PollWriteRemoteResponse.COMMITTED,
            w.PollWriteRemoteResponse.ALL_EXIST,
        ):
            return IOResult(all_indices)
        if state == w.PollWriteRemoteResponse.PARTIAL_EXIST:
            return IOResult(self._subset(hashes, result.existing_hashes))
        if state == w.PollWriteRemoteResponse.STORED_UNREGISTERED:
            # The copy finished, but future readers cannot discover it.
            return IOResult()
        # FAILED can be an expired operation whose native pull is still running.
        return IOResult(quarantine=all_indices)

    def spill(self, group, targets, source_id, endpoints, hashes, pages):
        remaining = list(range(len(hashes)))
        done, quarantine = set(), set()
        for target in targets:
            try:
                address = self.resolve(target, group)
            except (grpc.RpcError, ValueError) as exc:
                logger.warning("Raiden spill target resolution failed: %s", exc)
                continue
            if address is None:
                continue
            # PARTIAL_EXIST can make progress without copying. Re-offer only
            # the missing suffix/subset, bounded by progress on every iteration.
            while remaining:
                result = self._write(
                    address,
                    source_id,
                    endpoints,
                    [hashes[i] for i in remaining],
                    [pages[i] for i in remaining],
                )
                done.update(remaining[i] for i in result.done)
                quarantine.update(remaining[i] for i in result.quarantine)
                old = remaining
                remaining = [i for i in old if i not in done and i not in quarantine]
                if len(remaining) == len(old):
                    break
            if not remaining:
                break
        return IOResult(frozenset(done), frozenset(quarantine))

    def recall(self, group, source_id, endpoints, hashes, pages):
        done, quarantine = set(), set()
        try:
            hits = self.lookup(hashes, source_id)
        except (grpc.RpcError, RuntimeError) as exc:
            logger.warning("Raiden recall lookup failed: %s", exc)
            return IOResult()
        owners = defaultdict(list)
        for i, hit in enumerate(hits):
            owners[hit.raiden_id.SerializeToString()].append(i)
        for owner, indices in owners.items():
            try:
                address = self.resolve(w.RaidenIdProto.FromString(owner), group)
                if address is None:
                    continue
            except (grpc.RpcError, ValueError) as exc:
                logger.warning("Raiden recall owner resolution failed: %s", exc)
                continue
            try:
                response = self._rpc(address, STORE_SERVICE, "Fetch", w.FetchResponse)(
                    w.FetchRequest(
                        block_hashes=[hashes[i] for i in indices],
                        host_block_ids=[pages[i] for i in indices],
                        client_raiden_id=source_id,
                        client_worker_endpoints=endpoints,
                    ),
                    timeout=self.timeout_s,
                )
                expected = {hashes[i] for i in indices}
                # Native Fetch returns only after copying and pins its source
                # through the entire operation. Validate the complete verdict.
                if (
                    response.error_message
                    or response.failed_block_hashes
                    or set(response.done_block_hashes) != expected
                ):
                    raise ValueError("Incomplete Raiden Fetch result")
                done.update(indices)
            except (grpc.RpcError, ValueError) as exc:
                logger.warning("Raiden Fetch failed: %s", exc)
                if isinstance(exc, grpc.RpcError) and exc.code() in (
                    grpc.StatusCode.NOT_FOUND,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    grpc.StatusCode.UNIMPLEMENTED,
                ):
                    continue  # Native validation failed before writing.
                # Timeout/transport failure may leave a remote write running.
                quarantine.update(indices)
        return IOResult(frozenset(done), frozenset(quarantine))

    def close(self):
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()
