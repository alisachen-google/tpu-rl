# SPDX-License-Identifier: Apache-2.0
"""Reverse prefix pull: a prefill rank reads an evicted prefix back from its
decode peer's HiCache.

The decode engine keeps finished requests in its radix tree, in HBM and in its
host pool. A prefill rank that misses a prefix in its own tiers sends the
tokens to the matching decode rank. The decode matches them in its tree, locks
the nodes, restores host pages to HBM when it has to, registers the device
pages for a Raiden read and replies with the block ids. The prefill pulls the
pages into its own HBM with the same consumer-pull transfer the P/D path uses,
inserts them into its radix tree and prefills only the rest.

Raiden does three things here: H2D on the decode for the restore, then D2H,
network and H2D for the pull. HiCache on each side decides what is kept and
for how long. Nothing lands in the prefill host pool.

Threads: the ZMQ listener only enqueues. Every tree operation runs on the
scheduler thread, through ``peer_prefix_serve`` on the decode and the
``peer_fetch_*`` methods on the prefill.
"""

from __future__ import annotations

import logging
import threading
import time
from array import array
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams, MatchPrefixParams
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixBlocks,
    PrefixRelease,
    PrefixReq,
)

logger = logging.getLogger("sglang_tpu.disaggregation.raiden.peer_prefix")

# Raiden ids of a reverse pull. The room's own P/D transfer uses uuid == room on
# the same engines, so the reverse pull takes a disjoint range.
PEER_UUID_BASE = 1 << 40

STAGE_SENT = 0  # PrefixReq sent, no answer yet
STAGE_REPLIED = 1  # PrefixBlocks received (hit_len may be 0)
STAGE_READING = 2  # start_read issued on this rank
STAGE_DONE = 3  # every read of this rank settled
STAGE_FAILED = 9


def peer_req_id(room: int) -> str:
    return f"peer:{room}"


def peer_swa_req_id(room: int) -> str:
    return f"peerswa:{room}"


def peer_uuid(room: int) -> int:
    return PEER_UUID_BASE + room


def page_floor(n: int, page_size: int) -> int:
    return (n // page_size) * page_size


@dataclass
class PeerFetch:
    """Prefill-side state of one reverse pull."""

    room: int
    matched_len: int
    num_tokens: int
    sent_at: float
    stage: int = STAGE_SENT
    hit_len: int = 0
    uuid: int = 0
    blocks: List[int] = field(default_factory=list)
    endpoint: str = ""
    swa_blocks: List[int] = field(default_factory=list)
    swa_endpoint: str = ""
    error: str = ""
    req_ids: tuple = ()
    # Set by the scheduler hook: the device slots the pull lands in.
    new_indices: Optional[torch.Tensor] = None
    lock_node: Any = None
    lock_params: Any = None
    prefix_indices: Optional[torch.Tensor] = None


@dataclass
class PeerServe:
    """Decode-side state of one reverse pull it serves."""

    req: PrefixReq
    received_at: float
    hit_len: int = 0
    node_id: Any = None
    lock_params: Any = None
    restore_node_id: Any = None
    restore_lock_params: Any = None
    consumer_index: int = -1
    device_indices: Optional[torch.Tensor] = None
    replied: bool = False


@dataclass
class ProducerContext:
    """What the decode scheduler hands to ``peer_prefix_serve``."""

    tree_cache: Any
    allocator: Any
    page_size: int
    hybrid: bool = False
    sliding_window_size: Optional[int] = None
    restore_allowed: bool = True
    min_hit_tokens: int = 64
    lease_timeout_s: float = 180.0
    is_eagle: bool = False


class PeerPrefixMixin:
    """Mixed into ``RaidenKVManager``; both roles share the state and the
    message handlers, each role uses its own methods."""

    def _peer_init(self) -> None:
        self._peer_lock = threading.Lock()
        self._peer_inbox: deque = deque()
        self._peer_release_inbox: deque = deque()
        self._peer_serving: Dict[int, PeerServe] = {}
        self._peer_fetches: Dict[int, PeerFetch] = {}
        self._peer_decode_endpoint: Optional[str] = None
        self._message_handlers.update(
            {
                PrefixReq.TAG: self._on_prefix_req,
                PrefixBlocks.TAG: self._on_prefix_blocks,
                PrefixRelease.TAG: self._on_prefix_release,
            }
        )

    # ------------------------------------------------------------------ #
    # Listener thread: enqueue only.
    # ------------------------------------------------------------------ #
    def _on_prefix_req(self, msg: List[bytes]) -> None:
        req = PrefixReq.from_frames(msg)
        with self._peer_lock:
            self._peer_inbox.append(req)

    def _on_prefix_blocks(self, msg: List[bytes]) -> None:
        reply = PrefixBlocks.from_frames(msg)
        with self._peer_lock:
            fetch = self._peer_fetches.get(reply.room)
            if fetch is None or fetch.stage != STAGE_SENT:
                return
            if reply.error:
                fetch.error = reply.error
                fetch.stage = STAGE_FAILED
                return
            fetch.hit_len = reply.hit_len
            fetch.uuid = reply.uuid
            fetch.blocks = list(reply.blocks)
            fetch.endpoint = reply.endpoint
            fetch.swa_blocks = list(reply.swa_blocks)
            fetch.swa_endpoint = reply.swa_endpoint
            fetch.stage = STAGE_REPLIED

    def _on_prefix_release(self, msg: List[bytes]) -> None:
        release = PrefixRelease.from_frames(msg)
        with self._peer_lock:
            self._peer_release_inbox.append(release.room)

    def _on_pull_req(self, msg: List[bytes]) -> None:
        # The decode rank that pulls from this prefill rank is the rank that
        # serves its reverse pulls. Remember its control endpoint.
        try:
            from sglang_tpu.disaggregation.raiden.serde import PullReq

            req = PullReq.from_frames(msg)
            self._peer_decode_endpoint = _tcp_endpoint(
                ip=req.decode_ip, port=req.decode_port
            )
        except Exception:  # noqa: BLE001 - the real handler reports it
            pass
        super()._on_pull_req(msg)

    # ------------------------------------------------------------------ #
    # Prefill role (consumer). Scheduler thread.
    # ------------------------------------------------------------------ #
    def peer_known(self) -> bool:
        return self._peer_decode_endpoint is not None

    def peer_fetch_begin(
        self,
        room: int,
        token_ids: Sequence[int],
        matched_len: int,
        extra_key: Optional[str],
        cache_salt: Optional[str],
    ) -> bool:
        endpoint = self._peer_decode_endpoint
        if endpoint is None:
            return False
        payload = PrefixReq(
            room=room,
            prefill_ip=self.local_ip,
            prefill_port=self.rank_port,
            token_ids=[int(t) for t in token_ids],
            matched_len=matched_len,
            extra_key=extra_key,
            cache_salt=cache_salt,
        ).to_frames()
        fetch = PeerFetch(
            room=room,
            matched_len=matched_len,
            num_tokens=len(token_ids),
            sent_at=time.monotonic(),
        )
        with self._peer_lock:
            self._peer_fetches[room] = fetch
        try:
            self._send_multipart_locked(endpoint, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("peer prefix request for room %s failed: %s", room, exc)
            with self._peer_lock:
                self._peer_fetches.pop(room, None)
            return False
        return True

    def peer_fetch(self, room: int) -> Optional[PeerFetch]:
        with self._peer_lock:
            return self._peer_fetches.get(room)

    def peer_inflight_pages(self) -> int:
        """Pages of this rank's reverse reads that Raiden has not settled."""
        with self._peer_lock:
            return sum(
                len(f.blocks)
                for f in self._peer_fetches.values()
                if f.stage == STAGE_READING
            )

    def peer_read_budget_pages(self) -> int:
        """Reverse reads in flight on this rank stay under one max-size request,
        so they never starve the P/D transfers that share the same staging."""
        max_blocks = int(getattr(self, "_full_engine_max_blocks", 0) or 0)
        if max_blocks <= 0:
            return 1 << 30
        return max(1, max_blocks // 2)

    def peer_fetch_start_read(
        self, room: int, local_blocks: List[int], local_swa_blocks: List[int]
    ) -> bool:
        fetch = self.peer_fetch(room)
        if fetch is None or fetch.stage != STAGE_REPLIED:
            return False
        if len(local_blocks) != len(fetch.blocks) or len(local_swa_blocks) != len(
            fetch.swa_blocks
        ):
            fetch.error = "local and remote block counts differ"
            fetch.stage = STAGE_FAILED
            return False
        req_ids = [peer_req_id(room)]
        try:
            engine = self._ensure_engine()
            engine.start_read(
                peer_req_id(room),
                fetch.uuid,
                fetch.endpoint,
                list(fetch.blocks),
                list(local_blocks),
                self._transfer_parallelism,
            )
            if fetch.swa_blocks:
                if self._swa_engine is None:
                    raise RuntimeError(
                        "peer sent SWA blocks but this rank has no SWA engine"
                    )
                self._swa_engine.start_read(
                    peer_swa_req_id(room),
                    fetch.uuid,
                    fetch.swa_endpoint,
                    list(fetch.swa_blocks),
                    list(local_swa_blocks),
                    self._transfer_parallelism,
                )
                req_ids.append(peer_swa_req_id(room))
        except Exception as exc:  # noqa: BLE001
            logger.warning("peer prefix start_read for room %s raised: %s", room, exc)
            fetch.error = f"start_read: {exc}"
            fetch.req_ids = tuple(req_ids[:0])
            fetch.stage = STAGE_FAILED
            return False
        fetch.req_ids = tuple(req_ids)
        fetch.stage = STAGE_READING
        logger.info(
            "peer prefix read started | room=%s pages=%d swa_pages=%d waited=%.1fs",
            room,
            len(local_blocks),
            len(local_swa_blocks),
            time.monotonic() - fetch.sent_at,
        )
        return True

    def peer_fetch_poll(self, room: int, timeout_s: float) -> int:
        """Advance the fetch from what Raiden reported; returns the stage."""
        fetch = self.peer_fetch(room)
        if fetch is None:
            return STAGE_FAILED
        if fetch.stage == STAGE_READING:
            self._drain_engine()
            with self._state_lock:
                done = all(rid in self._completed_recv for rid in fetch.req_ids)
                failed = any(rid in self._failed for rid in fetch.req_ids)
            if failed:
                fetch.error = fetch.error or "raiden read failed"
                fetch.stage = STAGE_FAILED
                logger.warning(
                    "peer prefix read failed | room=%s after %.1fs",
                    room,
                    time.monotonic() - fetch.sent_at,
                )
            elif done:
                fetch.stage = STAGE_DONE
                logger.info(
                    "peer prefix read done | room=%s after %.1fs",
                    room,
                    time.monotonic() - fetch.sent_at,
                )
        if fetch.stage in (STAGE_SENT, STAGE_REPLIED) and (
            time.monotonic() - fetch.sent_at > timeout_s
        ):
            fetch.error = fetch.error or "timed out waiting for the decode"
            fetch.stage = STAGE_FAILED
        return fetch.stage

    def peer_fetch_reads_settled(self, room: int) -> bool:
        """True when no read of this fetch is still in flight on this rank."""
        fetch = self.peer_fetch(room)
        if fetch is None or not fetch.req_ids:
            return True
        self._drain_engine()
        with self._state_lock:
            return all(
                rid in self._completed_recv or rid in self._failed
                for rid in fetch.req_ids
            )

    def peer_fetch_end(self, room: int, notify: bool = True) -> None:
        with self._peer_lock:
            fetch = self._peer_fetches.pop(room, None)
        if fetch is None:
            return
        with self._state_lock:
            for rid in fetch.req_ids:
                self._completed_recv.discard(rid)
                self._failed.discard(rid)
                self._native_failed.discard(rid)
        endpoint = self._peer_decode_endpoint
        if notify and endpoint is not None:
            try:
                self._send_multipart_locked(
                    endpoint, PrefixRelease(room=room).to_frames()
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("peer prefix release for room %s failed: %s", room, exc)

    # ------------------------------------------------------------------ #
    # Decode role (producer). Scheduler thread.
    # ------------------------------------------------------------------ #
    def peer_prefix_serve(self, ctx: ProducerContext) -> None:
        with self._peer_lock:
            inbox = list(self._peer_inbox)
            self._peer_inbox.clear()
            releases = list(self._peer_release_inbox)
            self._peer_release_inbox.clear()
        if not inbox and not releases and not self._peer_serving:
            return
        for req in inbox:
            try:
                self._peer_serve_new(ctx, req)
            except Exception as exc:  # noqa: BLE001
                logger.exception("peer prefix serve failed for room %s", req.room)
                self._peer_reply_miss(req, f"serve: {exc}")
                self._peer_lease_release(ctx, req.room)
        for room, serve in list(self._peer_serving.items()):
            if not serve.replied:
                try:
                    self._peer_serve_advance(ctx, room, serve)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("peer prefix register failed for room %s", room)
                    self._peer_reply_miss(serve.req, f"register: {exc}")
                    self._peer_lease_release(ctx, room)
        for room in releases:
            self._peer_lease_release(ctx, room)
        if not self._peer_serving:
            return
        self._drain_engine()
        now = time.monotonic()
        with self._state_lock:
            done = set(self._completed_send)
            failed = set(self._failed)
        for room, serve in list(self._peer_serving.items()):
            if not serve.replied:
                continue
            rid = peer_req_id(room)
            settled = rid in done or rid in failed
            if settled or now - serve.received_at > ctx.lease_timeout_s:
                self._peer_lease_release(ctx, room)

    def _peer_match(self, ctx: ProducerContext, req: PrefixReq):
        # Tree keys are int64 arrays, as the tokenizer manager builds them.
        key = RadixKey(
            array("q", req.token_ids),
            req.extra_key,
            is_bigram=ctx.is_eagle,
            cache_salt=req.cache_salt,
        )
        return ctx.tree_cache.match_prefix(MatchPrefixParams(key=key))

    def _peer_serve_new(self, ctx: ProducerContext, req: PrefixReq) -> None:
        room = req.room
        if room in self._peer_serving:
            self._peer_lease_release(ctx, room)
        tree = ctx.tree_cache
        result = self._peer_match(ctx, req)
        device_len = page_floor(len(result.device_indices), ctx.page_size)
        host_len = result.host_hit_length if ctx.restore_allowed else 0
        limit = page_floor(len(req.token_ids), ctx.page_size)
        hit = min(page_floor(device_len + host_len, ctx.page_size), limit)
        if hit < req.matched_len + ctx.min_hit_tokens:
            self._peer_reply_miss(req)
            return
        serve = PeerServe(req=req, received_at=time.monotonic(), hit_len=hit)
        serve.node_id = result.last_device_node
        serve.lock_params = tree.inc_lock_ref(result.last_device_node).to_dec_params()
        self._peer_serving[room] = serve
        if hit > device_len:
            restored = self._peer_start_restore(ctx, serve, result, hit)
            if not restored:
                # Serve what is in HBM, or nothing.
                hit = device_len
                if hit < req.matched_len + ctx.min_hit_tokens:
                    self._peer_reply_miss(req, "restore unavailable")
                    self._peer_lease_release(ctx, room)
                    return
                serve.hit_len = hit
                serve.device_indices = result.device_indices[:hit]
        else:
            serve.device_indices = result.device_indices[:hit]
        self._peer_serve_advance(ctx, room, serve)

    def _peer_start_restore(self, ctx, serve, result, hit: int) -> bool:
        tree = ctx.tree_cache
        if not tree.has_free_load_back_slot():
            return False
        fake_req = SimpleNamespace(
            rid=peer_req_id(serve.req.room),
            last_node=result.last_device_node,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            kv=SimpleNamespace(mamba_pool_idx=None),
        )
        out = tree.init_load_back(
            InitLoadBackParams(
                best_match_node=result.best_match_node,
                host_hit_length=result.host_hit_length,
                req=fake_req,
            )
        )
        if out is None:
            return False
        new_indices, restored_node = out
        if len(new_indices) == 0:
            return False
        serve.restore_node_id = restored_node
        serve.restore_lock_params = tree.inc_lock_ref(restored_node).to_dec_params()
        serve.consumer_index = tree.ready_to_load_host_cache()
        serve.device_indices = torch.cat(
            [result.device_indices, new_indices.to(result.device_indices.device)]
        )[:hit]
        return True

    def _peer_serve_advance(
        self, ctx: ProducerContext, room: int, serve: PeerServe
    ) -> None:
        if serve.replied:
            return
        tree = ctx.tree_cache
        if serve.consumer_index >= 0 and not tree.is_load_back_event_done(
            serve.consumer_index
        ):
            return
        req = serve.req
        indices = serve.device_indices
        if indices is None or len(indices) < serve.hit_len:
            self._peer_reply_miss(req, "device indices short")
            self._peer_lease_release(ctx, room)
            return
        indices = indices[: serve.hit_len]
        # The prefill holds [0, matched_len) already; it pulls the rest.
        pulled = indices[req.matched_len :]
        blocks = [int(b) for b in kv_to_page_indices(pulled, ctx.page_size)]
        swa_blocks: List[int] = []
        if ctx.hybrid:
            if self._swa_engine is None or ctx.sliding_window_size is None:
                self._peer_reply_miss(req, "no SWA engine")
                self._peer_lease_release(ctx, room)
                return
            window_start = page_floor(
                max(0, serve.hit_len - ctx.sliding_window_size), ctx.page_size
            )
            swa_idx = ctx.allocator.translate_loc_from_full_to_swa(
                indices[window_start:]
            )
            if swa_idx.numel() == 0 or bool((swa_idx < 0).any()):
                self._peer_reply_miss(req, "SWA window not resident")
                self._peer_lease_release(ctx, room)
                return
            swa_blocks = [int(b) for b in kv_to_page_indices(swa_idx, ctx.page_size)]
        uuid = peer_uuid(room)
        engine = self._ensure_engine()
        accepted = bool(engine.register_read(peer_req_id(room), uuid, blocks))
        if not accepted:
            self._peer_reply_miss(req, "register_read refused")
            self._peer_lease_release(ctx, room)
            return
        if swa_blocks:
            self._swa_engine.register_read(peer_swa_req_id(room), uuid, swa_blocks)
        serve.replied = True
        serve.received_at = time.monotonic()
        payload = PrefixBlocks(
            room=room,
            hit_len=serve.hit_len,
            uuid=uuid,
            blocks=blocks,
            endpoint=self.raiden_endpoint or "",
            swa_blocks=swa_blocks,
            swa_endpoint=self.swa_raiden_endpoint or "",
        ).to_frames()
        self._send_multipart_locked(
            _tcp_endpoint(ip=req.prefill_ip, port=req.prefill_port), payload
        )
        logger.info(
            "peer prefix served | room=%s hit=%d pages=%d swa_pages=%d",
            room,
            serve.hit_len,
            len(blocks),
            len(swa_blocks),
        )

    def _peer_reply_miss(self, req: PrefixReq, error: str = "") -> None:
        logger.info(
            "peer prefix miss | room=%s matched=%d reason=%s",
            req.room,
            req.matched_len,
            error or "short hit",
        )
        payload = PrefixBlocks(
            room=req.room, hit_len=0, uuid=0, blocks=[], endpoint="", error=error
        ).to_frames()
        try:
            self._send_multipart_locked(
                _tcp_endpoint(ip=req.prefill_ip, port=req.prefill_port), payload
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "peer prefix miss reply for room %s failed: %s", req.room, exc
            )

    def _peer_lease_release(self, ctx: ProducerContext, room: int) -> None:
        serve = self._peer_serving.pop(room, None)
        if serve is None:
            return
        logger.info(
            "peer prefix lease released | room=%s hit=%d age=%.1fs",
            room,
            serve.hit_len,
            time.monotonic() - serve.received_at,
        )
        tree = ctx.tree_cache
        if serve.restore_node_id is not None and serve.restore_lock_params is not None:
            tree.dec_lock_ref(serve.restore_node_id, serve.restore_lock_params)
        if serve.node_id is not None and serve.lock_params is not None:
            tree.dec_lock_ref(serve.node_id, serve.lock_params)
        with self._state_lock:
            for rid in (peer_req_id(room), peer_swa_req_id(room)):
                self._completed_send.discard(rid)
                self._send_done_at.pop(rid, None)
                self._failed.discard(rid)
                self._native_failed.discard(rid)

    # ------------------------------------------------------------------ #
    # Introspection, for tests and metrics.
    # ------------------------------------------------------------------ #
    def peer_stats(self) -> dict:
        with self._peer_lock:
            return {
                "serving": len(self._peer_serving),
                "fetching": len(self._peer_fetches),
                "inbox": len(self._peer_inbox),
                "decode_endpoint": self._peer_decode_endpoint,
            }
