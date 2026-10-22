# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of the reverse prefix pull (see
``disaggregation/raiden/peer_prefix.py``).

Decode engine: after each ``process_decode_queue`` the manager serves the
prefix requests of its prefill peers from the decode radix tree.

Prefill engine: a request whose prefix is not in the prefill's own tiers asks
the decode peer when it enters the queue. ``pop_bootstrapped`` holds the
request until the pull lands, inserts the pulled pages into the radix tree and
only then lets the request be scheduled, so it prefills the remaining tokens.

All decisions that change what a request does are taken from values reduced
across the attention-TP ranks, so every rank inserts the same key and
schedules the same request in the same step.

Enable with ``SGLANG_TPU_PEER_PREFIX=1`` on both engines. The decode engine
needs ``--disaggregation-decode-enable-radix-cache --enable-hierarchical-cache``.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import _match_prefix_result
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_disagg
from sglang_tpu.disaggregation.raiden.peer_prefix import (
    STAGE_DONE,
    STAGE_FAILED,
    STAGE_REPLIED,
    ProducerContext,
    page_floor,
)

logger = logging.getLogger("sglang_tpu.pd_peer_prefix")

ENV_ENABLE = "SGLANG_TPU_PEER_PREFIX"
ENV_TIMEOUT = "SGLANG_TPU_PEER_PREFIX_TIMEOUT_S"
ENV_MIN_TOKENS = "SGLANG_TPU_PEER_PREFIX_MIN_TOKENS"


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0") == "1"


def _timeout_s() -> float:
    return float(os.environ.get(ENV_TIMEOUT, "120"))


def _min_tokens(page_size: int) -> int:
    return max(page_size, int(os.environ.get(ENV_MIN_TOKENS, "8192")))


def _all_reduce(values: List[int], group, op) -> List[int]:
    if not values:
        return values
    t = torch.tensor(values, dtype=torch.int64)
    if group is not None and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(t, op=op, group=group)
    return t.tolist()


# --------------------------------------------------------------------------- #
# Decode engine
# --------------------------------------------------------------------------- #
def _producer_context(scheduler) -> Optional[ProducerContext]:
    ctx = getattr(scheduler, "_peer_prefix_ctx", None)
    if ctx is not None:
        return ctx
    tree = scheduler.tree_cache
    allocator = scheduler.token_to_kv_pool_allocator
    hybrid = bool(getattr(tree, "is_swa_enabled", False))
    page_size = allocator.page_size
    ctx = ProducerContext(
        tree_cache=tree,
        allocator=allocator,
        page_size=page_size,
        hybrid=hybrid,
        sliding_window_size=getattr(scheduler, "sliding_window_size", None),
        # The decode's host copy of a hybrid model holds full-attention KV
        # only; a restored prefix would have no sliding-window pages and the
        # prefill could not use it. Serve HBM-resident prefixes for those.
        restore_allowed=bool(scheduler.enable_decode_hicache) and not hybrid,
        min_hit_tokens=_min_tokens(page_size),
        lease_timeout_s=max(
            _timeout_s() + 60.0,
            float(os.environ.get("SGLANG_RAIDEN_TIMEOUT_S", "600")) + 60.0,
        ),
        is_eagle=bool(getattr(getattr(tree, "tree_core", None), "is_eagle", False)),
    )
    scheduler._peer_prefix_ctx = ctx
    return ctx


def _process_decode_queue(upstream: Callable, scheduler) -> None:
    upstream(scheduler)
    queue = getattr(scheduler, "disagg_decode_prealloc_queue", None)
    manager = getattr(queue, "kv_manager", None)
    serve = getattr(manager, "peer_prefix_serve", None)
    if serve is None:
        return
    if not get_disagg().disaggregation_decode_enable_radix_cache:
        return
    ctx = _producer_context(scheduler)
    if ctx is not None:
        serve(ctx)


# --------------------------------------------------------------------------- #
# Prefill engine
# --------------------------------------------------------------------------- #
# Pages a pull has allocated but not yet handed to the radix tree. The idle
# pool invariant counts only tree and request pages, so these are reported as
# "uncached" through the wrapped invariant checks below.
_INFLIGHT: dict = {}


def _inflight_tokens() -> int:
    return sum(
        len(f.new_indices) for f in _INFLIGHT.values() if f.new_indices is not None
    )


def _check_full_pool(upstream: Callable, checker, ps, uncached: int = 0):
    return upstream(checker, ps, uncached=uncached + _inflight_tokens())


def _check_swa_pool(upstream: Callable, checker, ps, uncached: int = 0):
    return upstream(checker, ps, uncached=uncached + _inflight_tokens())


@dataclass
class _Fetch:
    room: int
    matched_len: int
    want_len: int
    hit_len: int = 0
    new_indices: Optional[torch.Tensor] = None
    lock_node: Any = None
    lock_params: Any = None
    prefix_indices: Optional[torch.Tensor] = None
    reading: bool = False
    remote_blocks: List[int] = None
    local_blocks: List[int] = None
    remote_swa_blocks: List[int] = None
    local_swa_blocks: List[int] = None


def _add_request_to_queue(
    upstream: Callable, scheduler, req, is_retracted: bool = False
):
    upstream(scheduler, req, is_retracted)
    try:
        _maybe_begin_fetch(scheduler, req, is_retracted)
    except Exception:  # noqa: BLE001
        logger.exception("peer prefix: begin failed for %s", getattr(req, "rid", "?"))


def _maybe_begin_fetch(scheduler, req, is_retracted: bool) -> None:
    if scheduler.disaggregation_mode != DisaggregationMode.PREFILL or is_retracted:
        return
    if getattr(req, "bootstrap_room", None) is None or req.finished():
        return
    queue = scheduler.disagg_prefill_bootstrap_queue
    manager = getattr(queue, "kv_manager", None)
    if manager is None or not hasattr(manager, "peer_fetch_begin"):
        return
    tree = scheduler.tree_cache
    allocator = scheduler.token_to_kv_pool_allocator
    page_size = allocator.page_size
    tokens = req.origin_input_ids
    result = _match_prefix_result(tree, req, tokens)
    matched = page_floor(len(result.device_indices), page_size)
    want = page_floor(req._compute_max_prefix_len(len(tokens)), page_size)
    # Every rank must take the same decision. The local match is in lockstep
    # already; whether the decode peer is known is not, until the first pull.
    local_ok = int(
        result.host_hit_length == 0
        and want - matched >= _min_tokens(page_size)
        and manager.peer_known()
    )
    ok = _all_reduce([local_ok], scheduler.attn_tp_cpu_group, dist.ReduceOp.MIN)[0]
    if not ok:
        return
    if not manager.peer_fetch_begin(
        req.bootstrap_room, tokens[:want], matched, req.extra_key, req.cache_salt
    ):
        # The send failed on this rank only; the hold below reduces stages, so
        # a missing fetch surfaces as a failure on every rank.
        pass
    fetch = _Fetch(room=req.bootstrap_room, matched_len=matched, want_len=want)
    # Snapshot of the local prefix. It is not locked while the pull is in
    # flight: a lock would pin a whole context per held request and starve
    # the prefill's own scheduling. The commit re-matches and gives up if the
    # prefix moved.
    fetch.prefix_indices = result.device_indices[:matched].cpu().to(torch.int64)
    req._peer_fetch = fetch


def _pop_bootstrapped(upstream: Callable, queue, *args, **kwargs):
    out = upstream(queue, *args, **kwargs)
    if isinstance(out, tuple):
        reqs, failed = out
    else:
        reqs, failed = out, None
    held = queue.__dict__.setdefault("_peer_held", [])
    passthrough = []
    for req in reqs:
        if getattr(req, "_peer_fetch", None) is not None:
            held.append(req)
        else:
            passthrough.append(req)
    # _advance_held reduces every decision across ranks and must not raise;
    # an escaping error here would desynchronize the ranks' collectives.
    passthrough.extend(_advance_held(queue, held))
    return passthrough if failed is None else (passthrough, failed)


def _advance_held(queue, held: List[Any]) -> List[Any]:
    """Advance every held request one step. Each step's decision is taken
    from values reduced over the attention-TP ranks, so the ranks hold, pull,
    insert and release the same requests in the same scheduler iteration. A
    rank-local problem only marks that rank's fetch failed; the next
    iteration's MAX reduce then makes every rank abandon the pull."""
    if not held:
        _reap_draining(queue)
        return []
    scheduler = queue.scheduler
    manager = queue.kv_manager
    group = scheduler.attn_tp_cpu_group
    timeout = _timeout_s()
    stages, hits = [], []
    for req in held:
        fetch = req._peer_fetch
        stage = manager.peer_fetch_poll(fetch.room, timeout)
        peer = manager.peer_fetch(fetch.room)
        stages.append(stage)
        hits.append(peer.hit_len if peer is not None else 0)
    min_stage = _all_reduce(stages, group, dist.ReduceOp.MIN)
    max_stage = _all_reduce(stages, group, dist.ReduceOp.MAX)
    min_hit = _all_reduce(hits, group, dist.ReduceOp.MIN)
    released = []
    # Replied on every rank: prepare the pull locally, then agree on it.
    # Every rank sees the same in-flight set, so the budget gate is consistent.
    budget = manager.peer_read_budget_pages() - manager.peer_inflight_pages()
    page_size = scheduler.token_to_kv_pool_allocator.page_size
    prepare_idx = []
    for i in range(len(held)):
        if not (min_stage[i] == STAGE_REPLIED and max_stage[i] == STAGE_REPLIED):
            continue
        need = max(0, min_hit[i] - held[i]._peer_fetch.matched_len) // page_size
        if need > budget:
            continue  # wait for in-flight reads to settle
        budget -= need
        prepare_idx.append(i)
    if prepare_idx:
        ready = []
        for i in prepare_idx:
            try:
                ready.append(int(_prepare_read(queue, held[i], min_hit[i])))
            except Exception:  # noqa: BLE001
                logger.exception("peer prefix: prepare failed for %s", held[i].rid)
                ready.append(0)
        agreed = _all_reduce(ready, group, dist.ReduceOp.MIN)
        for i, ok in zip(prepare_idx, agreed):
            req = held[i]
            if ok:
                try:
                    if not _issue_read(queue, req):
                        _mark_failed(manager, req, "start_read refused")
                except Exception:  # noqa: BLE001
                    logger.exception("peer prefix: start_read failed for %s", req.rid)
                    _mark_failed(manager, req, "start_read raised")
            else:
                _abandon(queue, req)
                released.append(req)
    for i, req in enumerate(list(held)):
        if req in released:
            continue
        if max_stage[i] == STAGE_FAILED:
            _abandon(queue, req)
            released.append(req)
        elif min_stage[i] == STAGE_DONE:
            try:
                _commit(queue, req)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "peer prefix: commit failed for %s; releasing", req.rid
                )
                _abandon(queue, req)
            released.append(req)
        # STAGE_SENT / STAGE_READING on some rank: wait.
    for req in released:
        held.remove(req)
    _reap_draining(queue)
    return released


def _mark_failed(manager, req, reason: str) -> None:
    fetch = getattr(req, "_peer_fetch", None)
    peer = manager.peer_fetch(fetch.room) if fetch is not None else None
    if peer is not None:
        peer.error = peer.error or reason
        peer.stage = STAGE_FAILED


def _prepare_read(queue, req, hit: int) -> bool:
    """Allocate this rank's landing pages for a pull of ``hit`` tokens and
    check them against the decode's reply. Rank-local; the caller reduces
    the result before any read starts."""
    scheduler = queue.scheduler
    manager = queue.kv_manager
    allocator = scheduler.token_to_kv_pool_allocator
    page_size = allocator.page_size
    fetch = req._peer_fetch
    hit = min(page_floor(hit, page_size), fetch.want_len)
    hybrid = bool(getattr(scheduler.tree_cache, "is_swa_enabled", False))
    window = getattr(scheduler, "sliding_window_size", None)
    if hybrid and (
        window is None
        or hit - fetch.matched_len < page_floor(window, page_size) + page_size
    ):
        return False
    if hit <= fetch.matched_len:
        return False
    peer = manager.peer_fetch(fetch.room)
    if peer is None:
        return False
    n_new = hit - fetch.matched_len
    if len(peer.blocks) != n_new // page_size:
        return False
    last_loc = int(fetch.prefix_indices[-1]) if fetch.matched_len > 0 else -1
    new_indices = allocator.alloc_extend(
        torch.tensor([fetch.matched_len], dtype=torch.int64),
        torch.tensor([fetch.matched_len], dtype=torch.int64),
        torch.tensor([hit], dtype=torch.int64),
        torch.tensor([hit], dtype=torch.int64),
        torch.tensor([last_loc], dtype=torch.int64),
        n_new,
    )
    if new_indices is None:
        return False
    new_indices = new_indices.cpu().to(torch.int64)
    fetch.hit_len = hit
    fetch.new_indices = new_indices
    _INFLIGHT[id(fetch)] = fetch
    local_blocks = [int(b) for b in kv_to_page_indices(new_indices, page_size)]
    swa_blocks: List[int] = []
    if hybrid:
        window_start = page_floor(max(0, hit - window), page_size)
        swa_idx = allocator.translate_loc_from_full_to_swa(
            new_indices[window_start - fetch.matched_len :]
        )
        swa_blocks = [int(b) for b in kv_to_page_indices(swa_idx, page_size)]
    # Every rank must pull the same token range, so the decode ranks must have
    # answered with the same hit. A rank whose decode found more or fewer pages
    # than the agreed hit makes every rank give up on this pull.
    remote_blocks = list(peer.blocks)
    remote_swa = list(peer.swa_blocks)
    if len(remote_blocks) != len(local_blocks) or len(remote_swa) != len(swa_blocks):
        _release_pages(allocator, fetch)
        return False
    fetch.remote_blocks = remote_blocks
    fetch.local_blocks = local_blocks
    fetch.remote_swa_blocks = remote_swa
    fetch.local_swa_blocks = swa_blocks
    return True


def _issue_read(queue, req) -> bool:
    manager = queue.kv_manager
    fetch = req._peer_fetch
    peer = manager.peer_fetch(fetch.room)
    if peer is None:
        return False
    peer.blocks = list(fetch.remote_blocks)
    peer.swa_blocks = list(fetch.remote_swa_blocks)
    if not manager.peer_fetch_start_read(
        fetch.room, fetch.local_blocks, fetch.local_swa_blocks
    ):
        return False
    fetch.reading = True
    return True


def _release_pages(allocator, fetch) -> None:
    """Return a fetch's pages to the allocator and drop them from the in-flight set."""
    _INFLIGHT.pop(id(fetch), None)
    if fetch.new_indices is not None:
        allocator.free(fetch.new_indices)
        fetch.new_indices = None


def _commit(queue, req) -> None:
    """Every rank's read settled: hand the pulled pages to the radix tree."""
    scheduler = queue.scheduler
    manager = queue.kv_manager
    tree = scheduler.tree_cache
    allocator = scheduler.token_to_kv_pool_allocator
    page_size = allocator.page_size
    fetch = req._peer_fetch
    hit = fetch.hit_len
    tokens = req.origin_input_ids[:hit]
    if fetch.matched_len > 0:
        # The local prefix was not locked during the pull. It must still be
        # the same device pages, or the pulled tail has nothing to attach to.
        again = _match_prefix_result(
            tree, req, req.origin_input_ids[: fetch.matched_len]
        )
        current = again.device_indices[: fetch.matched_len].cpu().to(torch.int64)
        if len(current) != fetch.matched_len or not torch.equal(
            current, fetch.prefix_indices
        ):
            logger.info(
                "peer prefix dropped | rid=%s local prefix moved (%d -> %d tokens)",
                req.rid,
                fetch.matched_len,
                len(current),
            )
            _abandon(queue, req)
            return
    values = torch.cat([fetch.prefix_indices, fetch.new_indices])
    params = InsertParams(
        key=RadixKey(
            tokens,
            req.extra_key,
            is_bigram=bool(
                getattr(getattr(tree, "tree_core", None), "is_eagle", False)
            ),
            cache_salt=req.cache_salt,
        ),
        value=values,
        prev_prefix_len=fetch.matched_len,
        inserted_len=fetch.matched_len,
    )
    hybrid = bool(getattr(tree, "is_swa_enabled", False))
    if hybrid:
        window = scheduler.sliding_window_size
        window_start = page_floor(max(0, hit - window), page_size)
        if window_start > fetch.matched_len:
            allocator.free_swa_segment(
                fetch.new_indices[: window_start - fetch.matched_len],
                start_pos=fetch.matched_len,
            )
        params.set_evicted_seqlen(ComponentType.SWA, window_start)
    result = tree.insert(params)
    declined = bool(getattr(result, "rotation_tail_declined", False))
    if declined:
        _release_pages(allocator, fetch)
    else:
        _INFLIGHT.pop(id(fetch), None)  # the tree owns the pages now
    _unlock(tree, fetch)
    manager.peer_fetch_end(fetch.room)
    req._peer_fetch = None
    logger.info(
        "peer prefix committed | rid=%s room=%s matched=%d hit=%d declined=%s",
        req.rid,
        fetch.room,
        fetch.matched_len,
        hit,
        declined,
    )


def _unlock(tree, fetch) -> None:
    if fetch.lock_node is not None and fetch.lock_params is not None:
        tree.dec_lock_ref(fetch.lock_node, fetch.lock_params)
        fetch.lock_node = None
        fetch.lock_params = None


def _abandon(queue, req, free_pages: bool = True) -> None:
    """Give up on the pull; the request prefills what it misses."""
    try:
        _abandon_impl(queue, req, free_pages)
    except Exception:  # noqa: BLE001
        logger.exception("peer prefix: abandon failed for %s", getattr(req, "rid", "?"))
        req._peer_fetch = None


def _abandon_impl(queue, req, free_pages: bool) -> None:
    scheduler = queue.scheduler
    manager = queue.kv_manager
    fetch = getattr(req, "_peer_fetch", None)
    if fetch is None:
        return
    _unlock(scheduler.tree_cache, fetch)
    if fetch.new_indices is not None:
        if fetch.reading and not manager.peer_fetch_reads_settled(fetch.room):
            # A read may still land in these pages. Keep them out of the
            # allocator until Raiden settles it.
            queue.__dict__.setdefault("_peer_draining", []).append(fetch)
            manager.peer_fetch_end(fetch.room, notify=True)
            req._peer_fetch = None
            return
        if free_pages:
            _release_pages(scheduler.token_to_kv_pool_allocator, fetch)
    manager.peer_fetch_end(fetch.room, notify=True)
    req._peer_fetch = None


def _reap_draining(queue) -> None:
    draining = queue.__dict__.get("_peer_draining")
    if not draining:
        return
    manager = queue.kv_manager
    allocator = queue.scheduler.token_to_kv_pool_allocator
    keep = []
    for fetch in draining:
        # peer_fetch_end already dropped the fetch; settle by req id.
        settled = manager.peer_fetch_reads_settled(fetch.room)
        if settled:
            _release_pages(allocator, fetch)
        else:
            keep.append(fetch)
    queue._peer_draining = keep


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #
def _wrap(owner: Any, name: str, impl: Callable) -> None:
    upstream = getattr(owner, name)
    if getattr(upstream, "_peer_prefix_wrapped", False):
        return

    @functools.wraps(upstream)
    def wrapper(*args, **kwargs):
        return impl(upstream, *args, **kwargs)

    wrapper._peer_prefix_wrapped = True
    setattr(owner, name, wrapper)


def install() -> None:
    if not enabled():
        return
    from sglang.srt.disaggregation import decode as decode_mod
    from sglang.srt.disaggregation import prefill as prefill_mod
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.invariant_checker import (
        SchedulerInvariantChecker,
    )

    _wrap(
        decode_mod.SchedulerDisaggregationDecodeMixin,
        "process_decode_queue",
        _process_decode_queue,
    )
    _wrap(prefill_mod.PrefillBootstrapQueue, "pop_bootstrapped", _pop_bootstrapped)
    _wrap(Scheduler, "_add_request_to_queue", _add_request_to_queue)

    _wrap(SchedulerInvariantChecker, "_check_full_pool", _check_full_pool)
    _wrap(SchedulerInvariantChecker, "_check_swa_pool", _check_swa_pool)
    logger.info("peer prefix pull installed (timeout %.0fs)", _timeout_s())
