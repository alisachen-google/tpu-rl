"""_PACKER_FAST: numpy-first chunk assembly for tunix.rl.utils.pack_sequences (tunix 408ca1d9).
WHY: the stock packer builds every packed bin with ~18 jax ops (jnp.zeros/jnp.array/[None,:]/
concatenate) -> ~14k device dispatches per step (768 bins) = the +4.7 s actor_dequeue_time measured
in run 43c vs 42. FIX: fill [pack_size, budget] numpy arrays per feature and do ONE jnp.asarray per
feature per chunk (72 dispatches/step). Verified bit-identical output on synthetic data (bench_fast.py).
Replaces the (_flush_pack, chunk_capacity, _emit) block; FFD, _mark, main loop untouched. Fail-open."""
FAST_BLOCK = '\n  def _emit(chunk):\n    """[_PACKER_FAST] Merge one chunk (pack_size bins) into a [pack_size, budget] example, numpy-first."""\n    fi = first_item_for_dummy or {}\n    has_policy_version = fi.get("policy_version") is not None\n    tracked = [k for k in ("ref_per_token_logps", "old_per_token_logps", "returns", "old_values") if fi.get(k) is not None]\n    P = len(chunk); B = max_token_budget\n    c_ids = np.full((P, B), pad_id, dtype=np.int32)\n    c_mask = np.zeros((P, B), dtype=np.int32)\n    adv = np.zeros((P, B), dtype=np.float32)\n    seg = np.zeros((P, B), dtype=np.int32)\n    pos = np.zeros((P, B), dtype=np.int32)\n    feats = {k: np.zeros((P, B), dtype=np.float32) for k in tracked}\n    pv = None\n    for r, pack_items in enumerate(chunk):\n      if len(pack_items) > effective_max_segments:\n        raise ValueError(\n            f"pack_sequences: a packed row has {len(pack_items)} segments, "\n            f"exceeding max_segments_per_packed_row={effective_max_segments}; "\n            "increase it (or leave it None for the budget-derived safe default).")\n      off = 0\n      for i, item in enumerate(pack_items, start=1):\n        p_ids = item["prompt_ids"]; cc_ids = item["completion_ids"]\n        pl = len(p_ids); cl = len(cc_ids); L = pl + cl\n        c_ids[r, off:off + pl] = p_ids\n        c_ids[r, off + pl:off + L] = cc_ids\n        c_mask[r, off + pl:off + L] = item["completion_mask"]\n        if item["adv_is_per_token"]:\n          adv[r, off + pl:off + L] = item["advantages"]\n        else:\n          adv[r, off + pl:off + L] = item["advantages"]  # scalar broadcast == np.full(cl, adv)\n        for k in tracked:\n          feats[k][r, off + pl:off + L] = item[k]\n        seg[r, off:off + L] = i\n        pos[r, off:off + L] = np.arange(L, dtype=np.int32)\n        off += L\n      if pack_items and has_policy_version and pv is None:\n        pv = pack_items[0]["policy_version"]\n    kwargs = dict(\n        prompt_ids=jnp.zeros((P, 0), dtype=jnp.int32),\n        prompt_mask=jnp.zeros((P, 0), dtype=jnp.int32),\n        completion_ids=jnp.asarray(c_ids),\n        completion_mask=jnp.asarray(c_mask),\n        advantages=jnp.asarray(adv),\n        ref_per_token_logps=None,\n        old_per_token_logps=None,\n        segment_ids=jnp.asarray(seg),\n        segment_positions=jnp.asarray(pos),\n    )\n    for k in tracked:\n      kwargs[k] = jnp.asarray(feats[k])\n    if has_policy_version:\n      # original: per-bin policy_version concatenated along axis 0 (empty bins use first_item\'s)\n      pvs = [(b[0]["policy_version"] if b else fi["policy_version"]) for b in chunk]\n      kwargs["policy_version"] = jnp.concatenate([jnp.asarray(x) for x in pvs], axis=0)\n    return example_cls(**kwargs)  # pytype: disable=wrong-keyword-args\n\n  chunk_capacity = pack_size * max_token_budget\n'
def _patch():
    try:
        import tunix.rl.utils as U
        fp = U.__file__; src = open(fp).read()
        if "_PACKER_FAST" in src:
            print("PACKER-FAST: already applied", flush=True); return
        a = "  def _flush_pack("; b = "  def _mark("
        if src.count(a) != 1 or src.count(b) != 1:
            print("PACKER-FAST: anchors %d/%d (skipped)" % (src.count(a), src.count(b)), flush=True); return
        i = src.index(a); j = src.index(b)
        if not (i < j and "def pack_sequences(" in src[:i]):
            print("PACKER-FAST: anchor order unexpected (skipped)", flush=True); return
        new = src[:i] + FAST_BLOCK.lstrip("\n") + "\n" + src[j:]
        compile(new, fp, "exec"); open(fp, "w").write(new)
        print("PACKER-FAST: replaced _flush_pack/_emit with numpy-first chunk assembly (%d chars removed)" % (j - i), flush=True)
    except Exception as e:
        print("PACKER-FAST: failed %r" % (e,), flush=True)
_patch()
