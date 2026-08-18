"""Vocab-sharded sampling + logprobs for the fused decode window.

WHY. sampling.py force-replicates the logits over the vocab dim every decode
iteration ("Unshard the logits explicitly...", with an upstream TODO asking
for exactly this change): a (batch, 151936) f32 all-gather (~78 MB per DP
rank per iteration), followed by topk_mask/topp_mask -- 32-33 full vocab-axis
reduction sweeps each (the ~800us/call while-loops in every profile) -- and a
full-vocab categorical, all computed redundantly on every TP device. The
logprobs path (gather_logprobs) repeats full-vocab log_softmax + top-N + a
rank reduction on the same replicated array.

WHAT. Rebinds `sample` and `compute_and_gather_logprobs` with vocab-sharded
implementations under jax.shard_map:
  - per TP shard: one lax.top_k over vocab/8, all-gather only the top-64
    candidate (value, index) pairs (~0.4 MB vs ~78 MB);
  - exact top-k cutoff, top-p cutoff, and categorical on the merged
    candidates. EXACTNESS: temperature scaling is monotone (order preserved);
    the union of per-shard top-64 contains the global top-k for k <= 64; the
    cutoff mask keeps values >= cutoff exactly like topk_mask (ties
    included); masked/non-candidate entries underflow to exactly zero
    probability in f32 softmax, so top-p mass sums and the categorical
    distribution equal the full-vocab computation. Verified on CPU with an
    8-device mesh: kept sets identical on adversarial ties, greedy exact,
    draw distribution within statistical noise, logprob top-N ids exact,
    values within 1e-6, ranks exact (scratchpad/shard_proto.py).
  - logprobs: log_softmax via pmax/psum, top-N from the same candidates
    (log_softmax is monotone in logits), token logprob + rank via psum.

WHAT CHANGES NUMERICALLY. The categorical RNG stream differs (noise drawn
over candidates, not the full vocab), so sampled tokens are not bitwise equal
to the old path -- distribution-equal only. Logprob VALUES shift ~1e-6 from
reduction order. Validity gates for the run: LENDIST truncation band and
reward band, not bitwise reward equality.

GUARDS.
  - lax.cond falls back to the ORIGINAL implementation for the whole batch
    if any row has top_k <= 0 or top_k > 64 (e.g. eval "liberal" top_k=2000)
    or top_p <= 0; both branches are compiled, only the taken one runs.
  - compute_and_gather_logprobs falls back statically if max_logprobs > 48.
  - env SHARDED_SAMPLING=0 disables the rebinding entirely at patch time...
    is checked at RUNTIME by the wrappers, so the kill-switch works from the
    manifest without re-patching.
  - `sample`'s ret_logits in the sharded path is the RAW (still-sharded)
    logits: valid for logprobs_mode="raw" (our config). A processed_logprobs
    deployment must disable this patch.

FAIL-OPEN. Any missing anchor prints FAILED and exits 0.
Prints exactly one of:
    SHARDED-SAMPLING PATCH: APPLIED
    SHARDED-SAMPLING PATCH: FAILED <reason>
"""

import ast
import os
import sys

SITE = os.environ.get("PATCH_ROOT", "/usr/local/lib/python3.12/site-packages")
ROOT = os.path.join(SITE, "tpu_inference")
SP = os.path.join(ROOT, "layers/jax/sample/sampling.py")


def fail(msg):
    print(f"SHARDED-SAMPLING PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(SP):
    fail(f"{SP} not found")
src = open(SP, encoding="utf-8").read()

if "_SHARDED_SAMPLING" in src:
    print("SHARDED-SAMPLING PATCH: already patched")
    sys.exit(0)

for needle in ("def sample(", "def compute_and_gather_logprobs(",
               "def gather_logprobs(", "def _apply_sampling_transforms(",
               "ShardingAxisName"):
    if needle not in src:
        fail(f"anchor '{needle}' not present in sampling.py")

APPEND = '''

# --- _SHARDED_SAMPLING -------------------------------------------------------
# Appended by patch_sharded_sampling.py (see its docstring). Rebinds `sample`
# and `compute_and_gather_logprobs` below; originals are kept as fallbacks.
try:
  import os as _ss_os
  from jax import lax as _ss_lax
  try:
    from jax import shard_map as _ss_shard_map          # jax >= 0.8
  except ImportError:
    from jax.experimental.shard_map import shard_map as _ss_shard_map

  _SS_CAND = 64
  _SS_NEG = -1e12

  _ss_orig_sample = sample
  _ss_orig_cgl = compute_and_gather_logprobs

  def _ss_axes(mesh):
    """(batch_axes, vocab_axis) restricted to axes present in the mesh."""
    def present(name):
      if name is None:
        return ()
      if isinstance(name, (tuple, list)):
        return tuple(a for a in name if a in mesh.axis_names)
      return (name,) if name in mesh.axis_names else ()
    batch = present(ShardingAxisName.ATTN_DATA)
    vocab = present(ShardingAxisName.MODEL)
    return batch, (vocab[0] if vocab else None)

  def _ss_merged_candidates(x_blk, vocab_axis):
    v_shard = x_blk.shape[-1]
    midx = _ss_lax.axis_index(vocab_axis)
    vals, idx = _ss_lax.top_k(x_blk, _SS_CAND)
    gidx = idx + midx * v_shard
    av = _ss_lax.all_gather(vals, vocab_axis)
    ai = _ss_lax.all_gather(gidx, vocab_axis)
    n = av.shape[0]
    av = jnp.transpose(av, (1, 0, 2)).reshape(-1, n * _SS_CAND)
    ai = jnp.transpose(ai, (1, 0, 2)).reshape(-1, n * _SS_CAND)
    return av, ai

  def _ss_greedy(cv, ci):
    m = jnp.max(cv, axis=-1, keepdims=True)
    big = jnp.iinfo(jnp.int32).max
    return jnp.min(jnp.where(cv >= m, ci, big), axis=-1)

  def _ss_kth(cv, k):
    sv = -jnp.sort(-cv, axis=-1)
    kk = jnp.clip(k, 1, cv.shape[-1]) - 1
    return jnp.take_along_axis(sv, kk[:, None], axis=-1)

  def _ss_topp_cutoff(probs, p):
    sp = -jnp.sort(-probs, axis=-1)
    cum = jnp.cumsum(sp, axis=-1)
    jstar = jnp.argmax(cum >= p[:, None], axis=-1)
    return jnp.take_along_axis(sp, jstar[:, None], axis=-1)

  def _ss_sample_block(vocab_axis, batch_axes, rng, x, temperature, top_k,
                       top_p, do_argmax_only):
    # x: (b_local, v_shard) f32
    didx = 0
    for i, a in enumerate(batch_axes):
      didx = didx * _ss_lax.axis_size(a) + _ss_lax.axis_index(a)
    xt = x / temperature[:, None]
    cv, ci = _ss_merged_candidates(xt, vocab_axis)
    greedy = _ss_greedy(cv, ci)
    if do_argmax_only:
      return _ss_lax.pmax(greedy, vocab_axis)
    kth = _ss_kth(cv, top_k)
    mk = jnp.where((top_k > 0)[:, None] & ~(cv >= kth), _SS_NEG, cv)
    probs = jax.nn.softmax(mk, axis=-1)
    pcut = _ss_topp_cutoff(probs, top_p)
    mp = jnp.where((top_p < 1.0)[:, None] & ~(probs >= pcut), _SS_NEG, mk)
    r = jax.random.fold_in(rng, didx)
    choice = jax.random.categorical(r, mp)
    sampled = jnp.take_along_axis(ci, choice[:, None], axis=-1)[:, 0]
    out = jnp.where(temperature < _SAMPLING_EPS, greedy, sampled)
    # Values are identical on every vocab shard (built from all-gathered
    # candidates); the pmax is an identity that makes that replication
    # provable to shard_map's output check on every jax version.
    return _ss_lax.pmax(out, vocab_axis)

  def _ss_sample(rng, mesh, logits, meta):
    batch_axes, vocab_axis = _ss_axes(mesh)
    bspec = P(batch_axes if batch_axes else None)
    lspec = P(batch_axes if batch_axes else None, vocab_axis)
    logits = jax.lax.with_sharding_constraint(logits,
                                              NamedSharding(mesh, lspec))

    def blk(rng, x, t, k, p):
      return _ss_sample_block(vocab_axis, batch_axes, rng, x, t, k, p,
                              do_argmax_only=not meta.do_sampling)

    tokens = _ss_shard_map(blk, mesh=mesh,
                           in_specs=(P(), lspec, bspec, bspec, bspec),
                           out_specs=bspec)(
                               rng, logits, meta.temperature.astype(
                                   jnp.float32), meta.top_k, meta.top_p)
    tokens = jax.lax.with_sharding_constraint(tokens, NamedSharding(mesh, P()))
    return tokens, logits

  def sample(rng, mesh, logits, tpu_sampling_metadata):
    if _ss_os.environ.get("SHARDED_SAMPLING", "1") == "0":
      return _ss_orig_sample(rng, mesh, logits, tpu_sampling_metadata)
    meta = tpu_sampling_metadata
    if meta._cache_collision_dummy is not None:
      logits = logits + 0 * jnp.sum(meta._cache_collision_dummy)
    _, vocab_axis = _ss_axes(mesh)
    if vocab_axis is None:
      return _ss_orig_sample(rng, mesh, logits, meta)
    logits = logits.astype(jnp.float32)
    if not meta.do_sampling:
      return _ss_sample(rng, mesh, logits, meta)
    safe = jnp.all((meta.top_k > 0) & (meta.top_k <= _SS_CAND)
                   & (meta.top_p > 0.0))
    return _ss_cond_sample(safe, rng, mesh, logits, meta)

  def _ss_cond_sample(safe, rng, mesh, logits, meta):
    # mesh/meta are closed over rather than passed as lax.cond operands
    # (Mesh is not a valid JAX type). Both branches constrain ret_logits to
    # the SHARDED spec so cond-output unification cannot reintroduce the
    # vocab all-gather on the hot path (the fallback branch has already paid
    # its own gather internally; re-constraining its output is cheap).
    batch_axes, vocab_axis = _ss_axes(mesh)
    lspec = P(batch_axes if batch_axes else None, vocab_axis)

    def tru(op):
      r, lg = op
      return _ss_sample(r, mesh, lg, meta)

    def fls(op):
      r, lg = op
      toks, ret = _ss_orig_sample(r, mesh, lg, meta)
      ret = jax.lax.with_sharding_constraint(ret, NamedSharding(mesh, lspec))
      return toks, ret

    return jax.lax.cond(safe, tru, fls, (rng, logits))

  def compute_and_gather_logprobs(logits, next_tokens, max_logprobs):
    if (_ss_os.environ.get("SHARDED_SAMPLING", "1") == "0"
        or max_logprobs > 48):
      return _ss_orig_cgl(logits, next_tokens, max_logprobs)
    # Mesh is implicit here (no arg); derive from the array sharding if
    # available, else fall back.
    sh = getattr(logits, "sharding", None)
    mesh = getattr(sh, "mesh", None)
    if mesh is None or getattr(mesh, "empty", False):
      return _ss_orig_cgl(logits, next_tokens, max_logprobs)
    batch_axes, vocab_axis = _ss_axes(mesh)
    if vocab_axis is None:
      return _ss_orig_cgl(logits, next_tokens, max_logprobs)
    bspec = P(batch_axes if batch_axes else None)
    lspec = P(batch_axes if batch_axes else None, vocab_axis)
    logits = jax.lax.with_sharding_constraint(
        logits, NamedSharding(mesh, lspec))

    def blk(x, toks):
      v_shard = x.shape[-1]
      midx = _ss_lax.axis_index(vocab_axis)
      gmax = _ss_lax.pmax(jnp.max(x, axis=-1), vocab_axis)
      se = _ss_lax.psum(jnp.sum(jnp.exp(x - gmax[:, None]), axis=-1),
                        vocab_axis)
      lse = gmax + jnp.log(se)
      rv, ri = _ss_merged_candidates(x, vocab_axis)
      nv, ni = _ss_lax.top_k(rv, max_logprobs)
      top_ids = jnp.take_along_axis(ri, ni, axis=-1)
      top_lps = nv - lse[:, None]
      owner = (toks >= midx * v_shard) & (toks < (midx + 1) * v_shard)
      col = jnp.clip(toks - midx * v_shard, 0, v_shard - 1)
      lval = jnp.take_along_axis(x, col[:, None], axis=-1)[:, 0]
      tok_logit = _ss_lax.psum(jnp.where(owner, lval, 0.0), vocab_axis)
      tok_lp = tok_logit - lse
      rank = _ss_lax.psum(jnp.sum(x >= tok_logit[:, None], axis=-1),
                          vocab_axis)
      idx = jnp.concatenate((toks[:, None], top_ids), axis=1).astype(jnp.int32)
      lps = jnp.concatenate((tok_lp[:, None], top_lps), axis=1)
      # identity pmax: values already replicated across vocab shards; makes
      # the replication provable to shard_map's output check.
      return (_ss_lax.pmax(idx, vocab_axis), _ss_lax.pmax(lps, vocab_axis),
              _ss_lax.pmax(rank, vocab_axis))

    idx, lps, rank = _ss_shard_map(
        blk, mesh=mesh,
        in_specs=(lspec, bspec),
        out_specs=(P(batch_axes if batch_axes else None, None),
                   P(batch_axes if batch_axes else None, None), bspec),
        )(logits.astype(jnp.float32), next_tokens)
    return LogprobsTensors(idx, lps, rank)
except Exception as _e:
  print("SHARDED-SAMPLING ERROR (install): %r" % (_e,), flush=True)
# --- end _SHARDED_SAMPLING ---------------------------------------------------
'''

new = src + APPEND
try:
    ast.parse(new)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")

open(SP, "w", encoding="utf-8").write(new)

v = open(SP, encoding="utf-8").read()
problems = []
if "_ss_sample_block" not in v or "_ss_orig_cgl" not in v:
    problems.append("appended block incomplete")
if v.count("def sample(") != 2:
    problems.append("expected original + rebound sample definitions")
try:
    ast.parse(v)
except SyntaxError as e:
    problems.append(f"reparse failed: {e}")
if problems:
    fail("; ".join(problems))
print("SHARDED-SAMPLING PATCH: APPLIED")