"""Port tunix chunked log-probs (tunix b02231b8 / PR#4409's tunix half) onto the image's tunix.

WHY. maxtext PR #4409 (patch_pr4409) adds `skip_lm_head` + `compute_final_logits` on the
MaxText side and passes `compute_logps_chunk_size` into tunix's RLTrainingConfig — but the
image ships google-tunix 0.1.7-snapshot, which predates tunix commit b02231b8 ("enable
chunked logps computation"): RLTrainingConfig has no such field (TypeError at startup, run
g4n3s0840) and common.compute_per_token_logps always materializes the full
[B, S, 262144] f32 logits (Fix #21: tmbs>1 OOM at 8192 tokens).

WHAT (all default-off, byte-identical behaviour when chunk_size == 0):
  rl/rl_cluster.py   RLTrainingConfig.compute_logps_chunk_size: int = 0; old-logps call
                     passes chunk_size.
  rl/common.py       compute_per_token_logps(..., chunk_size=0): when >0 calls the model with
                     skip_lm_head=True, then projects+log-softmaxes hidden states in sequence
                     chunks under lax.scan+remat (_compute_chunked_logps), returning
                     per-token logps and per-token ENTROPY in the `logits` return slot
                     (2-D == entropy; callers check ndim).
  rl/algo_core.py    grpo_loss_fn passes chunk_size from kwargs and uses the precomputed
                     entropy when the second return is 2-D.
  rl/grpo/grpo_learner.py  loss lambda forwards compute_logps_chunk_size from the cluster's
                     RLTrainingConfig.

FAIL-OPEN. Every edit asserts its anchor; any miss prints FAILED and leaves ALL files
untouched (edits are staged in memory and written only if all succeed + compile).
Prints exactly one of:
    TUNIX-CHUNKED-LOGPS PATCH: APPLIED
    TUNIX-CHUNKED-LOGPS PATCH: already patched
    TUNIX-CHUNKED-LOGPS PATCH: FAILED <reason>
"""
import importlib.util
import os
import sys

MARK = "_compute_chunked_logps"


def fail(msg):
  print(f"TUNIX-CHUNKED-LOGPS PATCH: FAILED {msg}")
  sys.exit(0)


def rep(src, old, new, what):
  n = src.count(old)
  if n != 1:
    fail(f"{what}: anchor count {n} != 1")
  return src.replace(old, new, 1)


CHUNK_FN = '''

def _compute_chunked_logps(model, hidden_states, target_ids, temperature, chunk_size):
  """Per-token log-probs + entropy from hidden states in sequence chunks (scan+remat).

  Peak live logits are [B, chunk_size, V] instead of [B, S, V]. Requires the model to
  expose `compute_final_logits(hidden)` (maxtext PR #4409 TunixMaxTextAdapter).
  Returns (per_token_logps [B, S], per_token_entropy [B, S]).
  """
  batch_size, seq_len, hidden_dim = hidden_states.shape
  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:
    hidden_states = jnp.pad(hidden_states, ((0, 0), (0, pad_len), (0, 0)))
    target_ids = jnp.pad(target_ids, ((0, 0), (0, pad_len)))
  num_chunks = (seq_len + pad_len) // chunk_size
  hs = jnp.swapaxes(hidden_states.reshape(batch_size, num_chunks, chunk_size, hidden_dim), 0, 1)
  ids = jnp.swapaxes(target_ids.reshape(batch_size, num_chunks, chunk_size), 0, 1)

  @nnx.remat
  def logp_step(carry, xs):
    hs_chunk, ids_chunk = xs
    logits_chunk = model.compute_final_logits(hs_chunk).astype(jnp.float32)
    if temperature != 0.0 and temperature != 1.0:
      logits_chunk = logits_chunk / temperature
    logps_chunk = selective_log_softmax(logits_chunk, ids_chunk)
    log_probs = jax.nn.log_softmax(logits_chunk, axis=-1)
    entropy_chunk = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
    return carry, (logps_chunk, entropy_chunk)

  _, (logps, entropy) = jax.lax.scan(logp_step, None, (hs, ids))
  logps = jnp.swapaxes(logps, 0, 1)[:, :seq_len]
  entropy = jnp.swapaxes(entropy, 0, 1)[:, :seq_len]
  return logps, entropy

'''


def main():
  spec = importlib.util.find_spec("tunix")
  if spec is None or not spec.submodule_search_locations:
    fail("cannot locate tunix")
  root = list(spec.submodule_search_locations)[0]
  paths = {k: os.path.join(root, v) for k, v in {
      "common": "rl/common.py", "cluster": "rl/rl_cluster.py",
      "algo": "rl/algo_core.py", "grpo": "rl/grpo/grpo_learner.py"}.items()}
  for p in paths.values():
    if not os.path.exists(p):
      fail(f"{p} missing")
  src = {k: open(p).read() for k, p in paths.items()}
  if MARK in src["common"]:
    print("TUNIX-CHUNKED-LOGPS PATCH: already patched")
    return

  # ---- common.py ----
  c = src["common"]
  c = rep(c, '''    return_logits: bool = False,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
    temperature: float = 1.0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Computes the per-token log probabilities.''', '''    return_logits: bool = False,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
    temperature: float = 1.0,
    chunk_size: int = 0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Computes the per-token log probabilities.''', "common signature")
  c = rep(c, '''  logits, _ = model(input_tokens, **model_kwargs)

  if segment_ids is not None:
    # Packed Mode: Evaluate the full sequence (mixed prompts + completions).
    # Since predicting token[i] requires logit[i-1], we skip the first token.
    # This shrinks the output shape to [Batch, FullSeqLen - 1]
    logits_to_keep = input_tokens.shape[1] - 1
  else:
    logits_to_keep = completion_tokens.shape[1]

  logits = logits[:, -logits_to_keep - 1 : -1, :]
  if temperature != 0.0 and temperature != 1.0:
    logits /= temperature

  input_tokens_to_keep = input_tokens[:, -logits_to_keep:]
  per_token_logps = selective_log_softmax(logits, input_tokens_to_keep)
''', '''  if chunk_size > 0:
    # Chunked path (maxtext PR #4409 / tunix b02231b8): the model returns hidden
    # states instead of logits; project per chunk below.
    model_kwargs["skip_lm_head"] = True
  logits, _ = model(input_tokens, **model_kwargs)

  if segment_ids is not None:
    # Packed Mode: Evaluate the full sequence (mixed prompts + completions).
    # Since predicting token[i] requires logit[i-1], we skip the first token.
    # This shrinks the output shape to [Batch, FullSeqLen - 1]
    logits_to_keep = input_tokens.shape[1] - 1
  else:
    logits_to_keep = completion_tokens.shape[1]

  input_tokens_to_keep = input_tokens[:, -logits_to_keep:]
  if chunk_size > 0:
    hidden = logits[:, -logits_to_keep - 1 : -1, :]
    per_token_logps, logits = _compute_chunked_logps(
        model, hidden, input_tokens_to_keep, temperature, chunk_size
    )  # NOTE: `logits` now holds per-token ENTROPY [B, L] (2-D), not logits.
  else:
    logits = logits[:, -logits_to_keep - 1 : -1, :]
    if temperature != 0.0 and temperature != 1.0:
      logits /= temperature
    per_token_logps = selective_log_softmax(logits, input_tokens_to_keep)
''', "common body")
  c = rep(c, '''    if return_logits:
      logits = jnp.pad(logits, ((0, 0), (1, 0), (0, 0)), constant_values=0.0)
''', '''    if return_logits:
      if logits.ndim == 2:  # chunked path: entropy [B, L]
        logits = jnp.pad(logits, ((0, 0), (1, 0)), constant_values=0.0)
      else:
        logits = jnp.pad(logits, ((0, 0), (1, 0), (0, 0)), constant_values=0.0)
''', "common segment pad")
  # insert helper BEFORE the jit decorator of compute_per_token_logps, and make
  # chunk_size a static arg (python control flow on it inside the jitted fn).
  DECO = '''@partial(
    jax.jit,
    static_argnames=(
        "pad_id",
        "eos_id",
        "stop_gradient",
        "return_logits",
        "temperature",
    ),
)
def compute_per_token_logps(
'''
  DECO_NEW = '''@partial(
    jax.jit,
    static_argnames=(
        "pad_id",
        "eos_id",
        "stop_gradient",
        "return_logits",
        "temperature",
        "chunk_size",
    ),
)
def compute_per_token_logps(
'''
  c = rep(c, DECO, CHUNK_FN + DECO_NEW, "common helper insert + static chunk_size")

  # ---- rl_cluster.py ----
  r = src["cluster"]
  r = rep(r, '''  rollout_micro_batch_size: int | None = None
  compute_logps_micro_batch_size: int | None = None
''', '''  rollout_micro_batch_size: int | None = None
  compute_logps_micro_batch_size: int | None = None
  # Sequence chunk size for log-prob computation (0 = full logits). Requires a
  # model exposing skip_lm_head/compute_final_logits (maxtext PR #4409).
  compute_logps_chunk_size: int = 0
''', "cluster RLTrainingConfig field")
  r = rep(r, '''                stop_gradient=True,
                return_logits=False,
                temperature=temperature,
            )
        )
      actor_per_token_logps = jnp.concatenate(outs, axis=0)''', '''                stop_gradient=True,
                return_logits=False,
                temperature=temperature,
                chunk_size=self.cluster_config.training_config.compute_logps_chunk_size,
            )
        )
      actor_per_token_logps = jnp.concatenate(outs, axis=0)''', "cluster old-logps call")

  # ---- algo_core.py (grpo_loss_fn) ----
  a = src["algo"]
  a = rep(a, '''      segment_positions=getattr(train_example, "segment_positions", None),
      temperature=algo_config.temperature,
  )
  per_token_logps = jnp.astype(per_token_logps, jnp.float32)''', '''      segment_positions=getattr(train_example, "segment_positions", None),
      temperature=algo_config.temperature,
      chunk_size=kwargs.get("compute_logps_chunk_size", 0),
  )
  per_token_logps = jnp.astype(per_token_logps, jnp.float32)''', "algo grpo call")
  a = rep(a, '''  token_entropy = compute_entropy_from_logits(logits)
  entropy_loss = common.aggregate_loss(
      token_entropy, completion_mask, loss_aggregation_mode
  )
  aux["entropy"] = entropy_loss

  return loss, aux
''', '''  # chunked log-probs path returns per-token entropy (2-D) in the logits slot
  token_entropy = logits if logits.ndim == 2 else compute_entropy_from_logits(logits)
  entropy_loss = common.aggregate_loss(
      token_entropy, completion_mask, loss_aggregation_mode
  )
  aux["entropy"] = entropy_loss

  return loss, aux
''', "algo grpo entropy")

  # ---- grpo_learner.py ----
  g = src["grpo"]
  g = rep(g, '''        algo_config=self.algo_config,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
    )
''', '''        algo_config=self.algo_config,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
        compute_logps_chunk_size=getattr(
            self.rl_cluster.cluster_config.training_config,
            "compute_logps_chunk_size", 0),
    )
''', "grpo loss lambda")

  new = {"common": c, "cluster": r, "algo": a, "grpo": g}
  for k, s in new.items():
    try:
      compile(s, paths[k], "exec")
    except SyntaxError as e:
      fail(f"{k}: syntax error after patch: {e}")
  for k, s in new.items():
    open(paths[k], "w").write(s)
  print(f"TUNIX-CHUNKED-LOGPS PATCH: APPLIED -> {root}")


if __name__ == "__main__":
  main()
