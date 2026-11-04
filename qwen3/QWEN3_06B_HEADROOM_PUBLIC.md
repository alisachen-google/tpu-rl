# Qwen3-0.6B public headroom proposal

This is the shareable version of the headroom analysis. It contains:

- No links to private repositories, jobs, logs, buckets, or traces.
- No measured wall times, throughput, reward, or hardware-counter values.
- Only code pointers to public upstream repositories.
- Pseudocode for proposals that are not implemented in those public sources.

The performance model uses a normalized baseline of **100 arbitrary wall-time
units**. These units are not seconds and cannot be converted into the timing of
a particular job.

## Baseline contract

The baseline is a Qwen3-0.6B GRPO workload with:

- A fixed TPU fleet using TP4 x DP32 for rollout.
- Fixed prompts, generations per prompt, sampling policy, checkpoint, global
  batch, sequence limits, and output cap.
- Continuous decode enabled with a relatively shallow fused window.
- A fixed KV-cache page size.
- State-query-based DP request routing.
- A sampler that can fall back to replicated full-vocabulary processing.
- Synchronous output conversion and reward work.
- Dense padded actor training and general attention.
- KV-cache lifecycle RPCs and layer-by-layer parameter slicing during weight
  synchronization.

The baseline is intentionally a configuration contract rather than a timing
claim. A candidate is valid only if it processes equivalent work and remains
inside predeclared output-length, truncation, and reward bands.

## Estimated opportunity

| Planning case | Estimated remaining normalized time | Approximate speedup |
|---|---:|---:|
| Conservative | 70-75 units | 1.3-1.4x |
| Central | 55-65 units | 1.5-1.8x |
| Aggressive | 45-55 units | 1.8-2.2x |

The ranges overlap because several proposals shorten the same critical path.
They must not be added as independent wins.

## Public upstream map

The links below are pinned public source snapshots, except for explicitly
identified pull requests.

| Area | Public source |
|---|---|
| Continuous-decode state and loop condition | [TPU Inference `decode_loop.py`](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/runner/decode_loop.py#L83-L85) and [loop condition](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/runner/decode_loop.py#L252-L260) |
| Continue-decode compiler fix | [TPU Inference PR #3374](https://github.com/vllm-project/tpu-inference/pull/3374) |
| EOS-check interval | [TPU Inference PR #3174](https://github.com/vllm-project/tpu-inference/pull/3174) |
| RPA block-size override | [TPU Inference PR #3102](https://github.com/vllm-project/tpu-inference/pull/3102) |
| Batched-token padding boundary | [TPU Inference PR #3466](https://github.com/vllm-project/tpu-inference/pull/3466) |
| DP request routing | [TPU Inference `dp_scheduler.py`](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/core/sched/dp_scheduler.py#L701-L764) |
| Sampling transforms | [TPU Inference `sampling.py`](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/layers/jax/sample/sampling.py#L68-L131) |
| Host token extraction | [TPU Inference `tpu_runner.py`](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/runner/tpu_runner.py#L146-L275) |
| Sequence packing | [Tunix `pack_sequences`](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/rl/utils.py#L476-L740) |
| Serial reward loop | [Tunix `SequenceRewardManager`](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/rl/reward_manager.py#L93-L160) |
| Sampler update and KV lifecycle | [Tunix `VllmSampler.update_params`](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/generate/vllm_sampler.py#L185-L246) |
| Detokenization | [Tunix `VllmSampler.detokenize`](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/generate/vllm_sampler.py#L407-L443) |
| Scanned-layer unroll | [Tunix `_unroll_scanned_layers`](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/generate/utils.py#L464-L520) |
| TPU flash/splash attention | [MaxText `attention_op.py`](https://github.com/AI-Hypercomputer/maxtext/blob/2586d60b7467a84eb9ef8caad334f733d34a545e/src/maxtext/layers/attention_op.py#L570-L610) and [dispatch](https://github.com/AI-Hypercomputer/maxtext/blob/2586d60b7467a84eb9ef8caad334f733d34a545e/src/maxtext/layers/attention_op.py#L1490-L1505) |

At the time these links were verified, TPU Inference PRs #3374, #3102, and
#3174 were merged. PR #3466 was still open, so its behavior should be checked
against the selected runtime revision.

## Detailed evidence and proposals

### 1. Amortize fused-decode boundaries

Evidence for headroom:

- The internal profile showed repeated host engine windows around much shorter
  fused device programs.
- Public TPU Inference source makes the decode window a bounded loop and shows
  that EOS-check policy participates in its condition.
- Fixed scheduler, transfer, and output handling costs recur at every window
  boundary.

Public evidence:

- [`decode_loop.py` state update](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/runner/decode_loop.py#L68-L85)
- [`decode_loop.py` condition](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/runner/decode_loop.py#L252-L260)
- [Flax NNX continue-decode fix, PR #3374](https://github.com/vllm-project/tpu-inference/pull/3374)
- [EOS-check interval, PR #3174](https://github.com/vllm-project/tpu-inference/pull/3174)

Proposed configuration pseudocode:

```python
candidate.max_decode_steps = baseline.max_decode_steps * 4
candidate.max_num_batched_tokens = baseline.max_num_batched_tokens * 2
candidate.kv_page_size = baseline.kv_page_size

assert per_sequence_masking_is_correct
assert host_reads_each_sequence_last_real_token
```

Why it can improve performance:

A deeper useful window amortizes host scheduling, launch, transfer, and output
handling across more tokens. A larger admission budget reduces capacity waves.
Pinning the page size prevents the experiment from silently changing KV-cache
layout at the same time.

Required validation:

- Normalize engine-window count and duration by generated tokens.
- Check useful iterations per fused window.
- Require zero zombie requests and correct EOS retirement.
- Compare prompt identity, output-length distribution, truncation, and reward.

### 2. Avoid padding beyond the legal token budget

Evidence for headroom:

- Static-shape execution requires token buckets, but selecting a bucket above
  the configured maximum can execute padding that the scheduler never intended
  to admit.
- The issue and proposed boundary behavior are public in TPU Inference PR
  #3466.

Public evidence:
[TPU Inference PR #3466](https://github.com/vllm-project/tpu-inference/pull/3466).

Proposed pseudocode:

```python
def build_token_buckets(min_size, max_size, gap):
    buckets = geometric_or_gap_sequence(min_size, max_size, gap)
    exact_max_bucket = ceil_div(max_size, min_size) * min_size
    return sorted(set(buckets + [exact_max_bucket]))

selected = smallest_bucket_at_least(actual_tokens)
assert selected <= exact_max_bucket
```

Why it can improve performance:

The accelerator executes the selected static shape, including padding. An
exact maximum bucket avoids a large discontinuous jump in work near the token
budget boundary.

Required validation:

- Log actual tokens, selected bucket, and padding ratio for every engine step.
- Confirm no legal batch selects a shape above the exact maximum bucket.
- Keep admission decisions identical between control and candidate.

### 3. Replace all-rank request probing with optional round robin

Evidence for headroom:

- Public `DPScheduler` source sends several state queries to ranks before
  `_route_and_forward_request` chooses a destination.
- A homogeneous offline GRPO batch gives each rank similar work, reducing the
  value of per-request global state discovery.
- Scheduler IPC runs on the host-critical rollout path.

Public evidence:

- [Rank-state query path](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/core/sched/dp_scheduler.py#L701-L739)
- [Route and forward call site](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/core/sched/dp_scheduler.py#L760-L776)

Proposed pseudocode:

```python
def route(request):
    if routing_policy == "round_robin":
        rank = next_rank % dp_size
        next_rank += 1
    else:
        rank = find_best_rank_from_global_state(request)
    forward(rank, request)
```

Optional configuration:

```text
routing_policy = round_robin
prefix_caching = disabled_when_prompts_do_not_repeat
```

Why it can improve performance:

For the homogeneous case, route selection becomes constant host work instead
of request-by-rank communication. Disabling unproductive prefix caching also
avoids computed-block probes and cache bookkeeping.

Required validation:

- Count scheduler messages per admitted request.
- Compare admitted tokens and queue depth across ranks.
- Reject the policy if it materially worsens tail imbalance.
- Re-enable adaptive routing for heterogeneous or online traffic.

### 4. Keep sampling vocabulary-sharded

Evidence for headroom:

- Public sampling source explicitly unshards logits before applying top-k and
  top-p transforms.
- The requested candidate set is much smaller than the vocabulary.
- Padding/default metadata can make a whole-batch safety guard choose the
  replicated fallback even when all active rows support the sharded path.

Public evidence:

- [Sampling transforms](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/layers/jax/sample/sampling.py#L68-L104)
- [Explicit unshard in `sample`](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/layers/jax/sample/sampling.py#L106-L137)

Proposed pseudocode:

```python
def sharded_candidates(local_logits, model_axis, candidate_count):
    values, local_ids = lax.top_k(local_logits, candidate_count)
    global_ids = local_ids + axis_index(model_axis) * local_vocab_size
    values = lax.all_gather(values, model_axis)
    global_ids = lax.all_gather(global_ids, model_axis)
    return merge_candidates(values, global_ids)

safe_top_k = where(is_padding_row, candidate_count, top_k)
safe_top_p = where(is_padding_row, 1.0, top_p)
use_sharded = all(safe_top_k <= candidate_count) and all(safe_top_p > 0)
```

Why it can improve performance:

Candidate communication grows with the candidate set rather than the full
vocabulary. Top-k/top-p and categorical sampling then operate on the merged
candidates instead of replicated vocabulary arrays on every TP device.

Required validation:

- Add a sharded-versus-fallback path counter.
- Confirm the large logits unshard is absent for supported parameters.
- Test ties, greedy decoding, token logprobs, and token ranks.
- Use distributional output gates because categorical RNG order may change.

### 5. Remove an unused global EOS reduction

Evidence for headroom:

- Public `_update_loop_state` computes `any_hit_eos` as a batch-wide reduction
  every decode iteration.
- The public loop condition bypasses EOS checks when the interval policy makes
  the static window authoritative.
- Carrying an unused global scalar can preserve the collective in the compiled
  loop even when host logic does not consume it.

Public evidence:

- [`any_hit_eos` reduction](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/runner/decode_loop.py#L68-L85)
- [EOS-aware loop condition](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/runner/decode_loop.py#L252-L260)

Proposed pseudocode:

```python
if global_eos_early_exit_is_enabled:
    any_hit_eos = reduce_any(active_mask & is_eos)
else:
    any_hit_eos = False  # compile-time/static branch
```

Why it can improve performance:

A scalar collective has little computation to hide its communication latency,
and it lies on every loop iteration's dependency chain. Removing it is useful
only when the selected semantics provably do not consume the global flag.

Required validation:

- Confirm the scalar collective disappears from the device trace.
- Keep per-row `active_mask` updates unchanged.
- Test mixed EOS positions and full-window exhaustion.
- Never enable this branch with a loop policy that relies on global EOS exit.

### 6. Forward and tune RPA block sizes

Evidence for headroom:

- RPA is the long-context attention kernel on the decode path.
- Public PR #3102 adds environment-controlled decode/prefill/mixed block
  sizes, making the tuning surface explicit.
- An environment variable is useful only if the actual serving entry point
  forwards it to the kernel.

Public evidence:
[TPU Inference PR #3102](https://github.com/vllm-project/tpu-inference/pull/3102).

Proposed pseudocode:

```python
kwargs = existing_attention_kwargs()
if not using_specialized_hd64_kernel:
    if decode_block_sizes_from_env:
        kwargs["d_block_sizes"] = tuple(decode_block_sizes_from_env)
    if prefill_block_sizes_from_env:
        kwargs["p_block_sizes"] = tuple(prefill_block_sizes_from_env)
return ragged_paged_attention(*args, **kwargs)
```

Configuration experiment:

```text
kv_page_size = fixed_across_all_runs
decode_load_tile = sweep_around_baseline
decode_compute_tile = fixed_during_load_tile_sweep
```

Why it can improve performance:

The load tile determines how KV data is streamed and reused. A better tile can
reduce memory-transaction and tiling overhead, but it cannot remove host
scheduler gaps, so its expected end-to-end contribution is bounded.

Required validation:

- Read the effective page and block sizes from the compiled kernel name/HLO.
- Reject any run where the override is silently ignored.
- Compare device time per generated token, not only global step time.
- Sweep one block-size dimension at a time.

### 7. Vectorize host token extraction and batch detokenization

Evidence for headroom:

- Public `tpu_runner.py` extracts valid tokens separately for each request.
- Public Tunix detokenization iterates over outputs on the host.
- Both operations process independent rows and are suitable for batching.

Public evidence:

- [Per-request token extraction](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/runner/tpu_runner.py#L146-L167)
- [Output processing loop](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/runner/tpu_runner.py#L191-L280)
- [Tunix detokenization](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/generate/vllm_sampler.py#L407-L443)

Proposed pseudocode:

```python
is_eos = isin(generated_token_matrix, eos_ids)
first_eos = where(
    any(is_eos, axis=1), argmax(is_eos, axis=1) + 1, generated_token_matrix.width
)
valid_rows = [row[:first_eos[i]] for i, row in enumerate(generated_token_matrix)]
texts = tokenizer.batch_decode(valid_rows)
```

For overlap:

```python
future = bounded_executor.submit(convert_outputs, completed_request_batch)
continue_engine_work()
converted = future.result()  # propagate worker failures before consumption
```

Why it can improve performance:

One contiguous matrix scan replaces many Python/NumPy calls. Batch decoding
reduces tokenizer call overhead, while bounded asynchronous conversion can hide
CPU-only work behind independent rollout activity.

Required validation:

- Compare token arrays, first-EOS positions, and decoded strings exactly.
- Preserve request ordering.
- Track executor queue depth so overlap does not create an end-of-step backlog.
- Propagate worker errors rather than silently dropping outputs.

### 8. Parallelize and overlap reward work

Evidence for headroom:

- Public `SequenceRewardManager` calls each reward function over the completion
  batch serially.
- Sequence-level reward functions are commonly independent across examples.
- Parsing and mathematical verification are CPU work and can overlap request
  retirement when resources are isolated from the scheduler.

Public evidence:
[Tunix `SequenceRewardManager._compute_rewards`](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/rl/reward_manager.py#L114-L160).

Proposed pseudocode:

```python
def parallel_reward(reward_fn, prompts, completions, kwargs, workers):
    chunks = ordered_chunks(prompts, completions, kwargs, workers)
    parts = process_pool.map(reward_fn, chunks)
    return concatenate_in_input_order(parts)

try:
    rewards = parallel_reward(...)
except WorkerError:
    rewards = serial_reward(...)  # correctness-first fallback
```

For asynchronous overlap:

```python
on_request_finished(request):
    reward_future[request.id] = reward_pool.submit(score, request.output)

before_advantage_computation():
    rewards = gather_in_original_order(reward_future)
```

Why it can improve performance:

Processes bypass the Python GIL for independent parsing. Starting reward work
as outputs become available turns part of a serial post-rollout phase into
overlapped work.

Required validation:

- Require exact equality for deterministic rewards.
- Cross-check a configurable sample against the serial path.
- Slice per-example keyword arguments in lockstep with prompts/completions.
- Monitor CPU contention with scheduler and engine-driver processes.
- Fall back permanently to serial execution after a worker failure.

### 9. Pack real tokens and use flash-style attention

Evidence for headroom:

- Variable-length completions create padding in dense train batches.
- Tunix publicly implements first-fit-decreasing sequence packing with segment
  IDs and positions.
- MaxText publicly exposes TPU flash/splash attention and its block settings.

Public evidence:

- [Tunix `pack_sequences`](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/rl/utils.py#L476-L740)
- [MaxText flash/splash configuration](https://github.com/AI-Hypercomputer/maxtext/blob/2586d60b7467a84eb9ef8caad334f733d34a545e/src/maxtext/layers/attention_op.py#L570-L610)
- [MaxText flash-attention dispatch](https://github.com/AI-Hypercomputer/maxtext/blob/2586d60b7467a84eb9ef8caad334f733d34a545e/src/maxtext/layers/attention_op.py#L1490-L1505)

Proposed pseudocode:

```python
for packed_row in first_fit_decreasing(sequences, token_budget):
    token_ids = preallocated_numpy_array(token_budget, pad_id)
    segment_ids = zeros_numpy(token_budget)
    segment_positions = zeros_numpy(token_budget)
    fill_without_crossing_segment_boundaries(
        packed_row, token_ids, segment_ids, segment_positions
    )
    yield to_device_once(token_ids, segment_ids, segment_positions)

trainer.attention_kernel = "flash"
```

Why it can improve performance:

Packing avoids feed-forward and attention work on padding. Attention benefits
especially because dense sequence work grows with sequence area. Building a
packed chunk in NumPy and transferring once also avoids many small JAX or
remote dispatches.

Required validation:

- Assert segment masks block cross-example attention.
- Compare packed and unpacked losses on a deterministic micro-batch.
- Log real-token utilization per packed row.
- Confirm the intended flash/splash custom call appears in the compiled trace.

### 10. Remove unnecessary KV lifecycle and layer-slicing overhead

Evidence for headroom:

- Public Tunix `update_params` deletes and reinitializes the KV cache around
  weight synchronization.
- Public `_unroll_scanned_layers` slices each layer in a Python loop.
- Both mechanisms can generate control-plane work even though neither changes
  model mathematics.

Public evidence:

- [KV lifecycle in `update_params`](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/generate/vllm_sampler.py#L185-L246)
- [Layer-by-layer unroll](https://github.com/google/tunix/blob/7e83930703760e5158a4f867783e127b2b655807/tunix/generate/utils.py#L464-L520)

Proposed pseudocode:

```python
if live_kv_pages == 0 and enough_hbm_for_weight_update:
    keep_kv_allocation = True
else:
    delete_kv_cache()

reshard_and_update_weights()

if not keep_kv_allocation:
    reinitialize_kv_cache()
```

Compile scanned-layer extraction:

```python
@jit(static_argnums=(1,))
def unroll_all_layers(value, layer_axis):
    return tuple(take(value, i, axis=layer_axis)
                 for i in range(value.shape[layer_axis]))
```

Why it can improve performance:

Retaining an empty allocation avoids global lifecycle RPCs and allocator
churn. Compiled unrolling replaces many host-dispatched slices with a small
number of compiled operations.

Required validation:

- Assert no request owns a KV page at synchronization time.
- Verify HBM headroom through the transient update.
- Compare parameter trees and first post-sync logits.
- Count RPCs and host dispatches in the sync phase.

### 11. Explore TP2 and TP1 only after host-path cleanup

Evidence for headroom:

- Lower TP removes tensor-parallel collectives.
- Lower TP also makes each device stream more model weights, reduces local
  batch reuse, and increases DP-engine fan-out on a fixed fleet.
- Qwen3-0.6B has eight KV heads, so TP above eight has limited useful KV
  sharding.

Proposed experiment pseudocode:

```python
for topology in [(TP2, DP64), (TP1, DP128)]:
    assert topology.total_devices == reference.total_devices
    assert workload_and_sampler_are_fixed
    run_profiled_scout(topology)
    accept_only_if(
        normalized_time_per_token_improves
        and host_tail_is_healthy
        and semantic_gates_pass
    )
```

At TP1, use a local candidate sampler with the same semantics as the sharded
candidate path; otherwise the experiment confounds topology with a sampler
implementation change.

Why it could improve performance:

For a small model, collective latency may cost more than finer weight
partitioning saves. TP2 or TP1 can shorten the device dependency chain, but
extra HBM traffic and host fan-out may erase the benefit.

Required validation:

- Hold the fleet, prompts, sampling policy, page size, decode window, token
  budget, and routing policy fixed.
- Report normalized time per generated token and latency distributions.
- Track host CPU, queue depth, and DP imbalance.
- Treat output/reward distributions as gates because TP can shift sampling
  numerics.

## Recommended experiment sequence

1. Produce a clean candidate rollout profile with trainer profiling disabled.
2. A/B decode-window depth and token-bucket behavior.
3. A/B routing and prefix-cache policy.
4. Verify the sharded sampler is selected for active and padded mixtures.
5. Remove the unused EOS collective and sweep RPA tiling independently.
6. Add vectorized extraction, batched detokenization, and bounded overlap.
7. Add reward parallelism with a strict serial oracle.
8. Enable packing and flash attention with deterministic loss checks.
9. Optimize synchronization only after asserting KV ownership and HBM safety.
10. Scout TP2, then TP1, without changing any other rollout mechanism.

## Adoption gates

- Normalize performance by generated tokens and report a multi-step
  distribution.
- Keep prompts and input-length distributions identical.
- Predeclare acceptable output-length, truncation, and reward bands.
- Verify effective page size, RPA tiling, and sampler branch in the profile.
- Require zero correctness mismatches from async or fast paths.
- Keep an environment/config kill switch for each non-upstream proposal.
- Do not infer patch-level contribution from a bundled endpoint comparison.

## Disclosure note

This document contains only normalized estimates, public upstream links, and
pseudocode. It does not provide enough information to reconstruct the timing
or identity of a private benchmark job.
