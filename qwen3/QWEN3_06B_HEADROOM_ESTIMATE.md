# Qwen3-0.6B estimated performance headroom

This document intentionally contains no wall-clock timings, throughput values,
rewards, token counts, or device-counter values from production benchmark jobs.
All quantitative values below are normalized planning estimates.

The baseline is defined as **100 arbitrary wall-time units**. These units are
not seconds and should not be converted back into a specific run result.

## Executive estimate

For the software changes considered here, a reasonable planning range is:

- Conservative: roughly **1.3-1.4x** end-to-end speedup.
- Central estimate: roughly **1.5-1.8x**.
- Aggressive, if rollout and host orchestration both scale well: roughly
  **1.8-2.2x**.

The central thesis is that this is primarily a software and orchestration
opportunity, not a larger-hardware opportunity. Both configurations use the
same fixed TPU fleet and the same TP4 x DP32 rollout topology. Most estimated
headroom comes from reducing rollout boundaries, scheduler traffic,
full-vocabulary work, and CPU postprocessing. Packing and synchronization
improvements provide a smaller but still meaningful second source of savings.

## Estimation model

The following phase budget is deliberately broad. It represents what we would
expect for a small-model, long-generation GRPO workload before the proposed
optimizations.

| Phase | Estimated baseline share | Plausible phase reduction | Estimated contribution |
|---|---:|---:|---:|
| Rollout and generation | 70-80 units | 30-45% | 20-35 units |
| Actor training | 12-18 units | 40-65% | 5-12 units |
| Weight synchronization | 3-7 units | 60-85% | 2-6 units |
| Reward and other host work | 3-8 units | 15-35% | Less than 3 units |

The ranges should not be added mechanically. Several changes overlap: for
example, deeper decode windows, round-robin routing, and asynchronous output
processing can all shorten the same host-visible rollout interval.

Using the normalized baseline of 100:

| Planning case | Estimated remaining wall time | Approximate speedup |
|---|---:|---:|
| Conservative | 70-75 units | 1.3-1.4x |
| Central | 55-65 units | 1.5-1.8x |
| Aggressive | 45-55 units | 1.8-2.2x |

These estimates assume the prompt set, sampling policy, output cap, global
batch, checkpoint, and semantic validity gates remain fixed.

## Configuration comparison

### Invariants

The two configurations retain:

- The same Qwen3-0.6B GRPO workload and train batch.
- The same number of prompts and generations per prompt.
- The same sampling temperature, top-k, and top-p policy.
- TP4 x DP32 rollout on the same fixed fleet.
- The same sequence-length limits, output cap, and KV-cache page pin.
- Corrected continuous-decode masking and last-real-token reporting.
- The same checkpoint, optimizer, and synchronous engine scheduling policy.

Keeping these fixed matters because shorter outputs or a changed sampling
contract could look like a performance improvement without processing the same
effective work.

### Material changes

| Area | Baseline configuration | Candidate configuration | Estimated effect |
|---|---|---|---|
| Runtime | Earlier inference/training stack | Packing-enabled stack | Enables the remaining changes |
| Continue-decode window | Shallower | About 4x deeper | Fewer host/device boundaries |
| Batched-token budget | Baseline budget | About 2x larger | Fewer capacity waves |
| RPA decode tiling | Smaller KV load tile | Larger KV load tile | Better long-context kernel balance |
| Prefix caching | Inherited/default | Disabled for this workload | Less cache and router bookkeeping |
| DP routing | Query every rank for request state | Query-free round robin | Removes per-request fan-out |
| Sampling | Padding can trigger full-vocabulary fallback | Padding-safe sharded candidate path | Avoids replicated vocabulary work |
| EOS handling | Cross-DP scalar reduction | Unused reduction removed | Less per-token collective latency |
| Token extraction | Per-request scans | Vectorized batch scan | Lower host turnaround |
| Detokenization/logprobs | Synchronous | Asynchronous/overlapped | Hides CPU work behind rollout |
| Reward path | Smaller worker pool, serialized tail | Larger pool and async scoring | Reduces visible reward tail |
| Train representation | Dense padded sequences | Packed real-token sequences | Less padded attention work |
| Trainer attention | General/default implementation | Flash-style implementation | Better packed-sequence efficiency |
| Weight sync | KV reinitialization and Python slicing | KV retention and jitted unroll | Fewer dispatches and allocations |
| Profiling | Rollout device profile enabled | Rollout profile disabled | Evidence-scope difference only |

The candidate is a bundle, so the total estimate should not be allocated to
individual patches as though isolated A/B experiments had been performed.

## Rough profiling interpretation

### Internal trace map

The links in this subsection are access-controlled and lead to raw artifacts
that contain real benchmark measurements. The analysis below does not repeat
those values. Remove this subsection when sharing the document outside the
benchmark team.

| Artifact | Internal link | What to inspect |
|---|---|---|
| Baseline decode-only XPlane | [GCS browser](https://pantheon.corp.google.com/storage/browser/ubench-logs/tianyu-rl-23b/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_22_06_21_47) | Host `llm_engine.step` and `device_get`; device `jit__decode_core_impl`, RPA custom calls, `all-gather`, `all-reduce`, and sampling loops |
| Baseline prefill-heavy XPlane | [GCS browser](https://pantheon.corp.google.com/storage/browser/ubench-logs/tianyu-rl-23b/phased_profile/prefill_heavy/dp_rank_0/plugins/profile/2026_08_22_06_21_47) | Prefill/decode mix, RPA kernel selection, batching waves, and host gaps |
| Baseline trainer XPlane | [GCS browser](https://pantheon.corp.google.com/storage/browser/ubench-logs/tianyu-rl-23b/tensorboard/plugins/profile/2026_08_22_06_27_19) | `jit__train_step`, attention fusions, loop/dispatch structure, and collective operations |
| Baseline phase timeline | [GCS browser](https://pantheon.corp.google.com/storage/browser/ubench-logs/tianyu-rl-23b/perf_traces) | Ordering and overlap of rollout, reward, train, and weight synchronization |
| Candidate trainer XPlane | [GCS browser](https://pantheon.corp.google.com/storage/browser/ubench-logs/tianyu-rl-136/tensorboard/plugins/profile/2026_08_30_05_57_04) | Packed train program, flash-attention custom calls, host dispatches, and collectives |
| Baseline phase/quality logs | [GCS browser](https://pantheon.corp.google.com/storage/browser/ubench-logs/tianyu-rl-23b) | `phasetime`, `lendist`, TensorBoard events, patch banners, and engine statistics |
| Candidate phase/quality logs | [GCS browser](https://pantheon.corp.google.com/storage/browser/ubench-logs/tianyu-rl-136) | Same phase and semantic gates under the candidate stack |

The baseline rollout trace can be opened directly with the TensorBoard Profile
plugin:

```bash
tensorboard \
  --logdir=gs://ubench-logs/tianyu-rl-23b/phased_profile/decode_only/dp_rank_0 \
  --port=6006
```

The candidate trainer trace can be opened similarly:

```bash
tensorboard \
  --logdir=gs://ubench-logs/tianyu-rl-136/tensorboard \
  --port=6007
```

If direct GCS loading is unavailable, download the profile directory with
`gcloud storage cp --recursive` and point TensorBoard at the local directory.
The `*.trace.json.gz` files can also be downloaded and opened in Perfetto.

There is no candidate rollout phased trace: the candidate launch disabled
`PHASED_PROFILING_DIR` because that capture conflicted with the trainer
profiler. Consequently, phase-level comparisons are usable, but rollout
kernel attribution is one-sided. A profile-only candidate rerun is the first
recommended experiment.

### Baseline rollout signature

The baseline profile is consistent with a host/control-plane limited decode
path:

- Short TPU programs sit inside much larger host-visible engine boundaries.
- Blocking output materialization includes queueing, execution, transfer, and
  host conversion.
- Per-request DP state queries amplify scheduler traffic.
- Padding can send the batch through replicated full-vocabulary sampling.
- EOS checks and tensor-parallel collectives add latency at token granularity.

This means an accelerator roofline alone will overpredict achievable gains.
Even if the kernels are on the HBM side of the roofline, the end-to-end path
can still be dominated by launch latency, collectives, queueing, and Python
work.

### Candidate rollout expectation

The candidate should show:

- More useful decode iterations per host boundary.
- Fewer rank queries and fewer scalar collectives.
- Candidate-only vocabulary communication rather than full-vocabulary work.
- Less synchronous detokenization, logprob, and reward work on the critical
  path.
- A higher ratio of device-program time to engine-window time.

The candidate did not retain a directly comparable rollout device capture.
Therefore these are expected signatures, not measured endpoint deltas. A
short profile-only rerun is needed before making kernel-level attribution.

### Trainer expectation

The baseline trainer performs work over padded shapes and pays repeated host
dispatch overhead. Packing, a NumPy-first packer, and flash-style attention
should reduce both padded compute and dispatch count. Keeping the KV allocation
and synchronizing layers through one compiled operation should similarly
reduce the visible synchronization tail.

## Evidence-to-action matrix

The items below separate direct evidence from inference. “Trace evidence”
means the relevant operation is visible in an artifact above. “Source
evidence” means the cost mechanism follows from the executed code path but
still needs an isolated A/B to quantify.

### 1. Amortize decode-window boundaries

Evidence:

- The baseline decode trace shows repeated `llm_engine.step` host windows
  surrounding comparatively short device programs.
- Blocking `device_get` spans include queueing, device execution, transfer,
  and output materialization; they are not purely accelerator compute.
- The trace contains many repetitions of the fused decode program, so fixed
  boundary work is paid repeatedly.

Proposed solution:

- Increase the continue-decode window by roughly four times.
- Increase the batched-token budget by roughly two times.
- Keep the corrected per-sequence masking and last-real-token reporting
  patches enabled.
- Fix token buckets so the largest legal batch is represented exactly instead
  of being rounded into an unnecessarily larger bucket.

Why it should improve performance:

The scheduler, host/device handoff, output extraction, and synchronization
costs are paid once per window. A deeper useful window amortizes those fixed
costs over more generated tokens. A larger token budget also reduces the
number of capacity waves needed to admit the same requests.

Validation:

- Normalize `llm_engine.step` count and host-window time by generated tokens.
- Confirm the candidate executes more useful decode iterations per window.
- Require identical prompt inputs and bounded output-length, truncation, and
  reward distributions.
- Verify there are no zombie requests and that every EOS-ending request is
  retired.

Implementation references:

- [`patch_cd_proper.py`](../patches/patch_cd_proper.py)
- [`patch_cd_masking.py`](../patches/patch_cd_masking.py)
- [`patch_cd_report.py`](../patches/patch_cd_report.py)

### 2. Remove per-request all-rank routing work

Evidence:

- Source inspection shows `_find_best_rank_for_request` querying state across
  DP ranks for each newly admitted request.
- The workload is homogeneous enough that the expensive state discovery is
  unlikely to improve balance materially.
- The baseline host trace leaves substantial time outside device programs,
  which is where scheduler IPC and Python routing execute.

Proposed solution:

Use deterministic round-robin routing when `DP_ROUTING=rr`, while retaining
the stock heuristic as the fallback:

```diff
-rank = self._find_best_rank_for_request(request)
+if os.environ.get("DP_ROUTING") == "rr":
+    rank = getattr(self, "_rr_next", 0) % self.dp_size
+    self._rr_next = rank + 1
+else:
+    rank = self._find_best_rank_for_request(request)
```

Disable prefix caching for this benchmark when prompts do not repeat:

```bash
export DP_ROUTING=rr
export VLLM_PREFIX_CACHING=0
```

Why it should improve performance:

The routing decision changes from all-rank state discovery for every request
to constant host work. Disabling an unproductive prefix cache also avoids
computed-block probes, cache bookkeeping, and cache-router messages.

Validation:

- Count scheduler query/response messages per admitted request.
- Compare per-rank admitted token totals and queue-depth imbalance.
- Reject the change if round robin produces a materially worse long tail.
- Confirm prompt and generation semantics are unchanged.

Implementation reference:
[`patch_dp_rr_routing.py`](../patches/patch_dp_rr_routing.py).

### 3. Keep sampling vocabulary-sharded

Evidence:

- The baseline decode trace contains a logits `all-gather` followed by
  repeated full-vocabulary top-k/top-p reduction loops.
- Source inspection shows a whole-batch fallback: one padding row with an
  invalid/default top-k can send every row through the replicated sampler.
- The full vocabulary is much larger than the requested candidate set, making
  that communication and reduction structurally wasteful.

Proposed solution:

Take candidates locally on each TP shard, communicate only candidate values
and global IDs, and merge them before top-p and categorical sampling:

```python
candidate_values, candidate_ids = lax.top_k(local_logits, CANDIDATE_COUNT)
candidate_ids += lax.axis_index(model_axis) * local_vocab_size
candidate_values = lax.all_gather(candidate_values, model_axis)
candidate_ids = lax.all_gather(candidate_ids, model_axis)
```

Keep padding rows inside this path rather than allowing them to trip the
whole-batch fallback:

```python
safe_top_k = jnp.where(top_k <= 0, CANDIDATE_COUNT, top_k)
safe_top_p = jnp.where(top_k <= 0, 1.0, top_p)
use_sharded_path = jnp.all(
    (safe_top_k <= CANDIDATE_COUNT) & (safe_top_p > 0.0)
)
```

Why it should improve performance:

Communication scales with `batch x TP x candidate_count` instead of `batch x
vocabulary`, and top-k/top-p no longer sweep the full replicated vocabulary
on every TP device. The padding guard is essential; without it, the optimized
branch can exist in the binary but remain unused for mixed active/padded
batches.

Validation:

- The rollout trace should lose the large logits all-gather and full-vocabulary
  reduction loops on supported sampling parameters.
- Add a counter for sharded-path versus fallback-path selections.
- Compare kept candidate sets on adversarial ties and greedy inputs.
- Gate stochastic output by distribution, reward, length, and truncation—not
  byte identity, because the RNG draw order may change.

Implementation reference:
[`patch_sharded_sampling.py`](../patches/patch_sharded_sampling.py). The
candidate-specific padding guard is shown above because it is newer than the
generic checked-in patch.

### 4. Reduce decode collectives and tune RPA tiling

Evidence:

- The baseline device trace contains token-granularity scalar all-reduces in
  the fused loop.
- Its RPA custom-call name records the active KV page and decode block sizes,
  proving which configuration the kernel actually received.
- RPA and collectives occupy meaningful device timeline regions, but the much
  larger host envelope means kernel tuning alone cannot explain all headroom.

Proposed solution:

- Remove `any_hit_eos` from loop-carried state when the selected masking
  semantics never consult a global early-exit flag.
- Forward `RPA_V3_DECODE_BLOCK_SIZES` through the actual MaxText serving entry
  point and increase the KV load tile while keeping the compute tile fixed.
- Keep the KV page size pinned so topology/config experiments do not silently
  change paging behavior.

Representative EOS patch:

```diff
-any_hit_eos = jnp.any(jnp.logical_and(active_mask, is_eos))
+any_hit_eos = jnp.array(False)  # global flag unused by this loop condition
```

Representative configuration:

```bash
export VLLM_BLOCK_SIZE=128
export RPA_V3_DECODE_BLOCK_SIZES="1,8192,1,4096"
export CONTINUE_DECODE_EOS_CHECK_INTERVAL=8
```

Why it should improve performance:

Small global collectives have poor arithmetic intensity and impose a latency
floor at every decode iteration. Removing an unused one directly shortens the
critical dependency chain. A better RPA load tile can improve memory streaming
and reduce tiling overhead for long KV contexts.

Validation:

- Confirm the scalar all-reduce disappears without changing per-row active
  masks or termination behavior.
- Read the RPA custom-call name in the candidate trace and verify the intended
  page/load/compute sizes.
- Compare device program time per generated token; do not judge the RPA change
  only by end-to-end time.

Implementation references:

- [`patch_cd_noeos.py`](../patches/patch_cd_noeos.py)
- [`patch_rpa_blocks.py`](../patches/patch_rpa_blocks.py)

### 5. Vectorize output extraction and overlap postprocessing

Evidence:

- The baseline host path trims EOS separately for each request using repeated
  NumPy membership scans.
- `device_get` and engine-window spans include output transfer and
  materialization after the device program completes.
- Detokenization and per-token logprob object construction are CPU work that
  does not need to serialize the next independent engine work.

Proposed solution:

Compute the first EOS position for all rows with one matrix operation:

```python
is_eos = np.isin(generated_tokens_cpu, eos_ids)
first_eos = np.where(
    is_eos.any(axis=1), is_eos.argmax(axis=1) + 1, generated_tokens_cpu.shape[1]
)
valid_tokens = tokens[: int(first_eos[request_index])].tolist()
```

Batch detokenization across completed outputs, then move remaining
detokenization and logprob conversion to bounded worker executors. Retain
result ordering and surface worker exceptions before consuming the batch.

Why it should improve performance:

Vectorization replaces thousands of small Python/NumPy calls with one
contiguous scan. Asynchronous conversion overlaps CPU-only work with other
rollout activity and reduces the serial tail after device completion.

Validation:

- Compare extracted token arrays and EOS-hit counts against the stock path.
- Record executor queue depth and p95 task latency to ensure overlap is not
  merely moving a backlog to the end of the step.
- Require logprob IDs, ranks, and values to pass the declared tolerance.

Implementation references:

- [`patch_host_extract.py`](../patches/patch_host_extract.py)
- [`patch_fast_detok.py`](../patches/patch_fast_detok.py)

The asynchronous detokenization/logprob wrappers used by the candidate are
embedded in its launch manifest; the checked-in fast-detokenization patch
shows the independently useful batch-decode transformation.

### 6. Parallelize and overlap reward computation

Evidence:

- Phase instrumentation places reward parsing and verification after
  generation on the serial preparation path in the baseline.
- The work is independent across completions and largely CPU-bound.
- Source inspection shows repeated parsing, regex, and mathematical
  verification over a large list of independent strings.

Proposed solution:

- Chunk completion/reward inputs across a process pool, preserving input order
  and slicing per-example keyword arguments in lockstep.
- Use a linear boxed-answer extractor rather than repeatedly copying/scanning
  prefixes.
- Start scoring completed requests while later requests are still decoding;
  retain a synchronous fallback and sampled cross-checks.

Why it should improve performance:

Process workers bypass the Python GIL for independent parsing. Starting work
as completions retire converts much of reward latency from an additive phase
into overlapped work. Faster extraction reduces CPU demand and therefore
reduces contention with the scheduler.

Validation:

- Require exact reward equality for deterministic reward functions.
- Track async-versus-sync mismatches and make any mismatch adoption-blocking.
- Monitor scheduler CPU and rollout tail; too many reward workers can compete
  with engine-driver processes and reverse the gain.
- Measure the visible reward tail separately from total worker CPU time.

Implementation reference:
[`patch_reward_parallel.py`](../patches/patch_reward_parallel.py).

### 7. Pack real tokens and use flash-style trainer attention

Evidence:

- Variable-length completions imply substantial padding in dense train
  batches.
- The baseline trainer trace shows a different fusion/dispatch structure from
  the candidate packed trainer trace.
- Source inspection of the stock packer shows many small JAX array-building
  operations, while packing can be assembled on the host before one transfer.

Proposed solution:

```bash
export MAX_SEQ_TOKEN_PER_TPU=12288
# Trainer argument:
attention='flash'
```

- Build packed rows in preallocated NumPy arrays and convert each feature to a
  JAX array once per chunk.
- Carry segment IDs and positions so attention cannot cross sequence
  boundaries.
- Supply flash-attention block defaults through the RL configuration shim.

Why it should improve performance:

Packing removes attention and feed-forward work on padding. For attention,
avoiding padding is especially valuable because dense work grows with sequence
area rather than only token count. NumPy-first assembly also eliminates many
tiny Pathways/JAX dispatches before the compiled train step.

Validation:

- Compare real tokens per packed row and padding fraction.
- Assert segment masks prevent cross-example attention.
- Compare packed and unpacked losses on a small deterministic batch.
- In XPlane, verify flash-style custom calls appear and host dispatch count
  falls.

Implementation references:

- [`patch_packing_knob.py`](../patches/patch_packing_knob.py)
- [`patch_packer_fast.py`](../patches/patch_packer_fast.py)
- [`patch_flash_sa.py`](../patches/patch_flash_sa.py)

### 8. Remove weight-sync RPC and slicing overhead

Evidence:

- The phase timeline exposes weight synchronization as a separate serial span.
- Source inspection shows KV-cache deletion/reinitialization RPCs around every
  parameter update even when no requests are live.
- Scanned-layer unrolling slices parameters layer by layer, creating many
  remote dispatches through the proxy.

Proposed solution:

Gate KV teardown so the stock path remains available:

```python
if os.environ.get("SYNC_KEEPKV", "1") != "1":
    engine.collective_rpc("delete_kv_cache")
    # reshard weights
    engine.collective_rpc("reinitialize_kv_cache")
```

Slice all layers of each scanned parameter in one compiled helper instead of
one host-dispatched slice per layer:

```python
@functools.partial(jax.jit, static_argnums=1)
def unroll_layers(value, axis):
    return tuple(jnp.take(value, i, axis=axis) for i in range(value.shape[axis]))
```

Why it should improve performance:

Retaining an empty KV pool avoids global RPCs and allocator churn. Jitted
unrolling collapses many proxy round trips into a small number of compiled
operations. Neither change reduces model math; both remove orchestration from
the critical path.

Validation:

- Assert no live request owns a KV page at synchronization time.
- Monitor HBM headroom across the transient weight update.
- Compare parameter trees and first post-sync logits against the stock path.
- Count collective RPCs and host dispatches inside the sync span.

Implementation reference:
[`patch_sync_keepkv.py`](../patches/patch_sync_keepkv.py). The jitted unroll
helper is shown above because that candidate patch is embedded in its launch
manifest rather than checked in as a standalone file.

### 9. Explore lower TP only after host work is controlled

Evidence:

- The archived topology sweep contains TP4, TP8, and TP16, but no identified
  TP1 result.
- Qwen3-0.6B has eight KV heads, limiting useful KV sharding beyond TP8.
- Roofline reasoning predicts that lower TP removes collectives but also lowers
  local batch reuse and increases the number of DP engines handled by the host.

Proposed solution:

Run TP2 x DP64 first, then TP1 x DP128 only if host CPU, scheduler queues, and
tail latency remain healthy. Use a TP-independent local candidate sampler at
TP1 so the topology experiment does not also switch back to a full-vocabulary
sampling implementation.

Why it could improve performance:

For a small model, TP collective latency can exceed the benefit of splitting
weights more finely. TP2 or TP1 may therefore shorten the device critical path.
However, increased weight traffic and DP fan-out can erase the gain, which is
why this is an experiment rather than an assumed optimization.

Validation:

- Hold the total fleet, prompts, sampling policy, page size, decode depth,
  routing, and candidate sampler fixed.
- Report normalized wall time per generated token and p90 engine latency.
- Track CPU utilization, per-engine queue depth, and DP imbalance.
- Treat reward/length/truncation bands as semantic gates because TP can shift
  sampling numerics.

## Roofline reasoning without job counters

For a sub-billion-parameter decoder, generation is expected to sit on the
memory/latency side of the accelerator roofline. Each token reuses relatively
little of the model within a small local batch, while weights and KV state must
be read repeatedly. Tensor parallelism changes three competing terms:

1. Weight bytes read by each device.
2. Local batch size available to amortize those reads.
3. Collective latency needed to join tensor-parallel shards.

At a fixed fleet of 128 logical devices, a first-order sensitivity model is:

| Topology | DP engines | Relative local batch | Relative AI estimate | Main risk |
|---|---:|---:|---:|---|
| TP1 x DP128 | 128 | 0.25x | About 0.25x | Full model per device and 4x host fan-out |
| TP2 x DP64 | 64 | 0.5x | About 0.5x | More engines, but small TP collectives |
| **TP4 x DP32** | **32** | **1.0x** | **Reference** | Balanced current point |
| TP8 x DP16 | 16 | 2.0x | About 2.0x | Higher TP collective latency |

The arithmetic-intensity column is a directional estimate based on local
batch reuse, not a hardware measurement. Dynamic sequence retirement, KV
traffic, custom kernels, and collective topology all break perfect linear
scaling.

Qwen3-0.6B has eight KV heads, so TP above eight is especially unattractive:
K/V may need replication rather than further useful sharding. TP1 removes TP
collectives, but it also creates four times as many DP engines as TP4 and makes
each device stream the full model for a smaller local batch.

## TP1 x DP128 search result

No identifiable Tianyu TP1 x DP128 run was found in the available records.
The search covered:

- Checked-in Qwen launch configurations.
- GCS run and object names containing topology markers.
- Uploaded launcher/config-style artifacts discoverable in the run bucket.
- The archived topology-sweep notes.

The identifiable sweep covers TP4, TP8, and TP16 configurations, but not TP1.
Cloud Logging was not queryable from the current environment, so the precise
claim is **“no known run in the available artifacts,”** not proof that no
transient experiment ever existed.

TP1 is not automatically the best next point. The current sampling wrapper
also falls back to a local full-vocabulary path when no model axis exists,
which would confound a TP1 experiment. TP2 x DP64 is the safer first scout;
TP1 should follow only with a topology-independent local candidate sampler and
explicit host-load instrumentation.

## Optimization story

### 1. Make each host boundary do more useful decode work

Use a deeper continue-decode window, a larger batched-token budget, corrected
bucket sizing, vectorized output extraction, and no unused global EOS
reduction. The expected result is fewer control-plane boundaries per generated
token.

### 2. Stop doing global work for every request

Replace all-rank state discovery with deterministic round-robin routing for a
homogeneous DP fleet. Disable prefix-cache bookkeeping when the workload does
not benefit from prompt reuse. This attacks the part of rollout that a device
kernel optimization cannot fix.

### 3. Keep vocabulary and postprocessing off the serial path

Keep padding rows on the sharded candidate sampler, then overlap
detokenization, logprob extraction, and reward parsing with accelerator work.
This reduces both vocabulary communication and the CPU tail.

### 4. Train real tokens and synchronize in compiled batches

Pack variable-length sequences, use flash-style attention, avoid tiny JAX
dispatches during packing, retain reusable KV allocations, and compile layer
synchronization as a single operation.

## Consolidated candidate configuration diff

The following is the minimal configuration-level view of the proposed stack.
The numeric values are configuration parameters, not performance results.

```diff
-export RPA_V3_DECODE_BLOCK_SIZES="1,4096,1,4096"
+export RPA_V3_DECODE_BLOCK_SIZES="1,8192,1,4096"
 export CONTINUE_DECODE_EOS_CHECK_INTERVAL="8"
 export VLLM_BLOCK_SIZE=128
+export VLLM_PREFIX_CACHING=0
+export DP_ROUTING=rr
+export FAST_EXTRACT=fast
+export ASYNC_REWARD=on
+export ASYNC_LOGPROBS=on
+export MAX_SEQ_TOKEN_PER_TPU=12288

-math_verify_num_procs=32
+math_verify_num_procs=64
 rollout_data_parallelism=32
 rollout_tensor_parallelism=4
-max_num_batched_tokens=16640
+max_num_batched_tokens=33280
+attention='flash'
-vllm_additional_config='{"enable_continue_decode": true, "max_decode_steps": 128}'
+vllm_additional_config='{"enable_continue_decode": true, "max_decode_steps": 512}'
```

For the profile-only rerun, do not leave rollout profiling disabled:

```bash
export PHASED_PROFILING_DIR="gs://<restricted-bucket>/<profile-run>/phased_profile"
```

Disable the trainer profiler in that rerun so the two collectors do not race.
The goal is a clean candidate decode trace, not another benchmark score.

## Remaining estimated headroom

After the candidate stack, rollout is likely to remain roughly four-fifths or
more of total wall time. Consequently:

- A 10% rollout reduction is worth roughly an 8-9% end-to-end reduction.
- A 20% trainer-only reduction is likely worth only a low-single-digit global
  improvement.
- Eliminating every non-rollout phase is an impossible bound and would still
  leave most candidate wall time intact.

Future work should therefore prioritize rollout structure over small trainer
kernel wins.

## Recommended experiments

1. Run a short profile-only clone of the candidate with rollout profiling on
   and trainer profiling off. Compare normalized host-window composition,
   device duty cycle, collective share, and output-materialization share.
2. Run TP2 x DP64 on the same fixed fleet. Hold the workload, page size,
   decode depth, routing policy, and sampler implementation constant.
3. If head CPU utilization and engine-tail latency remain healthy, run
   TP1 x DP128 with an explicit local candidate sampler.
4. If topology is flat, shard engine-driver work across processes or heads and
   parallelize the remaining scheduler/output-materialization path.
5. Use isolated A/B runs for individual patches before assigning patch-level
   percentages.

## Validity gates

- Compare normalized time per generated token, not only time per step.
- Keep prompt identity and input-length distributions fixed.
- Require output-length, truncation, and reward distributions to remain within
  a predeclared control band.
- Verify the intended KV page and RPA kernel in the profile.
- Require async reward/logprob verification mismatches to remain zero.
- Report a multi-step distribution rather than a single best step.
- Treat each TP degree as a potentially shifted sampling-numerics contract.

## Disclosure note

This document is suitable for rough capacity planning and prioritization. It
does not reproduce exact benchmark values and is intentionally insufficient
for reconstructing the timing of any real job. The internal trace map points
to access-controlled raw artifacts at the user's request; remove that entire
subsection before external distribution.
