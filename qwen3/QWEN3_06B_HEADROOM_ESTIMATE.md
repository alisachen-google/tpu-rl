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
is intentionally insufficient for reconstructing the timing of any real job.
Exact benchmark values and raw artifact locations should remain in a separate,
access-controlled analysis.
