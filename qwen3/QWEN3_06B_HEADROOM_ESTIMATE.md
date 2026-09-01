# Qwen3-0.6B headroom from the tianyu-rl-23b baseline

This is the internal, evidence-backed headroom report. It uses restricted run
names and trace links, but keeps job performance in normalized units. It does
not reproduce production wall-clock times, throughput, rewards, token counts,
or hardware-counter rates.

The analysis baseline is **the `tianyu-rl-23b.yaml` configuration = 100
normalized wall-time units**. The unit is defined by that run's steady-step
median below; it is not a second and must not be converted into a job timing.
The current TP1 endpoint is approximately 53 of these baseline units.

The shareable, estimate-only report remains
[`QWEN3_06B_HEADROOM_PUBLIC.md`](QWEN3_06B_HEADROOM_PUBLIC.md). The last commit
that introduced our private patch pointers was `32e4ae4`; the sanitized public
document was introduced by `acc25f7`.

## Executive verdict

The current TP1 x DP128 package is the best measured endpoint for this stack.
Against the closest TP4 x DP32 profiled control, it lowers end-to-end time by
about 5% and rollout time by about 6%, while actor training is flat. Completion
volume between the two short profiled replicas differs by less than 0.1%, so
the gain is not a shorter-output artifact. The pair is not a pure topology
experiment, however: the TP1 recipe also carries required TP1 RPA decode,
mixed, and prefill tiles.

Most of the baseline-to-current gain was already delivered before TP1:

| Milestone | Normalized global time | What changed |
|---|---:|---|
| **Baseline, run 23b** | **100 units** | `tianyu-rl-23b.yaml`, TP4 x DP32 |
| Optimized TP4, run 136 | About 56 units | Decode, sampling, host, packing, train, and sync stack |
| Profiled TP4 replica, run 149c | About 56 units | Closest TP4 control and usable rollout profile |
| Profiled TP1 replica, run 154 | About 52 units | Same-as-152 TP1 recipe plus profilers |
| Current TP1 endpoint, run 152 | About 53 units | Sustained result plus a direct trainer XPlane; rollout/Perfetto capture off |

Approximately 94% of the reduction from the 23b baseline to the current level
came before the final TP1-plus-tile bundle; that bundle supplied the last
approximately 6% of the absolute reduction. The trace supports removal of the
TP collective tax as a mechanism, but the measured delta cannot be assigned to
topology alone because the RPA tiles also change.

Two large-looking levers are already inside that current endpoint and are not
remaining kernel headroom: the continue-decode window is already 512 rather
than 128, and packed-chunk assembly is already NumPy-first. The former has an
estimated latency-amortization benefit; the latter has a one-change measured
control. Their landed credit and residual ceilings are quantified below.

The remaining practical opportunity is much smaller than in the previous
report:

| Planning case | Remaining time in 23b-baseline units | Speedup from 23b | Incremental speedup from current |
|---|---:|---:|---:|
| Conservative | 50-51 units | About 1.95-2.00x | About 1.03-1.05x |
| Central | 48-50 units | About 2.00-2.10x | About 1.06-1.11x |
| Stretch | 44-48 units | About 2.10-2.25x | About 1.11-1.19x |

These ranges overlap. RPA tuning, sampling-guard removal, topology, and DP
tail work all shorten the same rollout critical path and must not be added as
independent wins.

Within that broader budget, TPU-side kernel/collective work alone is currently
estimated at about 2-5% of the run-152 endpoint. The main terms are the decode
sampling guard and RPA retuning; trainer-kernel tuning is sub-percent end to
end.

## Baseline and median contract

### Analysis baseline

The baseline for all end-to-end and remaining-headroom estimates is
**tianyu-rl-23b**, using the supplied `tianyu-rl-23b.yaml` manifest:

- Qwen3-0.6B GRPO on the fixed 128-device fleet.
- TP4 x DP32 rollout.
- Global train batch 256, eight generations per prompt, and rollout batch 256.
- Sampling temperature 0.8, top-k 50, and top-p 0.95.
- Prompt/prefill limit 4096, target length 12288, and output cap 8192.
- KV page size 128, HBM allocation fraction 0.3, and maximum 256 sequences.
- Batched-token budget 16640.
- Continue-decode depth 128 and EOS-check interval 8.
- RPA decode blocks `1,4096,1,4096`.
- Synchronous engine scheduling and no DP attention.
- Twenty global steps.
- Rollout phased profiling and a one-step trainer XPlane enabled.

The baseline statistic is TensorBoard
`perf/train/global_step_time`, global steps 3 through 20 inclusive. This gives
18 steady observations after the first two startup/warmup steps. Their median
is defined as 100 units.

- Use the 18-step median as the primary benchmark number.
- Report p10 and p90 over the same 18 steps.
- Report any stall separately rather than silently deleting it.
- Use an outlier-excluded mean only as a diagnostic, never as the headline.

The optimized TP1 run 152 is an endpoint, not the normalization baseline. Its
same-clock steady median is approximately 53 baseline units. It contains one
rollout-only stall; that observation is retained in its distribution, while
the median remains the comparison statistic.

### Clock alignment

The instrumentation uses two step conventions:

- TensorBoard global step `N+1` corresponds to phasetime/lendist step `N`.
- Both the baseline and sustained endpoint medians use TensorBoard steps 3-20.
- Per-token and phasetime analysis uses the aligned phasetime/lendist steps
  2-19.

Do not mix a phasetime median with a TensorBoard median without stating the
clock. The profiled TP1 run 154 is a trace surrogate, not the sustained
endpoint: it has only four post-warmup observations and carries profiling
instrumentation.

## Measurement roles

| Run | Role in this report | Usable evidence | Limitation |
|---|---|---|---|
| 23b | **Analysis baseline** | Eighteen steady steps, phase/quality, rollout XPlane, trainer XPlane, Perfetto | Earlier stack and TP4 |
| 136 | Optimized TP4 trainer reference | Long-run phase/quality and trainer XPlane | No usable rollout XPlane |
| 149c | Closest optimized TP4 rollout/device reference | Four steady steps, rollout XPlane, Perfetto | Short profiled replica; uses the 136 TP4 tile set |
| 152 | Current sustained TP1 endpoint | Eighteen steady steps, phase and semantic logs, and a direct trainer XPlane | No Perfetto; no phased-rollout session |
| 154 | Same-as-152 TP1 whole-cycle trace surrogate | Four steady steps and Perfetto | TP1 phased XPlane captured the 128-way reshard program rather than steady decode |

This bridge is necessary because no one run contains a clean long benchmark,
a valid TP1 decode profile, a trainer profile, and a whole-cycle timeline.
Run 152 supplies the sustained endpoint and the direct trainer profile; run
154 supplies the exact-recipe Perfetto timeline; run 149c supplies the closest
valid rollout profile at TP4. The last bridge changes topology and RPA tiles,
so it is suitable for mechanism and package-level attribution, not a numeric
topology-only claim.

## Work-equivalence gate

The TP1-package conclusion passes the available workload checks:

- Prompt/ISL arrays are identical step-for-step across the compared runs.
- The TP4 and TP1 profiled replicas differ in completion-token volume by less
  than 0.1% over their four aligned steady steps.
- The long TP4 and TP1 endpoints differ in average completion volume by less
  than 0.3% over the aligned steady window.
- Completion-length, cap-truncation, and reward distributions remain inside
  the established control band.
- Actor training and weight synchronization remain approximately flat in the
  near-matched TP4/TP1 profiled comparison.

TP degree changes reduction order and therefore sampling numerics. Prompt
identity is exact; generated sequences are not expected to be byte-identical.
The correct adoption gate is a predeclared distributional band for length,
truncation, reward, and token volume—not identical sampled tokens.

## Configuration comparison

### Baseline-to-current stack

The comparison below is deliberately more explicit than a feature summary.
The full `~/rl-repro/tianyu-rl-152-ub.yaml` manifest reproduced by the run
owner is the direct source for the 152 column; its header identifies
`tianyu-rl-150-ub.yaml` as the lineage and describes the recipe as run 136 at
TP1 x DP128 plus tile fixes. Values below come from that exact launch script
and patch ConfigMap, rather than being inferred from run 136 or from a different
model's manifest. Run-152 artifacts and the run-154 same-config profiled replica
cross-check execution and evidence roles. The supplied 23b YAML is the direct
source for the baseline column.

#### Exact performance-material configuration differences

| Knob | `tianyu-rl-23b` | `tianyu-rl-152` champion recipe | Status / consequence |
|---|---|---|---|
| Workload image | `maxtext-rl-ironwood:2026-08-03b` | Packing-enabled image pinned at `sha256:f7c957d0b286e03966fdd6f9b15e7609fae8460a933a9da0b9fbab3e04723c94` | Runtime and source pin changed; this enables the later Tunix/packing patches |
| Rollout tensor parallelism | `4` | `1` | Removes TP model collectives |
| Rollout data parallelism | `32` | `128` | Uses the same 128-device sampler fleet as four times as many replicas; increases DP control fan-out |
| Continue-decode depth | `max_decode_steps=128` | `max_decode_steps=512` | Four times more possible decode iterations per host boundary |
| Maximum batched tokens | `16640` | `33280` | Doubles the admission/token budget |
| RPA decode block tuple | `1,4096,1,4096` | `1,2048,1,2048` | TP1-specific decode tiles halve both KV load and compute dimensions relative to 23b |
| RPA mixed block tuple | Unset; kernel default | `512,1024,256,512` | TP1 safety tile avoids a mixed-attention working set above the 64 MiB VMEM limit |
| RPA prefill block tuple | Unset; kernel default | `512,1024,256,512` | Applies the matching TP1-safe prefill tile |
| Prefix caching | No explicit override; image/default behavior | `VLLM_PREFIX_CACHING=0` | Explicitly disabled in the later `block_size` patch variant |
| DP request routing | Stock state-aware `_find_best_rank_for_request` path | `DP_ROUTING=rr` | Selects deterministic round robin and avoids per-request all-rank state queries |
| Packed-token budget | Unset; packing disabled | `MAX_SEQ_TOKEN_PER_TPU=12288` | Enables packed real-token training |
| Trainer attention | No explicit `attention` override | `attention='flash'` | Selects the flash-style trainer attention path |
| Math-verification processes | `32` | `64` | More host-side reward verification capacity |
| Answer extraction mode | Stock | `FAST_EXTRACT=fast` | Selects the faster balanced-brace extractor |
| Reward overlap mode | None | `ASYNC_REWARD=on` | Scores completed requests while rollout continues |
| Logprob overlap mode | None | `ASYNC_LOGPROBS=on` | Extracts finished-request logprobs in the background |
| Pathways-head placement | No pod anti-affinity | Preferred anti-affinity from other `pathways-head` pods on the same host | Measurement-noise control for a run submitted concurrently with another job; no isolated speed credit |
| Phased rollout profiler | `PHASED_PROFILING_DIR=<run GCS path>` | `PHASED_PROFILING_DIR=""` | Enabled in 23b, disabled in the record recipe |
| Perfetto v2 exporter | Patch invoked | Patch file present in the 152 bundle but not invoked | 23b emits a Perfetto trace; 152 does not |

The performance-relevant launch delta can be summarized as the following
pseudo-diff. It is a configuration summary, not a claim that 152 was produced
by editing the supplied 23b file in one step:

```diff
-workload_image=maxtext-rl-ironwood:2026-08-03b
+workload_image=maxtext-rl-ironwood@sha256:f7c957d0b286e03966fdd6f9b15e7609fae8460a933a9da0b9fbab3e04723c94

-rollout_tensor_parallelism=4
-rollout_data_parallelism=32
+rollout_tensor_parallelism=1
+rollout_data_parallelism=128

-max_num_batched_tokens=16640
-vllm_additional_config={"enable_continue_decode":true,"max_decode_steps":128}
+max_num_batched_tokens=33280
+vllm_additional_config={"enable_continue_decode":true,"max_decode_steps":512}

-RPA_V3_DECODE_BLOCK_SIZES=1,4096,1,4096
+RPA_V3_DECODE_BLOCK_SIZES=1,2048,1,2048
+RPA_V3_MIXED_BLOCK_SIZES=512,1024,256,512
+RPA_V3_PREFILL_BLOCK_SIZES=512,1024,256,512

-math_verify_num_procs=32
+math_verify_num_procs=64
+attention=flash
+VLLM_PREFIX_CACHING=0
+DP_ROUTING=rr
+MAX_SEQ_TOKEN_PER_TPU=12288
+FAST_EXTRACT=fast
+ASYNC_REWARD=on
+ASYNC_LOGPROBS=on

-PHASED_PROFILING_DIR=<run GCS path>
+PHASED_PROFILING_DIR=""
```

After normalizing run names and artifact destinations, the exact manifest
audit finds one workload-image change, six training-command argument changes,
ten performance-relevant exported-environment changes, and the preferred
head-pod anti-affinity addition. TPU allocation, head/proxy CPU and memory
resources, Pathways images, proxy flags, and the other training arguments are
unchanged.

#### Explicit patch inventory

The following eleven patch **names** are active in both configurations. Eight
are behaviorally stable controls. Three carry material later revisions and
must be credited as version deltas rather than mislabeled as unchanged:
`patch_cd_masking.py`, `patch_sharded_sampling.py`, and
`patch_block_size.py`. The first-token addition to `patch_lendist.py` and the
Tunix class-anchor update to `patch_phasetime.py` are observability or
compatibility changes only.

| Patch | 23b | 152 | Relevant version note |
|---|---|---|---|
| `patch_vllm_sampler.py` | Applied | Applied | Same Pathways profiling-call compatibility and request-kwargs cap |
| `patch_cd_proper.py` | Applied | Applied | Same options-free model twin and top-level continue-decode JIT construction |
| `patch_lendist.py` | Applied | Applied, extended | Instrumentation only; the later version also records first-token class |
| `patch_rpa_blocks.py` | Applied | Applied | Same forwarding hook; the configured decode tuple changes separately |
| `patch_cd_masking.py` | Applied | Applied, extended | Both provide deep-window masking and inert re-feed; the later version also removes `jnp.any(active_mask)` from `cond_fn`, eliminating one per-iteration cross-DP reduction |
| `patch_cd_report.py` | Applied | Applied | Same per-sequence last-real-token reporting fix |
| `patch_phasetime.py` | Applied | Applied, re-anchored | Same phase timers; the later source accepts Tunix's `RLEngine` rename through the `RLCluster` alias |
| `patch_reward_parallel.py` | Applied | Applied | Same process-parallel base reward manager; 152 additionally overlaps scoring |
| `patch_sharded_sampling.py` | Applied | Applied, extended | The later `_SS_PADFIX` rewrites padded `top_k=0` rows so they no longer force the whole batch onto the stock full-vocabulary path; the residual dynamic `jnp.all` guard still remains |
| `patch_block_size.py` | Applied | Applied, extended | Both pin the KV page; the 152 variant also consumes `VLLM_PREFIX_CACHING=0` |
| `patch_log_stats.py` | Applied | Applied | Observability only: enables engine/KV statistics |

One patch is active only in 23b:

| Patch | 23b | 152 | Difference |
|---|---|---|---|
| `patch_perf_v2.py` | Applied by the launcher | Mounted in the 152 bundle but not called | Perfetto export is active in 23b and inactive in the 152 record recipe |

The following fifteen patches are additions in run 152. “Applied” here means
invoked by the exact supplied 152 launcher; each script still has its own
anchor/banner gate at runtime.

| Patch added for 152 | What it changes relative to 23b |
|---|---|
| `patch_profiler_fix.py` | Converts an invalid Tunix trainer-profile window into a disabled window instead of a startup failure; compatibility, not speed credit |
| `patch_gc_sync.py` | Gates unconditional per-step `gc.collect()` calls; default later behavior is collection off |
| `patch_packing_knob.py` | Plumbs `MAX_SEQ_TOKEN_PER_TPU` into Tunix so packing can be enabled despite the older RLConfig schema |
| `patch_packer_fast.py` | Replaces per-bin JAX assembly with NumPy-first packed-chunk construction and a small number of array conversions |
| `patch_async_detok.py` | Decodes finished requests in a background thread pool during the remaining rollout |
| `patch_async_logprobs.py` | Extracts finished-request token logprobs asynchronously and serves cached results to Tunix |
| `patch_dp_rr_routing.py` | Adds the `DP_ROUTING=rr` branch that avoids the stock all-rank rank-selection queries |
| `patch_fast_extract.py` | Adds the linear balanced-brace answer extractor selected by `FAST_EXTRACT=fast` |
| `patch_async_reward.py` | Scores completions during rollout in a dedicated process pool and consumes the cached scores later |
| `patch_flash_sa.py` | Supplies missing flash/splash-attention defaults to the older RL config so `attention='flash'` is usable |
| `patch_jit_unroll.py` | Replaces per-layer weight-sync slicing dispatches with one compiled unroll per scanned parameter |
| `patch_sync_keepkv.py` | Skips delete/reinitialize of an already-free vLLM KV cache around weight synchronization |
| `patch_host_extract.py` | Vectorizes EOS discovery and valid-token lengths over the entire continue-decode output matrix |
| `patch_cd_noglobal.py` | Removes the now-unused per-iteration cross-DP `any_hit_eos` scalar reduction |
| `patch_padding_bucket.py` | Adds the exact maximum token bucket so a full batch does not round up to the next power-of-two-like padding bucket |

Finally, `patch_fast_detok.py` is present in the 152 patch ConfigMap but is not
invoked by its launcher. The active path is `patch_async_detok.py`; no speedup
should be attributed to the uninvoked fast-detok file. Together with the
uninvoked 152 `patch_perf_v2.py`, this distinguishes “mounted in the ConfigMap”
from “executed by the launch script.” The manifest declares 28 patch files and
the launcher invokes exactly 26: 11 shared names plus the 15 additions above.

#### Landed run-152 optimization ladder

This is the requested mapping from the exact 152 launcher to the optimization
story. The stages are functional groupings, not claims that run 152 isolates
the timing of every patch. Every one of the 26 invoked patch files appears in
exactly one row; selectors shown in the configuration column are what activate
optional behavior.

| Landed step | Exact 152 configuration | Invoked patch implementation | What the step contributes; evidence status |
|---|---|---|---|
| S0. Runtime and continue-decode prerequisites | `enable_continue_decode=true`; trainer `profiler=xplane` requested | `patch_vllm_sampler.py`, `patch_cd_proper.py`, `patch_profiler_fix.py` | Makes Pathways profiling and the top-level fused-decode JIT runnable on the pinned image. This is primarily compatibility/foundation, not an isolated speed claim. |
| S1. Amortize decode boundaries and make deep windows correct | `max_decode_steps=512`, `max_num_batched_tokens=33280`, `CONTINUE_DECODE_EOS_CHECK_INTERVAL=8` | `patch_cd_masking.py`, `patch_cd_report.py`, `patch_cd_noglobal.py`, `patch_padding_bucket.py`, `patch_host_extract.py` | Runs a deeper fused window, preserves per-sequence EOS retirement, removes both unused per-iteration global EOS/active reductions, avoids oversize padding buckets, and vectorizes host EOS extraction. The 23b/149c host traces show the boundary and extraction mechanisms; 152 has no valid TP1 rollout XPlane. |
| S2. TP1 decode kernels and sharded sampling | TP1 x DP128; decode tiles `1,2048,1,2048`; mixed/prefill tiles `512,1024,256,512`; fixed top-k 50/top-p 0.95 | `patch_rpa_blocks.py`, `patch_sharded_sampling.py` | Forwards the three RPA tile families and keeps padded rows on the candidate sampler through `_SS_PADFIX`. TP1 removes TP collectives structurally. The valid 149c TP4 XPlane proves the kernel families and residual sampler guard, but cannot validate the TP1 tile shares. |
| S3. Admission and DP control | `VLLM_BLOCK_SIZE=128`, `VLLM_PREFIX_CACHING=0`, `DP_ROUTING=rr` | `patch_block_size.py`, `patch_dp_rr_routing.py` | Pins the generation contract, disables unhelpful prefix-cache work, and replaces per-request all-rank rank-selection queries with round robin. Host traces support the control-path mechanism; the cumulative run does not isolate each selector. |
| S4. Packed/flash actor training | `MAX_SEQ_TOKEN_PER_TPU=12288`, `attention='flash'`, `scan_layers=True` | `patch_packing_knob.py`, `patch_packer_fast.py`, `patch_flash_sa.py` | Enables real-token packing through the older schema, reduces packer dispatches, and supplies missing attention defaults. The direct run-152 trainer XPlane is the primary landed-program evidence. |
| S5. Move rollout postprocessing and reward work off the serial tail | `math_verify_num_procs=64`, `FAST_EXTRACT=fast`, `ASYNC_REWARD=on`, `ASYNC_LOGPROBS=on` | `patch_reward_parallel.py`, `patch_async_detok.py`, `patch_async_logprobs.py`, `patch_fast_extract.py`, `patch_async_reward.py` | Parallelizes/overlaps detokenization, token-logprob extraction, answer parsing, and reward scoring. Phasetime and run logs establish that these phases are now small; the async implementations retain fallback paths. |
| S6. Shorten weight synchronization and host pauses | `scan_layers=True`; `TUNIX_GC_COLLECT` unset, so patched default is off | `patch_gc_sync.py`, `patch_jit_unroll.py`, `patch_sync_keepkv.py` | Gates per-step Python GC, batches scanned-layer slicing dispatches, and retains the free KV allocation across sync. Perfetto places the resulting sync phase below 1% of the current cycle. |
| S7. Validate the contract and expose residuals | XPlane requested; `PHASED_PROFILING_DIR=""`; no Perfetto exporter call | `patch_lendist.py`, `patch_phasetime.py`, `patch_log_stats.py` | Records length/truncation, phase, and KV/engine evidence. These are measurement patches and receive no speed credit. Run 152 has the trainer XPlane; run 154 supplies same-config Perfetto; the valid rollout capture remains the TP4 run 149c. |

The two mounted-but-inactive files do not belong in a landed step:
`patch_fast_detok.py` is superseded by the invoked async detokenizer, and
`patch_perf_v2.py` is not called, which is why run 154—not run 152—provides the
same-config Perfetto timeline.

#### Configuration held constant

| Control | Both 23b and 152 |
|---|---|
| Workload | Qwen3-0.6B GRPO on OpenMathInstruct-2 |
| Global work | Batch 256, eight generations per prompt, 20 global steps |
| Sampling contract | Temperature 0.8, top-k 50, top-p 0.95 |
| Length contract | `max_target_length=12288`, prefill 4096, effective output cap 8192, KV buffer 4352 |
| KV page | `VLLM_BLOCK_SIZE=128` |
| EOS polling | `CONTINUE_DECODE_EOS_CHECK_INTERVAL=8` |
| Engine capacity controls | `max_num_seqs=256`, vLLM HBM utilization 0.3 |
| Microbatches | Train 32, rollout 256 |
| Trainer structure | `scan_layers=True`, physical-axis splitting allowed |
| Scheduler | `async_scheduling=False` |
| Mesh allocation | Trainer fraction 1.0, sampler fraction 1.0, DP attention disabled |
| Optimizer controls | Learning rate 1e-6, gradient clipping 1.0 |
| Trainer profile request | `profiler=xplane`, skip one step, capture one step |
| Head and proxy resources | Trainer 28 requested/44 limited CPU; proxy 8 requested/16 limited CPU; identical memory limits |
| Pathways/XLA | Same 20260730 JAX 0.10.2 server/proxy pins and proxy XLA flag set |

The attribution consequence is important. Run 23b already had continuous-
decode correctness, reward parallelism, RPA override forwarding, sharded
candidate sampling, and the page-size pin. The later result must not be
described as adding those capabilities from zero. The actual bundle delta is
deeper/broader rollout configuration, the fifteen host/train/synchronization
patches, and finally the TP1-specific topology-and-tile package. The 149c/154
pair isolates that final package from the earlier software ladder, but it does
not isolate topology from the tile change. The rest remains a cumulative
23b-to-152 bundle comparison.

### Near-matched TP4-to-TP1 bridge

Run 149c uses the 136 TP4 recipe; run 154 uses the exact 152 TP1 recipe with
profilers. They keep the fixed device fleet and the rest of the champion stack
constant, but change rollout partitioning **and** the RPA tile set:

```diff
-rollout_tensor_parallelism=4
-rollout_data_parallelism=32
+rollout_tensor_parallelism=1
+rollout_data_parallelism=128

-RPA_V3_DECODE_BLOCK_SIZES=1,8192,1,4096
+RPA_V3_DECODE_BLOCK_SIZES=1,2048,1,2048
+RPA_V3_MIXED_BLOCK_SIZES=512,1024,256,512
+RPA_V3_PREFILL_BLOCK_SIZES=512,1024,256,512
```

The near-matched shared recipe includes:

```text
model                         = qwen3-0.6b
global train batch            = 256 prompts
generations per prompt        = 8
sampling                      = temperature 0.8, top-k 50, top-p 0.95
output cap                    = 8192
KV page size                  = 128
continue-decode depth         = 512
EOS check interval            = 8
max batched-token budget      = 33280
prefix caching                = disabled
DP routing                    = round robin
trainer attention             = flash-style
packed-token budget per TPU   = 12288
```

The observed delta is therefore a TP1-package result. The disappearance of TP
model collectives still supplies direct mechanism evidence for TP1, but the
approximately 5% end-to-end number must not be labeled topology-only. Any
later strict topology retest should use one VMEM-safe common RPA tuple on both
sides; otherwise it must continue to be reported as a topology-plus-tiles
package comparison.

## Phase comparison

### Sustained endpoint

The current TP1 endpoint is approximately 53 baseline units, or about 47%
below the 23b steady median. The optimized TP4 long run is approximately 56
baseline units. Thus most of the baseline-to-current improvement predates the
final TP1-specific package: TP1 plus its required tile fixes contributes about
a 5% incremental end-to-end reduction, explained by a roughly 6% rollout
reduction while actor training and synchronization remain effectively
unchanged. The package delta is measured; a topology-only fraction is not.

### Profiled TP1-package bridge

The short, trace-enabled pair reproduces the same result:

| Component | TP4 run 149c | TP1 run 154 | Interpretation |
|---|---:|---:|---|
| Complete cycle | About 56 baseline units | About 52 baseline units | TP1-plus-tiles package is roughly 5% lower |
| Rollout | About 48 baseline units | About 45 baseline units | Nearly all of the package gain |
| Actor training | About 5 baseline units | About 5 baseline units | Flat |
| Weight sync | Less than 1 baseline unit | Less than 1 baseline unit | Flat/noisy |
| Other transitions | About 2 baseline units | About 2 baseline units | No material package difference |

The TP1 Perfetto complete-cycle decomposition is approximately:

| Component | Share of cycle | Headroom implication |
|---|---:|---|
| Rollout | 86% | Primary target |
| Actor train wall | 9% | Secondary target |
| Advantage/reward | About 1% | Already small |
| Weight sync | Less than 1% | Already small |
| Loading and phase transitions | About 3% | Small orchestration opportunity |

An impossible optimization that deletes all non-rollout work would improve
the current cycle by only about 16%. Any substantial next gain must shorten
rollout itself.

## Trace map

These links are access-controlled and expose raw benchmark measurements. Keep
them out of the public report.

| Artifact | Internal link | Use |
|---|---|---|
| **Baseline rollout XPlane, run 23b** | [decode-only session](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-23b/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_22_06_21_47/?project=cloud-tpu-shared-capacity) | Baseline Python host path, Pathways client/server activity, DP scheduler, and decode program |
| Baseline trainer XPlane, run 23b | [trainer session](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-23b/tensorboard/plugins/profile/2026_08_22_06_27_19/?project=cloud-tpu-shared-capacity) | Dense/padded trainer and collectives |
| Baseline phase timeline, run 23b | [Perfetto protobuf](https://console.cloud.google.com/storage/browser/_details/ubench-logs/tianyu-rl-23b/perf_traces/perfetto_trace_v2_1787379652.pb?project=cloud-tpu-shared-capacity) | Baseline whole-cycle phase attribution |
| Baseline phase and semantic logs, run 23b | [run artifacts](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-23b?project=cloud-tpu-shared-capacity) | Normalization metric, phasetime, lengths, rewards, and patch banners |
| TP4 rollout XPlane, run 149c | [decode-only session](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-149c/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_06_18_08/?project=cloud-tpu-shared-capacity) | Decode program, RPA, collectives, sampling, host engine window |
| **Current trainer XPlane, run 152** | [trainer session](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-152/tensorboard/plugins/profile/2026_08_31_08_55_05/?project=cloud-tpu-shared-capacity) | Direct evidence for the packed/flash actor program in the sustained endpoint |
| Optimized TP4 trainer XPlane, run 136 | [trainer session](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-136/tensorboard/plugins/profile/2026_08_30_05_57_04/?project=cloud-tpu-shared-capacity) | TP4 confirmation of the packed/flash train program and collectives |
| TP4 phase timeline, run 149c | [Perfetto protobuf](https://console.cloud.google.com/storage/browser/_details/ubench-logs/tianyu-rl-149c/perf_traces/perfetto_trace_v2_1788157043.pb?project=cloud-tpu-shared-capacity) | Whole-cycle TP4 phase attribution |
| TP1 phase timeline, run 154 | [Perfetto protobuf](https://console.cloud.google.com/storage/browser/_details/ubench-logs/tianyu-rl-154/perf_traces/perfetto_trace_v2_1788167363.pb?project=cloud-tpu-shared-capacity) | Whole-cycle TP1 phase attribution |
| TP1 diagnostic XPlane, run 154 | [decode-only session](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-154/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_09_10_14/?project=cloud-tpu-shared-capacity) | Profiler-failure diagnosis only; not valid for kernel shares |
| Sustained TP1 logs, run 152 | [run artifacts](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-152?project=cloud-tpu-shared-capacity) | Current-endpoint phase, length, reward, and TensorBoard distributions |
| Stock packed-assembly control, run 43c | [TensorBoard artifacts](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-43c/tensorboard?project=cloud-tpu-shared-capacity) | One-change control for the NumPy-first packer |
| NumPy-first packed-assembly probe, run 44 | [TensorBoard artifacts](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-44/tensorboard?project=cloud-tpu-shared-capacity) | One-change global result and phase cross-check for `patch_packer_fast.py` |
| DP-scheduler architecture cross-check | [Gemma G4R7B end-to-end study](https://github.com/alisachen-google/tpu-rl/blob/f259b6895cfb641286a189183a3d0e8fb4e61647/gemma4/G4R7B_DP_SCHEDULING_E2E.md) | Explains the shared parent/child scheduler, padding, Pathways, and barrier structure; Gemma timings are not imported into the Qwen budget |

Open the XPlane session by downloading the session directory and pointing
TensorBoard/XProf at its parent:

```bash
gcloud storage cp --recursive \
  'gs://ubench-logs/tianyu-rl-23b/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_22_06_21_47' \
  ./qwen23b-profile/
tensorboard --logdir=./qwen23b-profile --port=6006
```

Download either Perfetto `.pb` and drag it into
[ui.perfetto.dev](https://ui.perfetto.dev/).

## Trace findings

### Baseline CPU, Pathways, and DP-scheduler attribution

The baseline host analysis uses the complete run-23b merged host XPlane, not
only the capped Chrome JSON export. It takes the median over 15 complete
decode engine windows and excludes the final profile-stop window, which
ends during input preparation. The shares below are of one host-visible
`llm_engine.step`, not of the global RL step and not of total CPU capacity.

| Baseline main-thread component | Median share of engine window | Interpretation |
|---|---:|---|
| Blocking `device_get` | About 82% | Main thread waits for a remote result; includes queueing, device execution, rank tail, transfer, and materialization |
| Per-request valid-token extraction | About 5.5% | Real Python/NumPy postprocessing headroom |
| DP scheduler: schedule + grammar + output update | About 3.2% | Real but secondary control-plane wall; child send/receive spans are not added again |
| TPU-runner input preparation | About 2% | Device-put, sharding, metadata, and block-table preparation |
| vLLM output-object processing | About 1% | Request state and output construction after engine return |
| Stats/logging | Less than 1% | Visible but not a primary target |
| Other wrappers/conversion | Remainder | Small leaves and profiler overhead |

This table deliberately does not label the 82% blocked span “Pathways CPU.” A
Python `device_get` is an inclusive synchronization point. It says the result
was not ready for the caller; it does not say whether the missing time was TPU
execution, Pathways admission/launch, a slow DP rank, host-buffer transfer, or
CPU deserialization.

#### Evidence boundary for the combined host analysis

The [G4R7B DP-scheduling study](https://github.com/alisachen-google/tpu-rl/blob/f259b6895cfb641286a189183a3d0e8fb4e61647/gemma4/G4R7B_DP_SCHEDULING_E2E.md)
provides a code-level end-to-end model for the same TPU Inference Pathways
architecture. Its pinned source revision is newer than the run-23b image, so
it is used as an architectural cross-check, not as proof of byte-identical
code. The Qwen trace contains the corresponding `DPScheduler`, `_send_command`,
`_get_result`, `_prepare_inputs`, `_process_continue_decode_outputs`, and
Pathways host-buffer spans, which corroborates the control flow.

| Evidence class | What may be concluded here |
|---|---|
| **Qwen-measured** | Run-23b component shares, 15-window distribution, command/result counts, and the extraction-driven outlier |
| **Shared architecture, trace-corroborated** | One engine/model-runner parent, one `DPScheduler` parent, one CPU scheduler child per logical DP rank, and one global JAX/Pathways SPMD decode per window |
| **Gemma-measured only** | G4R7B's absolute window spans, occupancy sample, host-buffer event counts, scheduler stalls, and sampled CPU use; none enter Qwen's normalized budget |
| **Qwen hypothesis requiring new counters** | How much of Qwen's `device_get` is queueing, slow-rank tail, device execution, lookup, transfer, or host materialization |

The most important correction from the shared architecture is that DP32 does
not mean 32 independent model engines or 32 Pathways clients. Run 23b has 32
logical DP scheduler children, but the parent merges their decisions into one
TP4 x DP32 program over all 128 sampler devices. Likewise, TP1 x DP128 has 128
scheduler children feeding one global program, not 128 model engines.

#### What one long Qwen host window is waiting for

The host critical path is a sequence of three rendezvous around one global
device call:

```text
request admission (outside most profiled engine windows)
  stock 23b: query many ranks per request; 152: direct round-robin assignment
        |
        v
parent sends SCHEDULE to D CPU scheduler children
        |
        v
barrier 1 = wait for the slowest scheduler child
        |
        v
parent merges outputs, groups requests by sticky rank,
and builds rank-major D x bucket(busiest_rank) host arrays
        |
        v
device_put + one global JAX/Pathways continue-decode submission
        |
        v
barrier 2 = device_get waits for global completion and host visibility
        |
        v
parent restores request order, trims EOS, updates host state,
and splits the global output back into D rank-local objects
        |
        v
parent sends UPDATE_FROM_OUTPUT to D scheduler children
        |
        v
barrier 3 = wait for the slowest scheduler update before the next window
```

This can be summarized without double-counting overlapping trace spans as:

```text
T_window ~= max(rank schedule)
          + parent merge/pack/put
          + Pathways queue/launch
          + global device completion/tail
          + host-buffer fan-in/materialization
          + parent output/state work
          + max(rank update)
```

The children schedule and update in parallel, so their durations are not
summed; each barrier pays the maximum. Conversely, the global `device_get`
wait is inclusive of several middle terms and must not be added to a sampled
device interval. This is why a long host-visible window can coexist with no
large serial Python leaf and without saturating all head CPU cores. G4R7B
independently observed that low-core-use, long-wall pattern; Qwen still needs
sampled CPU counters before making the same utilization claim.

| Critical-path stage | Direct run-23b evidence | Architectural interpretation and scaling |
|---|---|---|
| Admission before decode windows | The manifest uses the stock state-aware rank finder; the 15-window table does not isolate admission | The stock path performs per-request rank-state probes, exposing approximately `O(requests * DP)` Pipe traffic. Run 152's `DP_ROUTING=rr` removes these all-rank queries. |
| Schedule and update rendezvous | Scheduler, grammar, and output-update envelopes total about 3.2% of a median window | A parent fans out to DP children and waits for every response twice per steady window; latency follows the slowest child, while serialization/message work grows with DP. |
| Merge, rank-major packing, and input construction | `_prepare_inputs` is about 2% | One parent groups all requests, finds the busiest rank, and pads every rank to the same supported request bucket before creating global NumPy/JAX metadata. |
| Global execution and result visibility | `device_get` is about 82% | One SPMD program spans the whole TP4 x DP32 mesh. The wait can include proxy admission, launch, the slowest device/rank, HostBufferStore lookup, transfer, and host materialization. |
| Parent output processing | First-EOS extraction is about 5.5%; output-object work is about 1% | One parent restores order and mutates global request/token/attention state. This remains serial even though the preceding model rows were data parallel. |
| Split and child-state update | Included in the scheduler envelope and command/result counts | The parent constructs rank-local output objects, sends one to every child, and cannot begin the next window until the slowest child updates KV/request state. |

The baseline trace also makes the per-window DP traffic concrete. A full
window contains 128 `_send_command` and 128 `_get_result` calls on the engine
thread. With DP32, that is four rank-wide command/response rounds per window.
The shared architecture establishes a minimum steady sequence of `SCHEDULE`
and `UPDATE_FROM_OUTPUT`; the two extra baseline rounds are consistent with
the grammar and unfinished-request broadcasts visible in the traced control
path. The nested send/receive spans are already inside scheduler wall and
therefore must not be summed as an additional cost.

#### Why higher DP can lengthen host orchestration

For this workload, the configured rollout contains 2,048 requests. The
average initial work per logical rank falls from 64 requests at DP32 to 16 at
DP128, but several surrounding structures do not shrink with useful work:

| Host/control term | Scaling model | Qwen implication |
|---|---|---|
| Stock admission probes | Approximately `O(requests * DP)` | A potentially material 23b setup cost; removed by 152 round-robin routing, though not isolated in the window shares |
| Minimum schedule/update messages | `O(DP)` per window | Two mandatory rank-wide command/result rounds remain in 152 and quadruple in count from DP32 to DP128 |
| Scheduler barriers | `max(child latency)` | More children increase exposure to a slow process, descheduling, pipe backpressure, or uneven KV/request state |
| Maximum host-batch capacity | `DP x max_num_seqs` | With `max_num_seqs=256`, configured capacity grows from 8,192 slots at DP32 to 32,768 at DP128 for 2,048 real requests; persistent metadata should be audited and right-sized |
| Executed request rows | `DP x bucket(max active requests on any rank)` | A single busy rank raises the padded shape for every rank; occupancy worsens as completion lengths diverge and ranks drain unevenly |
| Result dependency | One global sharded result | Host progress waits for all required shards and their materialization, even if most ranks finish early |

The `patch_padding_bucket.py` fix in 152 prevents a full token batch from
rounding to the next oversized token bucket; it does **not** remove
request-axis padding to the busiest DP rank. Similarly,
`patch_host_extract.py` removes repeated first-EOS scans, but it does not
compact the global result before `device_get` or eliminate the remaining
single-parent request-state updates.

#### Routing lesson: Qwen and Gemma are deliberately different

The Gemma study found that naive round-robin routing regressed its DP64
rollout because it discarded useful prefix/load locality. That result must not
be copied blindly to Qwen. Run 152 combines `DP_ROUTING=rr` with explicitly
disabled prefix caching and a homogeneous 2,048-request math batch; its patch
rationale records the avoided all-rank probes as material at Qwen scale. The
23b-to-152 comparison still cannot assign an isolated gain to routing because
prefix caching, DP degree, window depth, and the rest of the host stack change
together.

The transferable rule is therefore conditional:

- keep Qwen's low-overhead round robin as the current control;
- do not restore per-request all-rank probes;
- if rank-drain skew is material, test one periodically refreshed parent-side
  load snapshot with bulk assignment;
- if prefix caching is re-enabled, route all generations of a prompt as a
  locality group and measure cache hit rate as well as rank tail;
- judge routing on rollout median, mean, p90/p99, padded-row occupancy, and
  completion/reward gates—not on scheduler microbenchmarks alone.

#### What the Pathways planes do and do not prove

The native Pathways client plane distributes work across the IFRT request
processor, dispatcher queues, host-buffer store/lookup workers, and pipe
callbacks. There is no single serialized Pathways client CPU span covering the
engine window. On one sampled worker, the captured local decode module is much
shorter than the client-side `device_get` wait. That establishes a large
remote/control/tail envelope, but one worker cannot represent the slowest rank
across the fleet.

The profile contains traced function spans, not sampled per-core utilization.
It therefore cannot prove that the 28-core requested/44-core-limited head pod
was CPU saturated. The defensible conclusions are:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Pathways client CPU is the baseline bottleneck | **Not supported** | Native client work is distributed; no serial client leaf explains the engine window |
| A Pathways-mediated remote wait is the dominant host-visible critical path | **Supported** | `device_get` owns about four-fifths of the baseline engine window |
| Per-window DP schedule/update work is the primary bottleneck | **Rejected** | High-level DP scheduler wall is only about 3.2% of the baseline window; request admission occurs outside most profiled engine windows and is a separate term |
| DP scheduling is meaningful scaling headroom | **Supported** | Rank command count scales with DP degree and remains on the critical thread |
| DP32 means 32 independently progressing model engines | **Rejected** | The children own logical scheduler/KV state; one parent merges their work into one global Pathways model execution and result |
| Slowest-rank tail or busiest-rank padding explains most of `device_get` | **Unresolved** | The architecture exposes both mechanisms, but the Qwen trace lacks per-rank completion and real-versus-padded-row counters |
| Round robin is universally better than state-aware routing | **Rejected** | It avoids Qwen's all-rank admission probes, but the Gemma control shows that lost prefix/load locality can outweigh that saving in a different workload |
| Host output extraction is meaningful baseline headroom | **Supported** | Median share is about 5.5%; the trace outlier is almost entirely extraction growth |
| Async detokenization is a median critical-path bottleneck | **Not supported** | Aggregate CPU demand is visible on worker threads but overlaps the engine path |

One baseline capture window is especially useful for causality. Its
`device_get` duration stays in the normal band while valid-token extraction
grows by several times and explains almost all of the engine-window excursion.
That is a host postprocessing tail, not a Pathways or device stall. The current
stack removes `_extract_valid_tokens_host` from the captured engine thread via
batch extraction.

Representative delivered transformation:

```python
# Baseline shape: repeated Python scans and object construction by request.
host_tokens = np.asarray(jax.device_get(generated_tokens))  # [steps, batch]
for request_idx, request in enumerate(requests):
    valid = extract_valid_tokens_host(
        host_tokens[:, request_idx], request.decode_length
    )
    update_request(request, valid)

# Optimized shape: one batch conversion, vectorized last-real-token lookup,
# then only the unavoidable request-state updates remain in Python.
host_tokens = np.asarray(jax.device_get(generated_tokens))
rows = np.arange(host_tokens.shape[0])[:, None]
real = (host_tokens != pad_token_id) & (rows < generated_count[None, :])
last_row = np.max(np.where(real, rows, 0), axis=0)
last_token = host_tokens[last_row, np.arange(host_tokens.shape[1])]
```

Why this improves performance: it replaces thousands of short Python calls
with array operations, reduces interpreter/GIL overhead, and removes the
observed extraction-driven tail. The optimized host traces validate the
mechanism because the per-request extraction leaf is absent.

#### Baseline versus optimized host path

The engine windows have different static decode depths, so their absolute
durations must not be compared. Shares and rank-command counts expose the
structural change:

| Trace | Decode depth | `device_get` share | High-level DP wall | Valid-token extraction | Send/receive calls per captured full window |
|---|---:|---:|---:|---:|---:|
| 23b baseline, TP4 x DP32 | 128 | About 82% | About 3.2% | About 5.5% | 128 / 128 |
| 149c optimized TP4 x DP32 | 512 | About 93% | About 1.8% | Not present | 64 / 64 |
| 154 optimized TP1 x DP128 | 512 | About 92% | About 2.5% | Not present | 256 / 256 |

The baseline row is a 15-window median. The optimized rows are one complete
engine window from each capped Chrome export, so they support structural
attribution and command-count scaling rather than distributional claims. The
run-154 host window is usable even though its device XPlane captured the wrong
program.

At fixed TP4, the optimized stack halves rank command/response calls while
making a window four times deeper. That is roughly an eightfold reduction in
rank-control calls per maximum decode iteration. TP1 then quadruples the rank
count, so its two rank-wide rounds become 256 sends and receives. Scheduler
wall rises versus optimized TP4 but remains secondary; this is a scaling tax
to measure, not evidence that DP scheduling dominates.

The larger `device_get` *share* in the optimized traces is mechanically
consistent with removing host extraction and repeated control boundaries. It
is not evidence that Pathways regressed, but it is also not proof that all of
the wait is useful decode: queueing, rank tail, host-buffer fan-in, and
materialization remain unresolved inside the inclusive span.

### TP4 decode device composition

The usable run-149c decode capture attributes the decode XLA program
approximately as follows:

| Leaf group | Approximate share of decode self time | Meaning |
|---|---:|---|
| RPA decode custom calls | 45% | Largest kernel family |
| TP model/sampling reductions | 20% | Cost TP1 is intended to remove |
| Cross-DP sampling safety guard | 7% | Remaining avoidable per-token collective |
| Candidate sampling fusion | 5% | Candidate merge/top-k work |
| Candidate all-gather | Less than 1% | Sharded vocabulary communication is already small |
| Other compute/data movement | 22% | Matmul/fusions/formatting and smaller operations |

The active RPA custom-call name is
`RPAd-p_128-bq_1_1-bkv_8192_4096`, confirming that the intended page, load
tile, and compute tile reached the kernel.

The large scalar reduction is `all-reduce.125`, with source annotation
`jit(_decode_core_impl)/while/body/reduce_and`. Its replica group is the DP
axis, not the TP axis. Source correlation identifies it as the dynamic
whole-batch safety check in the sharded sampler. It survives even though the
benchmark sampling policy is fixed and safe for the candidate path.

#### Code-to-trace anchors for the decode-loop and sampling collectives

The numeric HLO suffix is not a stable code identifier. For example, the same
logical reduction can appear as `all-reduce.68`, `.125`, or `.127` after a
different topology, patch set, or compiler schedule. The source expression,
XLA source annotation, shape, and replica groups are the durable identity. The
four relevant anchors are:

| Trace operation | Source-level anchor | Run-152 interpretation |
|---|---|---|
| Deep-window active-row reduction | The earlier [`patch_cd_masking.py`](../patches/patch_cd_masking.py#L95-L105) generated `jnp.any(_cd_active)` in `_decode_core_impl::cond_fn`. This is patch-introduced behavior, not the current upstream condition. | The exact 152 revision replaces the condition with `return _cd_i < max_decode_steps`; this cross-DP scalar reduction is already removed. |
| Per-iteration EOS reduction | Upstream [`decode_loop.py::_update_loop_state`](https://github.com/vllm-project/tpu-inference/blob/f6ad5ecb1ca359c721b01fd0d7edd90b5cf0ac7e/tpu_inference/runner/decode_loop.py#L80-L85) computes `any_hit_eos = jnp.any(jnp.logical_and(active_mask, is_eos))`, returns it, and carries it through the while loop. The repository's guarded equivalent rewrite is visible in [`patch_cd_noeos.py`](../patches/patch_cd_noeos.py#L42-L49). | The supplied 152 `patch_cd_noglobal.py` replaces the expression with `jnp.array(False)`. This is valid because the 152 `cond_fn` is fixed-depth and no longer consumes the carried EOS flag. The trace-local `all-reduce.68` attribution refers to this expression when its source annotation and DP replica group agree. |
| Full-vocabulary logits gather | Stock [`sampling.py::sample`](https://github.com/vllm-project/tpu-inference/blob/f6ad5ecb1ca359c721b01fd0d7edd90b5cf0ac7e/tpu_inference/layers/jax/sample/sampling.py#L118-L123) explicitly constrains sampling logits to `P(ATTN_DATA, None)`. Replicating the vocabulary dimension induces the full-vocabulary all-gather when the stock/fallback branch runs. For 128 rows, Qwen's 151,936-entry vocabulary, and `f32`, the logical slab is `128 * 151936 * 4 = 77,791,232` bytes: 77.8 decimal MB or 74.2 MiB. | This is the source pointer for the pre-`_SS_PADFIX`/fallback `all-gather.5`, not for the small candidate gather. In a TP-greater-than-one control, that fingerprint means the guard selected fallback or the patch did not apply. Run 152 is TP1, so its size-one model axis removes this inter-device gather structurally; branch selection cannot be inferred from the absence of `all-gather.5` alone. |
| Dynamic sampling safety reduction | The sharded wrapper forms a DP-sharded Boolean vector and executes `safe = jnp.all(...)` before `lax.cond`; see [`patch_sharded_sampling.py`](https://github.com/alisachen-google/tpu-rl/blob/634a492658de243a69086cc60df2b7152d31932c/patches/patch_sharded_sampling.py#L196-L211). The exact 152 form applies `_SS_PADFIX` first and then evaluates `jnp.all((_tk <= 64) & (_tp > 0))`. | Reducing the global DP-sharded vector to one scalar produces `while/body/reduce_and`, observed as trace-local `all-reduce.125`. This guard remains in 152 and is the proposed static-plan optimization. |

The candidate path has different, deliberately small collectives: its
[`_ss_merged_candidates`](https://github.com/alisachen-google/tpu-rl/blob/634a492658de243a69086cc60df2b7152d31932c/patches/patch_sharded_sampling.py#L112-L122)
all-gathers only the per-TP-shard top-64 values and token IDs. It must not be
confused with the 78 MB full-vocabulary fallback gather. Consequently, the
pair “full-vocabulary `all-gather.5` plus dynamic `all-reduce.125`” describes
the guarded fallback mechanism in an earlier TP-greater-than-one trace, not two
current run-152 TP1 costs. The expected 152 hot path retains the cross-DP
scalar guard and the candidate-merge code; its TP top-64 gathers are
size-one identities rather than inter-device communication.

The remaining model all-reduces are across the TP axis and appear after the
attention/output projections. TP1 removes those by construction. This is one
directly observed mechanism behind the measured TP1-package gain; it does not
quantify the separate contribution of the RPA tile changes.

### Roofline interpretation

XProf classifies the captured TP4 decode program on the HBM side of its formal
roofline. Treat that classification as directional, not a utilization claim:
the RPA custom calls consume roughly 45% of device self time but expose zero
modeled FLOPs and bytes to the report. Consequently, the program-level
arithmetic intensity and roofline efficiency omit the largest kernel family.

The observed package result and the TP mechanism are consistent with the
roofline:

| Topology | Benefit | Cost | Observed conclusion |
|---|---|---|---|
| TP4 x DP32 | Sharded weights and more requests per model replica | About one-fifth of decode leaf time in TP reductions | Slower nearest-control rollout |
| TP1 x DP128 | No TP model collectives | Full weights per chip, fewer requests per replica, 128-way DP control | TP1-plus-tiles package is about 6% lower in rollout, not 20% |
| TP2 x DP64 | Potential middle point | Retains some TP communication | Worth one controlled scout |

The TP4 collective share must not be equated to the much smaller package-level
end-to-end delta. Even in a pure topology comparison, TP1 would give back part
of the communication benefit through less local batching/statistical
multiplexing, different matmul shapes, full-model weight traffic, and greater
DP fan-out; here the RPA tile change is an additional confound.

#### Boundary-amortization roofline: decode depth 128 to 512

Window depth is not a classical FLOP/byte roofline. It is a latency/Amdahl
roofline over a fixed cost paid at every host/device boundary. For fixed useful
token work, model one rollout as:

```text
T(d) = T_useful_tokens + W(d) * B + M(d)

d     static continue-decode depth
W(d) number of global windows
B     fixed schedule/merge/launch/fan-in/update cost per window
M(d) masked-row, padded-row, and delayed-refill cost at depth d
```

If long requests make `W(512) ~= W(128) / 4`, define
`H128 = W(128) * B`, `f = H128 / T(128)`, and
`m = (M(512) - M(128)) / T(128)`. Then:

```text
T(512) / T(128) ~= 1 - 0.75 * f + m
speedup             ~= 1 / (1 - 0.75 * f + m)
```

Thus “four times deeper” does **not** imply a fourfold rollout speedup. It can
remove at most three quarters of the old boundary term, and only wins when the
extra masking/refill tax is smaller than that removal.

The run-23b depth-128 host trace supplies a gross bound. The final column also
multiplies by the baseline rollout share, so it is expressed in this report's
100-unit global normalization:

| What is treated as fixed boundary cost | Share of a depth-128 engine window | Ideal engine-window reduction | Approximate gross global credit |
|---|---:|---:|---:|
| Schedule/update rendezvous only | 3.2% | 2.4% | About 1.9 baseline units |
| Rendezvous + input preparation + output-object work | About 6.2% | About 4.7% | About 3.6 baseline units |
| Add old per-request EOS extraction | About 11.7% | About 8.8% | About 6.8 baseline units, but this overlaps the separate vectorized-extraction patch |
| Treat every non-`device_get` span as fixed | About 18% | About 13.5% | About 10.5 baseline units; an intentionally loose ceiling |

The defensible depth-only planning credit is therefore about **2-4 baseline
units gross**, not the loose 10.5-unit envelope. The lower end counts only the
directly visible scheduler boundary; the upper end adds preparation and output
construction. EOS extraction is excluded because run 152 separately
vectorizes it, and the opaque part of `device_get` is excluded because it mixes
launch/fan-in overhead with useful device execution.

The optimized TP1 host capture gives a consistent reverse check. At depth 512,
high-level DP schedule/update wall is about 2.5% of one captured window. If the
same useful 512 iterations were split into four depth-128 calls with unchanged
per-boundary cost, the scheduler-only gross reduction would be:

```text
b = B / T512 = 0.025
reduction from shallow total = 3b / (1 + 3b) ~= 7% of rollout
global equivalent           ~= 7% * 86% ~= 6% of the current cycle
```

This is a one-window counterfactual, not an isolated run result. It excludes
the additional masked-slot work and delayed scheduler refill caused by the
deeper window. The continuous-decode correctness patches make finished slots
inert and retirement correct; they do not make masked TPU row-iterations free.
The shared Gemma scheduler study corroborates exactly where the repeated
barriers occur, but its absolute timings are not imported into this Qwen
budget.

The break-even test for any future depth change is therefore:

```text
boundary time removed > extra masked/padded device time + lost refill value
```

Record actual iterations, real and padded row-iterations, scheduler wall,
device-active time, and `engine_generate / generated_tokens` for every window.
For 512 to 1024, the scheduler-only gross ceiling from the current 2.5% share
is only about 1.25% of rollout, or about 1.1% of the whole cycle, before the
masking tax. That is diminishing-return territory rather than a new major
lever.

#### NumPy packed-chunk assembly: measured host-dispatch credit

The packer has stronger attribution than window depth. Run 44 changes exactly
one thing relative to run 43c: `patch_packer_fast.py`. The implementation
replaces per-bin JAX construction with NumPy fills followed by one array
conversion per feature.

| Packer quantity | Stock packed assembly | NumPy-first assembly | Interpretation |
|---|---:|---:|---|
| Host-triggered array operations per step | About 13,900 | About 72 | Approximately 193x fewer dispatches |
| Estimated packer host cost | About 4.2 baseline units | About 0.5 baseline units | About 3.7 units of causal host work removed |
| One-change global median | Run 43c control | Run 44 probe | About 4.5 baseline units lower, gross observed |

The gross global delta is slightly larger than the packer-cost delta because
rollout also moved within normal run noise. A conservative causal credit is
**3.5-4 baseline units**. Almost none of this is a faster actor TPU kernel: it
removes thousands of tiny host-triggered JAX dispatches from the dequeue/data
assembly path. The packed tensors are semantically identical.

This optimization is already active in run 152. Even deleting the recorded
fast-packer residual would be worth no more than about 0.5 baseline units, and
a realistic improvement is smaller. Combining the depth model and packer
control suggests roughly 5.5-8 landed baseline units, but that is a planning
ledger rather than a clean attribution of the full 23b-to-152 bundle.

### Optimized host rollout trace

The run-149c/run-154 host traces show:

- The blocking output-processing span is mostly an inclusive `device_get`, so
  it must not be counted again as independent host compute.
- Scheduler send/receive and output-object construction are visible but small
  relative to the captured engine window. DP128 raises scheduler wall versus
  optimized DP32, but not enough to become the primary bottleneck.
- Detokenization runs on asynchronous threads and largely overlaps the engine
  path.
- The remaining large host risk is scale, not one obvious serial Python leaf:
  TP1 increases the number of logical DP scheduler children from 32 to 128,
  while still feeding one global Pathways model execution.

This makes “optimize a random Python function” a weak next strategy. The next
host profile must measure per-rank fan-out, queue depth, active-sequence
imbalance, Pathways wait buckets, sampled CPU utilization, and critical-path
waits under TP1.

The asynchronous tokenizer pool carries nontrivial aggregate CPU demand, but
aggregate worker-thread time is not additive to wall time. Treat it as a
contention/tail hypothesis only. Before changing worker count or affinity,
record pool queue latency, runnable threads, process CPU, and whether engine
windows lengthen when the pool is busy. A useful controlled probe is:

```text
control: current detokenizer worker count
probe A: half the workers, bounded queue
probe B: current workers pinned away from Pathways/IFRT service threads
gate: lower engine p90/p99 with unchanged median tokens and no detokenizer tail
```

Without that correlation, reducing or increasing tokenizer workers can simply
move the same work between overlap and the end-of-rollout tail.

### Trainer and phase trace

The direct run-152 trainer XPlane is the primary actor-program record; run 136
is a TP4 confirmation. Across the eight TensorCore `XLA Ops` lines in the
run-152 profile, event/core time is distributed approximately as follows:

| Trainer device leaf | Share of event/core time |
|---|---:|
| Splash-attention custom call | 50.6% |
| All-gather | 25.1% |
| Dynamic-update-slice | 15.5% |
| Other fusions | 7.6% |
| Copy | 1.2% |

These are overlapping core-time proportions, not additive actor wall shares.
The active splash call uses query blocks 1024, KV blocks 2048, KV compute
blocks 2048, the corresponding backward blocks 1024/2048, head-dimension-minor
layouts, and an unfused backward kernel.

The current actor train wall is only about 9% of the cycle. Even mapping a
20% splash-attention leaf improvement directly onto actor wall gives a
generous Amdahl ceiling of `0.09 * 0.506 * 0.20 ~= 0.9%` end to end, before
accounting for overlap. Perfetto also shows that roughly one-fifth of train
wall is gap between successive TPU microbatch envelopes. Deleting every such
gap is only an approximately 2% global ceiling; a realistic compiled-dispatch
or trainer-kernel improvement is lower.

Advantage/reward and weight synchronization are each below 1% of the current
cycle. They should remain correctness-monitored but are no longer primary
performance projects.

### TP1 profiler failure

Three TP1 phased-profile attempts did not capture a representative steady
decode window. The run-154 diagnostic session is dominated by a 128-way
`jit_split`/reshard program and contains, at most, a tiny partial decode
envelope. It cannot support TP1 RPA, sampling, or collective percentages.

This is why the report uses run 149c for decode-kernel attribution and run 154
only for whole-cycle Perfetto attribution. Presenting the run-154 XPlane as a
valid TP1 decode profile would be misleading.

## Evidence-to-action matrix

The table below is the provenance key for every action. “Config” means the
next experiment changes a launch flag or environment value while keeping code
fixed. “Patch” means our source behavior must change. “Measurement” means the
first update adds observability and receives no performance credit. The final
column separately maps the action to the exact row labels in the historical
[configuration differences](#exact-performance-material-configuration-differences),
[patch inventory](#explicit-patch-inventory), and
[held-constant controls](#configuration-held-constant). “No landed row” means
the action is proposed headroom rather than a change already present in run
152.

| Item | Direct trace or log evidence | Proposed next update mechanism | Mapping to the 23b-versus-152 comparison tables |
|---:|---|---|---|
| 1. Remove sampling guard | [149c decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-149c/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_06_18_08/?project=cloud-tpu-shared-capacity) | **Our patch:** add a statically selected sampling executable; an optional config selector chooses it. Run 152 lands `_SS_PADFIX`, but not this static-plan change. | Held-constant row **Sampling contract**; revised shared-patch row **`patch_sharded_sampling.py`**. `_SS_PADFIX` is the landed precursor; removal of the residual `jnp.all`/`lax.cond` guard is proposed headroom. |
| 2. Validate/refine TP1 RPA tiles | [149c valid decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-149c/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_06_18_08/?project=cloud-tpu-shared-capacity); [154 diagnostic TP1 XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-154/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_09_10_14/?project=cloud-tpu-shared-capacity) | **Config:** sweep one of the decode/mixed/prefill RPA tuples at a time; no new kernel patch if forwarding remains active. | Config-difference rows **RPA decode block tuple**, **RPA mixed block tuple**, and **RPA prefill block tuple**; held-constant row **KV page**; shared-patch row **`patch_rpa_blocks.py`**. The TP1-specific tiles are already landed, but lack valid TP1 kernel evidence. |
| 3. Split Pathways wait | [23b baseline decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-23b/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_22_06_21_47/?project=cloud-tpu-shared-capacity); [149c optimized decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-149c/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_06_18_08/?project=cloud-tpu-shared-capacity); [shared scheduler architecture](https://github.com/alisachen-google/tpu-rl/blob/f259b6895cfb641286a189183a3d0e8fb4e61647/gemma4/G4R7B_DP_SCHEDULING_E2E.md) | **Our measurement patch first; our performance patch second:** add cross-layer wait buckets plus padded-row/host-buffer volume, then change transfer, capacity sizing, or launch behavior only if the corresponding bucket is material. | Config rows **Continue-decode depth** and **Maximum batched tokens**, held row **Engine capacity controls**, and 152-only patches **`patch_host_extract.py`** and **`patch_padding_bucket.py`** explain the changed envelope. Correlation and result compaction are not landed. |
| 4. DP128 fan-out/tail | [23b baseline decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-23b/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_22_06_21_47/?project=cloud-tpu-shared-capacity); [149c TP4 decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-149c/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_06_18_08/?project=cloud-tpu-shared-capacity); [154 TP1 diagnostic XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-154/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_09_10_14/?project=cloud-tpu-shared-capacity); [shared scheduler architecture](https://github.com/alisachen-google/tpu-rl/blob/f259b6895cfb641286a189183a3d0e8fb4e61647/gemma4/G4R7B_DP_SCHEDULING_E2E.md) | **Our measurement/performance patch:** instrument child barriers, serialized bytes, rank drain, and busiest-lane padding; then batch/fix-schema messages or test a once-per-window load snapshot. Do not restore per-request all-rank probes. | Config rows **Rollout data parallelism**, **DP request routing**, and **Prefix caching**; 152-only patches **`patch_dp_rr_routing.py`** and the prefix-control extension in **`patch_block_size.py`**. |
| 5. TP2 crossover scout | [149c TP4 Perfetto](https://console.cloud.google.com/storage/browser/_details/ubench-logs/tianyu-rl-149c/perf_traces/perfetto_trace_v2_1788157043.pb?project=cloud-tpu-shared-capacity); [154 TP1 Perfetto](https://console.cloud.google.com/storage/browser/_details/ubench-logs/tianyu-rl-154/perf_traces/perfetto_trace_v2_1788167363.pb?project=cloud-tpu-shared-capacity) | **Config only:** set TP2 x DP64 while retaining the current VMEM-safe TP1 tile set for the first strict scout. No code patch is proposed. | Config rows **Rollout tensor parallelism**, **Rollout data parallelism**, and the three **RPA** rows. |
| 6. Actor kernels and dispatch gaps | [23b trainer XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-23b/tensorboard/plugins/profile/2026_08_22_06_27_19/?project=cloud-tpu-shared-capacity); [152 direct trainer XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-152/tensorboard/plugins/profile/2026_08_31_08_55_05/?project=cloud-tpu-shared-capacity) | **Config/our patch:** first A/B the splash fused-backward and block choices one at a time; separately compile the microbatch loop or overlap already-compiled dispatches. | Config rows **Packed-token budget** and **Trainer attention**; held-constant rows **Microbatches** and **Trainer structure**; 152-only patch rows **`patch_packing_knob.py`**, **`patch_packer_fast.py`**, and **`patch_flash_sa.py`**. The new trainer probes and compiled loop have no landed row. |
| 7. Phase transitions | [149c TP4 Perfetto](https://console.cloud.google.com/storage/browser/_details/ubench-logs/tianyu-rl-149c/perf_traces/perfetto_trace_v2_1788157043.pb?project=cloud-tpu-shared-capacity); [154 TP1 Perfetto](https://console.cloud.google.com/storage/browser/_details/ubench-logs/tianyu-rl-154/perf_traces/perfetto_trace_v2_1788167363.pb?project=cloud-tpu-shared-capacity); [152 phasetime logs](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-152/phasetime?project=cloud-tpu-shared-capacity) | **Our patch:** tighten future ownership and phase handoffs. Existing async modes remain config-selected implementations. | Config rows **Math-verification processes**, **Answer extraction mode**, **Reward overlap mode**, and **Logprob overlap mode**; patch rows **`patch_reward_parallel.py`**, **`patch_async_detok.py`**, **`patch_async_logprobs.py`**, **`patch_fast_extract.py`**, **`patch_async_reward.py`**, **`patch_gc_sync.py`**, **`patch_jit_unroll.py`**, and **`patch_sync_keepkv.py`**. |
| 8. Isolated rollout stall | [152 phasetime logs](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-152/phasetime?project=cloud-tpu-shared-capacity); [152 TensorBoard artifacts](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-152/tensorboard?project=cloud-tpu-shared-capacity) | **Our measurement patch only:** add a threshold-triggered flight recorder. No mitigation is selected yet. | No landed config or patch row: this is a tail event observed on the complete 152 endpoint. No trace caught the stall, so these phase/TensorBoard logs are the strongest available evidence. |
| 9. Fix TP1 profiling | [154 failed/diagnostic TP1 XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-154/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_09_10_14/?project=cloud-tpu-shared-capacity); [149c valid TP4 reference XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-149c/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_06_18_08/?project=cloud-tpu-shared-capacity); [152 direct trainer XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-152/tensorboard/plugins/profile/2026_08_31_08_55_05/?project=cloud-tpu-shared-capacity) | **Profiling config plus our instrumentation patch:** move the phased capture trigger after reshard/setup; patch the profiler only if sequencing cannot express that. No performance credit. | Config row **Phased rollout profiler** and held-constant row **Trainer profile request**; `patch_profiler_fix.py` fixes Tunix trainer-window compatibility but does **not** fix the TP1 phased-capture race. The proposed phased fix has no landed row. |

### 1. Remove the per-token cross-DP sampling guard

Evidence for headroom:

- The TP4 decode XPlane assigns about 7% of device self time to the scalar
  `reduce_and` safety check.
- The HLO replica group is DP32. Under TP1 the same logical check spans DP128,
  although the invalid TP1 device profile prevents direct timing.
- Sampling parameters are invariant for the benchmark and satisfy the
  candidate-path contract: top-k is positive and no larger than the candidate
  count, and top-p is positive.

Proposed solution:

Validate once on the host and select one of two compiled executables. Do not
reduce the same metadata across every DP rank on every generated token.

```python
def choose_sampling_plan(requests, candidate_count):
    safe = all(
        0 < request.top_k <= candidate_count and request.top_p > 0.0
        for request in requests
    )
    return SamplingPlan.CANDIDATE if safe else SamplingPlan.STOCK

@functools.partial(jax.jit, static_argnames=("sampling_plan",))
def decode_window(state, metadata, sampling_plan):
    if sampling_plan is SamplingPlan.CANDIDATE:
        return sharded_candidate_sample(state, metadata)
    return stock_sample(state, metadata)
```

For a homogeneous offline benchmark, a launch-time contract is sufficient:

```text
sampling_plan = candidate_top64
assert 0 < decode_top_k <= 64
assert decode_top_p > 0
```

Why it can improve performance:

It removes a latency collective from every decode iteration. It also removes
the `lax.cond` dependency associated with the dynamic predicate. The direct
TP4 leaf ceiling is about 7%; the planning estimate is 0.5-1.5 baseline units,
with potentially more value at DP128.

Validation:

- Verify `while/body/reduce_and` and its DP all-reduce disappear in XPlane.
- Keep a stock executable for unsupported or mixed sampling policies.
- Reject the run if any request violates the host-side contract.
- Re-run ties, greedy, logprob, length, truncation, and reward gates.

Public source context:
[TPU Inference sampling](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/layers/jax/sample/sampling.py#L68-L137).

### 2. Validate and refine the TP1-specific RPA tiles

Evidence for headroom:

- RPA is about 45% of TP4 decode device self time, the largest leaf family.
- The trace proves that the TP4 `1,8192,1,4096` decode tuple reaches its
  kernel.
- Run 152 does not inherit that tuple. It deliberately switches decode to
  `1,2048,1,2048` and adds mixed/prefill `512,1024,256,512`; the manifest
  records the latter as a TP1 VMEM-safety fix for eight KV heads per core.
- There is no valid TP1 device trace proving the landed TP1 tuples are active
  in the intended kernels or locally optimal.

Proposed solution:

First obtain a valid profile of the exact TP1 control. Then run an isolated,
randomized sweep on that stack. Pin every other knob, especially KV page size,
token budget, and continue-decode depth, and vary only one RPA family at a
time.

```bash
export VLLM_BLOCK_SIZE=128
export RPA_V3_MIXED_BLOCK_SIZES="512,1024,256,512"
export RPA_V3_PREFILL_BLOCK_SIZES="512,1024,256,512"

# One VMEM-validated decode value per run; order should be randomized.
export RPA_V3_DECODE_BLOCK_SIZES="1,1024,1,1024"
export RPA_V3_DECODE_BLOCK_SIZES="1,2048,1,1024"
export RPA_V3_DECODE_BLOCK_SIZES="1,2048,1,2048"   # exact 152 control
export RPA_V3_DECODE_BLOCK_SIZES="1,4096,1,2048"
export RPA_V3_DECODE_BLOCK_SIZES="1,4096,1,4096"
```

For Qwen TP1's eight KV heads per core, two query heads per KV head, head
dimension 128, and BF16 query/KV inputs, the kernel's own
`get_vmem_estimate_bytes` model gives:

| Fetch block `(bq_sz, bkv_sz)` | Pallas scratch estimate | Use in the sweep |
|---|---:|---|
| `(1, 1024)` | About 9 MiB | Small decode control |
| `(1, 2048)` | About 18 MiB | Run-152 decode control; compare compute sizes 1024 and 2048 |
| `(1, 4096)` | About 36 MiB | Largest decode fetch candidate with useful static margin |
| `(1, 8192)` | About 72 MiB | Reject before compile against the 64 MiB scoped VMEM limit |
| `(512, 1024)` | About 29 MiB | Run-152 mixed/prefill control |
| `(512, 2048)` | About 38 MiB | Mixed/prefill candidate after decode tuning |
| `(1024, 1024)` | About 49 MiB | Mixed/prefill candidate after decode tuning |
| `(1024, 2048)` | About 58 MiB | Edge candidate; little margin for compiler/VREG overhead |

The source estimate covers Pallas scratch, not every compiler allocation, so a
value below 64 MiB still has to compile and profile successfully. Reject a
candidate before benchmarking if the static estimate exceeds the limit. After
the decode winner is known, vary mixed and prefill independently; do not
collapse all three tile changes into one timing.

Why it can improve performance:

A 5-10% RPA improvement is worth roughly 2-5% of decode device time. After
host/phase dilution, plan for 0.5-1.5 baseline units. The opportunity is
meaningful because RPA is large; it is not guaranteed because the current
values are already deliberate TP1 safety tiles.

Validation:

- Require a valid TP1 profile and read the decode, mixed, and prefill
  custom-call configurations from the trace.
- Compare rollout time per generated token, not only time per step.
- Hold prompt arrays and semantic bands fixed.
- Require at least several steady observations per rung and repeat the winner.

Public source context:
[RPA block-size override, TPU Inference PR #3102](https://github.com/vllm-project/tpu-inference/pull/3102) and
[the source-level VMEM estimator](https://github.com/vllm-project/tpu-inference/blob/f6ad5ecb1ca359c721b01fd0d7edd90b5cf0ac7e/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L227-L260).

### 3. Split the Pathways-mediated wait before tuning Pathways CPU

Evidence for headroom:

- Baseline `device_get` owns about 82% of a normal engine window; the optimized
  TP4 and TP1 captures put the same inclusive wait above 90% of their deeper
  windows.
- A sampled worker's local decode module is much shorter than the client wait,
  so the opaque envelope contains more than local kernel execution.
- The native client plane shows IFRT dispatch, host-buffer lookup/store, and
  pipe callbacks distributed across worker threads. It does not show one
  serialized Pathways CPU leaf large enough to explain the wait.
- The shared DP-scheduler architecture explains why: one parent submits one
  global program and later materializes one global result after two CPU-child
  barriers; there are not independent model engines that can progress around
  a slow rank.
- Rank-major input and output shapes are coupled to the busiest DP lane, but
  the Qwen trace lacks real-row versus padded-row occupancy counters.
- No sampled process/core-utilization counter is present, so CPU saturation is
  not established.

Proposed solution:

Carry one correlation ID from the engine window through client, scheduler,
worker, device, and result materialization, then emit one compact record per
window:

```python
trace = new_decode_trace_id()
t_submit = monotonic_ns()
future = executable.submit(inputs, trace_id=trace)

# Recorded at the owning layers, not synthesized from the caller's wait.
mark(trace, "rm_admitted")
mark(trace, "worker_launch")
mark(trace, "device_start")
mark(trace, "device_done")
mark(trace, "result_ready")

result = future.result()
mark(trace, "host_materialized")
emit_decode_latency_breakdown(trace)
```

Record request-shape and host-buffer volume in the same record, then derive per
window and per rank:

```text
admission_queue = rm_admitted - submit
launch_delay    = device_start - worker_launch
device_time     = device_done - device_start
rank_tail       = max(device_done_by_rank) - median(device_done_by_rank)
result_transfer = result_ready - max(device_done_by_rank)
host_finalize   = host_materialized - result_ready
row_occupancy   = real_request_rows / (DP * request_bucket)
result_bytes    = tokens + logprobs + metadata materialized on the host
```

If result transfer/materialization is large, compact padded rows and masked
steps before host visibility while preserving every valid generated token and
logprob needed by GRPO:

```python
compact_result = {
    "row_ids": live_row_ids,
    "row_offsets": packed_row_offsets,
    "valid_lengths": valid_lengths,
    "tokens": packed_valid_tokens,
    "logprobs": packed_valid_logprobs,
    "last_token": last_real_token,
    "finished": finished_mask,
}
copy_to_host_async(compact_result)
```

This is not permission to return only a scheduler summary: the learner still
needs the complete valid token/logprob stream. A viable implementation must
reduce dummy/padding transfer without dropping that payload.

If admission or launch delay is large, reuse the already-compiled/pre-sharded
decode executable and keep topology setup outside the profiled loop. If rank
tail is large, apply the DP instrumentation and routing work in the next item.
If preparation grows with capacity rather than live rows, right-size
`max_num_seqs` and make block-table updates persistent/sparse. If device time
is large, prioritize RPA and sampling. These are distinct fixes for distinct
buckets; “optimize Pathways” is not a sufficiently specific patch.

Why it can improve performance:

The instrumentation itself is a measurement enabler. It converts the largest
opaque host-visible span into actionable queue, launch, device, tail, transfer,
and finalize components. Compact result transfer can directly reduce the
transfer/finalize buckets, but it receives no headroom credit until those
buckets are measured. This avoids claiming that a blocked caller consumed CPU.

Validation:

- Check clock alignment or use duration pairs local to each clock domain.
- Emit one record per window, not per token, to keep overhead negligible.
- Confirm the sum of disjoint buckets closes against `device_get` wall within
  a small residual.
- Record real rows, padded rows, request bucket, host-buffer event counts, and
  bytes transferred for every captured window.
- Require byte-identical valid token IDs, EOS inclusion, lengths, finish
  reasons, and selected-token logprobs for any output compaction patch.
- Add sampled process CPU, runnable threads, executor queue depth, and
  involuntary context switches before making a CPU-saturation claim.

Public context: [JAX `device_get`](https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html).

### 4. Measure and reduce DP128 fan-out and tail imbalance

Evidence for headroom:

- Rollout remains about 86% of the TP1 cycle.
- TP1 creates four times as many logical DP scheduler children as TP4 and
  fewer requests per model replica; it does not create 128 independent model
  engines or Pathways clients.
- The baseline TP4 trace executes 128 send/receive calls per full depth-128
  window. Optimized TP4 executes 64 per depth-512 window, while optimized TP1
  executes 256 because the same two rank-wide rounds now span DP128.
- The minimum steady control sequence is two rank-wide request/reply barriers
  (`SCHEDULE` and `UPDATE_FROM_OUTPUT`). Baseline 23b has two additional
  rank-wide rounds; these are consistent with grammar and unfinished-request
  broadcasts rather than additional model executions.
- High-level scheduler wall is about 3.2% of the baseline engine window, about
  1.8% in optimized TP4, and about 2.5% in the TP1 capture. It is secondary,
  but the TP1 scaling tax is directly visible.
- `max_num_seqs=256` makes maximum host-batch capacity grow with DP even though
  the configured 2,048 requests average only 16 per rank at DP128. The trace
  does not yet show whether capacity-sized metadata or busiest-rank padding is
  material.
- Completion lengths are broad and many sequences reach the output cap, so a
  rank can drain materially later than its peers.
- The TP4 host trace has no single large CPU leaf after `device_get`; queueing,
  fan-out, and rank imbalance are the more plausible remaining host costs.

Proposed solution:

First add rank-level accounting without querying every rank per request:

```python
rank_state[rank].queued_requests += admitted
rank_state[rank].retired_requests += retired
rank_state[rank].active_tokens = reported_once_per_window

emit_histogram("active_sequences_by_rank", rank_state)
emit_histogram("window_finish_time_by_rank", rank_state)
emit("request_bucket", request_bucket)
emit("padded_row_fraction", 1.0 - real_rows / (dp_size * request_bucket))
```

If command fan-out is material, batch control messages and result collection:

```python
commands_by_rank = bucket_by_rank(pending_commands)
send_many(commands_by_rank)          # one envelope per rank/window
results = collect_many(expected_ranks)
```

If the tail is material, compare the current round robin against a parent-side
load snapshot refreshed once per window. Route a batch from that snapshot
rather than restoring an all-rank state probe for every request:

```python
predicted_outstanding_work = snapshot_rank_work_once_per_window()
rank = min(eligible_ranks, key=lambda r: predicted_outstanding_work[r])
predicted_outstanding_work[rank] += request_cost_estimate(request)
```

If prefix caching is ever enabled, assign all generations of one prompt as a
group so the alternative does not discard prefix locality. The Gemma study's
round-robin regression is the warning: fewer control messages can still lose
end to end when routing destroys useful locality or increases the drain tail.

Why it can improve performance:

Batched fan-out and fixed-schema messages reduce Pipe/cloudpickle work at
every engine boundary. A lightweight parent-side load estimate can retain
most of round robin's low overhead while avoiding pathological DP tails. Plan
for roughly 0.5-1.5 baseline units only after the new rank histograms prove the
mechanism.

Validation:

- Compare p50/p90/p99 rank finish skew and queue depth.
- Count command envelopes, serialized bytes, real rows, padded rows, and
  host-buffer bytes per generated token.
- Monitor head CPU saturation and executor queue latency.
- Keep deterministic round robin as the Qwen control; reject any alternative
  that restores `O(requests * DP)` admission queries.
- If prefix caching is enabled, report cache-hit/locality changes separately.

Public source context:
[TPU Inference DP scheduler](https://github.com/vllm-project/tpu-inference/blob/a32e183a676cf428cb1cea953f58fdd616accb3e/tpu_inference/core/sched/dp_scheduler.py#L701-L776).

### 5. Scout TP2 x DP64 as a crossover point

Evidence for headroom:

- TP4 spends about 20% of decode leaf time in TP model/sampling reductions.
- TP1 removes those collectives, while the measured TP1-plus-tiles package is
  about 6% lower in rollout. The tile confound prevents treating 6% as the
  pure topology benefit; full-weight traffic, smaller per-replica batches, and
  DP fan-out are still the expected topology costs.
- TP2 is the only unmeasured point between the two regimes.

Proposed configuration:

```diff
-rollout_tensor_parallelism=1
-rollout_data_parallelism=128
+rollout_tensor_parallelism=2
+rollout_data_parallelism=64
```

Keep sampling, page size, token budget, decode depth, and all three current
TP1 RPA tuples identical for the first scout. Retune RPA only after that strict
topology comparison.

Why it can improve performance:

TP2 halves full-model weight traffic and doubles the requests per model
replica relative to TP1 while paying smaller TP collectives than TP4. It may
sit closer to the memory/latency crossover. This is a 0-1.5-baseline-unit
experiment, not an assumed win; TP1 remains the current control until TP2
beats it repeatedly.

Validation:

- Match completion volume and semantic distributions.
- Measure rollout per generated token and rank tail skew.
- Profile the actual TP2 collective share.
- Repeat the winner against a concurrent or interleaved TP1 control.

### 6. Tune actor kernels, then compile or overlap dispatch gaps

Evidence for headroom:

- Actor training is about 9% of the current cycle.
- In both Perfetto traces, gaps between TPU microbatch envelopes occupy about
  one-fifth of actor-train wall.
- The device envelopes themselves are stable across TP1 and TP4, confirming
  that rollout topology is unrelated to this gap.
- The direct run-152 trainer XPlane assigns about half of event/core time to
  the splash-attention custom call, but actor wall is only about 9% of the
  complete cycle.

Proposed solution:

Before changing the training loop, run low-cost, one-change trainer-kernel
scouts against the exact 1024/2048 block control:

```text
control: current splash blocks, fused backward off
probe A: fused backward on
probe B: one VMEM-safe splash block tuple at a time
probe C: base-2 exponent path, with an explicit numerical-equivalence gate
```

Do not combine these flags in the first pass. Compare trainer wall and
attention leaf time, and reject any candidate that raises compile time, HBM,
loss drift, or optimizer-state drift.

Where memory permits, move the microbatch loop inside one compiled program:

```python
@jax.jit
def train_minibatches(state, packed_microbatches):
    def one_microbatch(carry, batch):
        state, grad_accum = carry
        loss, grad = loss_and_grad(state, batch)
        return (state, add(grad_accum, grad)), loss

    (state, grad_accum), losses = jax.lax.scan(
        one_microbatch, (state, zeros_like_grads(state)), packed_microbatches
    )
    return apply_gradients(state, grad_accum), losses
```

If a full scan is too large, queue the next already-compiled microbatch before
materializing host metrics from the previous one.

Why it can improve performance:

It converts repeated proxy dispatch and idle gaps into device-side loop work.
The absolute dispatch-gap ceiling is only about 1 baseline unit; the realistic
planning range is 0.25-0.5 baseline units. A 20% splash-leaf improvement has a
separate generous ceiling below 1% end to end, so trainer kernel tuning remains
lower priority than decode RPA and sampling.

Validation:

- Confirm gradient accumulation and optimizer update order are identical.
- Compare loss/gradients on a deterministic packed batch.
- Check peak HBM and compile time.
- Verify the inter-envelope gaps shrink in Perfetto.

### 7. Tighten phase transitions, not reward math

Evidence for headroom:

- Perfetto places advantage/reward at about 1% and synchronization below 1%.
- Several small gaps remain between rollout, advantage, train, sync, and the
  next data load; together they are a few percent of the cycle.
- Further reward-worker parallelism cannot recover time that is already
  overlapped or below the timing floor.

Proposed solution:

Make future ownership explicit and consume results only at the true data
dependency:

```python
rollout_future = engine.generate_async(batch)
prepare_static_train_metadata(batch)
rollout = rollout_future.result()

reward_future = reward_pool.submit(score, rollout.completed_text)
prepare_packed_tokens(rollout.tokens)
rewards = reward_future.result()
```

Begin next-step host data loading immediately after synchronization while the
next rollout executable is being selected. Do not overlap next-policy rollout
with the current update; that would change on-policy semantics.

Why it can improve performance:

It removes avoidable handoff and future-materialization gaps without changing
model math. The planning range is 0.25-0.8 baseline units.

Validation:

- Preserve step ordering and policy-version tags.
- Report phase gaps directly in Perfetto.
- Monitor executor backlog so work is not merely moved to a later tail.
- Require exact deterministic rewards and packed-token identity.

### 8. Diagnose the isolated rollout stall

Evidence for headroom:

- One of the 18 current-endpoint observations is approximately twice the steady
  median.
- The extra time is entirely inside engine generation; actor train and sync
  remain normal.
- That step processes ordinary completion volume and is not a checkpoint
  step, so work quantity does not explain it.

Proposed solution:

Install a lightweight threshold-triggered flight recorder instead of running
a full profiler on every benchmark:

```python
if engine_window_time > 1.5 * rolling_median(engine_window_time):
    dump({
        "rank_queue_depths": rank_queue_depths,
        "pending_futures": pending_future_summary(),
        "pathways_queue_age": pathways_queue_age(),
        "host_cpu": sampled_host_cpu(),
        "kv_usage": kv_usage_by_rank(),
    })
```

Why it can improve performance:

This does not improve the steady median by itself. It targets mean and tail
reliability by distinguishing device stall, Pathways queueing, scheduler
backlog, and host contention. Fixing a recurring version of this event could
matter more operationally than another sub-percent kernel win.

Validation:

- Keep recorder overhead negligible on normal windows.
- Correlate the next stall across Pathways, host, and rank clocks.
- Reproduce the identified cause before adopting a mitigation.

### 9. Fix TP1 profiling before making a TP1 kernel claim

Evidence for headroom:

- The TP1 phased sessions capture the 128-way split/reshard path rather than a
  representative decode window.
- The TP4 phased profiler works, so this is topology/session scheduling rather
  than absence of decode execution.

Proposed profiling pseudocode:

```python
# Warm and finish all topology setup before tracing.
for _ in range(profile_warmup_windows):
    block_until_ready(run_decode_window())

barrier("decode-profile-ready")
if selected_profile_rank:
    start_trace(profile_dir)

for _ in range(profile_capture_windows):
    block_until_ready(run_decode_window())

if selected_profile_rank:
    stop_trace()
```

Start the capture after any DP reshard/split future is complete, and tag the
decode executable ID in the trace request. If phased profiling still races,
run a decode-only harness with the already-compiled executable and fixed KV
fixtures.

Why it matters:

This is a measurement enabler, not direct performance headroom. Without it,
TP1 RPA and sampling estimates remain extrapolations from TP4.

## Consolidated next-run matrix

| Priority | Experiment | Control | Adoption gate |
|---:|---|---|---|
| P0 | Static/host-validated sampling plan | Current TP1 endpoint | DP `reduce_and` gone; semantic gates pass |
| P0 | Pathways wait-bucket correlation | Current TP1 endpoint | Opaque wait closes into queue/device/tail/transfer/finalize buckets; result bytes and padded-row occupancy are recorded |
| P0 | Valid TP1 decode capture | Same TP1 binary | Steady decode module, not `jit_split`, dominates session |
| P1 | TP1 RPA tile validation/sweep | Current decode `1,2048,1,2048` and mixed/prefill `512,1024,256,512` | Repeated per-token rollout win and correct kernel names |
| P1 | TP2 x DP64 scout | Interleaved TP1 x DP128 | Repeated matched-work win |
| P1 | DP rank/tail instrumentation | Current TP1 endpoint | Identifies material child-barrier skew, fan-out serialization, or busiest-lane padding |
| P2 | Batched DP commands or snapshot-based routing | Instrumented TP1 | Lower host critical path without worse tail, prefix locality, or `O(requests * DP)` probes |
| P2 | Trainer splash one-change scouts | Current 1024/2048 unfused-backward splash control | Lower trainer wall and attention leaf time; HBM and numerical gates pass |
| P2 | Compiled actor microbatch loop | Current actor loop | Gap reduction, HBM safe, exact update semantics |
| P2 | Phase-future cleanup | Current synchronous handoffs | Lower transition share, no policy-version drift |

## Remaining-headroom budget

The following landed-credit ledger prevents historical wins from being added
to the remaining budget:

| Already-landed lever | Best current estimate of delivered credit | Remaining credit in run 152 |
|---|---:|---:|
| Decode depth 128 to 512 | About 2-4 baseline units gross, model-based; no isolated rung | Zero for the already-applied change; 512 to 1024 is only about a 1.1% whole-cycle scheduler-only gross ceiling before masking |
| NumPy-first packed-chunk assembly | About 3.5-4 baseline units causal; about 4.5 gross observed in the one-change pair | Less than 0.5 baseline units even if the measured residual vanished |

Neither row is TPU-kernel headroom. Window depth amortizes host/device
boundaries while trading against masked device work; the packer removes host
dispatches before the compiled actor program.

| Opportunity | Evidence strength | Plausible reduction in 23b-baseline units | Notes |
|---|---|---:|---|
| Remove DP sampling guard | Strong TP4 trace/source; TP1 timing missing | 0.5-1.5 baseline units | First code target |
| TP1 RPA validation/retune | Strong TP4 leaf share; no valid TP1 kernel trace | 0.5-1.5 baseline units | Current TP1 safety tiles are landed; requires profiler fix |
| Trainer splash/collective tuning | Direct run-152 trainer XPlane; wall mapping is inclusive | 0.1-0.5 baseline units | Actor is only about 9% of the cycle; tune after rollout |
| Pathways wait split / compact result | Strong opaque-wait evidence; bucket split missing | 0-1.5 baseline units | No credit until transfer/finalize is measured |
| DP128 fan-out/tail | Strong structural case; instrumentation missing | 0.5-1.5 baseline units | Measure before patching |
| TP2 crossover | Direct topology model, no TP2 result | 0-1.5 baseline units | Can be negative |
| Actor dispatch gaps | Direct Perfetto evidence | 0.25-0.5 baseline units | Small global ceiling |
| Phase transitions | Direct Perfetto evidence | 0.25-0.8 baseline units | Preserve on-policy order |
| Isolated stall | Direct event, cause unknown | No median claim | Mean/p99 opportunity |

Do not sum the high ends. A central near-term target is 2.5-5 baseline units
below the current endpoint. Reaching the stretch case requires both a real
decode-kernel win and a demonstrated DP-control/tail win.

Restricting the budget to TPU-side decode kernels and collectives gives roughly
**1-3 baseline units**, or about **2-5% of the current endpoint**: the sampling
guard plus RPA retuning dominate, while trainer kernel work is sub-percent end
to end. There is no evidence for another 10-20% kernel-only win.

## Validity checklist

- Baseline metric is the run-23b TensorBoard median over steps 3-20.
- Startup steps are excluded by rule, not by observed value.
- The run-152 rollout stall remains in the endpoint distribution.
- Prompt arrays are byte-identical.
- Report completion-token volume and time per generated token.
- Predeclare length, truncation, and reward bands.
- Pin the KV page size during topology and RPA comparisons.
- Read the actual RPA custom-call name from every profile.
- Confirm sharded-sampler fallback and static-plan counters.
- Require zero zombie requests and correct EOS retirement.
- Report multiple steady observations and repeat every winner.
- Treat TP degree as a sampling-numerics contract change.

## Bottom line

Run 23b is the baseline, and its steady-step median is the 100-unit reference.
Run 152 is the sustained current endpoint at approximately 53 baseline units.
Run 154 supplies the same-config phase timeline; run 149c explains the closest
available optimized decode program at TP4. The direct run-152 XPlane is the
current trainer reference, with run 136 as TP4 corroboration.

The final TP1-plus-tile package is a real but modest win. The large reduction
from the 23b baseline came from the earlier software stack. From here, rollout
still owns roughly six-sevenths of the cycle, and the most defensible next
story is:

1. remove the per-token cross-DP sampling guard;
2. split the Pathways-mediated wait into queue/device/tail/transfer buckets;
3. obtain a valid TP1 decode trace;
4. validate and refine the already-landed TP1 RPA tiles;
5. measure and reduce DP128 fan-out/tail behavior;
6. scout TP2 as the only plausible topology crossover;
7. spend trainer effort only on the measured microbatch gaps.

That supports a credible additional 5-10% target, with 10-15% as a stretch
case—not another 2x claim. The narrower TPU-kernel-only target is about 2-5%;
the already-landed 128-to-512 window and NumPy packer must not be counted again.
