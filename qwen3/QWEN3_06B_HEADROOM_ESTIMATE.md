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

The baseline trace also makes the DP traffic concrete. A full window contains
128 `_send_command` and 128 `_get_result` calls on the engine thread. With
DP32, that is four rank-wide command/response rounds per window. The nested
send/receive spans are already inside scheduler wall and therefore must not be
summed as an additional cost.

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
| DP scheduling is the primary bottleneck | **Rejected** | High-level DP scheduler wall is only about 3.2% of the baseline window |
| DP scheduling is meaningful scaling headroom | **Supported** | Rank command count scales with DP degree and remains on the critical thread |
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

The larger `device_get` *share* in the optimized traces is healthy in context:
host extraction and repeated control boundaries have been removed, so more of
each deeper window is intentionally spent waiting for useful remote decode.
It is not evidence that Pathways regressed.

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
  TP1 increases the number of DP engines from 32 to 128.

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
is a TP4 confirmation. The current actor train wall is only about 9% of the
cycle. Perfetto shows that roughly one-fifth of that train wall is gap between
successive TPU microbatch envelopes. Even deleting every such gap is only an
approximately 2% global ceiling; a realistic compiled-dispatch improvement is
lower.

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
| 3. Split Pathways wait | [23b baseline decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-23b/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_22_06_21_47/?project=cloud-tpu-shared-capacity); [149c optimized decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-149c/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_06_18_08/?project=cloud-tpu-shared-capacity) | **Our measurement patch first; our performance patch second:** add cross-layer correlation buckets, then change transfer/launch behavior only if a bucket is material. | Config rows **Continue-decode depth** and **Maximum batched tokens** plus 152-only patch row **`patch_host_extract.py`** explain the changed envelope. The proposed Pathways correlation has no landed row. |
| 4. DP128 fan-out/tail | [23b baseline decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-23b/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_22_06_21_47/?project=cloud-tpu-shared-capacity); [149c TP4 decode XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-149c/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_06_18_08/?project=cloud-tpu-shared-capacity); [154 TP1 diagnostic XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-154/phased_profile/decode_only/dp_rank_0/plugins/profile/2026_08_31_09_10_14/?project=cloud-tpu-shared-capacity) | **Our measurement/performance patch:** instrument rank tails and command counts; batch messages or change routing only after attribution. The DP scale itself is config. | Config rows **Rollout data parallelism** and **DP request routing**; 152-only patch row **`patch_dp_rr_routing.py`**. |
| 5. TP2 crossover scout | [149c TP4 Perfetto](https://console.cloud.google.com/storage/browser/_details/ubench-logs/tianyu-rl-149c/perf_traces/perfetto_trace_v2_1788157043.pb?project=cloud-tpu-shared-capacity); [154 TP1 Perfetto](https://console.cloud.google.com/storage/browser/_details/ubench-logs/tianyu-rl-154/perf_traces/perfetto_trace_v2_1788167363.pb?project=cloud-tpu-shared-capacity) | **Config only:** set TP2 x DP64 while retaining the current VMEM-safe TP1 tile set for the first strict scout. No code patch is proposed. | Config rows **Rollout tensor parallelism**, **Rollout data parallelism**, and the three **RPA** rows. |
| 6. Actor dispatch gaps | [23b trainer XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-23b/tensorboard/plugins/profile/2026_08_22_06_27_19/?project=cloud-tpu-shared-capacity); [152 direct trainer XPlane](https://console.cloud.google.com/storage/browser/ubench-logs/tianyu-rl-152/tensorboard/plugins/profile/2026_08_31_08_55_05/?project=cloud-tpu-shared-capacity) | **Our patch:** compile the microbatch loop or overlap already-compiled dispatches. The first proposal does not change batch flags. | Config rows **Packed-token budget** and **Trainer attention**; held-constant rows **Microbatches** and **Trainer structure**; 152-only patch rows **`patch_packing_knob.py`**, **`patch_packer_fast.py`**, and **`patch_flash_sa.py`**. The new compiled loop has no landed row. |
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
export RPA_V3_DECODE_BLOCK_SIZES="1,2048,1,2048"   # exact 152 control
export RPA_V3_DECODE_BLOCK_SIZES="1,4096,1,2048"
```

Reject a candidate before benchmarking if its static VMEM estimate exceeds the
limit. After the decode winner is known, vary mixed and prefill independently;
do not collapse all three tile changes into one timing.

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
[RPA block-size override, TPU Inference PR #3102](https://github.com/vllm-project/tpu-inference/pull/3102).

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

Derive, per window and per rank:

```text
admission_queue = rm_admitted - submit
launch_delay    = device_start - worker_launch
device_time     = device_done - device_start
rank_tail       = max(device_done_by_rank) - median(device_done_by_rank)
result_transfer = result_ready - max(device_done_by_rank)
host_finalize   = host_materialized - result_ready
```

If result transfer/materialization is large, return only the compact scheduler
summary instead of a padded token-history payload:

```python
compact_result = {
    "slot": slot_ids,
    "last_token": last_real_token,
    "generated_count": generated_count,
    "finished": finished_mask,
}
copy_to_host_async(compact_result)
```

If admission or launch delay is large, reuse the already-compiled/pre-sharded
decode executable and keep topology setup outside the profiled loop. If rank
tail is large, apply the DP instrumentation and routing work in the next item.
If device time is large, prioritize RPA and sampling. These are distinct fixes
for distinct buckets; “optimize Pathways” is not a sufficiently specific patch.

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
- Add sampled process CPU, runnable threads, executor queue depth, and
  involuntary context switches before making a CPU-saturation claim.

Public context: [JAX `device_get`](https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html).

### 4. Measure and reduce DP128 fan-out and tail imbalance

Evidence for headroom:

- Rollout remains about 86% of the TP1 cycle.
- TP1 creates four times as many DP engines as TP4 and fewer requests per
  model replica.
- The baseline TP4 trace executes 128 send/receive calls per full depth-128
  window. Optimized TP4 executes 64 per depth-512 window, while optimized TP1
  executes 256 because the same two rank-wide rounds now span DP128.
- High-level scheduler wall is about 3.2% of the baseline engine window, about
  1.8% in optimized TP4, and about 2.5% in the TP1 capture. It is secondary,
  but the TP1 scaling tax is directly visible.
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
```

If command fan-out is material, batch control messages and result collection:

```python
commands_by_rank = bucket_by_rank(pending_commands)
send_many(commands_by_rank)          # one envelope per rank/window
results = collect_many(expected_ranks)
```

If the tail is material, route from host-maintained outstanding work rather
than an all-rank state probe for every request:

```python
rank = min(eligible_ranks, key=lambda r: predicted_outstanding_work[r])
predicted_outstanding_work[rank] += request_cost_estimate(request)
```

Why it can improve performance:

Batched fan-out reduces fixed IPC work at every engine boundary. A lightweight
central load estimate retains most of round robin's low overhead while
avoiding pathological DP tails. Plan for roughly 0.5-1.5 baseline units only
after the new rank histograms prove the mechanism.

Validation:

- Compare p50/p90/p99 rank finish skew and queue depth.
- Count command envelopes and bytes per generated token.
- Monitor head CPU saturation and executor queue latency.
- Keep a deterministic round-robin fallback.

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

### 6. Compile or overlap the actor microbatch dispatch gaps

Evidence for headroom:

- Actor training is about 9% of the current cycle.
- In both Perfetto traces, gaps between TPU microbatch envelopes occupy about
  one-fifth of actor-train wall.
- The device envelopes themselves are stable across TP1 and TP4, confirming
  that rollout topology is unrelated to this gap.

Proposed solution:

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
The absolute theoretical ceiling is only about 1 baseline unit; the realistic
planning range is 0.25-0.5 baseline units.

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
| P0 | Pathways wait-bucket correlation | Current TP1 endpoint | Opaque wait closes into queue/device/tail/transfer/finalize buckets |
| P0 | Valid TP1 decode capture | Same TP1 binary | Steady decode module, not `jit_split`, dominates session |
| P1 | TP1 RPA tile validation/sweep | Current decode `1,2048,1,2048` and mixed/prefill `512,1024,256,512` | Repeated per-token rollout win and correct kernel names |
| P1 | TP2 x DP64 scout | Interleaved TP1 x DP128 | Repeated matched-work win |
| P1 | DP rank/tail instrumentation | Current TP1 endpoint | Identifies material skew or fan-out cost |
| P2 | Batched DP commands or load-aware routing | Instrumented TP1 | Lower host critical path without worse tail |
| P2 | Compiled actor microbatch loop | Current actor loop | Gap reduction, HBM safe, exact update semantics |
| P2 | Phase-future cleanup | Current synchronous handoffs | Lower transition share, no policy-version drift |

## Remaining-headroom budget

| Opportunity | Evidence strength | Plausible reduction in 23b-baseline units | Notes |
|---|---|---:|---|
| Remove DP sampling guard | Strong TP4 trace/source; TP1 timing missing | 0.5-1.5 baseline units | First code target |
| TP1 RPA validation/retune | Strong TP4 leaf share; no valid TP1 kernel trace | 0.5-1.5 baseline units | Current TP1 safety tiles are landed; requires profiler fix |
| Pathways wait split / compact result | Strong opaque-wait evidence; bucket split missing | 0-1.5 baseline units | No credit until transfer/finalize is measured |
| DP128 fan-out/tail | Strong structural case; instrumentation missing | 0.5-1.5 baseline units | Measure before patching |
| TP2 crossover | Direct topology model, no TP2 result | 0-1.5 baseline units | Can be negative |
| Actor dispatch gaps | Direct Perfetto evidence | 0.25-0.5 baseline units | Small global ceiling |
| Phase transitions | Direct Perfetto evidence | 0.25-0.8 baseline units | Preserve on-policy order |
| Isolated stall | Direct event, cause unknown | No median claim | Mean/p99 opportunity |

Do not sum the high ends. A central near-term target is 2.5-5 baseline units
below the current endpoint. Reaching the stretch case requires both a real
decode-kernel win and a demonstrated DP-control/tail win.

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
case—not another 2x claim.
