# Gemma4-e2b on TPU v7x: performance headroom

Source run: `gs://ubench-logs/alisachen-tpu7x/runs/g4r7b`

Analysis date: 2026-08-30. The later 12:03–13:02 UTC session is the completed baseline. Step 0 is excluded from steady timing because profiling and compilation dominate it. XProf itself was collected inside that cold outer step, so the device result is a warmed decode invocation, not a full steady rollout.

## Executive conclusion

`g4r7b` is a synchronous on-policy GRPO run of Gemma4-E2B on a 64-chip TPU v7x allocation. The job exposes 128 TensorCore/JAX devices and splits them 50/50: 64 devices for training and 64 for rollout (`tp=1`, `dp=64`). The effective batch is 256 prompts x 8 generations = 2,048 sampled sequences per cycle.

The primary roof is host-side rollout serialization, not MXU or HBM saturation. A representative host trace shows a 774.10 ms `llm_engine.step` containing only 15.99 ms of captured TPU work, a device-coverage proxy of 2.07%. The head used only 0.9–2.1 of 44 CPU cores in the follow-up `g4r8b` measurement, while all engine drivers shared one Python process. The independent window-64 -> window-128 experiment halved round trips but did not improve rollout, confirming that total serialized host work, rather than the number of dispatches, is binding.

Inside the 15.99 ms device interval, the program is on the HBM side of the roofline but not at the HBM roof: arithmetic intensity is 15.709 FLOP/byte, achieved compute is 17.760 TFLOP/s, and the same-intensity modeled HBM roof is 57.907 TFLOP/s. That is 30.670% roof efficiency and 1.726% of compute peak. The handoff's older statement that the whole decode reaches about 109 TFLOP/s / 100% of the memory roof is not supported by the aggregate XProf data; it likely conflates the per-package theoretical roof or the near-roof logits fusion with the whole program.

Priority order from the combined log and roofline evidence:

1. Parallelize or shard head-side engine routing and output processing. Target DP round-robin routing, batched/vectorized output extraction, fewer `HostBufferStore` lookups, and eventually multiple engine-driver processes or heads.
2. Remove steady-cycle bubbles and tails. The outer-loop gap is about 8.93 s/cycle; one rollout stall plus four reward stalls consume 4.77% of steady logged time.
3. Reduce or overlap the approximately 10.38 s weight synchronization tax.
4. Improve useful decode work: correct stop-token behavior and test active-sequence compaction/refill. Only 81.20% of issued 4,096-token-cap slots become recorded output.
5. Tune device kernels only after host coverage rises. The two sampling reductions are the best kernel targets, but taking them to 60% HBM is worth only about 0.26 s/cycle under the current 2.07% device-coverage proxy.

The ceilings overlap and must not be summed. A near-term 1.10–1.15x end-to-end target is plausible from host/orchestration, tail, and partial-sync work. The current trace does not justify a 1.20–1.35x end-to-end claim from kernels alone.

## Baseline

| Clock or metric | Steady value |
|---|---:|
| Published benchmark timer (`phasetime.run_global_step`, steps 3–19) | 132.5465 s mean; **125.7963 s median** |
| Same timer, all non-cold steps 1–19 | 131.4702 s mean; 125.7963 s median |
| TensorBoard iteration including weight sync | 142.1705 s mean; 136.2982 s median |
| Observed artifact cadence | 151.0999 s mean; 145.440 s median |
| Samples | 2,048 per step |
| Logged sample throughput | 14.405 samples/s |
| Cadence sample throughput | 13.554 samples/s |
| Logged useful output throughput | 47,913.5 tokens/s |
| Cadence useful output throughput | 45,082.0 tokens/s |
| Engine-only useful output throughput | 69,558.7 tokens/s |
| Output length | 3,326.12 mean steady; cap 4,096 |
| Truncated at cap | 78.83% steady |

The published `125.8 s/step` is therefore a benchmark-timer median, not full wall cadence. Weight sync and the next-cycle gap are outside it.

TensorBoard logged-wall decomposition (mean, steps 1–19):

| Exclusive component | Mean | Logged-wall share |
|---|---:|---:|
| Engine generation | 97.930 s | 68.88% |
| Reward computation | 15.228 s | 10.71% |
| Actor/update | 17.517 s | 12.32% |
| Advantage and prepare overhead | 0.794 s | 0.56% |
| Weight synchronization | 10.376 s | 7.30% |
| Remaining logged residual | about 0.33 s | about 0.23% |

The 151.10 s observed cadence is another 8.93 s slower than the TensorBoard timer. Recorded actor dequeue explains 4.49 s; reporting/upload/loop work accounts for the other approximately 4.43 s, subject to GCS finalization latency. Perfetto independently measures 151.50 s mean / 145.59 s median over complete aligned steady cycles.

## Decode XProf

The full job allocated 64 physical TPU packages / 128 TensorCore devices. The configured 50/50 split assigns 64 devices to rollout and 64 to training. Decode has eight populated server XPlanes; each representative server contains eight TensorCore planes mapped to four physical chip IDs, so the profiled rollout half spans 32 packages / 64 TensorCores. XProf's per-TensorCore normalization reports 1,028.75 TFLOP/s peak compute and 3,686.156 GB/s peak HBM bandwidth.

Latest representative-host metrics:

| Metric | Value |
|---|---:|
| Overall op time, mean per TensorCore for captured invocation | 15.986 ms |
| Main decode module envelope, mean per TensorCore | 15.310 ms |
| Achieved normalized compute | 17.760 TFLOP/s |
| Peak compute utilization | 1.726% |
| HBM throughput | 1,052.88 GiB/s |
| Peak HBM utilization | 30.670% |
| Arithmetic intensity | 15.71 FLOP/byte |
| Roofline bound | HBM |
| Explicit op-profile idle | 0.447% |
| Overview device duty | 99.0% |

Low FLOP utilization therefore does not imply a 58x compute opportunity. During the captured interval, the compiled program is occupied with accounted memory movement, attention, formatting, and latency-bound collectives; host and orchestration stalls outside that interval remain possible.

Latest HLO/category shares:

| Category | Device-time share |
|---|---:|
| Loop fusion | 27.11% |
| Convolution fusion | 24.85% |
| RPAd/RPAm and other custom calls | 22.96% |
| Data formatting | 14.47% |
| All-reduce | 5.70% |
| Collective-permute start/done | 3.60% |

### Sampling bottleneck

The two largest HLOs scan full-vocabulary tensors shaped around `[262144, 32]`:

- `convert_reduce_fusion.3`: 10.695% of decode time, about 40.83 us average.
- `fusion.4516`: 9.924%, about 40.94 us average.

Together they consume 20.62%. Their XProf-accounted bandwidth is about 765 GiB/s (about 822 GB/s), only approximately 22.2–22.3% of peak HBM, whereas the top logits convolution fusion reaches about 3,022 GiB/s (about 3,244 GB/s), or 88.0% of peak. Experiments should test fused/partitioned sampling, lower-precision intermediates where numerically valid, and avoiding materialization or repeated passes over the full vocabulary.

### Collective bottleneck

`all-reduce.3` reduces a scalar `u32` across `mesh['axis_0'=64]` inside the loop/termination path. It is latency-bound:

- Early capture: 15 recorded core events x 202.52 us average, 2.48% of captured core-time.
- Later capture: 16 recorded core events x 415.98 us average, 5.204% of captured core-time.

The comparable latency more than doubled. At the later event count, restoring the older latency would save about 2.67% of captured device core-time. Under the measured 2.07% device-coverage proxy, its end-to-end impact is much smaller than a full-rollout extrapolation. Counts are aggregated across TensorCore planes and trace-boundary sensitive; they are not loop-trip counts. All directly identified communication is about 9.30% of latest decode, so even perfect removal has only a 10.3% captured-device speed ceiling.

### Attention and formatting

RPAd/RPAm attention custom calls consume about 22.7% of captured device time, split primarily into sliding-window RPAd (15.98%), global RPAd (4.37%), and RPAm variants (2.38%). Data formatting is another 14.47%. These percentages are kernel-local Amdahl shares, not end-to-end shares; the current host/device coverage prevents translating them directly to rollout wall time.

### Host-side decode

Across seven profiled host steps, the latest trace averages:

- `llm_engine.step`: 774.10 ms.
- `execute_model`: 700.36 ms.
- `_process_continue_decode_outputs`: 644.60 ms.
- Blocking `device_get`: 545.22 ms.
- `_prepare_inputs`: 33.53 ms.
- `continue_decode` dispatch: 2.95 ms.

`device_get` includes queueing, waiting for TPU completion, and transfer/materialization; it is not 545 ms of proven CPU compute. The 15.99 / 774.10 = 2.07% ratio is therefore a coverage proxy, not a synchronized hardware-utilization counter. It is nevertheless consistent with the follow-up low head-CPU observation and the neutral window-size experiment. The nesting and 130 `HostBufferStore::Lookup` events per host step make routing, output materialization, and synchronization the first targets.

Correcting the earlier end-to-end extrapolation gives:

| Device-only scenario | Captured-device speedup | Estimated observed-cadence speedup | Seconds saved/cycle |
|---|---:|---:|---:|
| Sampling reductions reach 60% HBM | 1.1490x | 1.0017x | 0.26 s |
| Sampling + formatting + restored all-reduce | 1.2388x | 1.0026x | 0.39 s |
| Loop + convolution reach per-HLO roofs | 1.4641x | 1.0043x | 0.64 s |
| Add formatting at its per-HLO roof | 1.6575x | 1.0053x | 0.80 s |
| Formal whole accounted HBM roof | 3.2606x | 1.0094x | 1.40 s |
| Delete all captured device work | infinite | 1.0136x | 2.02 s |

These estimates assume the representative 2.07% ratio applies throughout engine generation. XProf has no cross-host correlation markers, so treat the table as a sensitivity bound, not a direct measurement.

## Orchestration and training

The later Perfetto trace contains only application-created phase envelopes, not OS scheduling or hardware counters. Within that limitation:

- Rollout: 64.62% of steady critical cycles in the Perfetto alignment.
- Advantage/reward envelope: 10.04%.
- Train wall: 9.46%.
- Weight sync: 6.84%.
- Unlabelled transition gaps: 8.81%.
- Additional gaps between train TPU envelopes: 1.89% of the cycle.

The largest transitions are weight-sync end to next data load, median 9.26 s, and train end to weight sync, normally about 2.8 s. Train TPU-labelled envelopes last 443.9 ms median, but successive starts are 551.3 ms apart, leaving a 106.4 ms median bubble. TPU envelope occupancy within train wall is about 80%; closing this sequencing gap is worth approximately 1.9% end-to-end.

Weight sync itself is 10.44 s median. The model is 8.621 GiB bf16, and embeddings account for 5.151 GiB or 59.74% of parameters. If full weights are actually moved, double buffering, changed-shard-only sync, or special treatment of embeddings has disproportionate leverage. The trace does not prove full-model transfer semantics.

## Tails and cold start

Step 0 is not a steady baseline. The later step-0 rollout is 392.7 s, and the earlier run reaches 689.4 s rollout plus a 977.1 s first-train envelope. XProf collection starts about 24–25 s after the first Perfetto event and flushes near the end of step-0 rollout, so profiling contributes to the cold rollout.

The steady tails are separately actionable:

- Step 7 rollout is 138.87 s versus a roughly 96.09 s median. Its output-token count differs by only -0.031% from the mean of adjacent steps while duration is +45.23%. This is not explained by output length; cache miss/recompile/runtime stall is more likely.
- Reward time spikes at steps 8, 10, 11, and 16 to 27.14–36.02 s versus about 10.74 s on normal steps.

Removing the one rollout stall and four reward stalls lowers the TensorBoard iteration mean from 142.17 s to about 135.22 s and reduces CV from 9.05% to about 2.16%, a +5.0% throughput opportunity.

## Prefill instrumentation limitation

Neither `prefill_heavy` session contains TPU device planes or device OpMetricsDb data. The early coordinator XSpace contains only host metadata, `/host:CPU`, and task-environment planes; the later conversion reports 0.0% device duty.

The apparent improvement is host tracing/compilation only:

| Host event | Early | Later |
|---|---:|---:|
| `llm_engine.step` | 241.105 s | 80.851 s |
| `execute_model` | 151.837 s | 56.987 s |
| Wrapped model/compile | 135.822 s | 54.338 s |

These numbers must not be interpreted as steady prefill performance. Prefill TPU compute, HBM, ICI, and idle headroom are unknown. A post-warmup recapture with TPU device planes and explicit step markers is required.

## Recommended A/B order

1. A/B DP round-robin routing, vectorized EOS/output extraction, batched detokenization, and fewer cross-rank cache probes while preserving byte-identical prompts and sampling semantics.
2. Revisit the single-process engine-driver restriction under Pathways. Measure 4–8 driver processes or multi-head sharding with head CPU, per-window wall time, and useful tokens/s.
3. Add markers and CPU/scheduler samples around `device_get`, `HostBufferStore` lookups, actor dequeue, reporting/upload, reward RPCs, and train-to-sync boundaries.
4. A/B double-buffered or incremental weight sync, measuring actual overlap rather than only sync duration.
5. Test correct stop-token handling and active-sequence compaction/refill. Hold prompts, RNG, and policy snapshot fixed.
6. Only after host coverage improves, microbenchmark the `[262144,32]` sampling reductions and scalar 64-way all-reduce.
7. Recapture after at least five unprofiled warmup steps, with correlated host/device step IDs, all populated TPU planes, and a valid prefill device profile.

## Evidence and caveats

- Decode comparisons use one representative populated host per capture. HLO occurrence counts are aggregated across TensorCore planes and differ between captures, so per-op latency and work-normalized metrics are safer than raw counts.
- Both XProf sessions were collected during cold outer step 0. The later decode program itself is warmed, but the capture is not a steady full-step sample and has no step markers.
- XProf's roofline values are modeled/accounted from OpMetrics/HLO metadata; `performance_counter_result` is empty. RPAd/RPAm custom calls (22.96% of core-time) report zero FLOPs and bytes, so the aggregate arithmetic intensity and bandwidth omit their work.
- The 99% device duty and 0.447% explicit idle apply only inside the captured device interval; they do not rule out host or orchestration stalls elsewhere in rollout.
- The host Chrome traces are capped at exactly 1,000,000 duration events. Inclusive Python durations overlap and must not be summed.
- XProf reports no step markers.
- Perfetto has zero `sched`, `thread_state`, CPU/perf samples, counters, or real process/thread tracks. It supports phase-envelope timing only.
- All isolated speedup ceilings use Amdahl's law and are deliberately non-additive.

Supporting analyses:

- `rollout_roofline.md`
- `metrics/metric_findings.md`
- `perfetto/analyze_perfetto.py`
- Raw small artifacts under `raw/`
