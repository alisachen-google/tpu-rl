# Benchmark plan for SGLang TorchTPU disaggregation improvements

Prepared **2026-10-20 UTC** for the
[SGLang/vLLM feature comparison](DISAGG_FEATURE_COMPARISON.md).
**Status: proposed experiments; no runs or performance results recorded here.**
This plan excludes remote DRAM spill as the variable under test unless a case
explicitly names it.

## Questions to answer

1. How much time is spent computing prefill, transferring KV, waiting for
   admission, and waiting for the first decode forward?
2. Does block-major KV reduce end-to-end cost after enabling the existing DMA
   coalescing and overlap mechanisms?
3. Can a different P/D topology improve latency or capacity at the same total
   TPU budget without correctness or liveness regressions?
4. For hybrid models, which useful layouts become possible after state
   resharding?
5. When does short-suffix prefill on D beat another P/D handoff while preserving
   ongoing decode latency?
6. Do independent P/D block sizes improve useful capacity or transfer cost?

## Controls and reproducibility

Record the exact framework, core dependency, torch, torch_tpu, libtpu and
tpu-sync versions; container digest; model/tokenizer revisions; patch commits;
router build; and resolved settings on every rank. Start from the pinned
comparison revisions, or clearly label a later baseline. Never mix a PR's
published measurements with new baseline results.

For each A/B pair, keep these fixed unless they are the declared variable:

- Checkpoint, chat template, tokenized requests, output limit, sampling,
  weight quantization, KV dtype and speculative configuration.
- TPU generation and total chip count; explicitly distinguish chips, cores
  and framework ranks. Record the P/D split and all TP/PP/DP/PCP/DCP groups.
- Network placement, host CPU resources, host-staging budget and HBM budget.
- Prefix state, cache policy, router policy and offered traffic.
- Native runtime version, DMA coalescing, transfer parallelism and compiler
  settings.

Use two comparisons where useful: a controlled transfer experiment with
matched layouts, then the best *validated* topology for each engine under the
same total chip budget. Different optimized topologies are a serving-system
comparison, not an isolated kernel comparison.

Separate cold start, compilation and warmup from steady state. Warm every
shape the workload will use; retain serving-time compilation counts. Alternate
baseline/candidate run order and repeat enough independent trials to show
variance. Retain request-level measurements, not only an average.

## Workload matrix

The values below are proposed starting points. Trim them to the model's
validated context and capacity limits; do not bypass launch guards to fill
the matrix.

| Dimension | Initial cases | What it exposes |
| --- | --- | --- |
| Model family | GPT-OSS-120B; one Qwen3.5-class GDN hybrid | Attention/SWA transfer versus hybrid-state transfer |
| Prompt length | 4k, 32k, 128k tokens | Fixed handoff overhead versus bandwidth and long-context cost |
| Output length | Fixed 512-token cap; separate natural-stop run | Comparable decode demand versus realistic completion behavior |
| Concurrency | 1, 8, 32, plus the largest validated level | Isolated latency, overlap and saturation |
| Arrival pattern | Closed-loop concurrency sweep, then open-loop rate sweep | Peak service rate versus queue growth and latency targets |
| Prefix reuse | Cold, 75%, 95%; device hit versus deliberate host restore | Compute savings, suffix-only transfer and restore interference |
| Multi-turn suffix | 32, 128, 512 new tokens after an 8k/32k history | Break-even for D-local extension |
| Allocation state | Fresh contiguous pages, then controlled cache churn | DMA coalescing sensitivity and fragmentation |
| Boundaries | Page/chunk/SWA boundary ±1; partial final pages | Placement and accounting errors hidden by aligned prompts |

Measure actual reused prefix tokens and cache tier. A repeated prompt alone
does not prove a hit, and a router affinity setting alone does not prove that
the request reached the cached rank. For multi-turn tests, include prior
generated tokens in the history and establish whether their KV is retained.

## Experiment sequence

| ID | Comparison | Prerequisite | Decision evidence |
| --- | --- | --- | --- |
| E0 | Current SGLang P/D versus a matched vLLM P/D configuration | Both independently correct on the chosen model/topology | Bottleneck decomposition and latency-constrained goodput |
| E1 | Current SGLang baseline versus existing overlap/routing settings | Supported flag combinations; resolved configuration captured | Less exposed handoff or queue time without worse decode tails |
| E2 | Layer-major versus #689 block-major; exercise proposed #913 coverage | Rebased/reviewed patch, matching dtype, spec compatibility checked | DMA count, transferred bytes, transfer time, attention time, TTFT and TPOT |
| E3 | Supported baseline versus #1372 → #1373 → #1374 topology stack | Batched correctness and concurrency issues resolved/reproduced first | Capacity and goodput at fixed total TPUs; attention/collective cost |
| E4 | Symmetric hybrid P/D versus GDN-state resharding | New implementation plus state-placement and e2e correctness tests | Benefit of independently choosing P/D layouts |
| E5 | Always route through P versus bounded short-suffix extension on D | New D extend scheduler, warmed shapes, usable context, router policy | Follow-up TTFT and p99 TPOT of unrelated active requests |
| E6 | Equal versus independent P/D block sizes | Explicit remapping implementation and tail/window tests | Useful HBM capacity, fragmentation, DMA count and bytes/request |
| E7 | Existing P cache/load routing versus P and D locality/load selection | New D-aware policy and fresh cache/load signals | Cache hit rate, queue imbalance, transfer bytes and multi-turn goodput |

For E1, benchmark relevant existing mechanisms separately: chunk transfer,
pipeline early parts/stage groups/early close, prefix reuse and router
affinity/admission. Listener-driven starts and dynamic host staging already
default on in the reviewed SGLang tree. Record defaults instead of crediting
an unchanged flag with an improvement.

For E2, keep native coalescing on for the primary comparison. A disabled-
coalescing run is a diagnostic; it must be labeled as such. Add controlled
fragmentation with coalescing still enabled to test a realistic hard case.
Do not silently disable EAGLE3 or change KV dtype to make a candidate run.
Report a supported non-spec experiment separately from the deployment target.

For E3, preserve the exact tested PR stack and runtime. The author of #1373
reports batched DCP2 correctness problems and uses a modified guard for DCP8
experiments. That report is a validation requirement, not permission to
assume either configuration is ready. PP × PCP/TP concurrency also needs
its own liveness gate. See the [PR tracker](DISAGG_FEATURE_COMPARISON.md#existing-prs-to-evaluate).

For E5/E7, test hot context on the selected D, context only on another D,
evicted context, and an overloaded cache owner. Compare local extension with
the cost of queuing, restoring or transferring the context. Keep a fallback to
the normal P/D path. The spill store's prefill-layout payload is not itself a
decode-compatible cache hit.

## Correctness and liveness gates

Run these before using a candidate's performance numbers:

- Known-answer and needle probes, including long/chunked requests; model
  accuracy checks against an agreed baseline.
- Matched-batch greedy comparisons where reproducible. Batch-dependent
  numerics can invalidate strict cross-concurrency token equality; retain
  semantic and accuracy checks instead of declaring every divergence a KV
  corruption.
- Every supported TP/CP/PP rank receives the correct token, head and layer
  ranges. Use unique-byte marker tests for resharding, plus real TPU tests.
- Full and partial prefix hits, final partial pages, SWA windows, and GDN
  committed state with accepted/rejected speculative tokens where supported.
- Cancellation during prefill/transfer, delayed peers and cache churn:
  no stale DMA write into recycled pages, leaked reservations or stalled
  unrelated requests. Wait for native completion before reuse.
- Steady-state memory and queue recovery after traffic drains; no unexpected
  serving recompilation, worker restart or silent fallback to another layout.

CPU mapping tests do not establish TPU DMA completion, compiled layout
correctness or serving performance. Existing registered integration scripts
are useful starting points; capture the actual run outcomes.

## Metrics and timing boundaries

| Metric | Definition / use |
| --- | --- |
| TTFT | Client request admission to first generated-token event; use the same client/protocol semantics in both engines |
| TPOT / inter-token latency | Record per-request TPOT and individual streaming token gaps separately; include p50/p95/p99 |
| Goodput | Requests or output tokens served within predefined TTFT and TPOT targets; report throughput alongside it |
| Exposed handoff gap | From final required P KV becoming ready to the first D forward for the request |
| Transfer breakdown | Host-staging wait, coordination/registration, D2H, network movement, H2D and completion-to-scheduler delay |
| Transfer work | Payload bytes, DMA submissions, transferred pages and suffix/prefix bytes skipped |
| Memory | HBM peak/usable KV capacity, host staging peak, local host-cache use, fragmentation and reservation lifetime |
| Scheduling | P/D queue wait, active batch sizes, rejected/retracted requests and pending-transfer count |
| Runtime | Host submission time, CPU utilization, compilation count and attention/collective execution time |
| Cache/routing | Verified device/host hits, reused tokens, selected worker/rank and queue/load distribution |

Correlate P and D traces by request/transfer ID. Use synchronized clocks or a
trace alignment method before subtracting timestamps across hosts. Transfers
can overlap compute and each other; do not sum overlapping component durations
and call the result request latency.

Inspect p99 decode latency during heavy ingress, not only after prefill
finishes. A smaller average TTFT does not justify a candidate that violates the
decode service target. Capacity changes are valid results but should be
reported separately from speed at an equal active batch.

## Evidence to save for each run

Store completed runs under a new directory such as
`results/disagg_feature_comparison/<run_id>/`, leaving existing campaign
results intact. This plan does not create a result directory or placeholder
measurements.

Each run should include:

- Exact revisions, package versions, image digest, complete launch commands,
  environment and resolved per-rank configuration.
- Hardware/topology, weights/tokenizer IDs, prompt-set hash, traffic seed,
  intended cache state and the counters proving the actual state.
- Warmup/compile logs, request-level timings, correctness outputs, server
  metrics and correlated P/D profiles.
- Baseline/candidate pair, changed variable, trial index, failures and
  confidence/variance information.
- A decision: adopt, investigate further, or reject for this workload, with
  the measured benefit and compatibility limits.

Use this compact decision record:

```text
Experiment / run IDs:
Baseline revision / candidate revision / dependency versions:
Model / weights / KV dtype / spec mode:
TPUs and P/D topology:
Workload / offered load / cache state:
Changed variable:
Correctness and liveness outcome:
TTFT p50/p95/p99:
TPOT and token-gap p50/p95/p99:
Throughput / goodput / latency targets:
Exposed handoff / DMA count / transfer bytes:
HBM and staging peak:
Serving-time compilations:
Trials / variance / artifact paths:
Decision and remaining limits:
```

No speedup should be promoted to the feature comparison until its baseline,
configuration, correctness outcome and raw evidence are linked.
