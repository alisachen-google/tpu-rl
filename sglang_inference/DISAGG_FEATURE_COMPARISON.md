# SGLang TorchTPU versus vLLM TorchTPU: disaggregated inference

Reviewed on **2026-10-20 UTC**. Beyond remote DRAM spill, the strongest
candidates for improving SGLang are block-major KV transfers, broader P/D
parallelism, hybrid-state resharding, and short follow-up prefill on decode.
Several already have open SGLang PRs to evaluate.

This is a source and test-coverage review, not a measured ranking of the two
engines. Proposed benefits below are hypotheses until benchmarked. Existing
test files establish intended coverage, not that a particular hardware run
passed. No new serving code or TPU benchmark results accompany this document.

Companions:

- [Benchmark and validation plan](DISAGG_BENCHMARK_PLAN.md).
- [Existing spill/recall implementation](patches/decode_dram_spill/README.md).
- [Spill-specific ownership and API comparison](patches/decode_dram_spill/COMPARISON.md).

## Revisions and terminology

| Component | Revision inspected |
| --- | --- |
| SGLang TorchTPU | [`deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6`](https://github.com/sgl-project/sglang-torchtpu/tree/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6) |
| vLLM TorchTPU | [`e220041eab5802637e0c24a7c0e960279d7bef53`](https://github.com/vllm-project/vllm-torchtpu/tree/e220041eab5802637e0c24a7c0e960279d7bef53) |
| vLLM core dependency | [`v0.31.0`, `db9527a46873454610df6dbedf79a36d6bf1a7f6`](https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6) |

The SGLang repository is **`sgl-project/sglang-torchtpu`**. Upstream GPU SGLang
capabilities are not evidence that a combination works through this TPU fork.
The spill patch has an older, separately pinned base; this comparison does not
rebase or revalidate that patch.

P/D means prefill/decode. TP partitions tensors; PP partitions model layers
across pipeline stages; DP runs replicas. PCP distributes prefill context work;
DCP distributes a request's decode KV/attention work across ranks. Exact rank
ownership differs between frameworks, so matching flag values is not enough
to match topology.

## Feature differences

| Area | vLLM TorchTPU at the pinned revision | SGLang TorchTPU main at the pinned revision | Why investigate it |
| --- | --- | --- | --- |
| Block-major KV | Optional bundles make a logical block's layer regions contiguous. [Implementation][v-block] | Standard fused KV allocates a separate tensor per layer. [Allocation][s-pool] | Reduce separate DMA submissions and transfer overhead, especially with fragmented pages. |
| P/D with decode context parallelism | Registered Qwen3.5 integration path includes PCP4/TP1 P → TP4/DCP2 D. [Test][v-dcp], [CI registration][v-ci] | TPU validation rejects DCP > 1 with disaggregation. [Validation][s-platform] | Increase long-context decode capacity and distribute attention work; test TPOT and collective overhead. |
| Broader pipeline source layouts | Has pool/span-based PP and PCP transfer paths, subject to model and runtime admission. [Connector][v-connector] | Pipeline transfer requires attention CP=1 and a PP producer with attention TP=1. [Checks][s-pipeline] | Choose a prefill layout suited to prompt computation independently of decode. |
| GDN state resharding | Transfers full-attention KV plus convolution/recurrent state with separate geometry. [Pool manifest][v-manifest], [connector][v-connector] | Symmetric-TP GDN state transfer exists; asymmetric push resharding and pipeline transfer reject GDN state. [Receiver][s-receiver], [manager][s-manager] | Expand useful Qwen3.5-class P/D layouts. |
| Different P/D block sizes | Selected Raiden paths support different scheduler block sizes; one integration recipe uses 768 tokens on P and 2304 on D. [Recipe][v-block-test] | Bootstrap requires identical `--page-size` and KV dtype. [Peer checks][s-peer] | Tune allocation and transfer granularity independently; measure fragmentation and transferred bytes. |
| Local prompt/suffix work on D | A KV consumer can compute missing tokens or fall back to local prefill. [Admission/fallback][v-connector], [core scheduler][v-scheduler] | Dedicated P/D decode admits prebuilt transferred requests and runs decode; it does not schedule fresh prompt chunks. [Scheduler][s-decode] | Avoid a P/D round trip for short follow-up turns when D already has usable context. |

These are configuration-specific capabilities. In particular, vLLM still
rejects TP inside PP stages, requires an appropriate torch_tpu for stages
spanning multiple devices, and rejects async scheduling under PP. A planner
that can describe a topology does not prove the entire serving path admits it.
[vLLM PP validation][v-pp]

### Block-major storage: first candidate for an existing GPT-OSS deployment

SGLang's ordinary fused KV pool stores one tensor per layer. Its asymmetric
push path waits for D2H completion, then calls `push_registered_plan` for each
layer under a serialization lock. This identifies transfer submission work
worth profiling; it does not prove that the lock or DMA count is the dominant
bottleneck. [Allocation][s-pool], [push implementation][s-sender]

vLLM's optional block-major layouts group layer fragments into contiguous
block rows while attention kernels remap indices into the bundle. Supported
geometries can therefore move a block with fewer separate DMA operations.
Its hybrid P/D plus offload integration recipe disables speculative decoding
because that merged-pool combination refuses it. [Layout][v-block],
[dense offload contract][v-block-contract], [integration recipe][v-block-test]

SGLang already has [PR #689](https://github.com/sgl-project/sglang-torchtpu/pull/689)
and the proposed nightly coverage in
[#913](https://github.com/sgl-project/sglang-torchtpu/pull/913). The PR author
reports much larger gains with DMA coalescing disabled. Coalescing-enabled
results are smaller and variable, including regressions in some chunk-size
cases. Treat those as author-reported results on those configurations, not
current-head measurements or a general speedup promise. Evaluate the PR
against the current coalesced baseline and include kernel execution time and
speculative-decoding compatibility in the decision.

### P/D topology and GDN state: related but separate work

SGLang already reshards attention heads for supported asymmetric TP pairs.
The larger gap is composing P/D with DCP and more pipeline source layouts.
The current pipeline guard and DCP validation make the restriction explicit.
[Pipeline checks][s-pipeline], [DCP validation][s-platform]

For GDN hybrids, transferring attention KV alone is insufficient. Convolution
state, recurrent state, rank/head ownership, and the committed speculative
state must agree with the receiving model. vLLM's manifests and byte-span
transfer path provide a design reference. SGLang's existing third state engine
is a starting point, but its asymmetric and pipeline state paths need new
implementation and coverage. [vLLM manifest][v-manifest],
[vLLM byte-level reshard oracle][v-oracle], [SGLang receiver][s-receiver],
[SGLang state engine][s-manager]

For GPT-OSS and ordinary attention models, evaluate topology improvements after
the transfer baseline. For Qwen3.5-class hybrids, state resharding may take
priority because it enables the desired topology at all. Neither priority
implies that a larger DCP size always improves throughput.

### Independent page sizes: allocation granularity, not arbitrary conversion

vLLM's integration example with different P/D scheduler block sizes is evidence
for that admitted configuration, not arbitrary layout or dtype conversion.
Scheduler blocks may also contain multiple kernel blocks. SGLang currently
requires equal page sizes at bootstrap, so adding this feature needs explicit
token-to-byte mapping and receiver placement, including partial tails, cached
prefixes and SWA windows. [vLLM recipe][v-block-test], [SGLang checks][s-peer]

Benchmark this after topology and basic transfer efficiency. Larger pages can
reduce bookkeeping while increasing fragmentation or wasted transfer bytes;
the useful setting depends on request length and cache reuse.

### Short follow-up prefill on decode

The proposed multi-turn path is:

```text
usable cached prefix on decode
    -> restore to decode HBM if needed
    -> compute only the new user-turn suffix
    -> continue decoding
```

vLLM's consumer participates in the normal token-budget scheduler and can
compute tokens absent from the transferred prefix. Its example proxy still
sends the request to prefill before decode: automatic cache-aware bypass is a
separate routing policy, not a demonstrated default optimization.
[Consumer fallback][v-connector], [scheduler][v-scheduler], [proxy][v-proxy]

SGLang needs a way to admit an extend batch into dedicated P/D decode, compiled
shapes for those batches, compatible prefix reuse and a routing decision.
Bound suffix tokens and outstanding extend work so an optimization for one
conversation does not violate other requests' TPOT targets. Choose between
local extension and remote prefill using cache availability, suffix length
and current P/D load. This is a proposed design, not an existing feature.
[Current decode loop][s-decode]

The remote spill prototype does not supply this automatically. Its independent
decode-host store preserves the prefill layout, and it does not capture newly
generated decode tokens. A useful D-local continuation needs all context it
will reuse in a decode-compatible layout, potentially including retained
decode-generated KV. See the [prototype scope](patches/decode_dram_spill/README.md).

## Features SGLang already has

| Feature | Existing behavior and implication |
| --- | --- |
| Chunked prefill/transfer overlap | Commits and announces KV chunks during prefill. A request has one chunk in flight; the closing transfer includes the remaining pages and settled SWA window. Tune and profile this before building another overlap path. [Raiden design][s-raiden] |
| Listener-driven transfer starts | The decode listener can start the next read when announced; `SGLANG_RAIDEN_LISTENER_START_READ` defaults to 1. A scheduler-poll delay is not an unconditional limitation. [Raiden tunables][s-raiden] |
| Pipeline transfer overlap | Optional early parts, stage groups and early close overlap different parts of PP handoff. These options default off and have compatibility constraints. [Pipeline modes][s-raiden] |
| Prefix reuse and speculative decoding | A registered GPT-OSS test combines prefix caching, HiCache, EAGLE3 and asymmetric TP P/D transfer. Adding these features from scratch is not the task. [Combined test][s-combined] |
| Host/device overlap | SGLang uses future-token/device acceptance machinery; vLLM uses async scheduling and deferred output copies. Compare actual stalls and admitted combinations, not a binary “async supported” label. [SGLang overlap][s-overlap], [vLLM output handling][v-async] |
| Cache-aware routing | SGLang's experimental router has prefix affinity, pending-prefix tracking and load admission. Its documented P/D decode selection is load-based, including DP-rank selection. [Router][s-router] |

vLLM's generic `save_kv_layer` and `wait_for_layer_load` hooks are explicitly
unsupported on TPU. Its native block chunking and registration-ahead mechanism
should not be described as generic layer-wise compute/transfer overlap.
[Connector hooks][v-connector]

For multi-turn traffic, extending routing to consider **decode cache locality
and load together** is worth investigating. Preserve the existing prefill
affinity/load policies and verify which cache events reach the selected router.
This is an additional SGLang opportunity, not an established vLLM advantage.
[Router policies and DP routing][s-router]

## Existing PRs to evaluate

Statuses were checked on **2026-10-20 UTC**; they can change. None of these PRs
is included in the SGLang main revision inspected above.

| PR | Status at review | Scope | Head inspected for status |
| --- | --- | --- | --- |
| [#689](https://github.com/sgl-project/sglang-torchtpu/pull/689) | Open, non-draft | GPT-OSS block-major KV | `b4c3881a005a216ba4f440a9d7fa3742f8f732ab` |
| [#913](https://github.com/sgl-project/sglang-torchtpu/pull/913) | Open, non-draft; stacked on #689 | Block-major P/D and offload nightly test | `542d7a0bbebea6845439f56aa61f88d1a81d2d1b` |
| [#1372](https://github.com/sgl-project/sglang-torchtpu/pull/1372) | Open, draft | PP × PCP prefill sources | `169e5223dd5795816bd0aa65ccd25053aa773426` |
| [#1373](https://github.com/sgl-project/sglang-torchtpu/pull/1373) | Open, draft; stacked on #1372 | DCP decode destination | `de9177c2da6deb19157cf4de05d39463f9b102c3` |
| [#1374](https://github.com/sgl-project/sglang-torchtpu/pull/1374) | Open, draft; stacked on #1373 | Attention-TP and one-stage sources | `e339e4d56b731709dd942de5ca6c9c6d333eec55` |

The #1373 author reports incorrect batched DCP2 outputs in the stack tested,
including a standalone control, and tests a DCP8 configuration with a launch
guard removed. #1372/#1374 also report concurrent stalls in some configurations.
These are reported limitations, not defects independently reproduced at the
current comparison revision. Fix or reproduce the relevant correctness and
liveness issues before using those configurations as performance candidates.
Do not infer supported DCP8 serving from byte-mapping unit tests alone.

## Recommended sequence

1. **Establish and tune the current SGLang baseline:** existing prefix reuse,
   chunk/PP overlap, cache-aware routing, DMA coalescing and staging budgets.
2. **Evaluate #689 with #913 for GPT-OSS:** check transfer cost, kernel cost,
   fragmentation and feature compatibility on the actual deployment.
3. **Evaluate the #1372 → #1373 → #1374 stack:** validate batched attention,
   transfer placement and concurrency before comparing topology performance.
4. **Prioritize GDN-state resharding for hybrid targets:** use explicit pool,
   state and ownership contracts rather than treating every state as KV pages.
5. **Prototype D-local short-suffix prefill for multi-turn traffic:** add
   routing, decode-compatible context reuse and a TPOT-aware token budget.
6. **Then tune independent P/D block sizes:** use the measured allocation,
   fragmentation and transfer costs to choose the experiment.

The [benchmark plan](DISAGG_BENCHMARK_PLAN.md) defines comparable workloads,
correctness gates, metrics and the evidence to record for each decision.

[v-block]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/src/vllm_torchtpu/block_major_pool.py
[v-block-contract]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/src/vllm_torchtpu/offload/block_major_layout.py
[v-block-test]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/scripts/vllm/integration/run_qwen35_block_major_disagg_offload.sh
[v-dcp]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/scripts/vllm/integration/run_qwen35_pcp_tp_dcp_disagg_correctness.sh
[v-ci]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/.buildkite/pipeline_integration.yml
[v-connector]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/src/vllm_torchtpu/distributed/kv_transfer/tpu_connector.py
[v-manifest]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/src/vllm_torchtpu/distributed/kv_transfer/raiden/pool_manifest.py
[v-oracle]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/tests/distributed/test_pd_reshard_oracle.py
[v-pp]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/src/vllm_torchtpu/platforms/pp_validation.py
[v-scheduler]: https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py
[v-proxy]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/examples/disagg/toy_proxy_server.py
[v-async]: https://github.com/vllm-project/vllm-torchtpu/blob/e220041eab5802637e0c24a7c0e960279d7bef53/src/vllm_torchtpu/runner/tpu_runner_async_output.py
[s-pool]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/sglang_tpu/kv_pool.py#L310
[s-platform]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/sglang_tpu/tpu_platform.py#L614
[s-pipeline]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/sglang_tpu/disaggregation/raiden/pipeline.py#L439
[s-receiver]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/sglang_tpu/disaggregation/raiden/receiver.py#L197
[s-manager]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/sglang_tpu/disaggregation/raiden/conn.py#L365
[s-sender]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/sglang_tpu/disaggregation/raiden/sender.py#L1078
[s-peer]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/python/sglang/srt/disaggregation/common/conn.py#L1172
[s-decode]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/python/sglang/srt/disaggregation/decode.py#L3148
[s-raiden]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/sglang_tpu/disaggregation/raiden/README.md
[s-combined]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/test/registered/tpu/e2e/test_tpu_gpt_oss_120b_pd_multihost_prefix_cache_kv_offload_spec_reshard.py
[s-overlap]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/sglang_tpu/overlap_utils_tpu.py
[s-router]: https://github.com/sgl-project/sglang-torchtpu/blob/deec4fbada51ba3dc72c935095ed3fa4aa1bb6d6/experimental/sgl-router/README.md
