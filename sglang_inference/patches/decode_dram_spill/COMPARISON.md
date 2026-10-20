# SGLang implementation versus vLLM TorchTPU

For the broader feature review beyond remote spill, see
[SGLang versus vLLM disaggregation](../../DISAGG_FEATURE_COMPARISON.md) and its
[benchmark plan](../../DISAGG_BENCHMARK_PLAN.md). That review uses newer
framework revisions; the spill-specific comparison below retains its original
implementation pins.

Both paths aim to preserve prefill-generated KV in remote host DRAM and reuse
it on a later request. Both rely on Raiden for native transport. This SGLang
prototype integrates that mechanism through HiCache; it is not a full port of
vLLM's offloading controller or its native cache-management policy.

The comparison is pinned to vLLM TorchTPU
`e785dfa6e8367ffa92cd5031e7f3116ec6a74764`, SGLang TorchTPU
`f2588cdf793423442fcebc1c816f399739dd2f9f` plus this patch, and tpu-sync
`73499f1d463608d1337be92a9ef5eae8f7f431c0`.

| Aspect | Referenced vLLM TorchTPU path | This SGLang TorchTPU implementation |
| --- | --- | --- |
| Framework integration | Scheduler-side offloading manager and Raiden connector | Dynamic `HiCacheStorage` backend |
| Prefill host-cache owner | Native `KVCacheStore` owns the directory, allocation, LRU, pins, and transfer lifecycle | SGLang HiCache owns the prefill host slots; only the remote process creates a `KVCacheStore` |
| Remote spill trigger | When configured, the native monitor invokes `KVCacheStore::SweepOnce` under host-pool pressure | HiCache's backup/write policy invokes `batch_set_v1` or `batch_set_v2` after D2H completes |
| Destination choice | Native pressure path asks the registry for `GetPlacementTargets` | Explicit ordered decode-store replicas; capacity fallback, without replication or load balancing |
| Recall API | `prepare_load` pins local hits and records remote slices; `submit_load_job` uses native `read_remote` with destination HBM block IDs | Registry lookup and `Fetch` into prefill HiCache host slots, followed by the existing H2D path |
| Buffer lifetime | Native store pins plus connector admission/completion bookkeeping | Existing HiCache host references/fences plus quarantine of pages with uncertain DMA completion |

## Why the ownership differs

SGLang already allocates and frees its local HiCache host pages. Adding another
native store allocator over the same memory would let two allocators hand out
the same slots. The adapter therefore exports the existing transport endpoints
and page geometry while preserving HiCache as the local owner. Separate native
stores on decode hosts own their own DRAM allocations.

In HiCache terminology, prefill HBM is L1, prefill host DRAM is L2, and the
decode-host Raiden store is L3. L3 describes the storage-backend role: its physical
medium can be another machine's DRAM. It does not have to be disk.

## Recall and spill policy are different

The vLLM connector supplies HBM destinations to `read_remote`; its native runtime
manages that transfer. The SGLang adapter explicitly recalls through its L2 host
pool so that the existing HiCache H2D completion gate remains in control. This
is an API/ownership distinction, not evidence of a measured performance gap or
a claim that the native vLLM transport never stages data internally.

For SGLang, `write_through` can create the remote copy after a prefix is cached,
without waiting for local DRAM pressure. `write_back` follows the existing
HiCache policy, including local copy-ahead in the pinned base. Neither reproduces
vLLM's exact native low-watermark spill policy. Porting that policy would be
additional work.

## Scope and validation

The SGLang prototype initially covers fused MHA/GQA, PP=1, attention CP=1, and
`kv`/`swa` pools with per-rank/pool isolation. Native TPU DMA and two-host model
serving remain unverified; the 126 passing CPU tests cannot establish hardware
correctness or throughput. No comparative performance claim has been tested.

Code references:

- [vLLM Raiden offloading manager](https://github.com/vllm-project/vllm-torchtpu/blob/e785dfa6e8367ffa92cd5031e7f3116ec6a74764/src/vllm_torchtpu/offload/raiden_store.py): `prepare_load`, `submit_load_job`, and `KVCacheStore` construction.
- [Native Raiden cache store](https://github.com/google/tpu-sync/blob/73499f1d463608d1337be92a9ef5eae8f7f431c0/tpu_sync/kv_cache/kv_cache_store.cc): `KVCacheStore::SweepOnce`, target selection, and remote spill scheduling.
- [SGLang adapter](files/sglang_tpu/raiden_dram/storage.py) and [RPC client](files/sglang_tpu/raiden_dram/client.py): host-page registration, backup, recall, and failure handling.
