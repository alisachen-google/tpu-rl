# SGLang TorchTPU: decode-host DRAM spill and recall

This experimental implementation saves prefill-generated KV pages on decode
hosts and recalls them to prefill for a later request. It extends SGLang's
HiCache storage interface using Raiden's native remote-store protocol.

**Status: 126 CPU tests passed; native TPU and two-host model-serving validation
remain pending.** This is a source-and-patch bundle for `sglang-torchtpu`, not an
enabled feature of the existing `tpu-rl` inference or RL launchers.

## How it works

HiCache is SGLang's hierarchical KV cache. Its prefix matching and cache
controller coordinate reuse across local device memory, local host memory, and
an optional storage backend. This implementation supplies that storage backend.

| Tier in this deployment | Owner and location |
| --- | --- |
| L1 | Prefill engine's TPU HBM |
| L2 | Prefill engine's HiCache host slots in local CPU DRAM |
| L3 | Separate native Raiden host-store processes on decode hosts |

```text
Spill:  prefill HBM -> prefill host DRAM -> WriteRemote -> decode-host DRAM
Recall: decode-host DRAM -> Lookup/Fetch -> prefill host DRAM -> prefill HBM
```

The existing Raiden D2H fence protects the first copy. HiCache holds the source
host pages while `WriteRemote` tells the destination store to allocate and pull
them. Backup succeeds only after the remote pages are committed and discoverable
in the registry. A later prefix lookup finds the remote pages; `Fetch` copies
them into HiCache-owned prefill host slots, and the existing H2D path restores
prefill HBM before computation.

The decode machine lends CPU DRAM through a separate store process. The store
does not use the decode engine's private HiCache allocator. The path preserves
prefill-generated KV; it does not add capture of decode-generated tokens.
Python sends page identities and transfer metadata; native Raiden moves the KV
payload. An uncertain DMA result quarantines the affected local slots to prevent
reuse while a transfer may still access them.

## Files to read

| File | Purpose |
| --- | --- |
| [storage.py](files/sglang_tpu/raiden_dram/storage.py) | `RaidenDramStorage`: HiCache v1/v2 adapter, pool registration, namespace isolation, backup and recall |
| [client.py](files/sglang_tpu/raiden_dram/client.py) | Registry lookup, `WriteRemote`/`PollWriteRemote`, `Fetch`, and completion handling |
| [kv_pool_host.py](files/sglang_tpu/kv_pool_host.py) | Host-slot quarantine that survives cache clearing |
| [raiden_transfer.py](files/sglang_tpu/raiden_transfer.py) | Export of existing native transport endpoints and page geometry |
| [launch_stores.py](files/sglang_tpu/raiden_dram/launch_stores.py) | Starts native stores on decode hosts from per-rank/pool manifests |
| [Implementation README](files/sglang_tpu/raiden_dram/README.md) | Runtime dependencies, configuration, launch commands, failure semantics, and serving verification |
| [Unit/integration tests](files/test/registered/unit/hardware_backend/tpu/test_raiden_dram_storage.py) | Real gRPC with simulated payload DMA, including failure and ownership cases |
| [Native TPU test](files/test/manual/tpu/test_raiden_dram_spill_recall_ondevice.py) | Hardware round-trip test; explicitly skips without the required environment |
| [verify_engine.py](files/sglang_tpu/raiden_dram/verify_engine.py) | Serving verifier that flushes local caches and requires remote storage hits |

`files/` contains all 13 implementation, test, and documentation files from the
locally tested snapshot, with the implementation README clarified during
publication. Runtime code and tests are unchanged.
[changes.patch](changes.patch) contains the same changes
against the pinned base. [manifest.json](manifest.json) records source revisions
and SHA-256 hashes. Upstream licensing and notices are retained in
[SOURCE_LICENSE](SOURCE_LICENSE) and [SOURCE_NOTICE](SOURCE_NOTICE).

## Apply to SGLang TorchTPU

Use a clean checkout of
[`sgl-project/sglang-torchtpu`](https://github.com/sgl-project/sglang-torchtpu) at
**`f2588cdf793423442fcebc1c816f399739dd2f9f`**. The older build documented in the
inference campaign runbook is not the patch base.

```bash
# Start from the root of this tpu-rl checkout.
bundle="$(pwd)/sglang_inference/patches/decode_dram_spill"

# Choose a new directory for the SGLang checkout.
git clone https://github.com/sgl-project/sglang-torchtpu.git /path/to/sglang-spill
cd /path/to/sglang-spill
git switch --detach f2588cdf793423442fcebc1c816f399739dd2f9f
git apply --check "$bundle/changes.patch"
git apply "$bundle/changes.patch"
```

Install the normal SGLang TorchTPU environment for that revision, then follow
the [launch configuration](files/sglang_tpu/raiden_dram/README.md#launch-configuration).
It uses `--hicache-storage-backend dynamic` with
`sglang_tpu.raiden_dram.storage.RaidenDramStorage`. Start a registry, enable this
backend on prefill, and use its manifests to launch separate stores on decode
hosts. The documented validation recipe uses `write_through` and
`wait_complete` to make backup and recall observable.

The checked Raiden protocol revision is
[`google/tpu-sync@73499f1`](https://github.com/google/tpu-sync/tree/73499f1d463608d1337be92a9ef5eae8f7f431c0).
The mechanism reference is
[`vllm-project/vllm-torchtpu@e785dfa`](https://github.com/vllm-project/vllm-torchtpu/tree/e785dfa6e8367ffa92cd5031e7f3116ec6a74764).
Both full revisions are recorded in the manifest.

## PR #1517 and scope

[SGLang TorchTPU PR #1517](https://github.com/sgl-project/sglang-torchtpu/pull/1517)
improves local `write_back` behavior with copy-ahead. It does not provide the
remote spill/recall path by itself, and is not a functional prerequisite for the
remote storage backend. The pinned base already includes its merge, so no
separate cherry-pick is needed. In that base, local write-back copying can start
ahead of eviction; remote backup still follows completed D2H copies.

This implementation follows HiCache's backup policy. It does not reproduce
vLLM's exact prefill-DRAM pressure policy (`KVCacheStore::SweepOnce`, driven by
`StoreMonitor`). Initial
scope is fused MHA/GQA, PP=1, attention CP=1, and `kv`/`swa` pools, with independent
namespaces and stores per rank/pool. A configured checkpoint namespace must
change when weights or KV-affecting configuration change.

See [COMPARISON.md](COMPARISON.md) for the differences from the referenced
vLLM TorchTPU implementation, including allocation ownership and recall paths.

## Validation

The local implementation run recorded 41 new CPU unit/gRPC integration tests,
47 existing Raiden host-pool/transfer tests, and 38 platform contract tests
passing. Payload DMA is simulated in the CPU integration tests. The native TPU
test skipped because `torch_tpu` and TPU hardware were unavailable; the full
two-host serving check has not run. The upstream strict support-matrix check
reports 10 label problems, reproduced on the untouched base.

See [validation/README.md](validation/README.md) for commands, archived evidence,
and remaining hardware checks. The bundle was also verified by applying the
patch at the pinned base and comparing all resulting file hashes to the manifest.
