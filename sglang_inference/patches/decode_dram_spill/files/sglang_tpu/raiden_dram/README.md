# Experimental decode-host DRAM spill and recall

This dynamic HiCache storage backend lets a TPU prefill engine spill cached KV
pages into Raiden host-store processes running on decode hosts. A later request
can discover those pages through the global registry and recall them to prefill.
The decode host lends **CPU DRAM**; its model engine does not recompute those
pages or supply decode-generated tokens to this path.

This is an experimental prototype. CPU tests exercise real gRPC with simulated payload
movement. Native TPU DMA and multi-host model-serving validation are still
required before treating this as a supported feature.

## Data flow and ownership

```text
Prefill HBM
  -> existing Raiden D2H + HiCache completion fence
  -> prefill HiCache host slots (held by the backup operation)
  -> WriteRemote: decode-host store allocates slots and pulls from prefill
  -> COMMITTED: remote index and global registry contain the pages

Next turn / local cache eviction
  -> registry Lookup + ResolveStore
  -> Fetch: remote store pins its source and copies to new prefill host slots
  -> existing HiCache H2D + forward completion gate
  -> prefill HBM, then ordinary prefill execution and P/D handoff
```

`storage.py` implements SGLang's dynamic `HiCacheStorage` v1/v2 interfaces.
`client.py` implements the Raiden store RPC lifecycle. KV bytes travel through
Raiden's native block transport; Python sends identities and page IDs only.
The existing local `KVCacheManager` exposes its endpoints and geometry through
`RaidenKVTransfer.host_store_descriptor()`.

Only the remote process creates a `KVCacheStore`. SGLang remains the allocator
for the prefill host pool; attaching another Raiden allocator to that pool would
allow both owners to allocate the same slots.

This follows vLLM TorchTPU's remote DRAM storage and recall mechanism, with a
different spill trigger. SGLang's `write_back` can copy ahead of HBM eviction;
`write_through` backs up after caching a prefix. Both then enqueue the remote
write after D2H completion. This implementation does **not** port vLLM's native
`KVCacheStore::SweepOnce` low-watermark policy for prefill DRAM. It also uses an
explicit ordered target list instead of `GetPlacementTargets`, because the
prefill HiCache allocator is not a registered native store.

Initial scope: fused MHA/GQA KV, PP=1, attention CP=1, and separate `kv`/`swa`
host pools. Each TP/DP rank and pool has an independent remote store identity.
MLA, draft/state pools, heterogeneous prefill layout reuse, and flat-tensor L3
interfaces are rejected or outside this implementation. Use host-memory mode
`cache`, without page-envelope/unified-memory layouts, for initial validation.

## Runtime and protocol version

Install the normal SGLang TorchTPU environment, including `torch_tpu`, `tpu_sync`,
`grpcio`, and `protobuf>=6.31.1`. The Raiden wheel/source build must provide
`global_registry_server` and `kv_cache_host_store_node_main`.

The wire subset and transfer lifecycle were checked against
[google/tpu-sync at 73499f1d463608d1337be92a9ef5eae8f7f431c0](https://github.com/google/tpu-sync/tree/73499f1d463608d1337be92a9ef5eae8f7f431c0).
Raiden describes its API as unstable: rerun the wire check when changing that
dependency, and review the native transfer/lifetime semantics too. The private
protobuf package avoids descriptor-name collisions with tpu_sync; RPC service
paths and field numbers remain upstream-compatible.

```bash
python -m sglang_tpu.raiden_dram.check_wire \
  --tpu-sync-source /path/to/tpu-sync
```

This developer check requires `grpcio-tools`. Serving does not. To regenerate
`wire_pb2.py`, use `grpcio-tools==1.76.0` (protobuf compiler 6.31.1):

```bash
python -m grpc_tools.protoc -I . --python_out=. sglang_tpu/raiden_dram/wire.proto
```

## Launch configuration

1. Start a reachable native registry:

   ```bash
   global_registry_server --port=50051
   ```

2. Add these flags to a working **prefill** launch command. Keep the existing
   model, topology and Raiden P/D transport flags. Replace the address and cache
   namespace before launching:

   ```bash
   --enable-hierarchical-cache \
   --hicache-io-backend raiden \
   --hicache-mem-layout page_first_fused \
   --hicache-host-memory-mode cache \
   --hicache-write-policy write_through \
   --hicache-storage-backend dynamic \
   --hicache-storage-prefetch-policy wait_complete \
   --hicache-storage-backend-extra-config '{
     "backend_name": "raiden_dram",
     "module_path": "sglang_tpu.raiden_dram.storage",
     "class_name": "RaidenDramStorage",
     "interface_v1": true,
     "global_registry_address": "REGISTRY_HOST:50051",
     "kv_pool_group": "sglang-prefill-cache",
     "cache_namespace": "CHECKPOINT_REVISION_AND_KV_CONFIG",
     "decode_store_job": "sglang-decode-dram",
     "decode_store_replicas": ["0"],
     "manifest_dir": "/tmp/raiden-dram-manifests",
     "rpc_timeout_s": 30,
     "prefetch_threshold": 64
   }'
   ```

   `interface_v1` selects SGLang's indexed host-pool interface. The backend also
   implements v2 for hybrid KV/SWA models. It does not serialize flat tensors.
   Choose a host-pool size suitable for the deployment with the usual HiCache
   sizing flags. Keep `write_through` for the validation recipe so the first
   request creates remote data without needing a pressure workload.

   `cache_namespace` must identify the immutable checkpoint, quantization and
   KV scales, RoPE configuration and adapters. Change it for any weight/config
   change, and restart/re-attach with the new namespace. Physical layout, model
   name, page size, rank and pool are added automatically. This prototype does
   not automatically version remote entries on hot weight updates; do not reuse
   a namespace after one. Namespace isolation is not an authentication boundary.

3. Prefill startup publishes each pool's `KVTransferSpec` and writes one JSON
   manifest per rank/pool. Copy those manifests to the corresponding decode
   host, then launch the stores **on that host**, alongside its decode engine:

   ```bash
   python -m sglang_tpu.raiden_dram.launch_stores \
     --manifest /path/to/copied-manifests/*.json \
     --replica 0 \
     --store-server-ip DECODE_HOST_IP \
     --dram-budget-bytes-per-store 1073741824
   ```

   `--dry-run` prints commands without starting processes. The DRAM budget is
   **per manifest/process**, not a total: eight rank manifests at the example
   budget allocate about 8 GiB in aggregate, plus process/transport overhead.
   KV and SWA each add a store. Reserve this capacity separately from decode's
   own local cache. The native stores use ephemeral, registry-advertised ports;
   prefill and decode hosts must reach each other's native transport endpoints.

   To add target hosts, list their distinct replica strings in the prefill
   configuration and launch the same group manifests on each, with its own
   replica and IP. Writes try them in order; this is capacity fallback, not
   replication or load balancing. A read follows the registry's current owner.

4. Send traffic only after all stores have registered. A missing/full store
   makes the backup fail; SGLang can still serve from local cache or recompute.
   Entries are best-effort and subject to the native store's eviction policy.
   Restarting a host store loses that process's cached DRAM.

## Completion and failure behavior

Write admission is not completion. `WriteRemote` must report `COMMITTED` or
existing data before backup success is returned. Partial existing offers retry
only missing pages. A lost result stream triggers `PollWriteRemote` recovery.
`STORED_UNREGISTERED` is not a successful discoverable backup.

The native `Fetch` implementation pins the source during the transfer and
returns success only after its copy completes. Registry lookup alone does not
pin data; eviction between lookup and fetch produces a miss and recomputation.

A gRPC timeout does not cancel native DMA. On an uncertain result, the backend
quarantines source/destination slots before HiCache can free them. Quarantined
slots remain unavailable across `/flush_cache` and are reclaimed only by process
restart. Repeated transport failures can reduce local host-cache capacity;
`get_stats()` exposes `quarantined_pages`, `failed_pages`, `spill_pages` and
`recall_pages`. These are per-backend counters, not new Prometheus metrics.
Storage operations are currently serialized per rank; no throughput claims
have been validated.

## Verification

CPU unit and gRPC integration tests (payload DMA is a test double):

```bash
PYTHONPATH=python:. python -m pytest -q \
  test/registered/unit/hardware_backend/tpu/test_raiden_dram_storage.py
```

Native transport round trip on a TPU, with real registry/store binaries on PATH:

```bash
PYTHONPATH=python:. python -m pytest -v -rs \
  test/manual/tpu/test_raiden_dram_spill_recall_ondevice.py
```

This test erases every prefill host slot after spilling, recalls into different
slots, reloads different TPU pages and checks every value. Its store is a
separate local process. It skips explicitly if the native environment is absent;
a skip does not establish hardware correctness.

For a full serving check, launch an isolated P/D deployment with the flags above
and native host stores on the decode machines. Use a prompt substantially longer
than the page size and prefetch threshold, then run:

```bash
python -m sglang_tpu.raiden_dram.verify_engine \
  --generate-url http://ROUTER:30000 \
  --flush-url http://PREFILL:30001 \
  --flush-url http://DECODE:30002 \
  --prompt-file /path/to/long-prompt.txt
```

Include **every** engine the router may select in `--flush-url`. This script
clears their local caches twice, so run it on a dedicated test deployment.
It makes a unique cold request, waits for write-through backup, flushes HBM and
local DRAM, then repeats it. It requires positive `cached_tokens_details.storage`,
zero local-tier hits and identical greedy output. Remote stores are preserved.
For a single combined engine, use its URL for both generate and flush.

The implementation reference was
[vllm-torchtpu at e785dfa6e8367ffa92cd5031e7f3116ec6a74764](https://github.com/vllm-project/vllm-torchtpu/tree/e785dfa6e8367ffa92cd5031e7f3116ec6a74764);
the destination-pull protocol, commit handling and recall source pinning live
in Raiden's `kv_cache_store_service.cc` at the pinned tpu-sync revision above.
