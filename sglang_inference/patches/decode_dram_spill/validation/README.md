# Validation record

Results below are from the local implementation run on 2026-10-19, before the
bundle was copied to `tpu-rl`. The bundled implementation files are byte-identical
to that tested snapshot. [environment.json](environment.json) records the CPU
environment; it is not a production TPU dependency lockfile.

| Check | Result |
| --- | --- |
| New CPU unit and real-gRPC integration tests | 41 passed; native payload DMA simulated |
| Existing Raiden host-pool, memory-budget, and completion tests | 47 passed; 61 unrelated tests deselected; 19 additional subtests |
| Existing TPU out-of-tree platform contract tests | 38 passed; 10 additional subtests |
| CPU total | **126 passed**, excluding subtests |
| Native TPU spill/recall test | 1 skipped: native TPU environment unavailable |
| Full model-serving / two-host P-D check | Not run |
| Raiden wire compatibility | 62 protobuf fields, enum values, and all 6 RPC signatures compatible with the pinned tpu-sync revision |
| Python compilation and source diff whitespace checks | Passed |
| Strict support matrix | 10 pre-existing label problems, also present on the untouched base |
| Export and publication integrity | All 13 file hashes match; patch applies at the pinned base and reproduces those files |

Archived evidence:

- [New-test output](sglang-spill-new-tests.log) and [JUnit report](sglang-spill-new-tests.xml).
- [Platform contract output](sglang-spill-platform-tests.log).
- [Untouched-base support-matrix errors](sglang-spill-baseline-matrix.err).

## Reproduce the checks

Run from the patched **SGLang TorchTPU** checkout after installing its test
dependencies. The original CPU run installed additional dependencies in an
isolated directory and prepended it to `PYTHONPATH`; that VM-specific path is
not required when dependencies are installed in your own environment.

```bash
PYTHONPATH=python:. python -m pytest -q -rs \
  test/registered/unit/hardware_backend/tpu/test_raiden_dram_storage.py \
  test/manual/tpu/test_raiden_dram_spill_recall_ondevice.py

PYTHONPATH=python:. python -m pytest -q \
  test/registered/unit/hardware_backend/tpu/test_tpu_raiden_offload_unit.py \
  -k 'HostPoolPaging or HostMemoryBudget or KVTransferEmptyPageList or FutureEvent or CombinedFutureEvent'

PYTHONPATH=python:. python -m pytest -q \
  test/registered/unit/plugins/test_tpu_oot_platform_contract.py

# Use tpu-sync revision 73499f1d463608d1337be92a9ef5eae8f7f431c0.
# This developer-only checker also requires grpcio-tools.
PYTHONPATH=. python -m sglang_tpu.raiden_dram.check_wire \
  --tpu-sync-source /path/to/tpu-sync

python scripts/ci/tpu/support_matrix.py --strict
```

CPU test success establishes adapter, RPC, completion-state, and buffer-ownership
behavior under the tested simulations. It does not establish native DMA,
multi-host transport, model correctness, or TPU performance.

## Hardware checks still required

Run the included native test with `torch_tpu`, an available TPU, and the native
registry/store executables. It spills pages, erases the prefill host copy,
recalls into different slots, reloads different device pages, and checks values.
Its store is a separate process on the same host, so it does not cover physical
two-host networking.

Then run [verify_engine.py](../files/sglang_tpu/raiden_dram/verify_engine.py)
against a dedicated prefill/decode deployment with stores on the decode hosts.
It flushes every selected engine's local caches, requires positive storage hits
with zero local-tier hits, and compares greedy output before and after recall.
The [implementation README](../files/sglang_tpu/raiden_dram/README.md#verification)
provides commands and the full setup.
