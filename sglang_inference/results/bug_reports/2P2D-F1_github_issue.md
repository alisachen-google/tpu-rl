# [TPU/Launcher] Two engines cannot share a host: torch_tpu ignores TPU_VISIBLE_CHIPS and opens device nodes by PCI order, so a second TP-4 engine aborts on "device node already opened"

## Summary
On a tpu7x host (4 chips, 2 cores each, 8 `/dev/vfio` nodes), a second sglang engine started with `tp_size 4` and `TPU_VISIBLE_CHIPS=2,3` (or `TPU_VISIBLE_DEVICES=4,5,6,7`) dies at scheduler initialization: every rank tries to open the same `/dev/vfio` nodes the first engine holds. `benchmark/tpu/launch_server.py` forwards `TPU_VISIBLE_CHIPS`, `TPU_CHIPS_PER_HOST_BOUNDS` and `TPU_HOST_BOUNDS` for `tp_size > 1`, which suggests chip subsets are supported; they are not honoured by torch_tpu's device acquisition. Result: one engine per host at any TP, so a 2-prefill/2-decode layout needs four hosts.

## Environment
- sglang-torchtpu main 2e14706a62, torch_tpu from the project venv, libtpu as shipped
- host: tpu7x-16 slice worker (`t1v-n-eb299eb1-w-1`), 8 vfio nodes; first engine: gpt-oss-120b, `tp_size 4`, `TPU_VISIBLE_CHIPS=0,1`, healthy, holding `/dev/vfio/0,3,4,5`

## Reproduction
1. Start engine 1: `python -m benchmark.tpu.launch_server --config @engine1.json` with `runtime_envs.TPU_VISIBLE_CHIPS=0,1`, `server_args.tp_size=4`, `port 30000`. It comes up healthy.
2. Start engine 2 on the same host: same config with `TPU_VISIBLE_CHIPS=2,3`, `port 30001`, distinct `MASTER_PORT`, `SGLANG_RAIDEN_KV_PORT`, compile-cache roots.
3. Engine 2 ranks abort (SIGABRT):
```
E0000 pjrt_state.cc:303] PjrtBackend::GetClient failed to initialize: FAILED_PRECONDITION: [TorchTPU] Failed to acquire a TPU device node '/dev/vfio/5' because it is already opened by another process ...
E1012 vfio_device_access.cc:241] FAILED_PRECONDITION: open(/dev/vfio/0): 16: Device or resource busy; Couldn't open iommu group /dev/vfio/0
F0000 device_rt.cc:295] Check failed: client != nullptr PjRtClient is null after initialization.
RuntimeError: Rank 0 scheduler died during initialization (exit code: -6)
```
4. Variants, same result or worse: `TPU_VISIBLE_DEVICES=4,5,6,7` + `ALLOW_MULTIPLE_LIBTPU_LOAD=1` collides identically; `TPU_VISIBLE_DEVICE_PATHS=/dev/vfio/1,/dev/vfio/2,/dev/vfio/6,/dev/vfio/7` + `ALLOW_MULTIPLE_LIBTPU_LOAD=1` passes the lock check but libtpu reports `TPU initialization failed: No jellyfish device found`.

## Device layout (from /sys/bus/pci)
Each chip is one PCI device (0x1ae0:0x0076) with two functions = two cores, each its own vfio group:
`00:03.0 -> 4`, `00:03.1 -> 3`, `00:04.0 -> 5`, `00:04.1 -> 0`, `c0:00.0 -> 1`, `c0:00.1 -> 2`, `c0:02.0 -> 6`, `c0:02.1 -> 7`.
A 4-rank engine always receives the first four cores in PCI order (groups 0,3,4,5 = chips A,B); chips C,D stay idle.

## Expected
Either `TPU_VISIBLE_CHIPS` (or a torch_tpu equivalent) selects which cores a process group opens, so two TP-4 engines can share a host, or the launcher refuses `TPU_VISIBLE_CHIPS` with `tp_size > 1` with a clear message instead of forwarding it.

## Impact
Blocks multi-engine-per-host P/D testing (2P/2D on a 2-host slice) and any colocation of engines on one host; also wastes half the chips when a TP-4 engine is the right size for a model.

## Notes
`torch_tpu._internal.utils.hardware.get_tpu_topology(4)` returns `1,2,1,2` on this host, so the sub-slice topology itself is supported; the gap is device selection. Memory is not the constraint: at TP-4 gpt-oss-120b weights take 56.9 GB of 94.7 GB per device (MXFP4 held as bf16), so `mem_fraction_static` must be >= 0.61 (0.75 works).
