# Test plan: 2-prefill + 2-decode disaggregation on TPU v7x

Written 2026-10-12 from the 1P/1D campaign (`TEST_CASES.md`, `FINDINGS.md`) and a capacity check of the three hosts named below. No code changed.

## 1. Host capacity (checked with gcloud and ssh over the internal IPs; the external IPs are unreachable from the test VM)

| Host name in the ssh config | What it is | Chips | State on 2026-10-12 |
|---|---|---|---|
| wyzhang-tpu7x-16-a (34.56.86.215) | worker 0 of slice `wyzhang-tpu7x-16` (tpu7x-16, topology 2x2x2), internal 10.128.15.217 | 8 | healthy; the current prefill host a |
| wyzhang-tpu7x-16-b (34.68.1.148) | worker 1 of the same slice, internal 10.128.0.90 | 8 | healthy; the current decode host b |
| wyzhang-tpu7x-8 (34.31.248.180) | separate slice `wyzhang-tpu7x-8` (2x2x1), internal 10.128.0.20 | 8 | `UNHEALTHY_TENSORFLOW`; root disk 0 bytes free; no venv; weights NFS not mounted at `/mnt/data` |

24 chips on two slices; each worker has 224 CPUs and 944 GB RAM. What that gives:

- **TP-8 engines (the 1P/1D recipes):** at most 3, so 2P/1D or 1P/2D, not 2P/2D. The third engine needs the tpu7x-8 host repaired first (disk, venv, NFS mount, TPU health).
- **TP-4 engines:** 6 (two per worker), enough for 2P/2D plus spares or 3P/3D. gpt-oss-120b weights fit in 4 chips, but the launcher's single-host topology code (`sglang_tpu/patches.py`) has no chip-subset support (`TPU_VISIBLE_CHIPS` / process bounds), and the compile buckets and 1M-context KV sizing would need re-deriving for 4 chips. Budget this as launcher work before any TP-4 run.
- **True 2P/2D at TP-8** needs a fourth 8-chip worker. The project has about 30 `tpu7x-8` slices owned by others (`gcloud compute tpus tpu-vm list --zone us-central1-c`); borrowing one for the test window is the quickest route.

**Why TP-8 is one engine per worker.** HBM is not the limit: each v7x chip has 96 GiB, and the TP-8 gpt-oss-120b engine uses about 31.7 GB per chip after load with the KV pool allocated (weights roughly 8 GB per chip at TP-8, the 786k-token pool under 2 GB per chip because only the 18 full-attention layers carry the full-length pool). The limit is chip exclusivity: a rank pins its chip through `/dev/vfio`, so two engines cannot share a chip. At TP-4, two engines per worker fit comfortably in HBM (about 16 GB of weights plus KV and compile workspace per chip); the cost is launcher work and losing direct comparability with the TP-8 1P/1D baselines.

Recommendation: get one more tpu7x-8 (or repair wyzhang-tpu7x-8) and keep TP-8 so every 1P/1D result transfers unchanged; use the TP-4 route only if no fourth host appears.

## 2. What is new over 1P/1D

Choice and fan-out: a router picks a P and a D per request, any P may transfer to any D, and one engine's death must not take the others down. Every 1P/1D finding is the regression floor; the new risks are routing correctness, cross-pair consistency, partial failure, and shared-host resource contention.

## 3. Phases

| Phase | Cases | Tools (existing unless noted) | Pass criteria |
|---|---|---|---|
| 0 Bring-up | 4 engines over 3 or 4 hosts; distinct server, raiden and bootstrap ports per engine; per-engine compile audit; Rust PD gateway or mini-LB in multi-worker mode | `pd.sh` extended to N engines, `compile_audit.sh` | all engines and the router healthy; 0 serving-time recompiles; router lists 4 workers |
| 1 Regression floor | TC-1 differential 1k/8k/32k/128k at c=1, TC-2 boundary sweep, TC-6 params, TC-9 chat, GSM8K-100, and one probe per 1P/1D finding (F-5 #1244, F-6 #1147, F-8 #1145, F-10 #1146, F-11 #1142, F-12 #1148, F-15 #1189, F-16 #1188 flag refused) | `golden_diff.py`, `params_matrix.py`, `chat_suite.py`, existing phase probes | same identical/numeric classification as 1P/1D; GSM8K at or above 0.98; every carried-over probe behaves as its fix PR validated |
| 2 Routing correctness | 200 requests with the chosen (P, D) recorded per request; all four pairings exercised; outputs independent of pairing | new `route_probe.py` (router logs or per-engine `prompt_tokens_total` deltas) | all 4 pairings seen; 0 output differences across pairings |
| 3 Rooms and n>1 | n=2/3/8, batches, consecutive calls, chat n=2 spread over both Ps | `_phase_1217.sh` battery against the router | every request returns; rooms unique per sample per P in the prefill logs |
| 4 Partial failure | kill P1 mid-prefill, kill D1 mid-decode, kill and restart the router, abort storm 40 in 10 s across pairs, client disconnect during handshake | `abort_test.py`, per-engine kill, state snapshots | survivors keep serving; dead rooms released within the bootstrap timeout (#1230 class, #1271); router stops routing to the dead engine (#1274 says it may not: file if so); no hang over 300 s |
| 5 Load and admission | bench_serving 131k/256 at c=8 and 16; small KV pool on D1 only; retraction on D; router-side shedding (`--max-queued-requests` is refused under PD by #1191) | `bench_serving`, `queue_limit_probe.py` at the router URL | 0 failures; no F-11 / #1140 / #1267 deadlock; shed requests get 503s, never hangs; all engines idle after |
| 6 Prefix cache and HiCache | MT-Bench 80x2 at c=1/4 with cache-aware routing on and off; host offload on both Ps; evict then reload | `mtbench_runner.py`, hicache metrics | turn-2 cached tokens higher with cache-aware routing; reload bit-exact (O-3: PD caches only the prompt) |
| 7 Speculative decoding | EAGLE3 on all four engines, TC-4 acceptance, outputs vs the bf16 golden | config D recipes | acceptance and NUM classification as in 1P/1D |
| 8 Mixed-config pair, deliberate | D2 with a different page size or KV layout | recipe variant | the handshake must fail loudly; #1077 predicts it will not, which is the test |
| 9 Soak | 2 h mixed traffic (random, MT-Bench, aborts) with the dashboard per engine | soak phase script | queue gauges drain; 0 bootstrap/transfer failures outside injected faults; no host RAM growth (#1286) |

## 4. Carry-in issues to watch at 2P/2D

#1273 (gateway drops `priority`), #1274 and #1276 (health 200 while broken), #1115 (decode death ends the stream silently), #1077 and #1076 (layout and SWA head-count mismatches only bite between unequal engines), #1113/#1114 (120 s holds after a peer fails), #1269 and #1286 (host RAM when two engines share a host).

## 5. Metrics

Give every panel of the PD dashboard an engine label so per-P and per-D queue depths, bootstrap and transfer failure rates and `kv_transfer_oldest_inflight_age_seconds` sit side by side, and add router-side request counts per worker to spot imbalance. The reading guide in the hardening report applies per engine.

## 6. Before the first run

1. Secure the fourth 8-chip worker, or repair wyzhang-tpu7x-8 (root disk full, TPU runtime unhealthy, venv and NFS missing).
2. Extend `pd.sh` and `launch_role.sh` to N engines with per-engine ports and run directories.
3. Decide the router (Rust PD gateway vs mini-LB multi-worker); verify it relays the decode's status (#1273, #1274 are open against it).
4. Land or carry the 1P/1D fixes the regression floor depends on (#1181, #1191, #1245, #1237, #1217, #1230).

## 7. Implementation on the existing tpu7x-16 pair (2026-10-12)

Decision: run 2P/2D on the two hosts we already have by using TP-4 engines, two per host, instead of waiting for a fourth worker. Each host has 4 chips with 2 TensorCores each (8 `/dev/vfio` devices, 96 GiB HBM per device); a TP-4 engine takes 2 chips. torch_tpu supports the sub-slice (`get_tpu_topology(4)` = `1,2,1,2`) and the repo launcher passes `TPU_VISIBLE_CHIPS` through for `tp_size > 1`, so no code change is needed in sglang-torchtpu; the launcher's P/D *group* mode still assumes one engine per host, so the engines are launched individually and the router is started by hand.

| Engine | Host | Chips (`TPU_VISIBLE_CHIPS`) | Server port | Bootstrap port | Raiden KV base port | MASTER_PORT |
|---|---|---|---|---|---|---|
| p1 | a 10.128.15.217 | 0,1 | 30000 | 8998 | 31000 | 29500 |
| p2 | a 10.128.15.217 | 2,3 | 30001 | 8999 | 31100 | 29501 |
| d1 | b 10.128.0.90 | 0,1 | 30000 | – | 31000 | 29500 |
| d2 | b 10.128.0.90 | 2,3 | 30001 | – | 31100 | 29501 |
| router | a | – | 8080 | – | – | – |

Raiden reserves `base + 4 * local_rank` control/data ports per rank, so the two engines on a host use bases 100 apart. Compile caches are keyed per engine (`TORCH_TPU_TIER2_NAME=pd2_<engine>_tp4_chips<..>`, separate inductor and tier-3 roots) so TP-4 graphs never collide with the TP-8 caches.

Recipes: `recipes/2p2d/gpt-oss_B_128k_{p1,p2,d1,d2}.json`, derived from the 1P/1D Config-B recipe (`gpt-oss_C_128k_ps_q.json` minus HiCache, priority scheduling and the queue limit, which #1191 refuses under PD) with `tp_size 4`, `ep_size 4`, `max_total_tokens 786432`, `mem_fraction_static 0.45`, same compile buckets. Scripts: `scripts/launch_engine.sh` (one engine, chip subset), `scripts/pd2.sh start|stop|status` (four engines + `sglang_router --pd-disaggregation --mini-lb` with two `--prefill url bootstrap_port` and two `--decode url`), `scripts/route_probe.py` (phase 2 routing checks), `scripts/_phase_2p2d_p0.sh` (phases 0-2 plus a decode-phase abort). Router policy: default (random) for prefill and decode selection.

Known caveats of this layout: TP-4 results are not byte-comparable with the TP-8 goldens (different reduction order; expect the F-3 class of numerical difference, so the TC-1 oracle here is needles plus GSM8K, not token identity), and two engines share one host's CPUs, RAM and host-RAM pools (#1286, #1269 are in scope).

## 8. Run log

| When (UTC) | Run | What | Result |
|---|---|---|---|
| 2026-10-12 08:36 | `2p2d_bringup` | first launch of p1,p2,d1,d2 on main 2e14706a62, `mem_fraction_static 0.45` (the TP-8 value) | all four engines died at KV-pool sizing: at TP-4 the gpt-oss-120b weights take 56.9 GB of the 94.7 GB available per device (`Load weight end ... mem usage=56.86 GB`; the MXFP4 checkpoint is held as bf16 on TPU, so 1/4 of ~230 GB), leaving 37.9 GB; the launcher refused: "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.45. Raise above 0.601". TP-4 needs `mem_fraction_static` >= 0.61; set to 0.75 (71 GB budget, ~14 GB for KV, enough for 786k tokens) |
| 2026-10-12 08:42 | `2p2d_bringup2` | relaunch with `mem_fraction_static 0.75` | p1 and d1 (the first engine on each host) came up healthy in 225 s with the 786k-token KV pool (K 3.38 GB + V 3.38 GB per device). **p2 and d2 died at scheduler init (SIGABRT):** `[TorchTPU] Failed to acquire a TPU device node '/dev/vfio/5' because it is already opened by ...`, `vfio_device_access.cc: open(/dev/vfio/0): Device or resource busy`. `TPU_VISIBLE_CHIPS` is passed through by the launcher but torch_tpu's PjRt device acquisition ignores it: both engines on a host try to open the same device nodes (d1 holds /dev/vfio/0,3,4,5 on b; p1 holds 4,5,6,7 on a). Finding 2P2D-F1, see section 9. The router was started over p1+d1 as a TP-4 1P/1D pair for a smoke test |
| 2026-10-12 09:18 | `tp4_1p1d` | router on b:8081 over p1 (a, chips A,B) + d1 (b, chips A,B): TP-4 1P/1D baseline | router up in 6 s; smoke 200 with the same completion as TP-8; routing probe: 12 requests at c=4 all 200 (0.3-1.0 s), 8 repeats of one prompt give 1 distinct output, decode idle after (R2/R3 were a probe bug: counters carry no `tp_rank` label; fixed); TC-7 two decode-phase aborts PASS, follow-ups served with 8,128 cached tokens; 0 serving-time recompiles. TC-1 8k vs the **TP-8** golden: both prompts complete 1024 tokens, but tokens diverge after 15-18 (agreement 2-4%, first-token logprob delta 0.03-0.15): the expected TP-change divergence (F-3 class), so a TP-4 monolithic golden is being produced as the proper control |
| 2026-10-12 09:33 | `2p2d_visdev` | second decode on b with `TPU_VISIBLE_DEVICES=4,5,6,7` + `ALLOW_MULTIPLE_LIBTPU_LOAD=1` (PCI-order indices of chips C,D) | same collision: ranks still open /dev/vfio/5,3,4 held by the first engine. 2P2D-F1 confirmed blocking |
| 2026-10-12 09:28 | `2p2d_vis` | second decode on b with `TPU_VISIBLE_DEVICE_PATHS=/dev/vfio/1,2,6,7` + `ALLOW_MULTIPLE_LIBTPU_LOAD=1` | lockfile check passed but libtpu found no device: `TPU initialization failed: No jellyfish device found` (path format not accepted). Device mapping established from sysfs: each TPU chip is one PCI device with two functions (cores), vfio groups 00:03.{0,1} -> 4,3; 00:04.{0,1} -> 5,0; c0:00.{0,1} -> 1,2; c0:02.{0,1} -> 6,7; torch_tpu assigns rank i to the i-th core in PCI order, so an engine with ranks 0-3 always takes chips A,B (groups 0,3,4,5) |

## 9. Findings

| Id | Finding | Evidence | Status |
|---|---|---|---|
| 2P2D-F1 | **Two torch_tpu engines cannot share a host.** torch_tpu assigns rank i the i-th TPU core in PCI order and opens that `/dev/vfio` node directly; none of the libtpu visibility variables change it. Tried on the second decode next to a running first one: `TPU_VISIBLE_CHIPS=2,3` (forwarded by `benchmark/tpu/launch_server.py`): "[TorchTPU] Failed to acquire a TPU device node /dev/vfio/5 because it is already opened by ..."; `TPU_VISIBLE_DEVICE_PATHS=/dev/vfio/1,2,6,7` + `ALLOW_MULTIPLE_LIBTPU_LOAD=1`: lock check passes, then "TPU initialization failed: No jellyfish device found"; `TPU_VISIBLE_DEVICES=4,5,6,7` + `ALLOW_MULTIPLE_LIBTPU_LOAD=1`: same node collision as the first variant. Device layout (sysfs): chip = PCI device with two core functions; vfio groups 00:03.{0,1} -> 4,3; 00:04.{0,1} -> 5,0; c0:00.{0,1} -> 1,2; c0:02.{0,1} -> 6,7, so every 4-rank engine takes chips A,B (groups 0,3,4,5) and chips C,D stay idle. Consequence: on this hardware one engine per host at any TP, so 2P/2D needs four hosts; the launcher accepting `TPU_VISIBLE_CHIPS` is misleading | `results/runs/2p2d_bringup2/{d2,p2}/`, `results/runs/2p2d_vis/d2/`, `results/runs/2p2d_visdev/d2/` (server.log + tpu_logs) | open, blocking; report filed as `bug_reports/2P2D-F1_github_issue.md` for sglang-torchtpu (launcher) and torch_tpu (device selection) |
| 2P2D-F2 | TP-4 memory: gpt-oss-120b weights occupy 56.9 GB of 94.7 GB per device at TP-4 (MXFP4 held as bf16), so `mem_fraction_static` must be at least 0.61; the TP-8 value 0.45 makes the launcher refuse to start | `results/runs/2p2d_bringup/*/server.log` | resolved in the recipes (0.75) |
