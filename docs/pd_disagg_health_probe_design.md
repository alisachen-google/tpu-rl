# P/D health probe design: `/health` and `/health_generate`

Design conclusion for sglang-torchtpu issue #1276, written 2026-10-13 after the 2P/2D
campaign. Interactive version with diagrams and a fault simulator:
https://claude.ai/artifact/4SLGSg2rHqw9xceLFwFBsR

## 1. Problem

A P/D decode keeps answering 200 on `/health` and `/health_generate` while every KV transfer
into it fails. Orchestrators keep the pod in service and routers keep pairing it, so one
broken decode silently takes a share of all traffic.

Evidence on our cluster (four hosts, 2P/2D TP-8, gpt-oss-120b):

| Run | Fault on d2 | Requests paired with d2 | d2 health during the fault |
|---|---|---|---|
| `2p2d_tp8_i1276b` | receive completion never signalled (issue's own repro hook), 60 s waiting timeout | 5/5 failed with 500 after 60 to 62 s | 53/53 polls 200, 0 x 503 |
| phase 8, finding 2P2D-F8 | `--page-size 128` against page-64 prefills | first request kills every rank | 200 for 211 s until the first request |
| phase 8b, finding 2P2D-F9 | lane-major KV layout against token-major prefills | 9/9 failed with 500 in 0.1 s | 200 throughout |

Three different causes, one picture. The gap is independent of why transfers fail.

## 2. Why today's probe lies

`/health` and `/health_generate` are one handler (`python/sglang/srt/entrypoints/http_server.py:704`).
With `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` on by default, both submit a one-token
generation request. On a P/D decode:

| Defect | Where | What happens |
|---|---|---|
| A | `http_server.py:745`, `decode.py:941`, `fake/conn.py:142` | The probe gets the fake bootstrap host `2.2.2.2`, so the decode builds a `FakeKVReceiver` that reports Success after two flags and zero bytes. The Raiden receiver, the prefill handshake and the NIC are never touched. |
| B | `scheduler.py:2314`, `scheduler.py:5366` | The scheduler runs the probe only when "fully idle", and that test excludes the prealloc and transfer queues. A decode with every request parked waiting for KV is "idle", so the probe is admitted and generates. The watchdog is disarmed in the same state. |
| C | `tokenizer_manager.py:2366`, `http_server.py:760` | The handler returns 200 as soon as any message arrives from the detokenizer, whichever request it belongs to. 503 only if nothing arrives for 20 s. |
| D | both exit paths, PR #826 open | The handler drops the probe's client state without aborting the scheduler side; a timed-out probe leaks a prealloc slot. This alone took down a 20-engine tier. |

Kubernetes templates in the repo disagree: `docker/k8s-sglang-service.yaml` separates the two
paths; the DeepSeek P/D template points both probes at `/health` with liveness failure
thresholds of 3000 and 30000 (liveness switched off); the LWS template uses a TCP check.

## 3. Design: three questions, three answers

The current probe fails because one generate request tries to answer all three.

### 3.1 Liveness: can the process make progress at all?

Endpoint: `GET /health`. Source: heartbeats and a fatal-error flag. No request is created.

- Every process on the request path writes a heartbeat into its own slot in a shared-memory
  region on each loop iteration, idle iterations included: each scheduler rank, the
  detokenizer, the tokenizer workers. The heartbeat is the monotonic clock reading at the top
  of that iteration. When a probe arrives, the HTTP process reads every slot and computes its
  age against the clock now. It never sends anything into the scheduler, so a hung process is
  detected by a stale slot rather than by a request that is never answered.
- 503 when any heartbeat is older than the threshold (3 x the longest legitimate step). The
  scheduler publishes an "in compile" or "in long chunk" flag that pauses the clock, with its
  own cap so a wedged recompile is still caught.
- Fatal errors exit the process (from TensorRT-LLM): when a rank has died or the scheduler
  has recorded an unrecoverable exception, the server sends itself SIGINT once and lets the
  orchestrator restart it. A zombie that accepts connections must not wait to be noticed.
- Never generates, never touches the KV pool, never fails under load.

### 3.2 Readiness: can this decode take part in real P/D requests?

Endpoint: `GET /health_generate`, alias `GET /readiness`. Same path as today, new contents,
no generation. Source: real outcomes, an idle-only canary, compatibility and start-up gates.

This is #1276's fix 1 with three additions.

The issue's rule, adopted as written:

- 503 once N transfers in a row have failed over at least T seconds with zero successes.
  A success clears it.
- A decode taken out of service gets no requests, so after T seconds with no new failure it
  passes on probation, failing again if a new request has no output within P seconds.
- Validated values N=20, T=300 s, P=60 s: handled the repro without tripping on a 25-failure
  burst or cancel-heavy traffic.

Additions:

1. Peer attribution. Failures count toward 503 only when they span two or more peers or are
   locally attributable (a pull failing on this host; one rank with no responses while its
   sibling ranks succeed). Failures confined to one peer put the decode in a Degraded state:
   still 200, with the peer named in the reason body. Without this, one dead prefill makes
   every decode in the group open together and the pool is empty. Phase 4c showed that
   pairing pattern.
2. Backoff on probation. Each failed probation doubles the next wait, capped at 30 minutes,
   so a decode that keeps failing re-admission stops oscillating every T seconds.
3. Stricter success. A success is the first token of a request that completed a real KV
   transfer. A fake-bootstrap probe or a fully cached request cannot clear a transfer-failure
   state.

Idle case (from Dynamo): activity is the evidence, so no probe runs while traffic flows.
After 15 s of silence the engine runs a loopback transfer over its real NIC path (a dead
device on an idle engine still goes 503), and the router sends a real tiny paired request to
idle pairs, since only the router can reach a prefill and a decode together.

Compatibility: page size, dtype, KV layout and a protocol version validated at bootstrap
(PR #1166 adds layout) and at start-up against the registered peers. Incompatible with every
peer is a start-up failure with a reason, never a ready engine. Mixed fleets during a rolling
upgrade need the per-peer matrix in the router, not a readiness bit.

Start-up gates: ready only after weights, warm-up, compile, bootstrap server bound, transfer
buffers registered.

### 3.3 Recovery

State machine: Starting -> Ready <-> Degraded; Ready or Degraded -> Unhealthy; Unhealthy ->
Probation -> Ready or back to Unhealthy with the wait doubled. A panic floor keeps the last
ready decode in a group on probation rather than removing it.

### 3.4 Parameters

| Parameter | Value | Note |
|---|---|---|
| N | 20 | validated in #1276; at low concurrency 20 sequential 60 s failures take 20 min, so N is the knob to lower after the burst test passes on our harness |
| T | 300 s | both N and T must be met |
| P | 60 s | probation request must produce output within P |
| T_recover | T, x2 per failed probation, cap 30 min | the issue's rule plus backoff |
| S | 1 | one real transfer success clears |
| heartbeat max | 3 x step | paused by the in-compile flag, with a cap |
| canary | after 15 s silence, k=3 failures to open | engine loopback and router paired request, idle only |
| fatal | SIGINT once | guarded against repeated probes |

## 4. Consumers and what each answer does

Two paths exist because Kubernetes does two different things with them. A failing liveness
probe restarts the container; a failing readiness probe removes the pod from the Service and
waits. A decode whose peer is dead or whose transfers all fail is alive and must not be
restarted, but is not ready.

| Consumer | Asks | Action |
|---|---|---|
| kubelet startup probe | `/health`, up to 1 h | suspends the other two until the first 200; covers cold compile without raising any threshold |
| kubelet liveness | `/health` | 503 x3 -> restart; exit -> restart at once; never a peer's fault |
| kubelet readiness | `/health_generate` | 503 x3 -> removed from the Service; 200 -> back; probation 200 -> traffic resumes and proves recovery |
| router active checker | `/health_generate` | 503 -> ineligible for pairing; reason "peer p1 failing" -> de-prioritise p1, not d2; owns the idle paired canary |
| router circuit breaker | real responses | 5 failures in 60 s -> open; independent of probes; the only protection today |
| alerts and operators | `/metrics` + JSON reason body | stall gauges (#1328), per-peer and per-rank counts name the broken engine |

Liveness answers can only cause a restart. Readiness answers can only move traffic. Keeping
the existing paths means no manifest, router or CI script changes. The DeepSeek P/D template
should move its readiness probe to `/health_generate` in the same change.

## 5. What exists elsewhere

| Stack | Liveness | Readiness | Borrowed |
|---|---|---|---|
| upstream sglang e0c957e | identical handler | identical | the fix applies unchanged |
| vLLM 656d0d4 | `/health` reads an engine-dead flag; no request | same flag | liveness needs no request |
| tpu-inference 88ddf9a | inherits vLLM | inherits; the Raiden connector withholds failed receives from vLLM so a bad completion cannot crash the engine core | failed receives must be surfaced through a path the scheduler expects, never injected as a completion |
| TensorRT-LLM | fatal-error flag plus batcher thread; sends itself SIGINT on a fatal error | real one-token chat request; disagg server reports topology only | self-exit; auxiliary processes in liveness |
| NVIDIA Dynamo | `/live`: up and not shutting down | per-endpoint task: real traffic marks Ready; canary payload after 10 s silence with 3 s timeout; endpoints start NotReady | activity as evidence; idle-only canary through the real path |

Not borrowed: Dynamo's single-failure flip to NotReady (no attribution, no backoff) and
TensorRT-LLM's real-prompt generate probe (occupies a slot, fails under load).

## 6. Relation to open work

| Item | State | Covers |
|---|---|---|
| #1301 | merged | one root cause (#1285, ZMQ monitor wedge); no detection change |
| #1328 | merged | stall gauges publish every 30 s; the watchdog half of #1303 is not opened |
| #826 | open | abort the probe before dropping its state (defect D); any redesign keeps this ordering |
| #843 | open | transfer deadline; main already has a 60 s chunk deadline |
| #1166 | open | KV layout check at bootstrap (F9, #1077); per request, not a start-up gate; needs a version field |
| #1139 | draft | fault-injection presets; the regression test should use these instead of our one-off hook |
| #1276 fix 1 and 2 | no PR | outcome-based readiness; probe waits for its own rid |
| heartbeat liveness, self-exit, loopback, start-up compatibility gate, router canary | no PR | new work; heartbeat and the start-up gate are the smallest pieces with the largest effect |

## 7. Validation plan (ten minutes each on the four-host harness)

1. Reproduction hook from `2p2d_tp8_i1276b`: 53 x 200 becomes 503 once N=20 failures span
   300 s, then 200 on probation after restore. Then the 25-failure burst and the cancel storm,
   which must not open it.
2. One prefill killed: both decodes stay 200 in Degraded with the peer named.
3. Phase 8b layout mismatch: d2 fails the start-up gate and never reports ready.
4. Idle decode with the transfer device disabled: readiness 503 from loopback within k x 15 s.
5. bench_serving at full concurrency for 30 minutes: liveness never leaves 200.
6. Probation flapping: fault toggled faster than T; backoff grows, state does not oscillate.
7. Cold compile on config D: liveness 200 through the window with the in-compile flag set.
8. Rank crashed: process exits within a second of the recorded error.

Sources: `sglang_inference/results/TEST_PLAN_2P2D.md` (findings F4, F7, F8, F9, F11),
`sglang_inference/results/2p2d_tp8/i1276/`, `sglang_inference/scripts/hooks/1276/`.
