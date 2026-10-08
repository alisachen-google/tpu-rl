# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [TPU/PD/Raiden] P/D handshake never completes on some pairs: decode stays in `Bootstrapping` with no deadline, prefill holds the request in its inflight queue, nothing times out (3 of 3 fresh pairs with one recipe)

**Labels:** bug, tpu, pd

---

## Summary

On three freshly started gpt-oss-120b pairs with the same recipe (`--max-running-requests 1`, 140k context), the very first request never
produced a token, and neither did any later request. The prefill engine ran the prompt (one 8,192-token chunk), put the request in its inflight
queue and then idled. The decode engine never ran a batch: its scheduler polled the request's KV receiver in the handshake wait, and the receiver
stayed in `KVPoll.Bootstrapping` for 30 minutes. Both engines answered `/health`, all queues looked empty in the metrics, and no warning or error
was logged until the client disconnected, at which point both sides logged the abort at once.

Pairs with neighbouring recipes (the same 140k recipe with two or eight slots; the 1M-context recipe with one slot) served every request; see
"Trigger conditions" below for the full table. A single request slot on both engines is necessary but not sufficient. The code explains why the
stall never resolves once it happens:

- The raiden **sender** (prefill) does not wait for the decode. It moves its room straight to `WaitingForInput` and bounds the handshake with
  `fail_decode_silence(room, init_time, bootstrap_timeout)`, which returns early once the decode has "asked" for the room. Note that our recipe
  sets `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` and `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` to 7200 s (to survive 1M-token prefills), so this
  deadline could not have fired within our 30-minute observation even if the decode never asked; with the 300 s default it would have.
- The raiden **receiver** (decode) has exactly one deadline, `_check_waiting_timeout`, and its clock `init_time` is set in `send_metadata`, which
  runs only after the room leaves `Bootstrapping`. While the room is still bootstrapping, `poll()` returns the status and nothing counts time.
  The decode prealloc queue treats `Bootstrapping` as "keep waiting".
- `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` (300 s) exists and is enforced by the Mooncake and Mori backends and by the raiden sender, but not
  by the raiden receiver.

So on the decode side nothing bounds a room that never leaves `Bootstrapping`, whatever the timeout settings; on the prefill side the bound is
the bootstrap timeout, 7200 s in our recipe. With one request slot the pair is unusable for the whole period.

**Expected behavior.** Either the handshake completes and the request is served, or both sides give up within the configured bootstrap timeout:
the decode fails the request with a clear error, the prefill releases its pages, the client gets HTTP 500, and the next request works. Today only
the prefill side times out (verified with 300 s deadlines); the decode holds the request and the client connection until the client disconnects.

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips / 8 TensorCores each), prefill on host a, decode on host b, TP 8, EP 8 |
| Model | `openai/gpt-oss-120b` (MXFP4), KV cache fp8_e4m3, page size 64 |
| Recipe that stalls (3 of 3 pairs) | both engines `--max-running-requests 1 --context-length 140288 --max-total-tokens 786432 --chunked-prefill-size 8192 --disable-radix-cache`, `SGLANG_RAIDEN_HOST_POOL_GB=128`, compile batch sizes `[1]`, `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=7200`, `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=7200` |
| Recipes that work | the same 140k recipe with `--max-running-requests 2` (tested right after the third stall: 8k, 32k and 131k served normally, 6 of 6); the 140k recipe with `--max-running-requests 8`; and the single-slot recipe with `--context-length 1056768 --max-total-tokens 1114112`, host pool 320 GiB |

## With default-length deadlines (300 s) the hang is still user-visible

We re-ran the stalling single-slot recipe with `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=300` and `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=300`
(our other runs used 7200 s). Result for an 8k request:

| t | Prefill | Decode | Client |
|---|---|---|---|
| 0 s | prefills the prompt in one chunk, request enters the inflight queue | request in the prealloc queue, receiver in `Bootstrapping` | waiting |
| 300 s | `Prefill transfer failed … Request <room> timed out after 300.0s waiting for the decode to ask for its KV` on all ranks; slot freed | still `Bootstrapping`, no log line | still waiting (the mini-LB returns the decode's response) |
| 720 s | – | `Decode handshake failed … Aborted by AbortReq` (client disconnect) | `Read timed out` (client's own timeout) |

So the prefill deadline works and tells us the cause: **the decode never asked for the KV**, i.e. its bootstrap for the room never completed.
The decode has no deadline of its own, so the request is held until the client disconnects, and the client sees a hang of its own timeout
length regardless of the server-side setting. Both requests in the run behaved identically.

## Trigger conditions (observed)

The stall occurred on four fresh single-slot pairs (140k shape radix off, 1M shape radix on) on their first request, and on one two-slot 1M pair after a burst of very long requests; eight-slot pairs never stalled. All pairs: gpt-oss-120b, raiden, radix cache off on
the prefill, both engines freshly started, first request sent within a minute of the load balancer becoming healthy.

| Pair | `--max-running-requests` (both engines) | `--context-length` / `--max-total-tokens` | `SGLANG_RAIDEN_HOST_POOL_GB` | Compile batch sizes | Requests | Outcome |
|---|---|---|---|---|---|---|
| F11_gpt-oss_A_128k_mr1 | 1 | 140,288 / 786,432 | 128 | [1] | 8k, 8k | **stall on the first request; the next stalls too** |
| F12_…_mr1_try1 (fresh pair) | 1 | 140,288 / 786,432 | 128 | [1] | 8k, 8k, 32k | **stall on every request** |
| F12_…_mr1_try2 (fresh pair) | 1 | 140,288 / 786,432 | 128 | [1] | 8k | **stall on the first request** |
| B_gpt-oss_1m (radix cache **on**) | 1 | 1,056,768 / 1,114,112 | 320 | [1] | 32k | **stall on the first request** (prefill holds it in the inflight queue; decode never runs a batch) |
| B_gpt-oss_1m_mr2 (radix on, **2 slots**) | 2 | 1,056,768 / 1,114,112 | 320 | [1, 2] | 14 long prompts 32k…1M served, then ~705k + ~1M boundary prompts, then 256k reuse, then GSM8K | **stall on GSM8K's first requests** (decode in `_update_handshake_waiters` 30 min; prefill holds 2 in the inflight queue; paired abort lines when the clients died) |
| F11_gpt-oss_A_128k_mr2 | 2 | 140,288 / 786,432 | 128 | [1, 2] | 8k, 32k, 131k | served, 6 of 6 |
| Config A campaign (A_gptoss_128k_run1/run2) | 8 | 140,288 / 786,432 | 128 | [1, 2, 4, 8] | ~200 requests over two days | served |
| A_gptoss_1m_run1/run2 | 1 | 1,056,768 / 1,114,112 | 320 | [1] | 1k, 8k (longer prompts hit F-11) | served |
| F11_gpt-oss_A_1m_mr4 | 4 | 1,056,768 / 1,114,112 | 320 | [1, 2, 4] | 8k, 32k, 131k | served |

What this says about the trigger:

- **A single request slot on both engines is necessary.** Changing only `--max-running-requests` from 1 to 2 on the stalling recipe removes the
  stall. A single slot also means a decode SWA pool of 512 tokens (768 rounded down to 8 pages), a request table of one row on each engine, and
  compile batch size 1 only.
- **A single slot is not required.** Four single-slot pairs stalled on their first request, but a two-slot 1M pair also stalled after serving
  14 long prompts (up to 1M tokens) and two near-context-length prompts. The common factor across all five stalls is a decode prealloc queue
  that never leaves `Bootstrapping` for a room the prefill considers live; what triggers that disagreement (fresh start with one slot, or a
  burst of very long requests with two) is still open. Eight-slot pairs never stalled in ~400 requests.
- **It does not depend on the prompt.** 8k (one chunk) and 32k prompts stall alike; the request is prefilled completely before the stall.
- **It does not depend on warm-up state.** Three pairs started from scratch stalled on their very first request.
- **It is not timing-sensitive in the usual sense.** The handshake never completes even after 30 minutes, so this is not a slow path but a
  missed or misrouted handshake step.

We did not isolate which of the three remaining differences matters. The next experiments, in order of cost: the stalling recipe with
`SGLANG_RAIDEN_HOST_POOL_GB=320`; then with `--max-total-tokens 1114112`; then with `--context-length 1056768`. Each is one pair start and one
request. With the handshake logging proposed in fix 3, one run would be enough.

## Code trace

All links at commit 09d1244c26.

**Decode side.** The prealloc queue polls every receiver; a room in `Bootstrapping` is simply skipped and polled again next step:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/decode.py#L1126-L1175

The raiden receiver's `poll` checks failure, success and `_check_waiting_timeout`, then returns the room status:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/raiden/receiver.py#L1146-L1170

`_check_waiting_timeout` does nothing while `init_time` is `None`:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/common/conn.py#L2003-L2008

and `init_time` is set only in `send_metadata`, i.e. after preallocation, after the room has left `Bootstrapping`:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/raiden/receiver.py#L1125-L1144

The bootstrap timeout env var is defined but the raiden receiver never consults it (Mooncake and Mori do, via `_check_bootstrap_timeout`):
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/environ.py#L698-L702
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/common/conn.py#L1696-L1710

**Prefill side.** The sender is consumer-pull: it advances to `WaitingForInput` without a handshake and bounds the wait with `fail_decode_silence`:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/raiden/sender.py#L997-L1007
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/raiden/sender.py#L1085-L1097

`fail_decode_silence` returns `False` as soon as the decode has asked for the room, after which no prefill-side deadline applies:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/raiden/sender.py#L286-L310

Observed combination: the decode never left `Bootstrapping` (no clock runs there); the prefill's clock is the bootstrap timeout, which our
recipe sets to 7200 s. Whether the decode had "asked" (which would also stop the prefill's clock) is not visible in the logs; see fix 3.

## How we detected it

The stall appeared while isolating a different bug (F-11) and was then reproduced deliberately.

- Pair 1 (05:16 UTC): first 8k request, no token in 30 minutes. Prefill metrics: `num_running_reqs 1`, `num_prefill_inflight_queue_reqs 1`,
  `swa_token_usage 0.5`; `py-spy dump` of the prefill TP0 scheduler: idle in `recv_requests`. Decode metrics: all queues 0, no `Decode batch`
  line; `py-spy dump` of the decode TP0 scheduler: `pop_preallocated → _update_handshake_waiters → poll_and_all_reduce`. When the client
  process exited, both logs printed within the same second: prefill `Prefill transfer failed for request ... with exception Aborted by AbortReq.`,
  decode `Decode handshake failed for request ... with exception Aborted by AbortReq.` The second request then stalled identically.
- Pairs 2 and 3 (06:36 and 06:59 UTC, fresh starts of both engines): first 8k request stalled within 45 s each time, same stacks. On pair 2 the
  decode's HTTP layer still held the first request's id ten minutes later (`Duplicate request ID detected` when the id was reused).
- Working controls: a pair with the 1M-context single-slot recipe served 8k, 32k and 131k prompts (10 minutes earlier, same hosts); the
  140k recipe with eight slots served the whole campaign.

## How to reproduce (as observed; trigger not isolated)

1. Start the pair with the "recipe that stalls" flags above (both engines freshly started).
2. Send one 8,192-token prompt with `max_new_tokens 64` through the load balancer.
3. After 60 s: decode `curl -s http://<decode>:30000/metrics | grep -E 'num_running|prealloc|transfer'`; prefill
   `curl -s http://<prefill>:30000/metrics | grep -E 'num_running|inflight'`.
4. `py-spy dump --pid <decode TP0 scheduler> --nonblocking`.

**Observed.** No answer. Prefill: one running request in the inflight queue. Decode: queues 0, scheduler in `_update_handshake_waiters`.
Nothing is logged until the client disconnects.
**Expected.** An answer in seconds, or a failed request within 300 s with an error on both sides.

## Fix proposal

1. **Give the raiden receiver a bootstrap deadline.** In `RaidenKVReceiver.poll`, while the room status is `Bootstrapping`, apply
   `kv_mgr.bootstrap_timeout` from the receiver's creation time, as `_check_bootstrap_timeout` does for the other backends, and conclude the room
   as failed with a message naming the room and the elapsed time. This alone turns a 30-minute silent hold into a 300-second logged failure.
2. **Keep a prefill-side deadline until the transfer actually starts.** `fail_decode_silence` stops counting once the decode has asked; add a
   second bound from that moment until `maybe_start_pull` or the first chunk read, so a decode that asked and then stalled cannot pin the
   prefill's pages forever.
3. **Log the handshake.** At INFO level, log on both sides when a room is registered, when the decode's request is received, and when the
   reply is sent. Today the only lines are the abort messages, which made the trigger impossible to isolate from the logs. With that logging we
   can re-run the three recipes and report what differs.
4. Investigate the trigger. Changing only `--max-running-requests` from 1 to 2 on the stalling recipe makes the stall disappear, so the single
   request slot is necessary; it is not sufficient, because the single-slot 1M recipe (larger `max_total_tokens`, host pool 320 GiB) works.
   A single slot on both engines also means a decode SWA pool of 512 tokens (768 rounded down to 8 pages) and a request table of one row.

## Related

- #1113 (decode waits the full 120 s transfer deadline when the prefill fails a transfer) and #1078 (reshard handler exceptions dropped; room waits
  out the 300 s timeout): in both, a deadline eventually fires. Here none does.
- #1112 (reconnect and error-log storm after a failed transfer) is the opposite symptom: too much logging; here there is none.

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/F11/handshake_stall/` (py-spy dumps of both schedulers, pair 1),
`results/logs/runs/F11_gpt-oss_A_128k_mr1/` (both engine logs, pair 1), `results/F12/` (client logs and decode stack dump, pair 2),
`results/logs/runs/F12_gptoss_A_128k_mr1_try2/` (decode log, pair 3), `results/FINDINGS.md` (F-12 entries), `results/bug_reports/F-12.md`.
