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

Two pairs with other recipes started the same way (1M context with the same single slot; 140k context with eight slots) served every request.
We have not isolated what makes this recipe deterministic for the stall, but the code explains why it never resolves:

- The raiden **sender** (prefill) does not wait for the decode. It moves its room straight to `WaitingForInput` and bounds the handshake with
  `fail_decode_silence(room, init_time, bootstrap_timeout)`, which returns early once the decode has "asked" for the room. That deadline did not
  fire, so from the prefill's point of view the decode had engaged.
- The raiden **receiver** (decode) has exactly one deadline, `_check_waiting_timeout`, and its clock `init_time` is set in `send_metadata`, which
  runs only after the room leaves `Bootstrapping`. While the room is still bootstrapping, `poll()` returns the status and nothing counts time.
  The decode prealloc queue treats `Bootstrapping` as "keep waiting".
- `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` (300 s) exists and is enforced by the Mooncake and Mori backends and by the raiden sender, but not
  by the raiden receiver.

So whenever the two sides disagree about the handshake state, the request is held on both engines forever, and with one request slot the pair
is unusable.

**Expected behavior.** Either the handshake completes and the request is served, or one side gives up within the configured bootstrap timeout
(300 s): the decode fails the request with a clear error, the prefill releases its pages, the client gets HTTP 500, and the next request works.

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips / 8 TensorCores each), prefill on host a, decode on host b, TP 8, EP 8 |
| Model | `openai/gpt-oss-120b` (MXFP4), KV cache fp8_e4m3, page size 64 |
| Recipe that stalls (3 of 3 pairs) | both engines `--max-running-requests 1 --context-length 140288 --max-total-tokens 786432 --chunked-prefill-size 8192 --disable-radix-cache`, `SGLANG_RAIDEN_HOST_POOL_GB=128`, compile batch sizes `[1]` |
| Recipes that work | same flags with `--context-length 1056768 --max-total-tokens 1114112`, host pool 320 GiB (single slot); and the 140k recipe with `--max-running-requests 8` |

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

The combination observed: the prefill believes the decode asked; the decode believes it is still bootstrapping; no clock is running on either side.

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
4. Investigate the trigger: the only recipe differences between the stalling and the working single-slot pair are context length (140k vs 1M),
   `max_total_tokens` (786k vs 1.1M) and `SGLANG_RAIDEN_HOST_POOL_GB` (128 vs 320).

## Related

- #1113 (decode waits the full 120 s transfer deadline when the prefill fails a transfer) and #1078 (reshard handler exceptions dropped; room waits
  out the 300 s timeout): in both, a deadline eventually fires. Here none does.
- #1112 (reconnect and error-log storm after a failed transfer) is the opposite symptom: too much logging; here there is none.

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/F11/handshake_stall/` (py-spy dumps of both schedulers, pair 1),
`results/logs/runs/F11_gpt-oss_A_128k_mr1/` (both engine logs, pair 1), `results/F12/` (client logs and decode stack dump, pair 2),
`results/logs/runs/F12_gptoss_A_128k_mr1_try2/` (decode log, pair 3), `results/FINDINGS.md` (F-12 entries), `results/bug_reports/F-12.md`.
