# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [PD] `n>1` (parallel sampling) never returns through the P/D pair: every sub-request of a prompt reuses the same bootstrap room, and the decode engine holds the leftover request until the client disconnects

**Labels:** bug, pd

---

## Summary

A `/generate` request with `"n": 2` through a 1-prefill + 1-decode pair gets no response. Every other sampling-parameter case in the same
matrix answers in under two seconds. Reproduced 3 of 3, with greedy and with sampled decoding. During the stall the prefill engine is idle,
the decode engine shows one running request with 128 used tokens (two pages: the prompt, no generated tokens) and zero throughput, both
engines report healthy, and nothing is logged. The request disappears only when the client disconnects.

The cause is in how parallel sampling is expanded for disaggregated serving. The tokenizer manager turns one `n=2` request into three
sub-requests: a prefix request with `max_new_tokens = 0`, then two sampling requests. The load balancer attaches one integer bootstrap room to
the HTTP request. `normalize_batch_and_arguments` expands that integer into `[room, room + 1]`, one per sample. But the expansion loop builds all
three sub-requests from `obj[i]` with `i` = prompt index 0, so all three carry `bootstrap_room = room`. The second room is never used. Three
requests with one room collide in the transfer bookkeeping on both engines, and the decode keeps whichever request lost the race, with the
prompt's KV pages allocated and nothing to generate.

**Expected behavior.** Two completions within a few seconds, exactly as in monolithic mode. Or, if parallel sampling is not supported in
disaggregated mode, HTTP 400 at validation. In no case should a request be held indefinitely with resources allocated.

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips / 8 TensorCores each), prefill on host a, decode on host b, TP 8, EP 8 |
| Model | `openai/gpt-oss-120b` (MXFP4), KV cache fp8_e4m3, page size 64 |
| Topology | raiden transfer backend, Python mini load balancer (`sglang_router.mini_lb`) on port 8080, radix cache off on the prefill |

## Code trace

All links at commit 09d1244c26.

**1. The load balancer attaches one room per HTTP request.** For a single prompt (no batch) the mini-LB adds a single integer `bootstrap_room`
(and host and port), regardless of `n`:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py#L490-L513

**2. Normalization expands the room per sample.** With `parallel_sample_num = 2`, `num = batch_size × parallel_sample_num = 2`, and an integer
room becomes `[room, room + 1]`:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/io_struct.py#L583-L603
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/io_struct.py#L886-L910

**3. The expansion loop uses the prompt index, not the sample index.** `_handle_batch_request` takes `objs = [obj[i] for i in range(batch_size)]`
(here one object, carrying `bootstrap_room[0]`), sends a prefix request built from it with `max_new_tokens = 0`, then sends `parallel_sample_num`
copies of the same tokenized object with new rids but the same bootstrap fields:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/tokenizer_manager.py#L2003-L2060

So the prefill engine and the decode engine each see three requests with identical `bootstrap_room`. The room is the key of the per-request
transfer state on both sides (`register_decode_request`, `check_status`, `send_kv_chunk`), so two of the three registrations overwrite or wait on
the third.

**4. The prefix request has no business on the decode side.** Its purpose is to populate the prefix cache once so the samples reuse it. In
disaggregated mode it is still forwarded to both engines, allocates KV on the decode (`is_prefill_only` is true, nothing will be generated) and
then depends on a transfer whose room is shared with two other requests:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/schedule_batch.py#L1430-L1435

Step 3 is confirmed by reading the code; the exact race among the three same-room requests (which one the decode ends up holding) is inferred
from the observed state: one running request on the decode with exactly the prompt's pages and no output.

## How we detected it

Sampling-parameter matrix of our validation campaign. Each case sends one request through the mini-LB with a timeout and records the result;
a watchdog polls both engines' `/metrics` every few seconds.

- Case `n_2` (`"n": 2, temperature 0, max_new_tokens 128`) returned nothing within 300 s on the first run; the 17 other cases answered in < 2 s.
- Dedicated repeat on a fresh pair: `n=2` greedy and `n=2` at temperature 0.7, each no response within 120 s. During both stalls:
  prefill `num_running_reqs 0`, `num_used_tokens 0`; decode `num_running_reqs 1`, `num_used_tokens 128`, `num_queue_reqs 0`, no `Decode batch`
  log lines, both `/health` 200.
- The decode released the request only when the client process exited. No error was logged on either side at any point.

## How to reproduce

1. Start a gpt-oss-120b pair (prefill + decode, raiden, mini-LB).
2. Send:
   ```
   curl -s -m 60 $LB/generate -H 'Content-Type: application/json' \
     -d '{"text":"The three primary colors are","sampling_params":{"temperature":0,"max_new_tokens":32,"n":2}}'
   ```
3. While it waits: `curl -s http://<decode>:30000/metrics | grep -E 'num_running_reqs|num_used_tokens'`.

**Observed.** Step 2 times out. Step 3 shows one running request with 128 used tokens and no progress; the prefill shows nothing running.
**Expected.** Two completions in a few seconds.

Control: the same request against a monolithic server returns two completions.

## Fix proposal

1. **One room per sub-request.** In `_handle_batch_request`, index the bootstrap fields by the sub-request number: the sample `k` of prompt `i`
   should use `bootstrap_room[i * parallel_sample_num + k]`, which `_normalize_bootstrap_params` already produces. The prefix request needs its
   own room as well (allocate `num + batch_size` rooms, or derive one from the prompt's room with an offset that cannot collide).
2. **Do not send the prefix request to the decode in disaggregated mode.** A `max_new_tokens = 0` request has nothing to decode. Either skip the
   prefix request entirely when `disaggregation_mode != null` (the prefill's radix cache, if enabled, is warmed by the first sample anyway), or
   let the router drop prefill-only requests on the decode side.
3. **Until fixed, validate.** Reject `n > 1` with HTTP 400 in disaggregated mode so clients get an immediate answer instead of a silent hold.
4. **Bound the hold.** Independently of the above, a decode request that never receives its KV should fail at the bootstrap or waiting timeout.
   Our recipe sets both to 7200 s, so the 120 s observation cannot say whether the waiting timeout would have fired; the raiden receiver has no
   deadline at all while a room is still in `Bootstrapping` (see our handshake-stall report, F-12).

## Related

- No open issue mentions parallel sampling in P/D mode.
- Our F-12 report (decode waits forever in `Bootstrapping`) describes the missing deadline that lets this hang persist.

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/A/params/gpt-oss/params_pd_A.json` (case `n_2`), `results/phase_A_gptoss_redo.log`
(section "(4) F-6 repro", with the per-engine metrics during both stalls), `results/bug_reports/F-6.md`.
