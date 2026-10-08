# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [TPU/PD/Raiden] Client disconnect during chunked prefill kills the prefill engine: aborted request's KV row is released twice (`AssertionError: Only a mamba-capable tree cache allows freeing before alloc`)

**Labels:** bug, tpu, pd

---

## Summary

A client opens a streaming request with a long prompt (131k tokens) against a 1-prefill + 1-decode pair and closes the connection two
seconds later, while the prefill engine is still processing the prompt in chunks. Within a few seconds all eight prefill scheduler ranks exit
with the same assertion, the launcher kills the process group, the load balancer goes down with it, and every in-flight and later request fails.
The decode engine survives. Reproduced 2 of 2. Disconnects during the decode phase (6 of 6, including two concurrent) are handled correctly.

The cause is a double release of the aborted request's KV row. The chunked-abort path defers the page release because the raiden transfer
backend needs its outstanding transfer group to drain first. But the result of the chunk that was already in flight lands one step later and
takes the regular aborted-result path, which frees the row immediately. When the deferred path later runs, the row is already gone, and the
general release routine treats "release without a row" as legal only for Mamba-style caches. The configuration uses the plain chunk cache
(radix cache off), so the assertion fires inside the scheduler loop and the process exits.

The assertion did its job: it caught a double free instead of corrupting the pool. But a client disconnect is a routine event and must not take
the engine down.

**Expected behavior.** A disconnect ends one request and nothing else:

1. No further chunks are launched for that request; the chunk already in flight completes and its result is discarded (this part works today).
2. The request's KV pages, request slot, metadata buffer and transfer room are released **exactly once**, after the raiden transfer group has
   drained, so no remote read is still touching the pages when they are reused.
3. Every other request is unaffected: queued requests, requests in flight on the decode side, and the client's next request all proceed normally.
4. Both engines stay healthy and the load balancer stays up; running requests and token usage return to their pre-request values within a
   bounded time (our test uses 60 s), and a follow-up request succeeds.
5. The event is logged at info level. If the abort bookkeeping fails for any reason, that one request is marked failed and the traceback is
   logged; the exception never escapes the scheduler loop.

Decode-phase disconnects in the same test already behave this way (6 of 6: usage back to baseline, follow-up request correct). A prefill-phase
disconnect should look identical from the outside, only with the page release delayed until the transfer drain completes.

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips / 8 TensorCores each). Host a: prefill engine. Host b: decode engine. TP 8, EP 8 on each |
| Model | `openai/gpt-oss-120b` (MXFP4 weights), KV cache fp8_e4m3, page size 64 |
| Prefill flags that matter | `--disaggregation-mode prefill --disaggregation-transfer-backend raiden --disable-radix-cache --chunked-prefill-size 8192 --context-length 140288` (overlap scheduling on; tree cache is `SWAChunkCache`) |

## Code trace

All links at commit 09d1244c26.

**1. The disconnect becomes an abort for the request that is currently being chunked.** The HTTP layer logs
`Request is disconnected from the client side (type 1). Abort request obj.rid=...` and sends an `AbortReq`. The scheduler sees that the rid is
its current `chunked_req` and only records it, because tearing the request down in the middle of an iteration is unsafe:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/scheduler.py#L5765-L5768

**2. Next scheduling step: the pending abort is processed and the page release is deferred.** `process_pending_chunked_abort` marks the
request aborted, calls `disagg_kv_sender.abort()`, and because the raiden sender declares `group_failure_requires_drain = True` and is not yet
drained, it sets `defer_page_release`, drops the cache-side state, and appends the request to `_deferred_chunked_sender_aborts` instead of
freeing its pages. It then clears `chunked_req`, so no further chunk is launched:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/scheduler.py#L3720-L3777
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/raiden/sender.py#L973

The docstring states the intent for the chunk already in flight: "the chunk already launched is drained when its result is resolved ... the
batch-result processors ... skip the aborted chunk".

**3. The in-flight chunk's result lands and frees the row anyway.** Under overlap scheduling the result of the launched chunk arrives one step
later in `process_batch_result_disagg_prefill`. For an aborted request it calls `_retire_aborted_prefill_result`:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/prefill.py#L905-L930

That routine checks `req.kv.holds_kv`, which is still true because step 2 deferred the release, and therefore releases the KV row immediately.
It does not consult the sender's drain state and does not know the request is on the deferred list:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/prefill.py#L1198-L1222

After this, `req.kv.req_pool_idx` is `None` and `holds_kv` is false:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/schedule_batch.py#L969-L970

**4. The deferred path runs and releases the row a second time.** At the top of a later step, `get_next_disagg_prefill_batch_to_run` calls
`process_pending_chunked_abort`, which first drains the deferred list. The sender reports drained, and the code calls `release_kv_cache` with
no `holds_kv` guard:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/prefill.py#L672
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/scheduler.py#L3779-L3796

**5. The release routine asserts.** `release_kv_cache` allows a release without a row only for a Mamba-capable cache. `SWAChunkCache` is not
one, so it raises:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/mem_cache/common.py#L292-L300

```
[TPx] Scheduler hit an exception: Traceback (most recent call last):
  File ".../sglang/srt/disaggregation/prefill.py", line 757, in event_loop_overlap_disagg_prefill
  File ".../sglang/srt/disaggregation/prefill.py", line 672, in get_next_disagg_prefill_batch_to_run
    self.process_pending_chunked_abort()
  File ".../sglang/srt/managers/scheduler.py", line 3734, in process_pending_chunked_abort
    self._drain_deferred_chunked_sender_aborts()
  File ".../sglang/srt/managers/scheduler.py", line 3795, in _drain_deferred_chunked_sender_aborts
    release_kv_cache(req, self.tree_cache, is_insert=False)
  File ".../sglang/srt/mem_cache/common.py", line 298, in release_kv_cache
    assert tree_cache.supports_mamba(), (
AssertionError: Only a mamba-capable tree cache allows freeing before alloc
```

The exception is not caught in the scheduler loop. All eight TP ranks hit the same path, the launcher kills the group (`exited with -9`), and the
load balancer that runs in the same process group dies with it. The tokenizer manager additionally logs
`Received output for rid='abort-0-...' but the state was deleted in TokenizerManager`.

**Why decode-phase disconnects are fine.** A request in the decode phase is no longer `chunked_req`, so the abort takes the running-batch path,
which releases once.

**Secondary concern.** Step 3 frees the pages while the raiden group may still be draining, which is exactly what the deferral in step 2 was
meant to prevent. On this run the sender had drained before step 4, so the crash is the only symptom we saw, but a slow drain could let the
pages be reused while a remote read is still in progress.

## How we detected it

Test case "client abort" in our 1-prefill + 1-decode validation campaign, run with `scripts/abort_test.py`. The tool opens a streaming
`/generate` request with a 131,072-token prompt and `max_new_tokens 1024`, waits 2 s (first token has not arrived, so the request is in
chunked prefill), then shuts the TCP socket down hard. Before the abort, right after it, and 30 s later it reads both engines' server info
and compares running requests, queued requests and token usage with the pre-test baseline. Pass criterion: both engines back to baseline within
60 s and a follow-up request succeeds.

- The post-abort poll of the prefill engine returned connection refused; `/health` returned no HTTP status.
- The prefill log carried the traceback above once per rank, then `Rank 0 scheduler died`, then the launcher's `exited with -9`.
- The decode engine stayed healthy; the load balancer was gone.
- Repeated on a fresh pair the next day: identical. The decode-phase variant (abort 10 s in, 6 aborts, 2 concurrent) passed with resources
  back to baseline.

## How to reproduce

1. Start a gpt-oss-120b pair as in the environment table (prefill with `--disable-radix-cache --chunked-prefill-size 8192`, raiden transfer).
2. Build a prompt of about 131k tokens (any text; needle content is irrelevant).
3. Open a streaming request and close the socket 2 s later. With Python `requests`:
   ```python
   import requests, socket, time
   r = requests.post(LB + "/generate", json={"text": PROMPT, "stream": True,
       "sampling_params": {"temperature": 0, "max_new_tokens": 1024, "ignore_eos": True}}, stream=True)
   time.sleep(2)
   s = r.raw._connection.sock; s.shutdown(socket.SHUT_RDWR); s.close(); r.close()
   ```
   or: `python scripts/abort_test.py --url $LB --prompt-file prompts/gpt-oss/131072.jsonl --phase prefill --abort-after 2 --repeat 1`.
4. Wait 15 s. `curl -s -o /dev/null -w '%{http_code}\n' $PREFILL/health`.
5. `grep -A12 'Scheduler hit an exception' <prefill log> | head -20`.

**Observed.** Step 4 prints `000`. Step 5 shows the traceback above on every rank.
**Expected.** `200`; the prefill engine frees the request's pages after the raiden group drains and serves the next request.

Also worth running: the same abort with the radix cache on (`UnifiedRadixCache`), which takes a different branch in `release_kv_cache`, and
with `--disable-overlap-schedule`, where the in-flight chunk's result lands in the same step.

## Fix proposal

1. **Make the deferred list the single owner of the release.** In `_retire_aborted_prefill_result`, if the request is on
   `_deferred_chunked_sender_aborts` (or, equivalently, if `sender.group_failure_requires_drain and not sender.drained()`), do the metadata
   and bookkeeping cleanup but skip `release_kv_cache`; the drain will release once the transfer group is quiescent. This also closes the
   secondary concern above, because pages are then never freed before the drain.
2. **Guard the drain.** In `_drain_deferred_chunked_sender_aborts`, release only `if req.kv.holds_kv or req.kv.holds_mamba`, matching the guard
   that `_retire_aborted_prefill_result` and `release_aborted_prefill_waiting_req` already use. On its own this prevents the crash but leaves
   the early free of step 3 in place, so it should go together with fix 1.
3. **Contain the blast radius.** A failure in the abort bookkeeping of one request should fail that request, not the scheduler. Wrapping the
   deferred-abort drain so that an exception marks the request failed and logs the traceback would keep the engine up even if another ordering
   bug of this kind exists.

A regression test: chunked prefill of a multi-chunk prompt on a `ChunkCache`, abort between chunks with a sender whose `drained()` returns
false for one step, and assert that `release_kv_cache` is called exactly once for the request.

## Related

- #1114 (prefill keeps a request after the decode gives up) is a different hold on the same abort path; no open issue reports this assertion.
- #1113 (decode waits the full transfer timeout) is the decode-side counterpart of a failed transfer; not the same.

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/logs/runs/A_gptoss_128k_run1/prefill_server.attempt2_abort_crash.log.gz`
(full traceback, all ranks), `results/A/abort/gpt-oss/abort_prefill.json` (tool output with the server-info snapshots), `results/phase_A_gptoss_redo.log`
(second reproduction), `results/bug_reports/F-8.md`.
