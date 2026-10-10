## Live validation on a TPU v7x P/D pair (1 prefill + 1 decode, gpt-oss-120b, raiden, mini-LB)

Head 683a25e34f deployed on both hosts; control = upstream main 4f7aae5369. "Orphan room" = a request sent straight to the decode with bootstrap fields no prefill will serve. Probe and recipes: `sglang_inference/scripts/handshake_timeout_probe.py` in alisachen-google/tpu-rl.

| # | Case (BOOTSTRAP_TIMEOUT = 60 s unless noted) | main | this PR |
|---|---|---|---|
| B3 | orphan room with an **unreachable** bootstrap host (stays in Bootstrapping) | hangs until the client gives up | **500 after 60.0 s**: "Request … timed out after 60.0s in KVPoll.Bootstrapping before its KV metadata was sent to the prefill. Raise SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT to wait longer." |
| B1/B2 | 1 and 16 orphan rooms against the **real** prefill (handshake succeeds, KV never comes) | hang until the client gives up (240 s) | same: these rooms send metadata at once, so they wait in the KV window, which this PR leaves to `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` (7200 s in this recipe). With WAITING_TIMEOUT = 90 s: __WT_RESULT__ |
| B4 | orphan whose client disconnects after 10 s, then the deadline passes | – | clean: prealloc queue 0, health 200, 0 exceptions 75 s later |
| B5 | normal traffic (8k ×4 at c=4, 32k, 131k) | – | all 200 (1.6–7.4 s), decode KV back to 0 |
| C | the #1148 trigger: single-slot 128k pair, 300 s timeouts, first 8k request | stalled 5/5 pairs on 10-08/09 | did **not** reproduce on this head (200 in 1 s; second request 200) |
| **B6** | **KV pressure**: decode pool 262k, 8 × 131k prompts / 2048 tokens at c=8 (one prompt fits at a time) | – | requests 0–4 served sequentially (20–70 s); **requests 5–7 failed with 500 at ~60 s** with the new message, state `KVPoll.WaitingForInput` (18 log lines). They were waiting for decode KV room, not for a dead prefill. |

Zero scheduler exceptions on either engine in every run.

**Regression risk (B6).** The deadline runs from receiver creation at intake until `send_metadata`, and `send_metadata` happens only after pre-allocation. So any decode-side wait for KV room longer than `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` (default 300 s) now fails the request, where before it would have completed once room freed. Under sustained pool pressure with long prompts this is reachable in production. Two ways out: start the clock when the handshake completes rather than at receiver creation (so only the Bootstrapping window is bounded), or exclude `WaitingForInput` from `_check_handshake_timeout` and leave that window to the waiting timeout.

Evidence: `results/fix_validate_1230_1222/` (probe outputs, server logs) in alisachen-google/tpu-rl.
