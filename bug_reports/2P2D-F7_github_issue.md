# Draft issue (not filed): [P0] [TPU/Spec] A request with `return_logprob` kills an EAGLE3 decode: `fail_on_recompile` trips in `_fused_draft_steps`

**Build:** main 2e14706a62. **Setup:** gpt-oss-120b, EAGLE3 (`lmsys/EAGLE3-gpt-oss-120b-bf16`, topk 1, 3 steps, 4 draft tokens), bf16 KV, `--enable-torch-compile`, P/D over raiden (also reproduces the guard in 1P/1D). TP-8 on one tpu7x host per engine.

**What happens.** The first `/generate` request with `"return_logprob": true` (output logprobs only, `logprob_start_len` default) that reaches the decode makes every TP rank raise:

```
Scheduler hit an exception: ...
RuntimeError: Detected recompile when torch.compile stance is 'fail_on_recompile'.
  filename: python/sglang/srt/speculative/eagle_worker_v2.py, function name: '_fused_draft_steps'
  triggered by the following guard failure(s):
  - 1/0: forward_batch.return_logprob == False   # if not forward_batch.return_logprob:  # sglang_tpu/compilation/compiled_forward_shim.py:440 (needs_input_logprob)
  - 1/1: tensor 'topk_p' size mismatch at index 0. expected 2, actual 1
  - 1/2: tensor 'topk_p' size mismatch at index 0. expected 4, actual 1
  - 1/3: tensor 'topk_p' size mismatch at index 0. expected 8, actual 1
  User stack: eagle_worker_v2.py:865 _fused_draft_steps -> model.forward -> CompiledForwardShim.__call__ -> _call -> _forward -> needs_input_logprob
```

The launcher then kills the process tree (`An engine or router exited with -9`). Every request routed to that decode afterwards fails; in a 2P/2D group the prefills count a bootstrap failure per request (79 and 61 in ten minutes) until the decode is restarted.

**Why.** `CompiledForwardShim._forward` calls `needs_input_logprob(forward_batch)`, which branches on `forward_batch.return_logprob`. For the EAGLE3 draft loop the shim is invoked *inside* the torch-compiled `_fused_draft_steps`, so that Python branch becomes a dynamo guard on the compiled draft graph. Warm-up only ever runs with `return_logprob=False`; the first `True` batch fails the guard, and with the production stance that is fatal. Non-speculative decodes are unaffected because the shim's branch runs outside any enclosing compiled region.

**Regression.** The same guard failure appears in a 2026-10-08 config-D log (build 09d1244c26) as an *allowed* recompile (`[__recompiles] Recompiling function _fused_draft_steps ... forward_batch.return_logprob == False`), after which the request completed. On 2e14706a62 the identical request kills the engine, so something between the two builds made the stance apply to the fused draft steps (candidates: #1289 "Fix speculative nightly warmup and startup failures", #1178 "Attribute serving-time eager compiles to sglang source lines").

**Reproduce.** Start a config-D pair (or a monolithic EAGLE3 server with `--enable-torch-compile`), warm up, then:

```
curl -s $LB/generate -H 'Content-Type: application/json' -d '{"text":"<any 8k prompt>","sampling_params":{"max_new_tokens":64,"temperature":0},"return_logprob":true}'
```

**Expected.** Either the logprobs are served (the draft graph should not depend on `return_logprob`: read it outside the traced region or warm both values), or the request gets a 400 at intake like prompt logprobs do on the compiled backend (#1245 pattern). A request field must not take down the engine.

**Evidence.** `results/2p2d_tp8/p7.log`; decode `server.log` of run `2p2d_tp8_D` at 22:47:02 UTC; 1P/1D decode log of 2026-10-08 12:00:48 UTC with the allowed recompile. Accept length and routing were otherwise normal (mean 2.49 at c=1, 5/5 routing probe).
