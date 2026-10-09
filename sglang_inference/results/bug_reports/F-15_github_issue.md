# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [PD/mini-LB] `/generate` with an empty `input_ids` list returns a bare 500 from the mini load balancer (IndexError in `_get_request_batch_size`) while the engines return a clean 400

**Labels:** bug, pd, router

---

## Summary

The PD mini load balancer (`sglang_router/mini_lb.py`) inspects each `/generate` body to decide whether it is a batch. For `input_ids` it
evaluates `isinstance(input_ids[0], int)` without checking that the list is non-empty, so a request with `"input_ids": []` raises
`IndexError: list index out of range` inside the handler and the client receives `Internal Server Error` (HTTP 500) with no JSON body.
Sent directly to either engine the same body is rejected properly: `{"error":{"message":"input_ids cannot be empty."}}`, HTTP 400.

**Expected behavior.** The load balancer should either forward the body unchanged and let the engine's validation answer, or reject it
itself with a 400 that names the problem, exactly as the engine does. A client mistake must never surface as a 500 from the router, because
500s are what operators alert on.

**Impact.** Low: the engines are unaffected, nothing leaks, and the router keeps serving. It is an input-validation gap that produces a
misleading error class and an unstructured body. Priority P3.

## Environment

- `sglang-torchtpu` main @ `09d1244c26`, PD pair (1 prefill + 1 decode, TPU v7x), mini-LB on port 8080 started by `benchmark.tpu.launch_server`.
- Model irrelevant (router-side); observed with gpt-oss-120b.

## Code trace

`_get_request_batch_size` in
[mini_lb.py#L571-L577](https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py#L571-L577):

```python
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)   # IndexError on []
    return None
```

called from `handle_generate_request`
([mini_lb.py#L496](https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py#L496))
before the request is forwarded. The engine-side normaliser already handles this case
([io_struct.py#L484-L492](https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/managers/io_struct.py#L484-L492),
`raise ValueError("input_ids cannot be empty.")`), which is why the direct-to-engine request gets a 400.

Traceback from the prefill host's mini-LB log:

```
  File ".../sglang_router/mini_lb.py", line 496, in handle_generate_request
    batch_size = _get_request_batch_size(modified_request)
  File ".../sglang_router/mini_lb.py", line 576, in _get_request_batch_size
    return None if isinstance(input_ids[0], int) else len(input_ids)
IndexError: list index out of range
```

## How we detected it

Boundary-case sweep of malformed prompts against the PD pair (`_phase_boundary2.sh`, case 2c): empty `text` → 400, whitespace-only
`text` → 200, empty `input_ids` → 500 in about 1 ms. The engines' `/health` stayed 200 and the next requests were served normally.
Repeating the body against each engine's own `/generate` returned the 400 above, isolating the fault to the router.

## How to reproduce

```bash
curl -s http://<mini-lb>:8080/generate -H 'Content-Type: application/json' \
  -d '{"input_ids":[],"sampling_params":{"max_new_tokens":8}}' -w ' HTTP %{http_code}\n'
# Internal Server Error HTTP 500
curl -s http://<prefill>:30000/generate -H 'Content-Type: application/json' \
  -d '{"input_ids":[],"sampling_params":{"max_new_tokens":8}}' -w ' HTTP %{http_code}\n'
# {"error":{"message":"input_ids cannot be empty."}} HTTP 400
```

## Fix proposal

Guard the empty list (and `text == []`, which has the same shape) and return a 400:

```python
    if (input_ids := request.get("input_ids")) is not None:
        if len(input_ids) == 0:
            raise HTTPException(status_code=400, detail="input_ids cannot be empty.")
        return None if isinstance(input_ids[0], int) else len(input_ids)
```

or, as the file's own comment suggests, reuse `GenerateReqInput.normalize_batch_and_arguments` so the router and the engine share one
validation path. Add a unit test for `[]`, `[[]]` and `[1,2,3]`.

## Related

- No open issue in the repository covers router-side validation of `/generate` bodies.
- Companion observations from the same sweep: empty `text` is already rejected with 400, and `ISL + max_new_tokens > context` is rejected
  with a 400 by the engine within 0.2 s (correct behaviour).

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/phase_boundary2.log` (case 2c), `results/runs/A_128k_smallpool/prefill/server.log`
lines 1328–1382 on the prefill host (mini-LB traceback), `results/FINDINGS.md` (boundary phase 2).
