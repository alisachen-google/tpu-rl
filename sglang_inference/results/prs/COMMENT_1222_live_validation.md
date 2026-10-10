## Live validation on a TPU v7x host (gpt-oss-120b, `--enable-torch-compile`)

Head 4914689cb1 deployed; control = upstream main 4f7aae5369. Launch time is from the first server log line to the error (our launcher then waits on `/health`, so its own exit took longer).

| # | Launch | main | this PR |
|---|---|---|---|
| A1 | monolithic, `--kv-canary log` | weights load (90 s), memory pool built, then every rank dies with `AttributeError: 'TpuFusedMHATokenToKVPool' object has no attribute 'k_buffer'` (the #1176 crash) | rejected at argument validation **in the same second as launch**: `ValueError: --kv-canary log is not supported on TPU: the canary cannot attach to the TPU fused KV pools. Use --kv-canary none.`; weights never loaded; no leftover scheduler processes |
| A2 | monolithic, `--kv-canary raise` | – | same rejection, same timing |
| A3 | monolithic, `--kv-canary bogus` | – | argparse rejects it first (`invalid choice: 'bogus'`), before the TPU check |
| A4 | monolithic, `--kv-canary none` (explicit) | – | starts normally (healthy after 331 s), `/generate` → 200 |
| B | P/D pair, `--kv-canary log` on the **decode only** (prefill `none`) | – | decode rejected at launch with the same message; the prefill (canary `none`) came up healthy and unaffected (`/health` 200) |

No scheduler exceptions in any run; the default (flag unset) path is exercised by every other run in our campaign and is unchanged.

Note on overlap: #1193 (ours) also handles #1176, by attaching the canary to the fused pools and refusing at `install_canary` through a platform capability flag. If both merge, this PR's argument-time rejection fires first and the install-time refusal becomes unreachable on TPU, which is fine; the attacher code in #1193 remains the prerequisite for running the canary on TPU later.

