# Bug reports: sglang-torchtpu, 1 prefill + 1 decode on TPU v7x

The chronological raw log is `../FINDINGS.md`. These reports use ASD-STE100 style: short sentences, active voice, one instruction per step. Each report has a numbered
"How to reproduce" section with the exact commands. The tools are in `../../scripts/`. The server recipes are in `../../recipes/`.

| ID | Title | Type | Severity | Status |
|---|---|---|---|---|
| [F-1](F-1.md) | Hard abort when the tier-2 compile cache directory cannot be created | bug (robustness) | low | reproduced, work-around |
| [F-2](F-2.md) | Qwen3-VL-30B-A3B language model is not compiled on TPU | bug (perf, missing policy) | high for perf tests | reproduced on every start |
| [F-3](F-3.md) | Greedy decoding is not batch-invariant | test-method note | medium | measured |
| [F-4](F-4.md) | gpt-oss PD output differs from monolithic for long prompts (chunk schedule differs) | numerical difference | high until accuracy check | reproduced 6/6, cause identified |
| [F-5](F-5.md) | A prompt-logprob request stops the prefill engine | bug (crash) | high | reproduced 1/1 |
| [F-6](F-6.md) | `n=2` request gets no response for 300 s | bug candidate (hang) | high if confirmed | observed 1/1 |
| [F-7](F-7.md) | Decode engine recovers about 2 min after the prefill engine dies | observation | low | observed |
| [F-8](F-8.md) | Client disconnect during chunked prefill stops the prefill engine | bug (crash) | high | reproduced 1/1 |
| [F-9](F-9.md) | Unaligned prompt lengths give a different first token in PD mode | bug candidate (correctness) | medium | observed 4/4 |
| [F-10](F-10.md) | One 1920×1080 image stops the Qwen3-VL prefill engine | bug (crash) | high | reproduced 1/1 |

## Common steps

1. Make sure that no other user holds the TPU: `bash scripts/tpu_free.sh`.
2. Start a server pair: `bash scripts/pd.sh start recipes/<model>_A_128k.json <run-name>`. Wait 4 to 8 minutes.
3. Start a single (monolithic) server: `bash scripts/pd.sh golden recipes/<model>_G_128k_fp8_e4m3.json <run-name> a|b`.
4. Stop all servers that you own: `bash scripts/pd.sh stop`.
5. Logs are in `results/runs/<run-name>/<role>/server.log` on the host that runs the role. Compressed copies are in `results/logs/`.
