# Bug reports: sglang-torchtpu, 1 prefill + 1 decode on TPU v7x

The chronological raw log is `../FINDINGS.md`. These reports use ASD-STE100 style: short sentences, active voice, one instruction per step. Each report has a numbered
"How to reproduce" section with the exact commands. The tools are in `../../scripts/`. The server recipes are in `../../recipes/`.

| ID | Title | Type | Severity | Status | Overlap with open issues |
|---|---|---|---|---|---|
| [F-1](F-1.md) | Hard abort when the tier-2 compile cache directory cannot be created | bug (robustness) | low | reproduced, work-around  none |
| [F-2](F-2.md) | Qwen3-VL-30B-A3B language model is not compiled on TPU | bug (perf, missing policy) | high for perf tests | reproduced on every start  #1069 (coverage request only) |
| [F-3](F-3.md) | Greedy decoding is not batch-invariant | test-method note | medium | measured  #321/#391 related |
| [F-4](F-4.md) | gpt-oss PD output differs from monolithic for long prompts (chunk schedule differs) | numerical difference | high until accuracy check | reproduced 6/6, cause identified  #321 related |
| [F-5](F-5.md) | A prompt-logprob request stops the prefill engine | bug (crash) | high | reproduced 2/2  #432 (same code, no crash reported) |
| [F-6](F-6.md) | `n=2` request gets no response | bug (hang) | high | reproduced 3/3  none |
| [F-7](F-7.md) | Decode engine recovers about 2 min after the prefill engine dies | observation | low | observed  #1113 (same mechanism) |
| [F-8](F-8.md) | Client disconnect during chunked prefill stops the prefill engine | bug (crash) | high | reproduced 2/2  none |
| [F-9](F-9.md) | Unaligned prompt lengths give a different first token in PD mode | resolved: not a bug (F-3 artefact) | none | identical 2/2 at c=1  none |
| [F-10](F-10.md) | One 1920×1080 image stops the Qwen3-VL prefill engine | bug (crash) | high | reproduced 2/2  #488 (same mechanism, other trigger); #1068 |
| [F-11](F-11.md) | Chunked prefill deadlocks when the SWA continuation budget hits 0 before eviction (gpt-oss, 1 slot, TPU rounding) | bug (hang) | high | reproduced 4/4, root cause in code  none (#1067 would hit it) |
| [F-12](F-12.md) | P/D handshake never completes; request held on both engines with no deadline | bug (hang) | high | reproduced on 3/3 fresh pairs (one recipe)  #1113/#1078 related (those expire) |
| [F-13](F-13_github_issue.md) | `--kv-canary` kills every TPU engine at start-up (fused pool not supported by the canary) | bug (start-up crash, observability) | medium | 2/2 models  none |
| [F-14](F-14_github_issue.md) | `--disaggregation-enable-kv-checksum` is CUDA-only; TPU prefill passes health then dies on first request | bug (crash, observability) | medium | 2/2 models  none |

Overlap check: all 69 open issues read on 2026-10-08 (`results/bug_reports/OVERLAP.md`). Production triage (severity, priority, filing order):
`PRIORITIES.md`. Ready-to-file GitHub issue drafts with code traces: `F-1_github_issue.md`, `F-2_github_issue.md`, `F-5_github_issue.md`,
`F-6_github_issue.md`, `F-8_github_issue.md`, `F-10_github_issue.md`, `F-11_github_issue.md`, `F-12_github_issue.md`, `F-13_github_issue.md`, `F-14_github_issue.md`.

## Common steps

1. Make sure that no other user holds the TPU: `bash scripts/tpu_free.sh`.
2. Start a server pair: `bash scripts/pd.sh start recipes/<model>_A_128k.json <run-name>`. Wait 4 to 8 minutes.
3. Start a single (monolithic) server: `bash scripts/pd.sh golden recipes/<model>_G_128k_fp8_e4m3.json <run-name> a|b`.
4. Stop all servers that you own: `bash scripts/pd.sh stop`.
5. Logs are in `results/runs/<run-name>/<role>/server.log` on the host that runs the role. Compressed copies are in `results/logs/`.
