# Progress-rule readiness for P/D engines: design page

Follow-up to `pd_disagg_health_probe_design.md` for sglang-torchtpu issue 1276. Written
2026-10-14 after the live campaign on the four-host 2P/2D TP-8 group.

Interactive page (permanent link, same URL for every version):
https://claude.ai/artifact/3GnECC2gY9jNyTgtBK1dYg

Offline copy of version 12, self-contained HTML (only the fonts load from Google Fonts):
`docs/pd_disagg_progress_rule_readiness.html`. Open it in a browser; it renders in light and
dark and the timeline simulator in section 1 runs without a network.

## What the page holds

| Section | Content |
|---|---|
| Rule | `UNHEALTHY <=> outstanding for T and nothing produced for T`; T = 600 s, 1200 s until the first output after start |
| 1 | Timeline simulator of the shipped tracker logic: healthy, transfers never complete, one prefill dead, idle then first request, cold start, scheduler hang; sliders for T, T_cold, transport timeout |
| 2 | Scheduler-side data flow (PR 1390): tracker in each rank, slot files in /dev/shm, HTTP reads |
| 3 | The hung-scheduler blind spot and the two ways to close it: staleness rule (1390) or HTTP-side clock (1392) |
| 3b | PR 1392 at a glance: the four wrapped TokenizerManager methods, the swapped route, the bookkeeping and tracker |
| 3c | How PR 1392 answers each case: starting, shutting down, fresh start, idle, hung while idle, requests producing, requests stalled, transfers aborting, recovery, client cancel, long chunked prefill, feature off, /health |
| 3d | The idle fallthrough: three checks in order, the core probe runs only when nothing is in flight |
| 4 | Files touched by each PR, plugin-only |
| 5 | Tunables |
| 6 | Unit tests and live results for both PRs |
| 6b | Per-peer rule on the decode (PR 1393) |
| 6c | Trade-offs HTTP side against scheduler side; what only each side can do |
| 7 | Recommendation |

## PRs

- sglang-torchtpu 1392, HTTP-side, head 0989ad530d: tracker in the HTTP process, idle fallthrough to the core probe.
- sglang-torchtpu 1390, scheduler-side, head 6a0211a844: tracker per rank, slot files, staleness rule.
- sglang-torchtpu 1393, per-peer rule on the decode, stacked on 1392.

## Live evidence referenced by the page

Results under `sglang_inference/results/2p2d_tp8/pr9*`, `pr11*`, `pr12*`, `pr13*`, `pr14*`
(decode and prefill phases, hang with work held, hang while idle, busy path), recipes under
`sglang_inference/recipes/gpt-oss_B_128k_2p2d_{d1,p1}{plug,prog,peers}*.json`, hooks under
`sglang_inference/scripts/hooks/{1276,1276p,hang}/`.
