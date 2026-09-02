# Forward Optimization Plan: g4r19 → optimal, step by step

Built from the exact config diff between g4r19 (our launcher) and g4r32b (champion).
Each step changes ONE thing (or one named bundle) and is measured with an 8-10 step run.

## The config diff (g4r19 → g4r32b)

| Dimension | g4r19 | g4r32b |
|---|---|---|
| Prompt | canonical `<|turn>` template | gemma-3-style textual (`gemma_data.py`) + stop-ids [1,106,50] |
| CD window | 64 | 512 |
| Textual `<end_of_turn>` stop | off | ON (our patch) |
| Async host work | none (serial rewards/detok/logprobs) | async reward+detok+logprobs |
| XLA proxy flags | none | 29-flag ladder (stability-required ≥ cd512) |
| Trainer | TMBS 4, HBM 0.25 | TMBS 8, HBM 0.3 |
| Engine shape | mnbt 8448, block auto, prefix on, best-rank routing | mnbt 33280, block 128 pinned, prefix OFF, rr routing, eos-interval 8 |
| Patch stack | our roster (26) | tianyu 34-patch stack + our stop/dump/vtmo/partial |
| Reward spike guard | vtmo only | vtmo + _AR_PARTIAL |

## The ladder (L-rungs, 2026-09-02 revision — one lever per queued run, all on OUR launcher)

Revised per direction: g4r19b baseline re-run dropped (L0 anchor = the original g4r19
measurement); each queued rung adds exactly one optimization bundle on top of the previous,
ending at the g4r32c champion config. Rungs L2+ auto-launch sequentially via
`ladder-chain.sh` (next rung installs only when the previous completes; chain stops on failure).

| Rung | Run | Delta vs previous | Status / expected |
|---|---|---|---|
| L0 | g4r19 ✅ | baseline (canonical prompt, cd64, serial host work) | **175.6 measured** |
| L1 | **g4r20b** (queued) | + end-of-turn/reward fix: gemma-3 textual prompt (`DL_DSPROC`), stop-ids [106,50], textual stop ON | OSL −45%, truncation ↓, first nonzero `check_numbers`; perf gain from shorter decode |
| L2 | **g4r38** (queued) | + host-work scheduling: async detok (`DL_ADETOK`), async logprobs (`DL_ALOGP`), rr DP routing (`DL_DPRR`), fast extract (`DL_FASTEX`) | shaves serial host time per window |
| L3 | g4r39 (staged, auto) | + decode-loop shape: CD window 512 (`DL_CD`), eos-interval 8, 29-flag XLA proxy ladder (`values-xla-ladder.yaml`) | 8× fewer head round-trips; KNOWN RISK: this lineage crashed at cd512 before (g4r21d) — this rung is the retest with the full ladder + current patch set |
| L4 | g4r40 (staged, auto) | + trainer/engine shape: TMBS 8, HBM 0.3 | should land ≈ champion band (~40 s) |
| L5 | **g4r32c** (queued, priority) | champion reference: g4r32b + spike fix (vtmo + _AR_PARTIAL) | the target: median ~40.7 without the 156 s step-8 spike |

If L3 reproduces the cd512 crash on our launcher, the chain stops there and the L3→L5 delta
stays priced by the reference-side measurements (window 512 −11.0 s via g4r33↔g4r32b; async
reward −5.6 s via g4r35↔g4r32b), with the launcher-parity gap documented as the open defect.

## Beyond F5 (the next frontier)
84% of the F5 step is rollout (33.5 s vs 2.6 s roofline): serial-head removal
(Pathways-compatible multiproc-DP / process-pooled window handling), window 1024 probe,
chunked-logps + HBM re-raise.
