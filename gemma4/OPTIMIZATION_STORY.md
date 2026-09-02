# From 175 s to 40 s: the gemma4-E2B RL optimization story

*(g4r19 → g4r32b, 2026-09-01/02. Every number below has a run, a manifest, and a GCS artifact behind it —
see `GEMMA4_PERF_REPORT.md` for the ledger and `gemma4/<run>-manifest/` for byte-exact reproduction.)*

## Act 1 — Earning the starting line (the OOM conquest)

The story begins before the first measurement: **tp1×dp128 on the full 128-device slice wouldn't run at
all.** The trainer and vLLM share every device in full-slice serial mode, and vLLM's default 60% HBM
reservation left 36.7 GB where the train step demanded 80.3 GB (the whale: bf16 logits + gradient over
the 262K vocab). Even after shrinking to fit, runs died at the *step-1 weight-sync boundary* — nine
straight times. The fix was three-part, learned from a working reference: **flash trainer attention**
(shrinks the resident program), **chips_per_vm=8** (correct device-per-VM mapping), **kv_cache_buffer=4352**.
g4r19 then ran 8/8 steps: **median 175.6 s**. Our baseline existed.

## Act 2 — Knowing the enemy (roofline says 30–50×)

The roofline said a step should cost **4–6 s**: decode sits on the HBM roof at ~0.8 ms/iteration,
the trainer needs under a second across 128 chips. We measured devices ~2–3% busy. Every trace —
xplane, Perfetto, phasetime — pointed the same way: **the host**. A single Python loop feeding 128
engines through the Pathways proxy, 2.4 s of host time per 64-token decode window against 52 ms of
device work. The enemy had a name: serialized host round-trips.

## Act 3 — The reference (make the target ours)

Tianyu's g4-003 ran the *same mesh, same image build, same checkpoint* at **~51 s**. Instead of
guessing toward it, we cloned its configmaps byte-exact from the cluster and ran it under our name:
**g4r30 = 50.2 s median, rewards byte-identical** to the original. The target stack was now ours to
dissect — and its levers were legible: CD windows of 512 (8× fewer round-trips), async host work
(scoring/detok during decode), a 29-flag XLA ladder, the textual gemma-3-style prompt.

## Act 4 — The crashes that taught us (why our launcher couldn't get there)

Porting those levers onto our own launcher produced a memorable villain: **synchronized worker
crashes at the step-1 boundary, 19 attempts in a row** at window 512. The investigation found a real
defect — our fused-loop exit condition ran a **cross-DP all-reduce every decode iteration** (512
chained collectives × 128 devices; tianyu's cond was count-only) — but fixing it wasn't enough, and
ablating the reference proved why: **g4r31 (g4r30 minus the XLA flags) died identically.** The XLA
proxy ladder isn't a perf garnish; it is *load-bearing* for deep-window stability. We stopped
patching toward the reference and adopted its lineage as our development base.

## Act 5 — Correctness excavations (the numbers had lies in them)

Three landmines were defused along the way, each invisible in step-time alone:

1. **The canonical `<|turn>` chat markers are untrained** in this checkpoint — under the standard
   template the model echoes prompts or loops "by definition" forever: **0.0 correct answers in
   131,000 completions** (g4r19). The gemma-3-style *textual* prompt is a correctness requirement.
2. **Our launcher had zeroed the correctness reward since bring-up** (`reward_exact_answer=0.0`
   nullified the exact-match *and* math-verify paths) — the RL signal was format-only by config.
3. The reward stack was **cross-validated against the GPU/verl port** (Summer's
   `maxtext_math_reward.py`): weights identical (+1.0/+0.1), equivalence paths agree at the worker
   level (one scientific-notation edge case documented). TPU and GPU reward curves are overlayable.

## Act 6 — The novel lever (49 `<end_of_turn>`s)

Reading raw completions revealed the campaign's best discovery: after answering (~1000 tokens), the
model emits the textual `<end_of_turn>` — then, unstopped, **invents a new user turn and talks to
itself, 49 markers deep, to the 4096 cap.** GPU stacks never see this (vLLM stop-*strings* cut at
marker #1); the TPU fused loop only stops on token IDs, and `<end_of_turn>` is seven plain BPE
pieces: `[236820, 643, 236779, 1340, 236779, 887, 236813]`. **`patch_textual_stop`** scans each
512-token window at the host boundary and retires the request at the first marker. Effect: OSL
3550→1900 (−45%), truncation 85%→40%, **and correct-answer density up ~40%** — ending at the real
answer instead of self-dialog is better RL, not just faster RL.

**g4r32 = g4r30 + our stop = 43.4 s. Confirmed by g4r32b = 40.7 s median, best step 38.8 s.**
The first configuration to beat the reference it was built on.

## Act 7 — The attribution ledger (measure one lever at a time)

Leave-one-out ablations on the champion, one knob per run:

| Lever | Δ warm median | Pair |
|---|---|---|
| tianyu base bundle @ win64 (prompt, async host work, TMBS8, …) | **−123.9 s** | g4r19 ↔ g4r33 |
| CD window 512 | **−11.0 s** | g4r33 ↔ g4r32b |
| Textual stop (ours) | **−6.8 s** (+quality) | g4r30 ↔ g4r32 |
| Async-reward | **−5.6 s** | g4r35 ↔ g4r32b |
| XLA ladder | stability-prerequisite | g4r31 (crash without) |
| Verify-timeout + partial-async | tail 156 s → **7.3 s** worst-case | g4r35 stress test |

(prompt / prefix / rr / TMBS rungs in flight or staged; the bundle row decomposes further.)

## Where the story stands

**175.6 → 40.7 s median (4.3× on this mesh; 3.1× vs the campaign's 125.8 origin), with healthier
training than when we started.** The clean step is now 84% rollout — trainer 4.5 s, weight-sync
0.45 s, rewards 0.2 s are effectively solved. What remains between 40 s and the 4–6 s roofline is a
single architectural fact: one head process serially driving 128 engines. Deep windows shortened the
conversation with the head; the sequel is removing the head from the inner loop entirely
(Pathways-compatible multiprocess-DP, or its equivalent). That is the next story.
