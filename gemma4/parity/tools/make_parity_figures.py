#!/usr/bin/env python3
"""gemma4-E2B GSM8K: TPU (MaxText+tunix, v7x) vs GPU (frozen reference) parity figures.

Inputs
  --gpu   aggregate_summary.json of the GPU reference (eval[].per_seed, timing, train)
  --tpu   one or more  NAME=DUMPDIR[:LOG1,LOG2,...]  specs, e.g.
          "seed42=evals_tpu/g4pt1:g4pt1-raw.log"
          DUMPDIR holds eval_KKK.jsonl / NAME_eval_KKK.jsonl (per-question scorer-v2 rows);
          LOGs are the head-pod stdout (Train step / Intermediate Eval / LENDIST / TIS-ACTIVE / PHASETIME).
  --out   output directory (PNG per panel + overview + parity_summary.md)

Metric: scorer-v2 accuracy (exclusive 1.0/0.1/0.0; acc = share of 1.0) on the 1319 GSM8K test
questions, greedy decode, both platforms. Gate: step-250 3-seed mean >= GPU mean - 1 pp (91.42).
"""
import argparse, glob, json, math, os, re, sys
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- design tokens (dataviz reference palette; identity = platform, seeds by marker) ----
GPU_C, TPU_C = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
MARKERS = ["o", "s", "D", "^", "v"]
GATE = None  # computed from GPU mean - 1pp
SKIP_DUMPS = set()  # dump indices to ignore (e.g. a startup eval scored on un-synced rollout weights)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------------------------------------------------------- GPU reference
def load_gpu(path):
    d = json.load(open(path))
    seeds = sorted(d["runs"].keys())
    ev = OrderedDict()
    for e in d["eval"]:
        ev[e["step"]] = e
    out = {"seeds": seeds, "steps": list(ev.keys()), "eval": ev,
           "train": d.get("train", {}), "timing": d.get("timing", {}), "runs": d["runs"]}
    return out


# ---------------------------------------------------------------- TPU run
STEP_RE = re.compile(r"Train step (\d+) training loss: ([0-9.eE+-]+)")
TS_RE = re.compile(r"^[IWEF](\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)")
EVAL_RE = re.compile(r"Intermediate Eval \(step=(\d+)\)|Pre RL Training:")
LEN_RE = re.compile(r"LENDIST step=(\d+) n=(\d+) cap=(\d+) truncated=(\d+) \(([0-9.]+)%\) eos_at_cap=(\d+) osl_mean=([0-9.]+) isl_mean=([0-9.]+)")
TIS_RE = re.compile(r"TIS-ACTIVE weight_mean=([0-9.]+) weight_min=([0-9.]+) weight_max=([0-9.]+) frac_clipped=([0-9.eE+-]+)(?: k3_kl=([0-9.eE+-]+) ess=([0-9.]+) logratio_mean=([0-9.eE+-]+) logratio_absmean=([0-9.eE+-]+) ntok=(\d+))?")
PHASE_RE = re.compile(r"PHASETIME step=(\d+) (.*)")
RLM_RE = re.compile(r"^RLMETRIC (.*)")


def parse_logs(logs):
    """Per-step records from head-pod stdout. Later logs override earlier (restarts)."""
    steps = {}          # step -> dict(loss, t(seconds-of-day), lendist, tis, phases)
    eval_steps = []     # in order of appearance (dedup)
    for lp in logs:
        if not os.path.exists(lp):
            print(f"  [warn] log missing: {lp}", file=sys.stderr)
            continue
        last_step = None
        with open(lp, errors="replace") as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    s = int(m.group(1)); rec = steps.setdefault(s, {})
                    rec["loss"] = float(m.group(2))
                    t = TS_RE.match(line)
                    if t:
                        mo, dd, hh, mi, ss, us = (int(x) for x in t.groups())
                        rec["t"] = ((dd * 24 + hh) * 60 + mi) * 60 + ss + us / 1e6
                    last_step = s
                    continue
                m = EVAL_RE.search(line)
                if m:
                    s = int(m.group(1)) if m.group(1) else 0
                    if s not in eval_steps:
                        eval_steps.append(s)
                    steps.setdefault(s, {})["eval"] = True
                    continue
                m = LEN_RE.search(line)
                if m:
                    s = int(m.group(1))
                    steps.setdefault(s, {})["len"] = dict(
                        n=int(m.group(2)), cap=int(m.group(3)), trunc=int(m.group(4)),
                        trunc_pct=float(m.group(5)), osl_mean=float(m.group(7)), isl_mean=float(m.group(8)))
                    last_step = s
                    continue
                m = TIS_RE.search(line)
                if m and last_step is not None:
                    g = m.groups()
                    steps.setdefault(last_step, {})["tis"] = dict(
                        wmean=float(g[0]), wmin=float(g[1]), wmax=float(g[2]), clip=float(g[3]),
                        k3=float(g[4]) if g[4] else None, ess=float(g[5]) if g[5] else None)
                    continue
                m = PHASE_RE.search(line)
                if m:
                    s = int(m.group(1)); ph = {}
                    for kv in m.group(2).split():
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            try: ph[k] = float(v)
                            except ValueError: pass
                    steps.setdefault(s, {})["phase"] = ph
                    continue
                m = RLM_RE.match(line)
                if m and last_step is not None:
                    toks = m.group(1).split()
                    if len(toks) >= 2:
                        steps.setdefault(last_step, {}).setdefault("rlm", {})[toks[0].strip("'\"")] = toks[1]
    return steps, eval_steps


def load_tpu(name, dumpdir, logs):
    files = sorted(glob.glob(os.path.join(dumpdir, "*eval_[0-9][0-9][0-9].jsonl")))
    if not files:
        raise SystemExit(f"no eval dumps in {dumpdir}")
    steps_log, eval_steps = parse_logs(logs)
    trained = sorted(s for s, r in steps_log.items() if "loss" in r)
    last_step = trained[-1] if trained else None
    evals = OrderedDict()
    prev = None
    for fp in files:
        k = int(re.search(r"eval_(\d{3})\.jsonl$", fp).group(1))  # file name = RL update step
        rows = [json.loads(l) for l in open(fp) if l.strip()]
        # step for eval k: 20k (eval period 20; k=0 is the pre-RL eval). The final eval runs at the last
        # trained step (250, or earlier if the run ended early): if 20k overshoots it, snap to it.
        # dumps are named eval_<train_steps>.jsonl; the periodic eval fires after the step counter has
        # advanced (20 -> "021"), so snap to the eval grid (multiples of 20, or the final step 250)
        step = k if (k % 20 == 0 or k == 250) else (k - 1 if ((k - 1) % 20 == 0 or k - 1 == 250) else k)
        if k in SKIP_DUMPS:
            print(f"  [skip] {os.path.basename(fp)} (explicitly excluded)", file=sys.stderr)
            continue
        if step in evals:
            # two dumps for one step (a first pass and a resumed re-run): keep the one from the LATER file
            # name is not meaningful, so prefer the re-run = the dump written last (mtime)
            if os.path.getmtime(fp) < evals[step].get("mtime", 0):
                print(f"  [skip] {os.path.basename(fp)}: older duplicate of step {step}", file=sys.stderr)
                continue
            print(f"  [replace] step {step}: {os.path.basename(fp)} supersedes the earlier dump", file=sys.stderr)
        n = len(rows)
        accs = np.array([float(r["acc"]) for r in rows])
        acc = accs.mean()
        boxed = np.mean([1.0 if (r.get("boxed") not in (None, "", "None")) else 0.0 for r in rows])
        nobox = np.mean([1.0 if r.get("fmt") in (0, 0.0, False) else 0.0 for r in rows]) if "fmt" in rows[0] else float("nan")
        chars = np.array([float(r.get("n_chars", float("nan"))) for r in rows])
        bykey = {r.get("qhash", r["i"]): float(r["acc"]) for r in rows}
        w2r = r2w = None
        if prev is not None:
            w2r = sum(1 for q, a in bykey.items() if a >= 1 and prev.get(q, 1) < 1)
            r2w = sum(1 for q, a in bykey.items() if a < 1 and prev.get(q, 0) >= 1)
        evals[step] = dict(k=k, n=n, acc=acc, wilson=wilson(acc, n), boxed=boxed, nobox=nobox, mtime=os.path.getmtime(fp),
                           chars_mean=float(np.nanmean(chars)), chars_p95=float(np.nanpercentile(chars, 95)),
                           w2r=w2r, r2w=r2w, bykey=bykey)
        if prev is None:
            prev = bykey  # transitions are vs step 0 (GPU convention)
    evals = OrderedDict(sorted(evals.items()))
    # per-step wall time: diff of consecutive "Train step" timestamps, excluding eval steps (both sides of an eval)
    times = {}
    for a, b in zip(trained, trained[1:]):
        if b == a + 1 and "t" in steps_log[a] and "t" in steps_log[b]:
            dt = steps_log[b]["t"] - steps_log[a]["t"]
            # drop sub-5s artifacts (duplicate step lines across restarts) and eval-adjacent steps
            if 5 < dt < 300 and not steps_log[a].get("eval") and not steps_log[b].get("eval"):  # >300 s = a restart/resume gap, not a step
                times[b] = dt
    return dict(name=name, evals=evals, steps=steps_log, times=times, trained=trained)


# ---------------------------------------------------------------- figures
def style_ax(ax, title, xlabel="RL update step", ylabel=None):
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def legend_platforms(ax, gpu_seeds, tpu_runs, loc="lower right"):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=GPU_C, lw=2, label=f"GPU reference ({len(gpu_seeds)} seeds, thin) · mean (thick)"),]
    for i, r in enumerate(tpu_runs):
        h.append(Line2D([], [], color=TPU_C, lw=2, marker=MARKERS[i % len(MARKERS)], ms=7, label=f"TPU v7x {r['name']}"))
    ax.legend(handles=h, loc=loc, fontsize=8.5)


def fig_accuracy(ax, gpu, tpu_runs, gate):
    steps = gpu["steps"]
    mean = [gpu["eval"][s]["acc_v2"]["mean"] * 100 for s in steps]
    lo = [gpu["eval"][s]["acc_v2"]["min"] * 100 for s in steps]
    hi = [gpu["eval"][s]["acc_v2"]["max"] * 100 for s in steps]
    ax.fill_between(steps, lo, hi, color=GPU_C, alpha=0.12, lw=0, label="GPU seed min–max")
    for sd in gpu["seeds"]:
        ax.plot(steps, [gpu["eval"][s]["per_seed"][sd]["acc_v2"] * 100 for s in steps], color=GPU_C, lw=0.9, alpha=0.6)
    ax.plot(steps, mean, color=GPU_C, lw=2.4)
    ax.annotate(f"GPU mean {mean[-1]:.2f}", (steps[-1], mean[-1]), xytext=(6, 4), textcoords="offset points", color=GPU_C, fontsize=8.5)
    for i, r in enumerate(tpu_runs):
        xs = list(r["evals"].keys()); ys = [r["evals"][s]["acc"] * 100 for s in xs]
        err = np.array([[ (r["evals"][s]["acc"] - r["evals"][s]["wilson"][0]) * 100, (r["evals"][s]["wilson"][1] - r["evals"][s]["acc"]) * 100] for s in xs]).T
        ax.errorbar(xs, ys, yerr=err, color=TPU_C, marker=MARKERS[i % len(MARKERS)], ms=7, lw=2, capsize=2.5, elinewidth=0.9, alpha=0.95)
        ax.annotate(f"{r['name']} {ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, -10 - 10 * i), textcoords="offset points", color=TPU_C, fontsize=8.5)
    if gate:
        ax.axhline(gate, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.annotate(f"gate {gate:.2f} (GPU step-250 mean − 1 pp)", (125, gate), xytext=(0, -11), textcoords="offset points", color=INK2, fontsize=8, ha="center")
    ax.set_xlim(-5, 262); ax.set_ylim(78, 95)
    style_ax(ax, "GSM8K test accuracy (scorer-v2, 1319 Qs, greedy)", ylabel="accuracy (%)")
    legend_platforms(ax, gpu["seeds"], tpu_runs)
    ax.text(0.01, 0.02, "TPU error bars: Wilson 95% (single run, n=1319); GPU band: min–max of 3 seeds",
            transform=ax.transAxes, fontsize=7.5, color=INK2)


def fig_delta(ax, gpu, tpu_runs):
    steps = gpu["steps"]
    m = [gpu["eval"][s]["delta"]["mean"] * 100 for s in steps]
    lo = [gpu["eval"][s]["delta_seed_t95"][0] * 100 for s in steps]
    hi = [gpu["eval"][s]["delta_seed_t95"][1] * 100 for s in steps]
    ax.fill_between(steps, lo, hi, color=GPU_C, alpha=0.12, lw=0)
    ax.plot(steps, m, color=GPU_C, lw=2.4)
    for i, r in enumerate(tpu_runs):
        xs = list(r["evals"].keys()); a0 = r["evals"][xs[0]]["acc"]
        ax.plot(xs, [(r["evals"][s]["acc"] - a0) * 100 for s in xs], color=TPU_C, marker=MARKERS[i % len(MARKERS)], ms=7)
    ax.axhline(0, color=GRID, lw=1)
    style_ax(ax, "Accuracy gain vs step 0 (pp)", ylabel="Δ accuracy vs step 0 (pp)")
    ax.text(0.01, 0.02, "GPU band: seed-mean ± t95", transform=ax.transAxes, fontsize=7.5, color=INK2)


def fig_boxed(ax, gpu, tpu_runs):
    steps = gpu["steps"]
    for sd in gpu["seeds"]:
        ax.plot(steps, [gpu["eval"][s]["per_seed"][sd]["boxed"] * 100 for s in steps], color=GPU_C, lw=0.9, alpha=0.6)
    ax.plot(steps, [np.mean([gpu["eval"][s]["per_seed"][sd]["boxed"] for sd in gpu["seeds"]]) * 100 for s in steps], color=GPU_C, lw=2.4)
    for i, r in enumerate(tpu_runs):
        xs = list(r["evals"].keys())
        ax.plot(xs, [r["evals"][s]["boxed"] * 100 for s in xs], color=TPU_C, marker=MARKERS[i % len(MARKERS)], ms=7)
    ax.set_ylim(80, 101)
    style_ax(ax, "Eval answers with \\boxed{} (%)", ylabel="boxed rate (%)")


def fig_transitions(ax, gpu, tpu_runs):
    steps = gpu["steps"][1:]
    for sd in gpu["seeds"]:
        ax.plot(steps, [gpu["eval"][s]["per_seed"][sd]["wrong_to_right"] for s in steps], color=GPU_C, lw=0.9, alpha=0.6)
        ax.plot(steps, [-gpu["eval"][s]["per_seed"][sd]["right_to_wrong"] for s in steps], color=GPU_C, lw=0.9, alpha=0.6, ls="--")
    ax.plot(steps, [np.mean([gpu["eval"][s]["per_seed"][sd]["wrong_to_right"] for sd in gpu["seeds"]]) for s in steps], color=GPU_C, lw=2.4)
    ax.plot(steps, [-np.mean([gpu["eval"][s]["per_seed"][sd]["right_to_wrong"] for sd in gpu["seeds"]]) for s in steps], color=GPU_C, lw=2.4, ls="--")
    for i, r in enumerate(tpu_runs):
        xs = [s for s in r["evals"] if r["evals"][s]["w2r"] is not None]
        ax.plot(xs, [r["evals"][s]["w2r"] for s in xs], color=TPU_C, marker=MARKERS[i % len(MARKERS)], ms=7)
        ax.plot(xs, [-r["evals"][s]["r2w"] for s in xs], color=TPU_C, marker=MARKERS[i % len(MARKERS)], ms=7, ls="--")
    ax.axhline(0, color=GRID, lw=1)
    style_ax(ax, "Questions changed vs step 0: wrong→right (solid, +) / right→wrong (dashed, −)", ylabel="questions")


def fig_train_len(ax, gpu, tpu_runs):
    # GPU: window means of rollout tokens (updates 1-20, 21-40, ...); TPU: LENDIST osl_mean per step
    for sd in gpu["seeds"]:
        w = gpu["train"].get(sd, {}).get("windows", [])
        xs = [int(x["updates"].split("-")[1]) - 10 for x in w]; ys = [x["tokens"] for x in w]
        ax.plot(xs, ys, color=GPU_C, marker="o", ms=6, lw=1.2, alpha=0.7)
    for i, r in enumerate(tpu_runs):
        xs = sorted(s for s, rec in r["steps"].items() if "len" in rec)
        if not xs: continue
        ys = [r["steps"][s]["len"]["osl_mean"] for s in xs]
        ax.plot(xs, ys, color=TPU_C, lw=1.2, alpha=0.9)
        # 20-step window means as markers (same granularity as GPU)
        for a in range(0, 260, 20):
            w = [r["steps"][s]["len"]["osl_mean"] for s in xs if a < s <= a + 20]
            if w: ax.plot([a + 10], [np.mean(w)], color=TPU_C, marker=MARKERS[i % len(MARKERS)], ms=7, ls="none")
    style_ax(ax, "Train rollout mean length (tokens; GPU = 20-update window means)", ylabel="tokens")


def fig_step_time(ax, gpu, tpu_runs):
    for i, r in enumerate(tpu_runs):
        xs = sorted(r["times"]); ys = [r["times"][s] for s in xs]
        if not xs: continue
        ax.plot(xs, ys, color=TPU_C, lw=1.2, alpha=0.9, marker=MARKERS[i % len(MARKERS)], ms=3)
        med = np.median(ys); ax.axhline(med, color=TPU_C, lw=1, ls=":")
        ax.annotate(f"TPU {r['name']} median {med:.1f}s (n={len(ys)})", (xs[-1], med), xytext=(-4, 4), textcoords="offset points", ha="right", color=TPU_C, fontsize=8)
    for j, sd in enumerate(gpu["seeds"]):
        t = gpu["timing"].get(sd, {}).get("internal_step_s", {})
        if "median" in t:
            ax.axhline(t["median"], color=GPU_C, lw=1, ls="--", alpha=0.8)
            ax.annotate(f"GPU {sd} median {t['median']:.1f}s", (0, t["median"]), xytext=(4, 3 + 0 * j), textcoords="offset points", color=GPU_C, fontsize=8)
    style_ax(ax, "Wall time per RL update (s), non-eval steps", ylabel="seconds / step")
    ax.set_ylim(bottom=0)


def fig_tis(ax, gpu, tpu_runs):
    for i, r in enumerate(tpu_runs):
        xs = sorted(s for s, rec in r["steps"].items() if "tis" in rec)
        if not xs: continue
        ax.plot(xs, [r["steps"][s]["tis"]["wmax"] for s in xs], color=TPU_C, lw=1.2, marker=MARKERS[i % len(MARKERS)], ms=3)
        ax.plot(xs, [r["steps"][s]["tis"]["wmin"] for s in xs], color=TPU_C, lw=1.2, ls="--")
        ax.plot(xs, [r["steps"][s]["tis"]["wmean"] for s in xs], color=TPU_C, lw=1.2, ls=":")
    lo = min(gpu["train"][sd]["is_max_range"][0] for sd in gpu["seeds"] if "is_max_range" in gpu["train"].get(sd, {}))
    hi = max(gpu["train"][sd]["is_max_range"][1] for sd in gpu["seeds"] if "is_max_range" in gpu["train"].get(sd, {}))
    ax.axhspan(lo, hi, color=GPU_C, alpha=0.15, lw=0)
    ax.annotate(f"GPU per-step max IS weight range {lo:.2f}–{hi:.2f}", (2, hi), xytext=(4, 3), textcoords="offset points", color=GPU_C, fontsize=8)
    ax.axhline(3.0, color=INK2, lw=1, ls=(0, (4, 3))); ax.annotate("TIS cap 3.0", (250, 3.0), xytext=(-4, 3), textcoords="offset points", ha="right", color=INK2, fontsize=8)
    ax.axhline(1.0, color=GRID, lw=1)
    style_ax(ax, "TIS weights per step: TPU max (solid) / min (dashed) / mean (dotted)", ylabel="weight")
    ax.set_ylim(0, 3.3)


def fig_k3_ess(ax, gpu, tpu_runs):
    have = False
    for i, r in enumerate(tpu_runs):
        xs = sorted(s for s, rec in r["steps"].items() if rec.get("tis", {}).get("k3") is not None)
        if not xs: continue
        have = True
        ax.plot(xs, [r["steps"][s]["tis"]["k3"] for s in xs], color=TPU_C, lw=1.2, marker=MARKERS[i % len(MARKERS)], ms=3)
    lo = min(gpu["train"][sd]["k3_kl_range"][0] for sd in gpu["seeds"]); hi = max(gpu["train"][sd]["k3_kl_range"][1] for sd in gpu["seeds"])
    ax.axhspan(lo, hi, color=GPU_C, alpha=0.15, lw=0)
    ax.annotate(f"GPU k3 KL range {lo:.1e}–{hi:.1e}", (2, hi), xytext=(4, 3), textcoords="offset points", color=GPU_C, fontsize=8)
    style_ax(ax, "Sampler↔trainer k3 KL per step (token mean)", ylabel="k3 KL")
    ax.set_xlim(-5, 262)
    if not have:
        ax.text(0.5, 0.5, "TPU k3/ESS logging added for seeds 43/44", ha="center", transform=ax.transAxes, color=INK2)
    ax.set_yscale("log")


def fig_loss(ax, gpu, tpu_runs):
    for i, r in enumerate(tpu_runs):
        xs = sorted(s for s, rec in r["steps"].items() if "loss" in rec)
        ax.plot(xs, [r["steps"][s]["loss"] for s in xs], color=TPU_C, lw=1, marker=MARKERS[i % len(MARKERS)], ms=3)
    style_ax(ax, "TPU training loss per update (−A·w·logπ, token mean)", ylabel="loss")


def summary_md(gpu, tpu_runs, gate, out):
    L = ["# gemma4-E2B GSM8K parity summary", "",
         "| step | GPU mean ± SD (min–max) | " + " | ".join(f"TPU {r['name']}" for r in tpu_runs) + " | TPU mean − GPU mean |", "|---:|---|" + "---|" * len(tpu_runs) + "---|"]
    allsteps = sorted(set(gpu["steps"]) | set(s for r in tpu_runs for s in r["evals"]))
    for s in allsteps:
        g = gpu["eval"].get(s)
        gs = f"{g['acc_v2']['mean']*100:.2f} ± {g['acc_v2']['sd']*100:.2f} ({g['acc_v2']['min']*100:.2f}–{g['acc_v2']['max']*100:.2f})" if g else "—"
        ts = []; vals = []
        for r in tpu_runs:
            e = r["evals"].get(s)
            if e: ts.append(f"{e['acc']*100:.2f} [{e['wilson'][0]*100:.1f}, {e['wilson'][1]*100:.1f}]"); vals.append(e["acc"])
            else: ts.append("—")
        d = f"{(np.mean(vals)-g['acc_v2']['mean'])*100:+.2f}" if (vals and g) else "—"
        L.append(f"| {s} | {gs} | " + " | ".join(ts) + f" | {d} |")
    L += ["", f"Gate: step-250 TPU mean ≥ {gate:.2f} (GPU step-250 mean {gpu['eval'][250]['acc_v2']['mean']*100:.2f} − 1 pp)."]
    fin = [r["evals"][250]["acc"] for r in tpu_runs if 250 in r["evals"]]
    if fin:
        m = np.mean(fin) * 100
        L.append(f"TPU step-250: {', '.join(f'{v*100:.2f}' for v in fin)} → mean **{m:.2f}** → **{'PASS' if m >= gate else 'FAIL'}** ({len(fin)} of 3 seeds).")
    else:
        L.append("TPU step-250: not yet available.")
    L += ["", "## Step time (non-eval updates)", "", "| run | median s | mean s | p90 s | n |", "|---|---:|---:|---:|---:|"]
    for sd in gpu["seeds"]:
        t = gpu["timing"].get(sd, {}).get("internal_step_s", {})
        if t: L.append(f"| GPU {sd} | {t['median']:.1f} | {t['mean']:.1f} | {t['p90']:.1f} | {t['n']} |")
    for r in tpu_runs:
        v = np.array(list(r["times"].values()))
        if len(v): L.append(f"| TPU {r['name']} | {np.median(v):.1f} | {v.mean():.1f} | {np.percentile(v,90):.1f} | {len(v)} |")
    open(os.path.join(out, "parity_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True); ap.add_argument("--tpu", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--skip-dumps", default="", help="comma-separated dump indices to ignore, e.g. 160")
    a = ap.parse_args()
    SKIP_DUMPS.update(int(x) for x in a.skip_dumps.split(",") if x.strip())
    os.makedirs(a.out, exist_ok=True)
    gpu = load_gpu(a.gpu)
    gate = gpu["eval"][250]["acc_v2"]["mean"] * 100 - 1.0
    runs = []
    for spec in a.tpu:
        name, rest = spec.split("=", 1)
        dumpdir, _, logs = rest.partition(":")
        runs.append(load_tpu(name, dumpdir, [l for l in logs.split(",") if l]))
        r = runs[-1]
        print(f"{name}: evals at steps {list(r['evals'].keys())}, trained steps {len(r['trained'])}, timed steps {len(r['times'])}")
    panels = [("01_accuracy", fig_accuracy, (gpu, runs, gate)), ("02_delta", fig_delta, (gpu, runs)),
              ("03_boxed", fig_boxed, (gpu, runs)), ("04_transitions", fig_transitions, (gpu, runs)),
              ("05_train_len", fig_train_len, (gpu, runs)), ("06_step_time", fig_step_time, (gpu, runs)),
              ("07_tis_weights", fig_tis, (gpu, runs)), ("08_k3_kl", fig_k3_ess, (gpu, runs)), ("09_loss", fig_loss, (gpu, runs))]
    for name, fn, args in panels:
        fig, ax = plt.subplots(figsize=(9, 4.6)); fn(ax, *args); fig.tight_layout()
        fig.savefig(os.path.join(a.out, name + ".png"), dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(22, 14.5))
    for (name, fn, args), ax in zip(panels, axes.flat):
        fn(ax, *args)
    fig.suptitle("gemma4-E2B · GSM8K · GRPO+TIS · TPU v7x (MaxText+tunix) vs GPU reference", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(a.out, "00_overview.png"), dpi=110); plt.close(fig)
    summary_md(gpu, runs, gate, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
