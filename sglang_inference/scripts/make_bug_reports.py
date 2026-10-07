#!/usr/bin/env python3
"""Split results/FINDINGS.md into one GitHub-issue-style file per finding under results/bug_reports/.

Each F-N section (## F-N — title ... up to the next ## or table row) becomes bug_reports/F-N.md with an issue header
(environment, status) prepended; an INDEX.md summarises severity/status. Idempotent.
"""
import re, os, sys, datetime

ROOT = os.path.expanduser("~/pd_disagg/results")
ENV = ("sglang-torchtpu main `09d1244c26` (2026-10-07), sglang 0.5.13.dev7249, torch-tpu 0.1.1+release.2026.9.22.20260927223741, "
       "libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, transformers 5.8.1; hosts wyzhang-tpu7x-16 a/b (TPU v7x, 4 chips / 8 cores "
       "per host, single-host isolation, TP8/EP8), 1 prefill + 1 decode via raiden, Python mini-LB. Recipes: `sglang_inference/recipes/`.")
BUG_CLASS = {"F-1": "bug", "F-2": "bug (missing compile policy / perf)", "F-3": "test-methodology note (not a bug)", "F-4": "bug candidate (correctness, under isolation)",
             "F-5": "bug (crash)", "F-6": "bug candidate (hang, repro pending)", "F-7": "observation (recovered; misleading log line)", "F-8": "bug (crash)"}


def main():
    text = open(f"{ROOT}/FINDINGS.md").read()
    out = f"{ROOT}/bug_reports"; os.makedirs(out, exist_ok=True)
    rows = re.findall(r"^\| (F-\d+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", text, re.M)
    sections = re.split(r"^## (?=F-\d+ — )", text, flags=re.M)[1:]
    secs = {}
    for s in sections:
        fid = s.split(" ", 1)[0]
        body = re.split(r"^\| F-\d+ \|", s, flags=re.M)[0]  # stop at the next summary-table row
        body = re.split(r"^## (?!F-)", body, flags=re.M)[0]
        secs.setdefault(fid, "")
        secs[fid] += ("## " + body if not secs[fid] else "\n" + body)
    # updates that appear after other sections (e.g. "**Update**", "**F-7 resolution**", "**F-8 traceback**") are attached by their F-id mention
    for fid in secs:
        for m in re.finditer(r"^\*\*(" + fid + r")[^*]*\*\*.*?(?=^\*\*F-|^## |^\| F-|\Z)", text, re.M | re.S):
            if m.group(0) not in secs[fid]:
                secs[fid] += "\n" + m.group(0)
    index = ["# Bug reports — sglang-torchtpu 1P/1D on TPU v7x", "", f"Generated {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC from FINDINGS.md.", "",
             "| ID | Class | First seen (UTC) | Severity | Area | Status | File |", "|---|---|---|---|---|---|---|"]
    for fid, date, sev, area, status in rows:
        body = secs.get(fid, "(no detail section)")
        title = re.search(r"^## " + fid + r" — (.+)$", body, re.M)
        title = title.group(1).strip() if title else fid
        with open(f"{out}/{fid}.md", "w") as f:
            f.write(f"# [{fid}] {title}\n\n**Class:** {BUG_CLASS.get(fid, 'bug')}  \n**Severity:** {sev.strip()}  \n**Area:** {area.strip()}  \n"
                    f"**Status:** {status.strip()}  \n**First observed:** {date.strip()}  \n**Target repo for upstream filing:** sgl-project/sglang-torchtpu (label `help wanted`)\n\n"
                    f"## Environment\n{ENV}\n\n{body.strip()}\n")
        index.append(f"| {fid} | {BUG_CLASS.get(fid, 'bug')} | {date.strip()} | {sev.strip()} | {area.strip()} | {status.strip()} | [{fid}.md]({fid}.md) |")
    open(f"{out}/INDEX.md", "w").write("\n".join(index) + "\n")
    print(f"wrote {len(rows)} reports to {out}")


if __name__ == "__main__":
    main()
