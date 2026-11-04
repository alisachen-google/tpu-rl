#!/usr/bin/env python3
"""phasetime_summary.py <raw.log> [min_step=2]: per-step phase table from PHASETIME lines + medians."""
import re, sys, statistics as st
f = sys.argv[1]; m0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2
rows = {}
for line in open(f, errors="replace"):
  m = re.search(r"PHASETIME step=(\d+) (.*)", line)
  if not m: continue
  d = {}
  for kv in m.group(2).split():
    k, _, v = kv.partition("=")
    try: d[k] = float(v)
    except ValueError: pass
  rows[int(m.group(1))] = d
print("step    gen   adv  prep  trainer  total   acc_v2")
tab = []
for s in sorted(rows):
  d = rows[s]
  gen = d["engine_generate"]; adv = d["generate_and_advantage"] - gen; prep = d["prepare_data"] - d["generate_and_advantage"]
  tr = d["micro_batch_steps"] - d["prepare_data"]; tot = d["run_global_step"]
  print(f"{s:4d} {gen:6.1f} {adv:5.1f} {prep:5.1f} {tr:8.1f} {tot:6.1f}   {d.get('acc_v2_mean', d.get('reward_v2_mean', float('nan'))):.3f}")
  if s >= m0: tab.append((gen, adv, prep, tr, tot))
if tab:
  med = [st.median(c) for c in zip(*tab)]
  print(f"median(step>={m0}, n={len(tab)}): gen {med[0]:.1f}  adv {med[1]:.1f}  prep {med[2]:.1f}  trainer {med[3]:.1f}  total {med[4]:.1f}")
