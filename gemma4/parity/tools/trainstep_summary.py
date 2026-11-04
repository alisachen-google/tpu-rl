#!/usr/bin/env python3
"""trainstep_summary.py <raw.log> [min_update=2]: per-update step time from consecutive
'peft_trainer ... Train step N training loss' timestamps (for manifests without the PHASETIME patch).
Update k's time = timestamp(Train step k+1) - timestamp(Train step k)."""
import re, sys, statistics as st, datetime as dt
f = sys.argv[1]; m0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ts = {}
for line in open(f, errors="replace"):
  m = re.search(r"^I(\d{2})(\d{2}) (\d{2}:\d{2}:\d{2}\.\d+) .*Train step (\d+) training loss", line)
  if m:
    t = dt.datetime.strptime(f"2026-{m.group(1)}-{m.group(2)} {m.group(3)}", "%Y-%m-%d %H:%M:%S.%f")
    ts.setdefault(int(m.group(4)), t)
steps = sorted(ts); rows = []
for a, b in zip(steps, steps[1:]):
  d = (ts[b] - ts[a]).total_seconds(); rows.append((a, d))
  print(f"update {a:3d}: {d:6.1f} s")
sel = [d for a, d in rows if a >= m0]
if sel: print(f"median(update>={m0}, n={len(sel)}): {st.median(sel):.1f} s   min {min(sel):.1f}  max {max(sel):.1f}")
