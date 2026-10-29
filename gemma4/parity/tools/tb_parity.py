#!/usr/bin/env python3
"""Parse a TPU-RL TensorBoard eventfile (pure stdlib, no numpy/tf) and print a sustained-parity
verdict for check_numbers / score/mean / kl against the GPU reference band (check_numbers 0.50-0.57,
target ~0.55; GPU score 0.54-0.65). Usage: tb_parity.py <eventfile>"""
import sys, struct
from collections import defaultdict


def rv(b, i):
    r = 0; s = 0
    while True:
        x = b[i]; i += 1; r |= (x & 0x7f) << s
        if not x & 0x80:
            break
        s += 7
    return r, i


def fields(b):
    i = 0; out = []
    while i < len(b):
        key, i = rv(b, i); fn = key >> 3; wt = key & 7
        if wt == 0:
            v, i = rv(b, i); out.append((fn, 0, v))
        elif wt == 1:
            out.append((fn, 1, b[i:i + 8])); i += 8
        elif wt == 2:
            ln, i = rv(b, i); out.append((fn, 2, b[i:i + ln])); i += ln
        elif wt == 5:
            out.append((fn, 5, b[i:i + 4])); i += 4
        else:
            return out
    return out


def parse(path):
    data = open(path, "rb").read()
    series = defaultdict(list); i = 0
    while i < len(data):
        if i + 12 > len(data):
            break
        ln = struct.unpack("<Q", data[i:i + 8])[0]; i += 12
        rec = data[i:i + ln]; i += ln + 4
        step = 0; summ = None
        for fn, wt, v in fields(rec):
            if fn == 2 and wt == 0:
                step = v
            elif fn == 5 and wt == 2:
                summ = v
        if summ is None:
            continue
        for fn, wt, v in fields(summ):
            if fn == 1 and wt == 2:
                tag = None; sv = None
                for a, b2, c in fields(v):
                    if a == 1 and b2 == 2:
                        tag = c.decode("utf-8", "replace")
                    elif a == 2 and b2 == 5:
                        sv = struct.unpack("<f", c)[0]
                if tag and sv is not None:
                    series[tag].append((int(step), sv))
    return series


def main():
    s = parse(sys.argv[1])

    def curve(tag):
        pts = sorted(s.get(tag, []), key=lambda p: p[0])
        return pts, [v for _, v in pts]

    cn_pts, cn = curve("rewards/train/check_numbers")
    sc_pts, sc = curve("rewards/train/score/mean")
    kl_pts, kl = curve("actor/train/kl")
    if not cn:
        print("NO check_numbers series found; tags:", sorted(s)[:40]); return
    n = len(cn)
    f10 = sum(cn[:10]) / min(10, n); l10 = sum(cn[-10:]) / min(10, n)
    ov = sum(cn) / n
    print(f"check_numbers: steps={n} first10={f10:.4f} last10={l10:.4f} overall={ov:.4f} "
          f"min={min(cn):.4f} max={max(cn):.4f}  trend(last-first)={l10 - f10:+.4f}")
    if sc:
        print(f"score/mean:    overall={sum(sc)/len(sc):.4f} last10={sum(sc[-10:])/min(10,len(sc)):.4f} "
              f"min={min(sc):.4f} max={max(sc):.4f}")
    if kl:
        print(f"kl:            overall={sum(kl)/len(kl):.6f} max={max(kl):.6f}  "
              f"({'FLAT ~0' if max(kl) < 1e-3 else 'NONZERO — check divergence'})")
    print("per-10-step check_numbers:", " ".join(f"{sum(cn[k:k+10])/len(cn[k:k+10]):.3f}" for k in range(0, n, 10)))
    # verdict vs GPU-parity band
    band_lo, band_hi, tgt = 0.50, 0.57, 0.55
    in_band = band_lo <= ov <= band_hi
    drift = abs(l10 - f10)
    print("\n=== PARITY VERDICT (GPU ref: check_numbers 0.50-0.57, ~0.55; score 0.54-0.65) ===")
    print(f"  overall check_numbers {ov:.4f} {'IN' if in_band else 'OUT of'} [0.50,0.57]  "
          f"| drift {drift:.4f} ({'flat' if drift < 0.02 else 'DRIFTING'})  "
          f"| kl {'ok' if (not kl or max(kl) < 1e-3) else 'NONZERO'}")
    verdict = "SUSTAINED GPU PARITY" if (in_band and drift < 0.03 and (not kl or max(kl) < 1e-3)) \
        else ("BELOW PARITY FLOOR" if ov < band_lo else "CHECK (drift/kl)")
    print(f"  => {verdict}")


if __name__ == "__main__":
    main()
