#!/usr/bin/env python3
"""Minimal pure-python TensorBoard tfevents scalar reader (no TF/TB deps).

Usage: tb_scalars.py <events file or gs:// path> [tag-regex]
Prints: step tag value   for every scalar (simple_value or tensor float) matching the regex.
"""
import re, struct, subprocess, sys, tempfile, os

def varint(b, i):
  r = 0; s = 0
  while True:
    c = b[i]; i += 1; r |= (c & 0x7f) << s; s += 7
    if not c & 0x80: return r, i

def fields(b):
  """Yield (field_no, wire_type, value/bytes) for a protobuf message."""
  i = 0; n = len(b)
  while i < n:
    key, i = varint(b, i); f, w = key >> 3, key & 7
    if w == 0: v, i = varint(b, i)
    elif w == 1: v = b[i:i+8]; i += 8
    elif w == 2: l, i = varint(b, i); v = b[i:i+l]; i += l
    elif w == 5: v = b[i:i+4]; i += 4
    else: raise ValueError(f"wire type {w}")
    yield f, w, v

def tensor_float(tb):
  # TensorProto: dtype(1) float_val(5, packed or repeated) tensor_content(4)
  dtype = None; vals = []; content = b""
  for f, w, v in fields(tb):
    if f == 1: dtype = v
    elif f == 5:
      if w == 2: vals += list(struct.unpack(f"<{len(v)//4}f", v))
      elif w == 5: vals.append(struct.unpack("<f", v)[0])
    elif f == 4: content = v
  if vals: return vals[0]
  if content and dtype == 1 and len(content) >= 4: return struct.unpack("<f", content[:4])[0]
  if content and dtype == 2 and len(content) >= 8: return struct.unpack("<d", content[:8])[0]
  return None

def read(path):
  if path.startswith("gs://"):
    tmp = tempfile.NamedTemporaryFile(delete=False).name
    subprocess.run(["gsutil", "-q", "cp", path, tmp], check=True); path = tmp
  b = open(path, "rb").read(); i = 0
  while i + 12 <= len(b):
    (l,) = struct.unpack("<Q", b[i:i+8]); i += 12  # len + crc
    rec = b[i:i+l]; i += l + 4  # data + crc
    step = 0; summ = None
    for f, w, v in fields(rec):
      if f == 3: step = v
      elif f == 5: summ = v
    if summ is None: continue
    for f, w, v in fields(summ):
      if f != 1: continue
      tag = None; val = None
      for ff, ww, vv in fields(v):
        if ff == 1: tag = vv.decode()
        elif ff == 2: val = struct.unpack("<f", vv)[0]
        elif ff == 8: val = tensor_float(vv)
      if tag is not None and val is not None: yield step, tag, val

if __name__ == "__main__":
  pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
  for step, tag, val in read(sys.argv[1]):
    if pat is None or pat.search(tag): print(step, tag, val)
