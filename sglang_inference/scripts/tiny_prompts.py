#!/usr/bin/env python3
"""tiny_prompts.py - build prompt sets with EXACT token lengths for sliding-window / page /
chunk-boundary tests of SGLang PD disaggregation.

  python3 tiny_prompts.py --model-path /path/to/hf/snapshot --out prompts.jsonl \
      [--lengths 1,2,16,63,64,65,...] [--seed 0] [--max-new-tokens 160] \
      [--input-ids --id-min 1000 --id-max 30000] [--no-special-tokens] [--trust-remote-code]

Text mode: lengths are hit iteratively with the model tokenizer (counted the way the server
tokenizes `text`, i.e. add_special_tokens=True unless --no-special-tokens).  Prompts >= 32
tokens carry a needle sentence "The secret code word is <word>." near the middle and end with
"Secret word:" so a greedy model will likely echo the needle.  Shorter prompts are plain words.
Input-ids mode: `input_ids` are random ids in [--id-min, --id-max] (never special ids), exact
by construction; needle/tail token ids are spliced in when they fit; `text` is the decode.
Each output line: {"id": "tiny<L>", "isl": exact, "target": L, "needle": word|null,
"text": ..., ["input_ids": [...],] "max_new_tokens": 160}.  Exit 1 if any length is inexact.
"""
import argparse
import json
import os
import random
import sys

DEFAULT_LENGTHS = ("1,2,16,63,64,65,100,127,128,129,130,191,192,193,255,256,257,383,384,385,"
                   "511,512,513,1023,1024,1025,8191,8192,8193,8255,8256,8257,16383,16384,16385")
FILLER = ("the quick brown fox jumps over a lazy dog while the river flows gently past the old "
          "stone bridge and children play in the warm afternoon sun near a small market where "
          "farmers sell fresh bread apples honey and cheese to travelers who rest under tall "
          "green trees before the long road home").split()
TUNE = [" a", " the", " of", " to", " and", " in", " is", " it", " on", " at", " we", " I", " so", ","]
NEEDLES = ("lantern velvet pistachio harbor quartz meadow saffron compass juniper marble orchid "
           "thistle walnut ember falcon glacier indigo kestrel lagoon nutmeg obsidian raven sable "
           "tundra umber willow zephyr cobalt dahlia fennel garnet pelican tulip anchor").split()


def parse_lengths(s):
    try:
        ls = [int(x) for x in s.split(",") if x.strip()]
    except ValueError:
        sys.exit(f"error: --lengths must be comma-separated ints, got {s!r}")
    if not ls or any(l < 1 for l in ls):
        sys.exit("error: --lengths must be positive")
    return ls


def load_tokenizer(path, trust):
    if not os.path.isdir(path):
        sys.exit(f"error: --model-path {path!r} is not a directory")
    try:
        from transformers import AutoTokenizer  # lazy: only needed here
    except ImportError:
        sys.exit("error: transformers not importable; run with the sglang venv python")
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
    except Exception as e:  # noqa: BLE001 - surface any loader failure clearly
        sys.exit(f"error: cannot load tokenizer from {path}: {type(e).__name__}: {e}")


def gen_filler(rng, n_words):
    """n_words of pseudo-random filler, grouped into capitalised sentences."""
    words, i = [], 0
    while i < n_words:
        n = min(rng.randint(8, 14), n_words - i)
        sent = [rng.choice(FILLER) for _ in range(n)]
        sent[0] = sent[0].capitalize()
        sent[-1] += "."
        words += sent
        i += n
    return " ".join(words)


def fit(count, rng, prefix, tail, L, max_iter=400):
    """Return prefix + filler + tail with exactly L tokens (or the closest reachable)."""
    def join(filler):
        mid = (" " if prefix and filler else "") + filler
        return prefix + mid + tail
    fixed = count(join(""))
    words = gen_filler(rng, max(0, int((L - fixed) * 0.7))).split()
    for _ in range(max_iter):
        text = join(" ".join(words))
        c = count(text)
        if c == L:
            return text
        if c > L:
            drop = max(1, int((c - L) * 0.6))
            if not words:
                return text  # fixed parts alone already exceed L
            words = words[:-drop]
            continue
        deficit = L - c
        if deficit > 8:
            words += gen_filler(rng, max(1, int(deficit * 0.6))).split()
            continue
        # fine tuning: pick the candidate that gets closest to L without exceeding it
        best, best_c = None, c
        for cand in TUNE + [" " + w for w in FILLER]:
            trial = join(" ".join(words) + cand)
            tc = count(trial)
            if best_c < tc <= L:
                best, best_c = cand, tc
                if tc == L:
                    break
        if best is None:
            return text  # no single word closes the gap exactly
        if words:
            words[-1] += best
        else:
            words.append(best.strip())
    return join(" ".join(words))


def build_text(count, rng, L, needle):
    if needle is None:
        return fit(count, rng, "", "", L)
    sentence = f"The secret code word is {needle}."
    # pre-filler gets exactly half of the filler budget so the needle lands mid-prompt
    bos = count("")
    budget = max(0, L - (count(sentence + " Secret word:") - bos))
    pre = fit(count, rng, "", "", bos + budget // 2)
    prefix = pre + " " + sentence if pre else sentence
    return fit(count, rng, prefix, " Secret word:", L)


def build_ids(tok, rng, L, needle, id_min, id_max):
    special = set(getattr(tok, "all_special_ids", None) or [])
    ids = []
    while len(ids) < L:
        t = rng.randint(id_min, id_max)
        if t not in special:
            ids.append(t)
    if needle is not None:
        nd = tok.encode(f" The secret code word is {needle}.", add_special_tokens=False)
        tl = tok.encode(" Secret word:", add_special_tokens=False)
        if len(nd) + len(tl) + 2 <= L:
            mid = (L - len(tl) - len(nd)) // 2
            ids[mid:mid + len(nd)] = nd
            ids[L - len(tl):] = tl
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="HF snapshot dir with tokenizer files")
    ap.add_argument("--out", default="prompts.jsonl")
    ap.add_argument("--lengths", default=DEFAULT_LENGTHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160, help="160 > 128 so decode crosses a SWA window")
    ap.add_argument("--needle-min", type=int, default=32, help="min length that gets a needle sentence")
    ap.add_argument("--input-ids", action="store_true", help="emit exact input_ids arrays instead of fitting text")
    ap.add_argument("--id-min", type=int, default=1000)
    ap.add_argument("--id-max", type=int, default=30000)
    ap.add_argument("--no-special-tokens", action="store_true",
                    help="count text with add_special_tokens=False (server default adds BOS where the tokenizer has one)")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    lengths = parse_lengths(args.lengths)
    tok = load_tokenizer(args.model_path, args.trust_remote_code)
    vocab = max(len(tok), getattr(tok, "vocab_size", 0) or 0, max(tok.get_vocab().values()) + 1)
    if args.input_ids and not (0 <= args.id_min <= args.id_max < vocab):
        sys.exit(f"error: need 0 <= --id-min <= --id-max < vocab size ({vocab})")
    add_special = not args.no_special_tokens

    def count(text):
        return len(tok(text, add_special_tokens=add_special).input_ids)

    rng = random.Random(args.seed)
    needles = rng.sample(NEEDLES, len(NEEDLES))
    rows, n_inexact = [], 0
    print(f"{'id':<10}{'target':>8}{'isl':>8}  exact  needle")
    for i, L in enumerate(lengths):
        needle = needles[i % len(needles)] if L >= args.needle_min else None
        prng = random.Random(f"{args.seed}:{L}")  # per-length rng: stable even if the list changes
        row = {"id": f"tiny{L}", "isl": None, "target": L, "needle": needle, "text": None,
               "max_new_tokens": args.max_new_tokens}
        if args.input_ids:
            ids = build_ids(tok, prng, L, needle, args.id_min, args.id_max)
            row.update({"text": tok.decode(ids), "input_ids": ids, "isl": len(ids)})
            if needle is not None and f"The secret code word is {needle}." not in row["text"]:
                row["needle"] = None  # did not fit / decode mangled it
        else:
            text = build_text(count, prng, L, needle)
            row.update({"text": text, "isl": count(text)})
            if needle is not None and text.count(needle) != 1:
                row["needle"] = None
        exact = row["isl"] == L
        n_inexact += not exact
        rows.append(row)
        print(f"{row['id']:<10}{L:>8}{row['isl']:>8}  {'yes' if exact else 'NO ':<5}  {row['needle']}", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    mode = "input_ids" if args.input_ids else f"text (add_special_tokens={add_special})"
    print(f"\n{len(rows)} prompts, {len(rows) - n_inexact} exact, {n_inexact} inexact [{mode}] -> {args.out}")
    if n_inexact:
        print("warning: some lengths could not be hit exactly (BOS tokenizers cannot produce isl<2 text)",
              file=sys.stderr)
    sys.exit(1 if n_inexact else 0)


if __name__ == "__main__":
    main()
