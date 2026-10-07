#!/usr/bin/env python3
"""params_matrix.py - differential test of sampling / decoding parameters on SGLang /generate.

  python3 params_matrix.py --a http://golden:port [--b http://pd-router:port] \
      [--prompt-file prompts.jsonl [--prompt-id tiny512]] [--cases greedy_baseline,n_2] \
      [--max-new-tokens 128] [--stop-token-id 13] [--bias-token-id 262] [--timeout 300] \
      [--report params_report.json]

Every case is sent to endpoint A (golden) and, if given, endpoint B (PD).  Per endpoint we record
http status, text, completion_tokens, finish_reason, error, first 5 output logprobs, elapsed.
A request that hits --timeout is reported as HANG?.  A vs B must agree on status, text,
completion_tokens and finish_reason (seeded-sampling case: status/finish_reason only; logprob case:
counts equal and first 5 logprobs within 1e-2).  Per-endpoint self checks: top_k_1 == greedy,
min_new_tokens honoured, stream == non-stream, grammar cases return a clean error or 200 (no hang).
Exit 1 if any case differs between endpoints, fails a self check, or hangs.
"""
import argparse
import json
import sys
import time

import requests

BUILTIN_PROMPT = (
    "The lighthouse keeper at Marrow Point had kept the same routine for thirty-one years. Each "
    "evening he climbed the one hundred and twelve iron steps, polished the great lens with a cloth "
    "his mother had sewn, and lit the lamp exactly nine minutes before sunset. The fishing boats "
    "relied on him, though few of the younger crews knew his name. In winter the wind came off the "
    "water like a wall, and the tower hummed with a low note that he had learned to read the way "
    "other men read barometers. When the hum rose half a tone, he knew ice was forming on the "
    "northern rocks; when it fell, the fog would be in by midnight. He kept a ledger of these "
    "observations in a hand so small that the harbor master once mistook it for printed type. "
    "The ledger also recorded every ship that passed, the color of its hull, the number of its "
    "lights, and whether it answered his signal. Over the decades, the record filled fourteen "
    "volumes, bound in canvas and stacked beneath the stair. Nobody asked to read them, and he "
    "never offered. On the morning the automation crew arrived with their crates of relays and "
    "photocells, he made them tea, showed them where the spare wicks were kept, and asked only "
    "that the lens be polished on Thursdays. They promised, politely, and forgot by the following "
    "spring. The secret code word is lantern. The ledgers remain under the stair to this day.\n\n"
    "Summarize the passage above in detail, then list three follow-up questions.\n\nSummary:")

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def load_prompt(path, pid):
    if not path:
        return "builtin", BUILTIN_PROMPT
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    sys.exit(f"error: {path}:{ln}: invalid JSON: {e}")
    rows = [r for r in rows if isinstance(r.get("text"), str)]
    if not rows:
        sys.exit(f"error: no prompt with 'text' in {path}")
    if pid is not None:
        match = [r for r in rows if str(r.get("id")) == pid]
        if not match:
            sys.exit(f"error: prompt id {pid!r} not found in {path}")
        rows = match
    return str(rows[0].get("id", "line1")), rows[0]["text"]


def build_cases(args):
    def sp(**kw):
        d = {"temperature": 0, "max_new_tokens": args.max_new_tokens}
        d.update(kw)
        return d
    stop = ["Secret", "\n\n"]
    cases = [
        dict(name="greedy_baseline", sp=sp()),
        dict(name="top_k_1", sp=sp(temperature=0.7, top_k=1), check="equals_greedy"),
        dict(name="repetition_penalty_1.3", sp=sp(repetition_penalty=1.3)),
        dict(name="frequency_penalty_0.8", sp=sp(frequency_penalty=0.8)),
        dict(name="presence_penalty_0.8", sp=sp(presence_penalty=0.8)),
        dict(name="min_new_tokens_100_ignore_eos_false", sp=sp(min_new_tokens=100, ignore_eos=False),
             check="min_tokens"),
        dict(name="stop_string", sp=sp(stop=stop)),
        dict(name="stop_token_ids", sp=sp(stop_token_ids=[args.stop_token_id])),
        dict(name="no_stop_trim", sp=sp(stop=stop, no_stop_trim=True)),
        dict(name="n_2", sp=sp(n=2)),
        dict(name="logit_bias", sp=sp(logit_bias={str(args.bias_token_id): -100})),
        dict(name="return_logprob_top5", sp=sp(),
             extra={"return_logprob": True, "top_logprobs_num": 5, "logprob_start_len": 0}),
        dict(name="max_new_tokens_1", sp=sp(max_new_tokens=1)),
        dict(name="max_new_tokens_0", sp=sp(max_new_tokens=0), error_ok=True),
        dict(name="skip_special_tokens_false", sp=sp(skip_special_tokens=False)),
        dict(name="sampling_seed_42_temp0.7", sp=sp(temperature=0.7, sampling_seed=42), mode="repeat"),
        dict(name="json_schema", sp=sp(json_schema=json.dumps(SCHEMA)), error_ok=True),
        dict(name="regex", sp=sp(regex=r"[A-Z][a-z]+( [a-z]+){0,8}\."), error_ok=True),
        dict(name="stream_vs_nonstream", sp=sp(), mode="stream_pair"),
    ]
    if args.cases:
        want = set(args.cases.split(","))
        unknown = want - {c["name"] for c in cases}
        if unknown:
            sys.exit(f"error: unknown case(s): {sorted(unknown)}")
        cases = [c for c in cases if c["name"] in want]
    return cases


def read_sse(resp):
    text, meta = "", {}
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[5:].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        piece = chunk.get("text", "")
        text = piece if len(piece) >= len(text) else text + piece  # cumulative text
        meta = chunk.get("meta_info") or meta
    return text, meta


def fin(meta):
    fr = (meta or {}).get("finish_reason")
    return fr.get("type") if isinstance(fr, dict) else fr


def lp5(meta):
    lp = (meta or {}).get("output_token_logprobs")
    if isinstance(lp, list):
        return [round(float(t[0]), 4) for t in lp[:5] if isinstance(t, (list, tuple)) and t[0] is not None]
    return None


def run_request(base, text, case, args, stream=False, tag=""):
    res = {"status": None, "text": None, "texts": None, "completion_tokens": None, "finish_reason": None,
           "error": None, "logprobs5": None, "n_input_logprobs": None, "n_output_logprobs": None,
           "elapsed": None, "hang": False}
    body = {"text": text, "sampling_params": case["sp"], "stream": stream,
            "rid": f"pm-{case['name']}{tag}-{int(time.time() * 1000)}"}
    body.update(case.get("extra", {}))
    t0 = time.time()
    try:
        resp = requests.post(base.rstrip("/") + "/generate", json=body, timeout=args.timeout, stream=stream)
        res["status"] = resp.status_code
        if resp.status_code != 200:
            res["error"] = resp.text[:300]
        elif stream:
            res["text"], meta = read_sse(resp)
            metas = [meta]
        else:
            j = resp.json()
            outs = j if isinstance(j, list) else [j]
            texts = [o.get("text", "") for o in outs]
            metas = [o.get("meta_info") or {} for o in outs]
            res["text"] = texts[0] if len(texts) == 1 else None
            res["texts"] = texts
        if res["status"] == 200:
            res["completion_tokens"] = [m.get("completion_tokens") for m in metas]
            res["finish_reason"] = [fin(m) for m in metas]
            if len(metas) == 1:
                res["completion_tokens"], res["finish_reason"] = res["completion_tokens"][0], res["finish_reason"][0]
            res["logprobs5"] = lp5(metas[0])
            for k in ("input_token_logprobs", "output_token_logprobs"):
                v = metas[0].get(k)
                res[f"n_{k.split('_')[0]}_logprobs"] = len(v) if isinstance(v, list) else None
    except requests.Timeout:
        res["hang"], res["error"] = True, f"HANG? no response within {args.timeout}s"
    except requests.RequestException as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        res["error"] = f"bad response: {type(e).__name__}: {str(e)[:300]}"
    res["elapsed"] = round(time.time() - t0, 3)
    return res


def run_case(base, text, case, args, greedy):
    """Run one case on one endpoint; returns the result with self-check notes."""
    res = run_request(base, text, case, args)
    notes = []
    mode = case.get("mode")
    if mode == "repeat":
        r2 = run_request(base, text, case, args, tag="-2")
        res["second_text"], res["self_identical"] = r2["text"], r2["text"] == res["text"]
        res["hang"] = res["hang"] or r2["hang"]
        notes.append("info: seeded-rerun " + ("identical" if res["self_identical"] else "DIFFERS"))
    elif mode == "stream_pair":
        r2 = run_request(base, text, case, args, stream=True, tag="-s")
        res["stream_text"], res["stream_equal"] = r2["text"], r2["text"] == res["text"]
        res["hang"] = res["hang"] or r2["hang"]
        if r2["error"]:
            notes.append(f"stream error: {r2['error'][:80]}")
        if not res["stream_equal"]:
            notes.append("stream != nonstream")
    if res["hang"]:
        notes.append("HANG?")
    elif res["status"] != 200:
        if case.get("error_ok") and res["status"] and 400 <= res["status"] < 500:
            res["clean_error"] = True
        else:
            notes.append(f"http {res['status']}: {(res['error'] or '')[:80]}")
    elif case.get("check") == "equals_greedy":
        if greedy is None or greedy.get("text") is None:
            notes.append("info: greedy_baseline not run; top_k_1 unchecked")
        elif res["text"] != greedy["text"]:
            notes.append("top_k_1 != greedy")
    elif case.get("check") == "min_tokens":
        ct = res["completion_tokens"]
        if not isinstance(ct, int) or ct < case["sp"]["min_new_tokens"]:
            notes.append(f"completion_tokens {ct} < min_new_tokens")
    res["notes"] = notes
    return res


def compare(case, a, b):
    """Return (dict of per-field equality, list of divergence notes)."""
    eq = {"status": a["status"] == b["status"]}
    diffs = [] if eq["status"] else [f"status {a['status']} vs {b['status']}"]
    if case.get("mode") == "repeat":
        eq["finish_reason"] = a["finish_reason"] == b["finish_reason"]
        if not eq["finish_reason"]:
            diffs.append("finish_reason")
        return eq, diffs
    ta = a["texts"] if a["texts"] is not None else a["text"]
    tb = b["texts"] if b["texts"] is not None else b["text"]
    eq["text"] = ta == tb
    eq["completion_tokens"] = a["completion_tokens"] == b["completion_tokens"]
    eq["finish_reason"] = a["finish_reason"] == b["finish_reason"]
    diffs += [k for k in ("text", "completion_tokens", "finish_reason") if not eq[k]]
    if "return_logprob" in case.get("extra", {}):
        la, lb = a["logprobs5"] or [], b["logprobs5"] or []
        eq["logprobs5"] = len(la) == len(lb) and all(abs(x - y) <= 1e-2 for x, y in zip(la, lb))
        eq["logprob_counts"] = (a["n_input_logprobs"], a["n_output_logprobs"]) == (b["n_input_logprobs"], b["n_output_logprobs"])
        diffs += [k for k in ("logprobs5", "logprob_counts") if not eq[k]]
    return eq, diffs


def cell(r):
    if r is None:
        return "-"
    if r["hang"]:
        return "HANG?"
    return f"{r['status']}/{r['completion_tokens']}/{r['finish_reason']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="golden endpoint http://host:port")
    ap.add_argument("--b", default=None, help="PD endpoint (omit to only report A)")
    ap.add_argument("--prompt-file", default=None, help="jsonl with 'text' lines; default: built-in ~300-token prompt")
    ap.add_argument("--prompt-id", default=None)
    ap.add_argument("--cases", default=None, help="comma list of case names to run (default all)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--stop-token-id", type=int, default=13, help="token id used by stop_token_ids case")
    ap.add_argument("--bias-token-id", type=int, default=262, help="token id biased to -100 by logit_bias case")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--report", default="params_report.json")
    args = ap.parse_args()

    pid, text = load_prompt(args.prompt_file, args.prompt_id)
    cases = build_cases(args)
    endpoints = [("a", args.a)] + ([("b", args.b)] if args.b else [])
    print(f"prompt={pid} ({len(text)} chars), {len(cases)} cases, endpoints: " +
          ", ".join(f"{k}={u}" for k, u in endpoints))
    hdr = f"{'case':<36}{'A st/tok/fin':<22}{'B st/tok/fin':<22}{'txt':<5}{'tok':<5}{'fin':<5}{'verdict':<8}notes"
    print(hdr + "\n" + "-" * len(hdr))
    rows, n_fail, greedy = [], 0, {}
    for case in cases:
        row = {"name": case["name"], "sampling_params": case["sp"], "extra": case.get("extra", {}),
               "a": None, "b": None, "compare": None, "verdict": "PASS", "notes": []}
        for key, url in endpoints:
            r = run_case(url, text, case, args, greedy.get(key))
            if case["name"] == "greedy_baseline":
                greedy[key] = r
            row[key] = r
            row["notes"] += [f"{key}: {n}" for n in r["notes"]]
        if row["b"] is not None:
            eq, diffs = compare(case, row["a"], row["b"])
            row["compare"] = eq
            row["notes"] += [f"a!=b: {d}" for d in diffs]
        fail = any("info:" not in n for n in row["notes"])  # every non-informational note fails the case
        row["verdict"] = "FAIL" if fail else "PASS"
        n_fail += fail
        rows.append(row)
        eq = row["compare"] or {}
        yn = lambda k: ("-" if k not in eq else ("y" if eq[k] else "N"))  # noqa: E731
        print(f"{case['name']:<36}{cell(row['a']):<22}{cell(row['b']):<22}{yn('text'):<5}"
              f"{yn('completion_tokens'):<5}{yn('finish_reason'):<5}{row['verdict']:<8}{'; '.join(row['notes'])}",
              flush=True)
    report = {"a": args.a, "b": args.b, "prompt_id": pid, "prompt_chars": len(text),
              "max_new_tokens": args.max_new_tokens, "timeout": args.timeout,
              "cases": len(rows), "failed": n_fail, "rows": rows}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n{len(rows)} cases, {n_fail} failed -> {args.report}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
