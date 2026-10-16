"""One P/D request sent to a prefill and a decode directly, the way MiniLB does.

Usage: pd_pair_request.py <prompt.jsonl> <max_new_tokens> <stream 0|1> <client_timeout_s> <prefill_ip> <decode_ip>
Prints: "<decode status> <total_s> <first_byte_s>" (first_byte_s is None when no chunk arrived).
"""

import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

pf, tok, stream, tmo, p1, d1 = (
    sys.argv[1],
    int(sys.argv[2]),
    sys.argv[3] == "1",
    float(sys.argv[4]),
    sys.argv[5],
    sys.argv[6],
)
text = json.loads(open(pf).readline())["text"]
body = {
    "text": f"{time.time():.3f} " + text,
    "sampling_params": {"max_new_tokens": tok, "temperature": 0, "ignore_eos": True},
    "stream": stream,
    "bootstrap_host": p1,
    "bootstrap_port": 8998,
    "bootstrap_room": random.randint(0, 2**63 - 1),
}
data = json.dumps(body).encode()
hdr = {"Content-Type": "application/json"}


def post_prefill():
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"http://{p1}:30000/generate", data=data, headers=hdr),
            timeout=tmo,
        ).read()
    except Exception:
        pass


threading.Thread(target=post_prefill, daemon=True).start()
t0 = time.time()
first = None
try:
    r = urllib.request.urlopen(
        urllib.request.Request(f"http://{d1}:30000/generate", data=data, headers=hdr),
        timeout=tmo,
    )
    while True:
        chunk = r.readline()
        if not chunk:
            break
        if first is None and chunk.strip():
            first = round(time.time() - t0, 1)
    code = r.status
except urllib.error.HTTPError as e:
    code = e.code
except Exception as e:
    code = "ERR:" + type(e).__name__
print(f"{code} {time.time() - t0:.1f} {first}")
