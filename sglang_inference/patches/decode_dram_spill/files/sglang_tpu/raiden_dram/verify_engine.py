# SPDX-License-Identifier: Apache-2.0
"""Verify an isolated, running SGLang deployment by flushing its local caches.

Run after starting prefill with RaidenDramStorage and the decode-host stores.
The generate URL can be a PD router; flush URLs must cover every serving engine
that router can choose. This intentionally clears engine HBM/local DRAM caches.
It never clears the remote host stores.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import requests


def verify(*, generate_url, flush_urls, prompt, spill_wait_s, max_tokens, timeout_s):
    session = requests.Session()
    request = {
        # Prevent a previous run's remote entry from satisfying the first pass.
        "text": f"Cache verification run {uuid.uuid4().hex}\n{prompt}",
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
        "stream": False,
    }

    def flush():
        for base in flush_urls:
            response = session.post(
                base.rstrip("/") + "/flush_cache",
                params={"timeout": 30},
                timeout=timeout_s,
            )
            response.raise_for_status()

    def generate():
        response = session.post(
            generate_url.rstrip("/") + "/generate", json=request, timeout=timeout_s
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or "text" not in result:
            raise AssertionError(f"Unexpected generate response: {result}")
        return result

    try:
        flush()
        first = generate()
        first_details = first.get("meta_info", {}).get("cached_tokens_details") or {}
        if first_details.get("storage", 0):
            raise AssertionError(
                "The unique cold prompt unexpectedly hit remote storage"
            )
        # Write-through backup is asynchronous. A shortfall is a test failure,
        # never reported as a successful recall. Increase for slow test hosts.
        time.sleep(spill_wait_s)
        flush()
        recalled = generate()
        details = recalled.get("meta_info", {}).get("cached_tokens_details") or {}
        if details.get("storage", 0) <= 0:
            raise AssertionError(
                f"No remote-storage recall after flushing local caches: {details}"
            )
        if details.get("device", 0) or details.get("host", 0):
            raise AssertionError(f"Local caches were not completely flushed: {details}")
        if first["text"] != recalled["text"]:
            raise AssertionError("Greedy output changed after spill and recall")
        return {
            "status": "passed",
            "cold": first["meta_info"],
            "recalled": recalled["meta_info"],
        }
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-url", required=True)
    parser.add_argument("--flush-url", action="append", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--spill-wait-s", type=float, default=10)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=300)
    args = parser.parse_args()
    if args.spill_wait_s < 0 or args.max_tokens <= 0 or args.timeout_s <= 0:
        parser.error("Invalid wait, token count, or request timeout")
    result = verify(
        generate_url=args.generate_url,
        flush_urls=args.flush_url,
        prompt=args.prompt_file.read_text(),
        spill_wait_s=args.spill_wait_s,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
