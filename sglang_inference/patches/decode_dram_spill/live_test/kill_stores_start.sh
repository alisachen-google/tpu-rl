#!/usr/bin/env bash
for p in $(pgrep -u $(id -un) -f "spill_stores.sh stores-sta""rt"); do kill $p 2>/dev/null && echo "killed $p"; done
