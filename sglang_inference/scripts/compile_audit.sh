#!/usr/bin/env bash
# TC-5 helper: count compile / recompile signatures in a server log, split at the "ready" line.
#   compile_audit.sh <server.log> [label]
log=${1:?server.log}; label=${2:-$(basename "$(dirname "$log")")}
ready_line=$(grep -nE 'The server is fired up and ready to roll|precompile done' "$log" | tail -n 1 | cut -d: -f1)
total=$(wc -l < "$log")
cnt() { # pattern from-line to-line
  sed -n "$2,$3p" "$log" | grep -cE "$1"
}
if [[ -z "$ready_line" ]]; then echo "$label: server not ready yet (no ready marker); lines=$total"; ready_line=$total; fi
printf '%s: ready_at_line=%s total_lines=%s\n' "$label" "$ready_line" "$total"
printf '  startup : __recompiles=%s  Compiling/compile=%s  cache_miss=%s  InductorError=%s\n' \
  "$(cnt '__recompiles\] Recompiling' 1 $ready_line)" "$(cnt 'Compil(ing|ed) |compile (begin|end|done)|TpuGraphRunner' 1 $ready_line)" "$(sed -n "1,${ready_line}p" "$log" | grep -ciE "cache miss")" "$(cnt 'InductorError|Detected recompile' 1 $ready_line)"
printf '  serving : __recompiles=%s  Compiling/compile=%s  cache_miss=%s  InductorError=%s  fail_on_recompile=%s\n' \
  "$(cnt '__recompiles\] Recompiling' $((ready_line+1)) $total)" "$(cnt 'Compil(ing|ed) |compile (begin|end|done)|TpuGraphRunner' $((ready_line+1)) $total)" "$(sed -n "$((ready_line+1)),${total}p" "$log" | grep -ciE "cache miss")" "$(cnt 'InductorError|Detected recompile' $((ready_line+1)) $total)" "$(cnt 'fail_on_recompile' $((ready_line+1)) $total)"
echo "  serving-time recompile lines (first 5):"; sed -n "$((ready_line+1)),${total}p" "$log" | grep -E '__recompiles\] Recompiling|InductorError|Detected recompile|Compiling' | head -5 | cut -c1-200
