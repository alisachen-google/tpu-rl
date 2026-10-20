#!/usr/bin/env bash
D=$HOME/pd_disagg/spill
until grep -q 'chain-agentic-b' $D/logs/chain_agentic_b.log 2>/dev/null; do sleep 60; done
bash $D/phase_agentic_c.sh > $D/logs/phase_agentic_c.log 2>&1; echo "[chain-agentic-c] exit $?"
