#!/usr/bin/env bash
# 1) native TPU round-trip test on b; 2) launch the baseline 1P/1D pair from the patched tree.
D=$HOME/pd_disagg/spill
echo "[chain] $(date -u +%FT%TZ) native test start"
bash $D/run_native_test.sh > $D/logs/native_test3.log 2>&1
echo "[chain] $(date -u +%FT%TZ) native test exit $? : $(grep -E -o '[0-9]+ (passed|failed)[^=]*' $D/logs/native_test3.log | tail -1)"
echo "[chain] $(date -u +%FT%TZ) launching base pair spill_base_1"
bash $D/pd_spill.sh start $HOME/pd_disagg/recipes/gpt-oss_C_spill_base.json spill_base_1
echo "[chain] $(date -u +%FT%TZ) base pair start returned $?"
