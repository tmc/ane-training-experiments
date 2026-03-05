#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ANE_CHAIN_SWEEP_OUT_DIR:-$ROOT/training/artifacts/seq_floor}"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
CSV_PATH="${ANE_CHAIN_SWEEP_CSV_OUT:-$OUT_DIR/ane_seq_floor_${TS}.csv}"
LOG_PATH="${ANE_CHAIN_SWEEP_LOG_OUT:-$OUT_DIR/ane_seq_floor_${TS}.log}"

CASE_NAME="TestANEClientSeqMapThresholdSweep"
SEQ_MIN="${ANE_CHAIN_SWEEP_MIN:-1}"
SEQ_MAX="${ANE_CHAIN_SWEEP_MAX:-32}"
CHANNELS="${ANE_CHAIN_SWEEP_CHANNELS:-64}"
EXPECTED_MIN="${ANE_CHAIN_EXPECTED_MIN_SEQ:-16}"

if ! [[ "$SEQ_MIN" =~ ^[0-9]+$ && "$SEQ_MAX" =~ ^[0-9]+$ && "$CHANNELS" =~ ^[0-9]+$ && "$EXPECTED_MIN" =~ ^[0-9]+$ ]]; then
  echo "invalid numeric config: min=$SEQ_MIN max=$SEQ_MAX channels=$CHANNELS expected=$EXPECTED_MIN" >&2
  exit 2
fi
if (( SEQ_MIN < 1 || SEQ_MAX < SEQ_MIN || CHANNELS < 1 || EXPECTED_MIN < 1 )); then
  echo "invalid range config: min=$SEQ_MIN max=$SEQ_MAX channels=$CHANNELS expected=$EXPECTED_MIN" >&2
  exit 2
fi

echo "running $CASE_NAME"
echo "  min_seq=$SEQ_MIN max_seq=$SEQ_MAX channels=$CHANNELS expected_min_seq=$EXPECTED_MIN"
echo "  csv=$CSV_PATH"
echo "  log=$LOG_PATH"

set +e
ANE_CHAIN_CASE_ONLY="$CASE_NAME" \
ANE_CHAIN_ENABLE_SEQ_SWEEP=1 \
ANE_CHAIN_SWEEP_MIN="$SEQ_MIN" \
ANE_CHAIN_SWEEP_MAX="$SEQ_MAX" \
ANE_CHAIN_SWEEP_CHANNELS="$CHANNELS" \
ANE_CHAIN_EXPECTED_MIN_SEQ="$EXPECTED_MIN" \
ANE_CHAIN_SWEEP_CSV="$CSV_PATH" \
"$ROOT/training/run_chaining_suite.sh" 2>&1 | tee "$LOG_PATH"
RC=${PIPESTATUS[0]}
set -e

ln -sfn "$(basename "$CSV_PATH")" "$OUT_DIR/latest.csv"
ln -sfn "$(basename "$LOG_PATH")" "$OUT_DIR/latest.log"

echo "exit_code=$RC"
echo "wrote_csv=$CSV_PATH"
echo "wrote_log=$LOG_PATH"
echo "latest_csv=$OUT_DIR/latest.csv"
echo "latest_log=$OUT_DIR/latest.log"

exit "$RC"
