#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STEPS=20
SKIP_BUILD=1
RUN_ORDER="c-go"
OUT_DIR="/tmp/ane_parity_suite"
PARITY_NAME="parity"
SEQ_NAME="seq_floor"
SEQ_MIN=1
SEQ_MAX=32
SEQ_CHANNELS=64
SEQ_EXPECTED_MIN=16

usage() {
  cat <<'USAGE'
Usage:
  run_full_parity_suite.sh [flags]

Flags:
  --steps N             steps for parity compare (default: 20)
  --skip-build 0|1      skip compile/build in compare path (default: 1)
  --run-order ORDER     c-go|go-c for compare run (default: c-go)
  --out-dir DIR         suite output directory (default: /tmp/ane_parity_suite)
  --seq-min N           seq floor sweep min (default: 1)
  --seq-max N           seq floor sweep max (default: 32)
  --seq-channels N      seq floor sweep channels (default: 64)
  --seq-expected-min N  expected floor for seq sweep (default: 16)
  --help                show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD="$2"; shift 2 ;;
    --run-order) RUN_ORDER="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --seq-min) SEQ_MIN="$2"; shift 2 ;;
    --seq-max) SEQ_MAX="$2"; shift 2 ;;
    --seq-channels) SEQ_CHANNELS="$2"; shift 2 ;;
    --seq-expected-min) SEQ_EXPECTED_MIN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for n in "$STEPS" "$SKIP_BUILD" "$SEQ_MIN" "$SEQ_MAX" "$SEQ_CHANNELS" "$SEQ_EXPECTED_MIN"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "numeric flag invalid: $n" >&2
    exit 2
  fi
done
if [[ "$RUN_ORDER" != "c-go" && "$RUN_ORDER" != "go-c" ]]; then
  echo "run-order must be c-go|go-c" >&2
  exit 2
fi
if (( SEQ_MIN < 1 || SEQ_MAX < SEQ_MIN || SEQ_CHANNELS < 1 || SEQ_EXPECTED_MIN < 1 )); then
  echo "invalid seq sweep config" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
SUITE_DIR="$OUT_DIR/suite_${TS}"
mkdir -p "$SUITE_DIR"

PARITY_OUT="$SUITE_DIR/$PARITY_NAME"
mkdir -p "$PARITY_OUT"
SEQ_OUT="$SUITE_DIR/$SEQ_NAME"
mkdir -p "$SEQ_OUT"

PARITY_LOG="$PARITY_OUT/compare.log"
SEQ_LOG="$SEQ_OUT/sweep.log"
SUMMARY="$SUITE_DIR/summary.txt"

compare_cmd=(
  "$ROOT/scripts/compare_c_objc_vs_go_training.sh"
  "--parity-mode"
  "--steps" "$STEPS"
  "--run-order" "$RUN_ORDER"
  "--out-dir" "$PARITY_OUT"
)
if [[ "$SKIP_BUILD" == "1" ]]; then
  compare_cmd+=("--skip-build")
fi

echo "[1/2] parity compare" | tee "$PARITY_LOG"
printf 'cmd:' | tee -a "$PARITY_LOG"
printf ' %q' "${compare_cmd[@]}" | tee -a "$PARITY_LOG"
echo | tee -a "$PARITY_LOG"
set +e
"${compare_cmd[@]}" 2>&1 | tee -a "$PARITY_LOG"
PARITY_RC=${PIPESTATUS[0]}
set -e

echo "[2/2] seq floor sweep" | tee "$SEQ_LOG"
set +e
ANE_CHAIN_SWEEP_OUT_DIR="$SEQ_OUT" \
ANE_CHAIN_SWEEP_MIN="$SEQ_MIN" \
ANE_CHAIN_SWEEP_MAX="$SEQ_MAX" \
ANE_CHAIN_SWEEP_CHANNELS="$SEQ_CHANNELS" \
ANE_CHAIN_EXPECTED_MIN_SEQ="$SEQ_EXPECTED_MIN" \
"$ROOT/training/run_seq_map_threshold_sweep.sh" 2>&1 | tee -a "$SEQ_LOG"
SEQ_RC=${PIPESTATUS[0]}
set -e

PARITY_SUMMARY_FILE="$(ls -1t "$PARITY_OUT"/summary_*.txt 2>/dev/null | head -n1 || true)"
SEQ_CSV_LINK="$SEQ_OUT/latest.csv"
SEQ_LOG_LINK="$SEQ_OUT/latest.log"

{
  echo "suite_dir=$SUITE_DIR"
  echo "parity_rc=$PARITY_RC"
  echo "seq_floor_rc=$SEQ_RC"
  echo "parity_log=$PARITY_LOG"
  echo "seq_log=$SEQ_LOG"
  echo "parity_summary=${PARITY_SUMMARY_FILE:-<none>}"
  echo "seq_csv_latest=$SEQ_CSV_LINK"
  echo "seq_log_latest=$SEQ_LOG_LINK"
  if [[ -n "$PARITY_SUMMARY_FILE" && -f "$PARITY_SUMMARY_FILE" ]]; then
    echo
    echo "==== parity summary excerpt ===="
    rg -n "avg_c_train_step_ms=|avg_go_train_step_ms=|go_vs_c_train_step_ratio=|c_ane_util_pct=|go_ane_util_pct=|profile=|parity_mode=" "$PARITY_SUMMARY_FILE" -N || true
  fi
  if [[ -f "$SEQ_CSV_LINK" ]]; then
    echo
    echo "==== seq floor csv tail ===="
    tail -n 12 "$SEQ_CSV_LINK" || true
  fi
} | tee "$SUMMARY"

if (( PARITY_RC != 0 || SEQ_RC != 0 )); then
  exit 1
fi
