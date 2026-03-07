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
STRICT=0
MAX_GO_VS_C_RATIO=""
MIN_GO_ANE_UTIL=""
MIN_C_ANE_UTIL=""

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
  --strict              enable threshold gates on parity metrics
  --max-go-vs-c-ratio F fail if go_vs_c_train_step_ratio > F (strict mode)
  --min-go-ane-util F   fail if go_ane_util_pct < F (strict mode)
  --min-c-ane-util F    fail if c_ane_util_pct < F (strict mode)
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
    --strict) STRICT=1; shift ;;
    --max-go-vs-c-ratio) MAX_GO_VS_C_RATIO="$2"; shift 2 ;;
    --min-go-ane-util) MIN_GO_ANE_UTIL="$2"; shift 2 ;;
    --min-c-ane-util) MIN_C_ANE_UTIL="$2"; shift 2 ;;
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
is_float() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}
if [[ -n "$MAX_GO_VS_C_RATIO" ]] && ! is_float "$MAX_GO_VS_C_RATIO"; then
  echo "max-go-vs-c-ratio must be numeric: $MAX_GO_VS_C_RATIO" >&2
  exit 2
fi
if [[ -n "$MIN_GO_ANE_UTIL" ]] && ! is_float "$MIN_GO_ANE_UTIL"; then
  echo "min-go-ane-util must be numeric: $MIN_GO_ANE_UTIL" >&2
  exit 2
fi
if [[ -n "$MIN_C_ANE_UTIL" ]] && ! is_float "$MIN_C_ANE_UTIL"; then
  echo "min-c-ane-util must be numeric: $MIN_C_ANE_UTIL" >&2
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

>"$PARITY_LOG"
echo "suite_config: steps=$STEPS skip_build=$SKIP_BUILD run_order=$RUN_ORDER strict=$STRICT" | tee -a "$PARITY_LOG"
echo "suite_config: compare_profile=high-util (resolved via --parity-mode)" | tee -a "$PARITY_LOG"
echo "suite_config: seq_sweep=min:$SEQ_MIN max:$SEQ_MAX channels:$SEQ_CHANNELS expected_floor:$SEQ_EXPECTED_MIN" | tee -a "$PARITY_LOG"

echo "[1/2] parity compare" | tee -a "$PARITY_LOG"
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
GATE_FAIL=0
GATE_MSGS=()

summary_value() {
  local key="$1"
  local file="$2"
  if [[ -z "$file" || ! -f "$file" ]]; then
    return 0
  fi
  rg -N "^${key}=" "$file" | head -n1 | sed "s/^${key}=//"
}

if (( STRICT == 1 )) && [[ -n "$PARITY_SUMMARY_FILE" && -f "$PARITY_SUMMARY_FILE" ]]; then
  GO_RATIO="$(summary_value "go_vs_c_train_step_ratio" "$PARITY_SUMMARY_FILE")"
  GO_UTIL="$(summary_value "go_ane_util_pct" "$PARITY_SUMMARY_FILE")"
  C_UTIL="$(summary_value "c_ane_util_pct" "$PARITY_SUMMARY_FILE")"

  if [[ -n "$MAX_GO_VS_C_RATIO" && -n "$GO_RATIO" ]]; then
    if ! awk -v v="$GO_RATIO" -v t="$MAX_GO_VS_C_RATIO" 'BEGIN{exit (v<=t)?0:1}'; then
      GATE_FAIL=1
      GATE_MSGS+=("gate_fail: go_vs_c_train_step_ratio=$GO_RATIO > max=$MAX_GO_VS_C_RATIO")
    fi
  fi
  if [[ -n "$MIN_GO_ANE_UTIL" && -n "$GO_UTIL" ]]; then
    if ! awk -v v="$GO_UTIL" -v t="$MIN_GO_ANE_UTIL" 'BEGIN{exit (v>=t)?0:1}'; then
      GATE_FAIL=1
      GATE_MSGS+=("gate_fail: go_ane_util_pct=$GO_UTIL < min=$MIN_GO_ANE_UTIL")
    fi
  fi
  if [[ -n "$MIN_C_ANE_UTIL" && -n "$C_UTIL" ]]; then
    if ! awk -v v="$C_UTIL" -v t="$MIN_C_ANE_UTIL" 'BEGIN{exit (v>=t)?0:1}'; then
      GATE_FAIL=1
      GATE_MSGS+=("gate_fail: c_ane_util_pct=$C_UTIL < min=$MIN_C_ANE_UTIL")
    fi
  fi
fi

{
  echo "suite_dir=$SUITE_DIR"
  echo "parity_rc=$PARITY_RC"
  echo "seq_floor_rc=$SEQ_RC"
  echo "parity_log=$PARITY_LOG"
  echo "seq_log=$SEQ_LOG"
  echo "parity_summary=${PARITY_SUMMARY_FILE:-<none>}"
  echo "seq_csv_latest=$SEQ_CSV_LINK"
  echo "seq_log_latest=$SEQ_LOG_LINK"
  echo "strict=$STRICT"
  echo "max_go_vs_c_ratio=${MAX_GO_VS_C_RATIO:-<unset>}"
  echo "min_go_ane_util=${MIN_GO_ANE_UTIL:-<unset>}"
  echo "min_c_ane_util=${MIN_C_ANE_UTIL:-<unset>}"
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
  if (( STRICT == 1 )); then
    echo
    if (( GATE_FAIL == 1 )); then
      echo "==== strict gates ===="
      for msg in "${GATE_MSGS[@]}"; do
        echo "$msg"
      done
    else
      echo "==== strict gates ===="
      echo "pass"
    fi
  fi
} | tee "$SUMMARY"

if (( PARITY_RC != 0 || SEQ_RC != 0 || GATE_FAIL != 0 )); then
  exit 1
fi
