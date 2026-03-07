#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ANE_RELEASE_OUT_DIR:-/tmp/ane_release_readiness}"
STEPS="${ANE_RELEASE_STEPS:-20}"
SKIP_BUILD="${ANE_RELEASE_SKIP_BUILD:-1}"
MAX_RATIO="${ANE_RELEASE_MAX_GO_VS_C_RATIO:-1.08}"
MIN_GO_UTIL="${ANE_RELEASE_MIN_GO_ANE_UTIL:-8.0}"
MIN_C_UTIL="${ANE_RELEASE_MIN_C_ANE_UTIL:-8.0}"

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_DIR/run_${TS}"
mkdir -p "$RUN_DIR"

COV_SUMMARY="$RUN_DIR/go_coverage_summary.txt"
PARITY_LOG="$RUN_DIR/parity_suite.log"
REPORT="$RUN_DIR/report.txt"

PKGS=(
  "./ane/bridge"
  "./ane/clientmodel"
  "./ane/storiestrainer"
  "./cmd/train-stories-ane-go"
)

echo "== go package coverage ==" | tee "$COV_SUMMARY"
for pkg in "${PKGS[@]}"; do
  out="$(cd "$ROOT" && go test "$pkg" -count=1 -cover 2>&1)"
  rc=$?
  echo "$out" | tee "$RUN_DIR/$(echo "$pkg" | tr '/.' '__').test.log" >/dev/null
  if (( rc != 0 )); then
    echo "FAIL $pkg" | tee -a "$COV_SUMMARY"
    exit 1
  fi
  cov="$(echo "$out" | rg -o 'coverage: [0-9.]+% of statements' | tail -n1 | sed 's/coverage: //')"
  if [[ -z "$cov" ]]; then
    cov="n/a"
  fi
  echo "$pkg $cov" | tee -a "$COV_SUMMARY"
done

echo "== parity + seq-floor strict suite ==" | tee "$PARITY_LOG"
set +e
"$ROOT/scripts/run_full_parity_suite.sh" \
  --steps "$STEPS" \
  --skip-build "$SKIP_BUILD" \
  --strict \
  --max-go-vs-c-ratio "$MAX_RATIO" \
  --min-go-ane-util "$MIN_GO_UTIL" \
  --min-c-ane-util "$MIN_C_UTIL" \
  2>&1 | tee -a "$PARITY_LOG"
SUITE_RC=${PIPESTATUS[0]}
set -e

SUITE_DIR="$(rg -N '^suite_dir=' "$PARITY_LOG" | tail -n1 | sed 's/^suite_dir=//')"
SUITE_SUMMARY=""
if [[ -n "$SUITE_DIR" && -f "$SUITE_DIR/summary.txt" ]]; then
  SUITE_SUMMARY="$SUITE_DIR/summary.txt"
fi
PARITY_SUMMARY=""
if [[ -n "$SUITE_SUMMARY" ]]; then
  PARITY_SUMMARY="$(rg -N '^parity_summary=' "$SUITE_SUMMARY" | tail -n1 | sed 's/^parity_summary=//')"
fi

{
  echo "release_readiness_dir=$RUN_DIR"
  echo "go_coverage_summary=$COV_SUMMARY"
  echo "parity_suite_log=$PARITY_LOG"
  echo "parity_suite_rc=$SUITE_RC"
  if [[ -n "$SUITE_SUMMARY" ]]; then
    echo "suite_summary=$SUITE_SUMMARY"
  fi
  if [[ -n "$PARITY_SUMMARY" && -f "$PARITY_SUMMARY" ]]; then
    echo "parity_summary=$PARITY_SUMMARY"
    rg -N 'avg_c_train_step_ms=|avg_go_train_step_ms=|go_vs_c_train_step_ratio=|c_ane_util_pct=|go_ane_util_pct=' "$PARITY_SUMMARY" || true
  fi
} | tee "$REPORT"

if (( SUITE_RC != 0 )); then
  exit "$SUITE_RC"
fi
