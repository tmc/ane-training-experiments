#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE="$ROOT/scripts/compare_c_objc_vs_go_training.sh"

RUNS=4
STEPS=20
WARMUP_STEPS=0
OUT_DIR="/tmp/ane_match_c_perf"
COMPARE_OUT_DIR="/tmp/ane_compare"
SEQ_OVERRIDE=384
FULL_ACCUM=80
VECLIB_THREADS=6
DW_CONCURRENCY=3
SKIP_BUILD=1
TARGET_RATIO=1.03

usage() {
  cat <<'EOF'
Usage:
  match_c_perf.sh [flags]

Flags:
  --runs N             number of compare runs (default: 4)
  --steps N            steps per run (default: 20)
  --warmup-steps N     warmup steps passed to compare script (default: 0)
  --seq-override N     compile-time seq override (default: 384)
  --full-accum N       full trainer accumulation (default: 80)
  --veclib-threads N   VECLIB threads (default: 6)
  --dw-concurrency N   dW async task concurrency (default: 3)
  --target-ratio F     pass if median go/c ratio <= F (default: 1.03)
  --skip-build 0|1     skip rebuilding binaries (default: 1)
  --out-dir DIR        output directory (default: /tmp/ane_match_c_perf)
  --compare-out-dir DIR compare script output dir (default: /tmp/ane_compare)
  --help               show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --seq-override) SEQ_OVERRIDE="$2"; shift 2 ;;
    --full-accum) FULL_ACCUM="$2"; shift 2 ;;
    --veclib-threads) VECLIB_THREADS="$2"; shift 2 ;;
    --dw-concurrency) DW_CONCURRENCY="$2"; shift 2 ;;
    --target-ratio) TARGET_RATIO="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --compare-out-dir) COMPARE_OUT_DIR="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for n in "$RUNS" "$STEPS" "$WARMUP_STEPS" "$SEQ_OVERRIDE" "$FULL_ACCUM" "$VECLIB_THREADS" "$DW_CONCURRENCY"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "invalid numeric flag: $n" >&2
    exit 2
  fi
done
if [[ "$SKIP_BUILD" != "0" && "$SKIP_BUILD" != "1" ]]; then
  echo "skip-build must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$TARGET_RATIO" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "target-ratio must be numeric" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$COMPARE_OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
ID="${TS}_seq${SEQ_OVERRIDE}_pid$$"
CSV="$OUT_DIR/match_c_perf_${ID}.csv"
LOG="$OUT_DIR/match_c_perf_${ID}.log"
SUMMARY="$OUT_DIR/match_c_perf_summary_${ID}.txt"

echo "run,order,ratio,c_util,go_util,summary_file" >"$CSV"

extract_field() {
  local file="$1"
  local key="$2"
  awk -F= -v k="$key" '$1==k {print $2; exit}' "$file"
}

for ((run = 1; run <= RUNS; run++)); do
  order="c-go"
  if (( run % 2 == 0 )); then
    order="go-c"
  fi
  cmd=(
    "$COMPARE"
    --steps "$STEPS"
    --c-mode ane
    --go-backend ane
    --seq-override "$SEQ_OVERRIDE"
    --full-accum "$FULL_ACCUM"
    --veclib-threads "$VECLIB_THREADS"
    --dw-concurrency "$DW_CONCURRENCY"
    --run-order "$order"
    --warmup-steps "$WARMUP_STEPS"
    --out-dir "$COMPARE_OUT_DIR"
  )
  if [[ "$SKIP_BUILD" == "1" ]]; then
    cmd+=(--skip-build)
  fi
  echo "[$(date +%H:%M:%S)] run=$run order=$order" | tee -a "$LOG"
  before="$(ls -1t "$COMPARE_OUT_DIR"/summary_*.txt 2>/dev/null | head -n1 || true)"
  "${cmd[@]}" | tee -a "$LOG"
  after="$(ls -1t "$COMPARE_OUT_DIR"/summary_*.txt 2>/dev/null | head -n1 || true)"
  if [[ -z "$after" || "$after" == "$before" ]]; then
    echo "failed to locate compare summary for run=$run" >&2
    exit 1
  fi
  ratio="$(extract_field "$after" "go_vs_c_train_step_ratio")"
  c_util="$(extract_field "$after" "c_ane_util_pct")"
  go_util="$(extract_field "$after" "go_ane_util_pct")"
  echo "$run,$order,$ratio,$c_util,$go_util,$after" >>"$CSV"
done

{
  echo "csv=$CSV"
  echo "log=$LOG"
  echo "target_ratio=$TARGET_RATIO"
  awk -F, '
  NR==1 { next }
  $3 ~ /^[0-9]+(\.[0-9]+)?$/ {
    ratios[++n]=$3+0
    sum+=$3
  }
  END {
    if (n == 0) {
      print "runs=0"
      print "mean_ratio=n/a"
      print "median_ratio=n/a"
      exit
    }
    for (i=1; i<=n; i++) {
      for (j=i+1; j<=n; j++) {
        if (ratios[j] < ratios[i]) {
          t=ratios[i]; ratios[i]=ratios[j]; ratios[j]=t
        }
      }
    }
    mean=sum/n
    if (n % 2 == 1) {
      med=ratios[(n+1)/2]
    } else {
      med=(ratios[n/2] + ratios[n/2+1]) / 2
    }
    printf "runs=%d\n", n
    printf "mean_ratio=%.6f\n", mean
    printf "median_ratio=%.6f\n", med
  }' "$CSV"
  echo "rows:"
  cat "$CSV"
} | tee "$SUMMARY"

MEDIAN="$(awk -F= '$1=="median_ratio"{print $2}' "$SUMMARY" | tail -n1)"
if [[ -n "$MEDIAN" ]] && awk -v m="$MEDIAN" -v t="$TARGET_RATIO" 'BEGIN{exit (m<=t)?0:1}'; then
  echo "PASS median_ratio=$MEDIAN <= target=$TARGET_RATIO"
else
  echo "FAIL median ratio above target (median=${MEDIAN:-n/a} target=$TARGET_RATIO)" >&2
  exit 1
fi
