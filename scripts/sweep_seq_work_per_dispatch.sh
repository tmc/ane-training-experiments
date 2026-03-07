#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE="$ROOT/scripts/compare_c_objc_vs_go_training.sh"

SEQ_LIST="256,512,1024"
STEPS=20
RUNS=1
SKIP_BUILD=1
OUT_DIR="/tmp/ane_seq_sweep"
COMPARE_OUT_DIR="/tmp/ane_compare"
DATA="$ROOT/training/tinystories_data00.bin"
MODEL="/Volumes/tmc/go/src/github.com/assets/models/stories110M.bin"

usage() {
	cat <<'EOF'
Usage:
  sweep_seq_work_per_dispatch.sh [flags]

Flags:
  --seq-list CSV         sequence overrides to sweep (default: 256,512,1024)
  --steps N              steps per run (default: 20)
  --runs N               runs per sequence (default: 1)
  --skip-build 0|1       pass through to compare script (default: 1)
  --out-dir DIR          output directory (default: /tmp/ane_seq_sweep)
  --compare-out-dir DIR  compare script output dir (default: /tmp/ane_compare)
  --data PATH            token data path
  --model PATH           stories model path (.bin)
  --help                 show this help
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--seq-list)
		SEQ_LIST="$2"
		shift 2
		;;
	--steps)
		STEPS="$2"
		shift 2
		;;
	--runs)
		RUNS="$2"
		shift 2
		;;
	--skip-build)
		SKIP_BUILD="$2"
		shift 2
		;;
	--out-dir)
		OUT_DIR="$2"
		shift 2
		;;
	--compare-out-dir)
		COMPARE_OUT_DIR="$2"
		shift 2
		;;
	--data)
		DATA="$2"
		shift 2
		;;
	--model)
		MODEL="$2"
		shift 2
		;;
	--help|-h)
		usage
		exit 0
		;;
	*)
		echo "unknown flag: $1" >&2
		usage >&2
		exit 2
		;;
	esac
done

if [[ ! -x "$COMPARE" ]]; then
	echo "compare script not executable: $COMPARE" >&2
	exit 1
fi
if [[ ! -f "$DATA" ]]; then
	echo "data file not found: $DATA" >&2
	exit 1
fi
if [[ ! -f "$MODEL" ]]; then
	echo "model file not found: $MODEL" >&2
	exit 1
fi
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
	echo "runs must be >=1: $RUNS" >&2
	exit 1
fi
if [[ "$SKIP_BUILD" != "0" && "$SKIP_BUILD" != "1" ]]; then
	echo "skip-build must be 0 or 1: $SKIP_BUILD" >&2
	exit 1
fi

mkdir -p "$OUT_DIR" "$COMPARE_OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
CSV="$OUT_DIR/seq_sweep_${TS}.csv"
LOG="$OUT_DIR/seq_sweep_${TS}.log"

extract_field() {
	local file="$1"
	local key="$2"
	awk -F= -v k="$key" '$1==k {print $2; exit}' "$file"
}

extract_ane_util() {
	local log_file="$1"
	local line
	line="$(grep -E "ANE utilization:" "$log_file" | tail -n1 || true)"
	if [[ -z "$line" ]]; then
		echo "n/a"
		return
	fi
	echo "$line" | sed -E 's/.*ANE utilization:[[:space:]]*([0-9.]+)%.*/\1/'
}

echo "seq_override,run,c_ane_util_pct,go_ane_util_pct,avg_c_train_step_ms,avg_go_train_step_ms,go_vs_c_ratio,summary_file" >"$CSV"

for seq in ${SEQ_LIST//,/ }; do
	if ! [[ "$seq" =~ ^[0-9]+$ ]] || [[ "$seq" -lt 1 ]]; then
		echo "invalid seq value: $seq" >&2
		exit 1
	fi
	for ((run = 1; run <= RUNS; run++)); do
		echo "[$(date +%H:%M:%S)] seq=$seq run=$run" | tee -a "$LOG"
		before_summary="$(ls -1t "$COMPARE_OUT_DIR"/summary_*.txt 2>/dev/null | head -n1 || true)"
		"$COMPARE" \
			--steps "$STEPS" \
			--data "$DATA" \
			--c-model "$MODEL" \
			--go-model "$MODEL" \
			--go-backend ane \
			--c-mode ane \
			--seq-override "$seq" \
			$( [[ "$SKIP_BUILD" == "1" ]] && echo --skip-build ) \
			2>&1 | tee -a "$LOG"
		after_summary="$(ls -1t "$COMPARE_OUT_DIR"/summary_*.txt 2>/dev/null | head -n1 || true)"
		if [[ -z "$after_summary" || "$after_summary" == "$before_summary" ]]; then
			echo "failed to discover summary file for seq=$seq run=$run" | tee -a "$LOG"
			exit 1
		fi

		c_log="$(extract_field "$after_summary" "c_log")"
		go_log="$(extract_field "$after_summary" "go_log")"
		c_util="$(extract_ane_util "$c_log")"
		go_util="$(extract_ane_util "$go_log")"
		c_avg="$(extract_field "$after_summary" "avg_c_train_step_ms")"
		go_avg="$(extract_field "$after_summary" "avg_go_train_step_ms")"
		ratio="$(extract_field "$after_summary" "go_vs_c_train_step_ratio")"
		echo "$seq,$run,$c_util,$go_util,$c_avg,$go_avg,$ratio,$after_summary" >>"$CSV"
	done
done

echo "csv=$CSV"
echo "log=$LOG"
