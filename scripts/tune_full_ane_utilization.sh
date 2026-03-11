#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE="$ROOT/scripts/compare_c_objc_vs_go_training.sh"

OUT_DIR="/tmp/ane_full_tune"
COMPARE_OUT_DIR="/tmp/ane_compare"
STEPS=40
RUNS=2
SEQ_LIST="256,384"
FULL_ACCUM_LIST="20,40"
WARMUP_STEPS=0
SKIP_BUILD=1
RUN_ORDER_MODE="alternate"
VECLIB_THREADS=0
DW_CONCURRENCY=0

usage() {
	cat <<'EOF'
Usage:
  tune_full_ane_utilization.sh [flags]

Flags:
  --steps N              training steps per run (default: 40)
  --runs N               replicate runs per (seq,accum) (default: 2)
  --seq-list CSV         sequence overrides to test (default: 256,384)
  --full-accum-list CSV  full trainer accumulation values (default: 20,40)
  --warmup-steps N       summary warmup steps passed to compare script (default: 0)
  --run-order MODE       alternate|go-c|c-go (default: alternate)
  --veclib-threads N     set VECLIB_MAXIMUM_THREADS for full trainer runs (default: process default)
  --dw-concurrency N     set dW async task concurrency for full trainer runs (default: trainer default)
  --skip-build 0|1       pass --skip-build to compare runs (default: 1)
  --out-dir DIR          tuner output directory (default: /tmp/ane_full_tune)
  --compare-out-dir DIR  compare script output directory (default: /tmp/ane_compare)
  --help                 show this help

Outputs:
  full_tune_<ts>.csv
  full_tune_<ts>.log
  full_tune_summary_<ts>.txt
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--steps)
		STEPS="$2"
		shift 2
		;;
	--runs)
		RUNS="$2"
		shift 2
		;;
	--seq-list)
		SEQ_LIST="$2"
		shift 2
		;;
	--full-accum-list)
		FULL_ACCUM_LIST="$2"
		shift 2
		;;
	--warmup-steps)
		WARMUP_STEPS="$2"
		shift 2
		;;
	--run-order)
		RUN_ORDER_MODE="$2"
		shift 2
		;;
	--veclib-threads)
		VECLIB_THREADS="$2"
		shift 2
		;;
	--dw-concurrency)
		DW_CONCURRENCY="$2"
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
if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || [[ "$STEPS" -lt 1 ]]; then
	echo "steps must be >= 1: $STEPS" >&2
	exit 1
fi
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
	echo "runs must be >= 1: $RUNS" >&2
	exit 1
fi
if ! [[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
	echo "warmup-steps must be >= 0: $WARMUP_STEPS" >&2
	exit 1
fi
if [[ "$SKIP_BUILD" != "0" && "$SKIP_BUILD" != "1" ]]; then
	echo "skip-build must be 0 or 1: $SKIP_BUILD" >&2
	exit 1
fi
if ! [[ "$VECLIB_THREADS" =~ ^[0-9]+$ ]]; then
	echo "veclib-threads must be >= 0: $VECLIB_THREADS" >&2
	exit 1
fi
if ! [[ "$DW_CONCURRENCY" =~ ^[0-9]+$ ]]; then
	echo "dw-concurrency must be >= 0: $DW_CONCURRENCY" >&2
	exit 1
fi
if [[ "$RUN_ORDER_MODE" != "alternate" && "$RUN_ORDER_MODE" != "go-c" && "$RUN_ORDER_MODE" != "c-go" ]]; then
	echo "run-order must be alternate, go-c, or c-go: $RUN_ORDER_MODE" >&2
	exit 1
fi

mkdir -p "$OUT_DIR" "$COMPARE_OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
CSV="$OUT_DIR/full_tune_${TS}.csv"
LOG="$OUT_DIR/full_tune_${TS}.log"
SUMMARY="$OUT_DIR/full_tune_summary_${TS}.txt"

echo "seq,full_accum,run,run_order,c_ane_util_pct,go_ane_util_pct,c_avg_step_ms,go_avg_step_ms,go_vs_c_ratio,summary_file" >"$CSV"

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

seq_values="${SEQ_LIST//,/ }"
accum_values="${FULL_ACCUM_LIST//,/ }"
current_skip_build="$SKIP_BUILD"

for seq in $seq_values; do
	if ! [[ "$seq" =~ ^[0-9]+$ ]] || [[ "$seq" -lt 1 ]]; then
		echo "invalid seq in --seq-list: $seq" >&2
		exit 1
	fi
	for accum in $accum_values; do
		if ! [[ "$accum" =~ ^[0-9]+$ ]] || [[ "$accum" -lt 1 ]]; then
			echo "invalid accum in --full-accum-list: $accum" >&2
			exit 1
		fi
		for ((run = 1; run <= RUNS; run++)); do
			run_order="$RUN_ORDER_MODE"
			if [[ "$RUN_ORDER_MODE" == "alternate" ]]; then
				run_order="c-go"
				if (( run % 2 == 0 )); then
					run_order="go-c"
				fi
			fi

			cmd=("$COMPARE"
				"--steps" "$STEPS"
				"--go-backend" "ane"
				"--c-mode" "ane"
				"--seq-override" "$seq"
				"--full-accum" "$accum"
				"--run-order" "$run_order"
				"--warmup-steps" "$WARMUP_STEPS"
				"--out-dir" "$COMPARE_OUT_DIR")
			if [[ "$VECLIB_THREADS" -gt 0 ]]; then
				cmd+=("--veclib-threads" "$VECLIB_THREADS")
			fi
			if [[ "$DW_CONCURRENCY" -gt 0 ]]; then
				cmd+=("--dw-concurrency" "$DW_CONCURRENCY")
			fi
			if [[ "$current_skip_build" == "1" ]]; then
				cmd+=("--skip-build")
			fi

			echo "[$(date +%H:%M:%S)] seq=$seq accum=$accum run=$run order=$run_order" | tee -a "$LOG"
			before_summary="$(ls -1t "$COMPARE_OUT_DIR"/summary_*.txt 2>/dev/null | head -n1 || true)"
			"${cmd[@]}" | tee -a "$LOG"
			after_summary="$(ls -1t "$COMPARE_OUT_DIR"/summary_*.txt 2>/dev/null | head -n1 || true)"

			if [[ -z "$after_summary" || "$after_summary" == "$before_summary" ]]; then
				echo "failed to discover summary file for seq=$seq accum=$accum run=$run" | tee -a "$LOG"
				exit 1
			fi

			c_log="$(extract_field "$after_summary" "c_log")"
			go_log="$(extract_field "$after_summary" "go_log")"
			c_step="$(extract_field "$after_summary" "avg_c_train_step_ms")"
			go_step="$(extract_field "$after_summary" "avg_go_train_step_ms")"
			ratio="$(extract_field "$after_summary" "go_vs_c_train_step_ratio")"
			c_util="$(extract_ane_util "$c_log")"
			go_util="$(extract_ane_util "$go_log")"
			echo "$seq,$accum,$run,$run_order,$c_util,$go_util,$c_step,$go_step,$ratio,$after_summary" >>"$CSV"

			current_skip_build=1
		done
	done
done

{
	echo "results_csv=$CSV"
	echo "raw_log=$LOG"
	echo "veclib_threads=$VECLIB_THREADS"
	echo "dw_concurrency=$DW_CONCURRENCY"
	echo
	echo "aggregate by seq+accum (mean over runs):"
	echo "seq,full_accum,runs,mean_c_util,mean_go_util,mean_c_step_ms,mean_go_step_ms,mean_go_vs_c_ratio"
	awk -F, 'NR>1 {
		key=$1","$2
		n[key]++
		if ($5 ~ /^[0-9]+(\.[0-9]+)?$/) { c_util[key]+=$5; c_util_n[key]++ }
		if ($6 ~ /^[0-9]+(\.[0-9]+)?$/) { go_util[key]+=$6; go_util_n[key]++ }
		if ($7 ~ /^[0-9]+(\.[0-9]+)?$/) { c_ms[key]+=$7; c_ms_n[key]++ }
		if ($8 ~ /^[0-9]+(\.[0-9]+)?$/) { go_ms[key]+=$8; go_ms_n[key]++ }
		if ($9 ~ /^[0-9]+(\.[0-9]+)?$/) { ratio[key]+=$9; ratio_n[key]++ }
	}
	END {
		for (k in n) {
			cu = (c_util_n[k] > 0) ? sprintf("%.4f", c_util[k]/c_util_n[k]) : "n/a"
			gu = (go_util_n[k] > 0) ? sprintf("%.4f", go_util[k]/go_util_n[k]) : "n/a"
			cm = (c_ms_n[k] > 0) ? sprintf("%.4f", c_ms[k]/c_ms_n[k]) : "n/a"
			gm = (go_ms_n[k] > 0) ? sprintf("%.4f", go_ms[k]/go_ms_n[k]) : "n/a"
			gr = (ratio_n[k] > 0) ? sprintf("%.6f", ratio[k]/ratio_n[k]) : "n/a"
			printf "%s,%d,%s,%s,%s,%s,%s\n", k, n[k], cu, gu, cm, gm, gr
		}
	}' "$CSV" | sort -t, -k1,1n -k2,2n
	echo
	echo "best_by_go_util:"
	awk -F, '
	BEGIN { best=-1 }
	NR>1 && $6 ~ /^[0-9]+(\.[0-9]+)?$/ {
		v=$6+0
		if (v > best) { best=v; row=$0 }
	}
	END { if (best < 0) print "n/a"; else print row }' "$CSV"
	echo
	echo "best_by_go_step:"
	awk -F, '
	BEGIN { best=-1 }
	NR>1 && $8 ~ /^[0-9]+(\.[0-9]+)?$/ {
		v=$8+0
		if (best < 0 || v < best) { best=v; row=$0 }
	}
	END { if (best < 0) print "n/a"; else print row }' "$CSV"
	echo
	echo "best_by_ratio_closeness_to_1:"
	awk -F, '
	BEGIN { best=-1 }
	NR>1 && $9 ~ /^[0-9]+(\.[0-9]+)?$/ {
		v=$9+0
		d=v-1.0
		if (d < 0) d=-d
		if (best < 0 || d < best) { best=d; row=$0 }
	}
	END { if (best < 0) print "n/a"; else print row }' "$CSV"
	echo
	echo "best_ratio_with_go_not_slower:"
	awk -F, '
	BEGIN { best=-1 }
	NR>1 && $9 ~ /^[0-9]+(\.[0-9]+)?$/ {
		r=$9+0
		if (r > 1.000001) next
		d=1.0-r
		if (best < 0 || d < best) { best=d; row=$0 }
	}
	END { if (best < 0) print "n/a"; else print row }' "$CSV"
	echo
	echo "recommended_compare_cmd (from best_by_ratio_closeness_to_1):"
	awk -F, '
	BEGIN { best=-1 }
	NR>1 && $9 ~ /^[0-9]+(\.[0-9]+)?$/ {
		r=$9+0
		d=r-1.0
		if (d < 0) d=-d
		if (best < 0 || d < best) {
			best=d
			seq=$1
			acc=$2
		}
	}
	END {
		if (best < 0) {
			print "n/a"
		} else {
			printf "./scripts/compare_c_objc_vs_go_training.sh --c-mode ane --go-backend ane --allow-mismatch --steps %d --seq-override %d --full-accum %d --profile high-util --run-order go-c --warmup-steps %d --skip-build\n", '"$STEPS"', seq, acc, '"$WARMUP_STEPS"'
		}
	}' "$CSV"
} | tee "$SUMMARY"

echo
echo "wrote:"
echo "  $CSV"
echo "  $SUMMARY"
echo "  $LOG"
