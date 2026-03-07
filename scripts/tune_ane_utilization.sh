#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE="$ROOT/scripts/compare_c_objc_vs_go_training.sh"

OUT_DIR="/tmp/ane_util_sweep"
COMPARE_OUT_DIR="/tmp/ane_compare"
DATA="$ROOT/training/tinystories_data00.bin"
MODEL="/Volumes/tmc/go/src/github.com/assets/models/stories110M.bin"
DYNAMIC_BIN="$ROOT/training/training_dynamic/train"
DYNAMIC_ACCUM_LIST="10,20,40"
STEPS_LIST="20,50,100"
MODES="ane,ane-dynamic"
RUNS=2
SKIP_BUILD=1
WARMUP_STEPS=0
NO_ANE_EXTRAS=0
ALLOW_MISMATCH=0

usage() {
	cat <<'EOF'
Usage:
  tune_ane_utilization.sh [flags]

Flags:
  --out-dir DIR          output dir for sweep results (default: /tmp/ane_util_sweep)
  --compare-out-dir DIR  output dir used by compare script (default: /tmp/ane_compare)
  --data PATH            TinyStories uint16 data path
  --model PATH           stories110M .bin model path
  --dynamic-bin PATH     dynamic trainer binary path
  --dynamic-accum-list CSV  comma-separated accum list for ane-dynamic (default: 10,20,40)
  --steps-list CSV       comma-separated steps list (default: 20,50,100)
  --modes CSV            comma-separated modes: ane,ane-dynamic (default: ane,ane-dynamic)
  --runs N               runs per (mode,steps) (default: 2)
  --warmup-steps N       pass through to compare summary warmup
  --skip-build 0|1       build before first run (default: 1)
  --no-ane-extras        pass to compare script
  --allow-mismatch       pass through to compare script
  --help                 show this help
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
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
	--dynamic-bin)
		DYNAMIC_BIN="$2"
		shift 2
		;;
	--dynamic-accum-list)
		DYNAMIC_ACCUM_LIST="$2"
		shift 2
		;;
	--steps-list)
		STEPS_LIST="$2"
		shift 2
		;;
	--modes)
		MODES="$2"
		shift 2
		;;
	--runs)
		RUNS="$2"
		shift 2
		;;
	--warmup-steps)
		WARMUP_STEPS="$2"
		shift 2
		;;
	--skip-build)
		SKIP_BUILD="$2"
		shift 2
		;;
	--no-ane-extras)
		NO_ANE_EXTRAS=1
		shift
		;;
	--allow-mismatch)
		ALLOW_MISMATCH=1
		shift
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
if [[ ! -x "$DYNAMIC_BIN" ]]; then
	echo "dynamic trainer binary not executable: $DYNAMIC_BIN" >&2
	echo "hint: make -C $ROOT/training/training_dynamic train" >&2
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

mkdir -p "$OUT_DIR" "$COMPARE_OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
CSV="$OUT_DIR/util_sweep_${TS}.csv"
LOG="$OUT_DIR/util_sweep_${TS}.log"
SUMMARY="$OUT_DIR/util_sweep_summary_${TS}.txt"

steps_values="${STEPS_LIST//,/ }"
mode_values="${MODES//,/ }"
dynamic_accum_values="${DYNAMIC_ACCUM_LIST//,/ }"

echo "mode,steps,dynamic_accum,run,run_order,c_ane_util_pct,go_ane_util_pct,c_compile_pct,go_compile_pct,c_avg_step_ms,go_avg_step_ms,go_vs_c_ratio,summary_file" >"$CSV"

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

extract_compile_pct() {
	local log_file="$1"
	local line
	line="$(grep -E "Compile time:" "$log_file" | tail -n1 || true)"
	if [[ -z "$line" ]]; then
		line="$(grep -E "^Compile:[[:space:]]+" "$log_file" | tail -n1 || true)"
	fi
	if [[ -z "$line" ]]; then
		echo "n/a"
		return
	fi
	echo "$line" | sed -E 's/.*\(([^)]*,[[:space:]]*)?([0-9]+(\.[0-9]+)?)%\).*/\2/'
}

current_skip_build="$SKIP_BUILD"

for mode in $mode_values; do
	if [[ "$mode" != "ane" && "$mode" != "ane-dynamic" ]]; then
		echo "unsupported mode in --modes: $mode (use ane or ane-dynamic)" >&2
		exit 1
	fi
	for steps in $steps_values; do
		if ! [[ "$steps" =~ ^[0-9]+$ ]] || [[ "$steps" -lt 1 ]]; then
			echo "invalid step count in --steps-list: $steps" >&2
			exit 1
		fi
		accum_iter_values="10"
		if [[ "$mode" == "ane-dynamic" ]]; then
			accum_iter_values="$dynamic_accum_values"
		fi
		for accum in $accum_iter_values; do
			if ! [[ "$accum" =~ ^[0-9]+$ ]] || [[ "$accum" -lt 1 ]]; then
				echo "invalid dynamic accum in --dynamic-accum-list: $accum" >&2
				exit 1
			fi
			for ((run = 1; run <= RUNS; run++)); do
				run_order="c-go"
				if (( run % 2 == 0 )); then
					run_order="go-c"
				fi
				cmd=("$COMPARE"
					"--steps" "$steps"
					"--data" "$DATA"
					"--c-model" "$MODEL"
					"--go-model" "$MODEL"
					"--go-backend" "$mode"
					"--c-mode" "$mode"
					"--dynamic-accum" "$accum"
					"--run-order" "$run_order"
					"--warmup-steps" "$WARMUP_STEPS"
					"--dynamic-bin" "$DYNAMIC_BIN"
					"--out-dir" "$COMPARE_OUT_DIR")
				if [[ "$current_skip_build" == "1" ]]; then
					cmd+=("--skip-build")
				fi
				if [[ "$NO_ANE_EXTRAS" == "1" ]]; then
					cmd+=("--no-ane-extras")
				fi
				if [[ "$ALLOW_MISMATCH" == "1" ]]; then
					cmd+=("--allow-mismatch")
				fi

				echo "[$(date +%H:%M:%S)] mode=$mode steps=$steps accum=$accum run=$run order=$run_order" | tee -a "$LOG"
				before_summary="$(ls -1t "$COMPARE_OUT_DIR"/summary_*.txt 2>/dev/null | head -n1 || true)"
				"${cmd[@]}" | tee -a "$LOG"
				after_summary="$(ls -1t "$COMPARE_OUT_DIR"/summary_*.txt 2>/dev/null | head -n1 || true)"
				if [[ -z "$after_summary" || "$after_summary" == "$before_summary" ]]; then
					echo "failed to discover summary file for mode=$mode steps=$steps accum=$accum run=$run" | tee -a "$LOG"
					exit 1
				fi

				c_log="$(extract_field "$after_summary" "c_log")"
				go_log="$(extract_field "$after_summary" "go_log")"
				c_avg="$(extract_field "$after_summary" "avg_c_train_step_ms")"
				go_avg="$(extract_field "$after_summary" "avg_go_train_step_ms")"
				ratio="$(extract_field "$after_summary" "go_vs_c_train_step_ratio")"
				c_util="$(extract_ane_util "$c_log")"
				go_util="$(extract_ane_util "$go_log")"
				c_compile_pct="$(extract_compile_pct "$c_log")"
				go_compile_pct="$(extract_compile_pct "$go_log")"

				echo "$mode,$steps,$accum,$run,$run_order,$c_util,$go_util,$c_compile_pct,$go_compile_pct,$c_avg,$go_avg,$ratio,$after_summary" >>"$CSV"

				current_skip_build=1
			done
		done
	done
done

{
	echo "results_csv=$CSV"
	echo "raw_log=$LOG"
	echo
	echo "aggregate by mode+steps+accum (mean over runs):"
	echo "mode,steps,dynamic_accum,runs,mean_c_util,mean_go_util,mean_c_compile_pct,mean_go_compile_pct,mean_c_step_ms,mean_go_step_ms,mean_go_vs_c_ratio"
	awk -F, 'NR>1 {
		key=$1","$2","$3
		n[key]++
		if ($6 ~ /^[0-9]+(\.[0-9]+)?$/) { c_util[key]+=$6; c_util_n[key]++ }
		if ($7 ~ /^[0-9]+(\.[0-9]+)?$/) { go_util[key]+=$7; go_util_n[key]++ }
		if ($8 ~ /^[0-9]+(\.[0-9]+)?$/) { c_comp[key]+=$8; c_comp_n[key]++ }
		if ($9 ~ /^[0-9]+(\.[0-9]+)?$/) { go_comp[key]+=$9; go_comp_n[key]++ }
		if ($10 ~ /^[0-9]+(\.[0-9]+)?$/) { c_ms[key]+=$10; c_ms_n[key]++ }
		if ($11 ~ /^[0-9]+(\.[0-9]+)?$/) { go_ms[key]+=$11; go_ms_n[key]++ }
		if ($12 ~ /^[0-9]+(\.[0-9]+)?$/) { ratio[key]+=$12; ratio_n[key]++ }
	}
	END {
		for (k in n) {
			cu = (c_util_n[k] > 0) ? sprintf("%.4f", c_util[k]/c_util_n[k]) : "n/a"
			gu = (go_util_n[k] > 0) ? sprintf("%.4f", go_util[k]/go_util_n[k]) : "n/a"
			cc = (c_comp_n[k] > 0) ? sprintf("%.4f", c_comp[k]/c_comp_n[k]) : "n/a"
			gc = (go_comp_n[k] > 0) ? sprintf("%.4f", go_comp[k]/go_comp_n[k]) : "n/a"
			cm = (c_ms_n[k] > 0) ? sprintf("%.4f", c_ms[k]/c_ms_n[k]) : "n/a"
			gm = (go_ms_n[k] > 0) ? sprintf("%.4f", go_ms[k]/go_ms_n[k]) : "n/a"
			gr = (ratio_n[k] > 0) ? sprintf("%.6f", ratio[k]/ratio_n[k]) : "n/a"
			printf "%s,%d,%s,%s,%s,%s,%s,%s,%s\n",
				k, n[k], cu, gu, cc, gc, cm, gm, gr
		}
	}' "$CSV" | sort
	echo
	echo "best_go_util_row:"
	awk -F, '
	BEGIN {
		best = -1
	}
	NR > 1 {
		if ($7 ~ /^[0-9]+(\.[0-9]+)?$/) {
			v = $7 + 0
			if (v > best) {
				best = v
				row = $0
			}
		}
	}
	END {
		if (best < 0) {
			print "n/a"
		} else {
			print row
		}
	}' "$CSV"
	echo
	echo "best_go_step_row:"
	awk -F, '
	BEGIN {
		best = -1
	}
	NR > 1 {
		if ($11 ~ /^[0-9]+(\.[0-9]+)?$/) {
			v = $11 + 0
			if (best < 0 || v < best) {
				best = v
				row = $0
			}
		}
	}
	END {
		if (best < 0) {
			print "n/a"
		} else {
			print row
		}
	}' "$CSV"
} | tee "$SUMMARY"

echo
echo "wrote:"
echo "  $CSV"
echo "  $SUMMARY"
echo "  $LOG"
