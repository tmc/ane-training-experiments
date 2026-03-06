#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="$ROOT/scripts/compare_c_objc_vs_go_training.sh"

if [[ ! -x "$BASE_SCRIPT" ]]; then
	echo "missing executable base script: $BASE_SCRIPT" >&2
	exit 1
fi

USER_ARGS=()
REPEATS=1
while [[ $# -gt 0 ]]; do
	case "$1" in
	--run-order)
		# This wrapper controls run order explicitly.
		shift 2
		;;
	--repeats)
		REPEATS="$2"
		shift 2
		;;
	*)
		USER_ARGS+=("$1")
		shift
		;;
	esac
done
if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
	echo "repeats must be a positive integer: $REPEATS" >&2
	exit 2
fi

run_once() {
	local order="$1"
	local out
	out="$("$BASE_SCRIPT" "${USER_ARGS[@]}" --run-order "$order")"
	printf '%s\n' "$out"
}

extract_field() {
	local key="$1"
	local text="$2"
	printf '%s\n' "$text" | awk -F= -v k="$key" '$1 == k { print $2 }' | tail -n1
}

mean_from_list() {
	printf '%s\n' "$1" | awk '/^-?[0-9]+([.][0-9]+)?$/ { sum+=$1; n++ } END { if (n==0) print "n/a"; else printf "%.6f", sum/n }'
}

median_from_list() {
	local sorted=()
	local line
	while IFS= read -r line; do
		[[ -z "$line" ]] && continue
		[[ "$line" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || continue
		sorted+=("$line")
	done < <(printf '%s\n' "$1" | awk 'NF { print $1 }' | sort -n)
	local n="${#sorted[@]}"
	if [[ "$n" -eq 0 ]]; then
		echo "n/a"
		return
	fi
	if (( n % 2 == 1 )); then
		printf "%.6f" "${sorted[$((n/2))]}"
		return
	fi
	awk -v a="${sorted[$((n/2-1))]}" -v b="${sorted[$((n/2))]}" 'BEGIN { printf "%.6f", (a+b)/2.0 }'
}

run_group() {
	local order="$1"
	local out ratio delta
	local ratios="" deltas=""
	local i=1
	while [[ "$i" -le "$REPEATS" ]]; do
		echo "== compare run-order=$order (run $i/$REPEATS) ==" >&2
		out="$(run_once "$order")"
		printf '%s\n' "$out" >&2
		ratio="$(extract_field "go_vs_c_train_step_ratio" "$out")"
		delta="$(extract_field "avg_delta_step_ms(go-c)" "$out")"
		if [[ -n "$ratio" ]]; then
			ratios+="$ratio"$'\n'
		fi
		if [[ -n "$delta" ]]; then
			deltas+="$delta"$'\n'
		fi
		echo >&2
		i=$((i + 1))
	done
	echo "$ratios"$'\x1f'"$deltas"
}

CGO_GROUP="$(run_group c-go)"
GOC_GROUP="$(run_group go-c)"

RATIO_CGO_LIST="${CGO_GROUP%%$'\x1f'*}"
DELTA_CGO_LIST="${CGO_GROUP#*$'\x1f'}"
RATIO_GOC_LIST="${GOC_GROUP%%$'\x1f'*}"
DELTA_GOC_LIST="${GOC_GROUP#*$'\x1f'}"

RATIO_CGO_MEAN="$(mean_from_list "$RATIO_CGO_LIST")"
RATIO_CGO_MEDIAN="$(median_from_list "$RATIO_CGO_LIST")"
RATIO_GOC_MEAN="$(mean_from_list "$RATIO_GOC_LIST")"
RATIO_GOC_MEDIAN="$(median_from_list "$RATIO_GOC_LIST")"
DELTA_CGO_MEAN="$(mean_from_list "$DELTA_CGO_LIST")"
DELTA_CGO_MEDIAN="$(median_from_list "$DELTA_CGO_LIST")"
DELTA_GOC_MEAN="$(mean_from_list "$DELTA_GOC_LIST")"
DELTA_GOC_MEDIAN="$(median_from_list "$DELTA_GOC_LIST")"

ALL_RATIO_LIST="$RATIO_CGO_LIST"$'\n'"$RATIO_GOC_LIST"
ALL_DELTA_LIST="$DELTA_CGO_LIST"$'\n'"$DELTA_GOC_LIST"
RATIO_ALL_MEAN="$(mean_from_list "$ALL_RATIO_LIST")"
RATIO_ALL_MEDIAN="$(median_from_list "$ALL_RATIO_LIST")"
DELTA_ALL_MEAN="$(mean_from_list "$ALL_DELTA_LIST")"
DELTA_ALL_MEDIAN="$(median_from_list "$ALL_DELTA_LIST")"

echo "== aggregate parity summary =="
echo "repeats=$REPEATS"
echo "c_go_ratio_mean=$RATIO_CGO_MEAN"
echo "c_go_ratio_median=$RATIO_CGO_MEDIAN"
echo "go_c_ratio_mean=$RATIO_GOC_MEAN"
echo "go_c_ratio_median=$RATIO_GOC_MEDIAN"
echo "all_ratio_mean=$RATIO_ALL_MEAN"
echo "all_ratio_median=$RATIO_ALL_MEDIAN"
echo "c_go_delta_step_ms_mean=$DELTA_CGO_MEAN"
echo "c_go_delta_step_ms_median=$DELTA_CGO_MEDIAN"
echo "go_c_delta_step_ms_mean=$DELTA_GOC_MEAN"
echo "go_c_delta_step_ms_median=$DELTA_GOC_MEDIAN"
echo "all_delta_step_ms_mean=$DELTA_ALL_MEAN"
echo "all_delta_step_ms_median=$DELTA_ALL_MEDIAN"
