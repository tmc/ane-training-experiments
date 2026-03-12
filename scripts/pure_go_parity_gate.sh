#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAP="$ROOT/scripts/compare_c_objc_vs_go_training_both_orders.sh"

if [[ ! -x "$WRAP" ]]; then
	echo "missing executable script: $WRAP" >&2
	exit 1
fi

THRESHOLD="1.05"
MAX_LOSS_DELTA="0.01"
STEPS="20"
REPEATS="3"
WARMUP="1"
PROFILE="c-match"
SKIP_BUILD=0
TRAINER_BACKEND="direct"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--threshold)
		THRESHOLD="$2"
		shift 2
		;;
	--steps)
		STEPS="$2"
		shift 2
		;;
	--repeats)
		REPEATS="$2"
		shift 2
		;;
	--warmup-steps)
		WARMUP="$2"
		shift 2
		;;
	--profile)
		PROFILE="$2"
		shift 2
		;;
	--trainer-backend)
		TRAINER_BACKEND="$2"
		shift 2
		;;
	--max-loss-delta)
		MAX_LOSS_DELTA="$2"
		shift 2
		;;
	--skip-build)
		SKIP_BUILD=1
		shift
		;;
	*)
		echo "unknown flag: $1" >&2
		exit 2
		;;
	esac
done

args=(--steps "$STEPS" --repeats "$REPEATS" --warmup-steps "$WARMUP" --profile "$PROFILE" --go-backend ane --trainer-backend "$TRAINER_BACKEND")
if [[ "$SKIP_BUILD" -eq 1 ]]; then
	args+=(--skip-build)
fi
OUT="$("$WRAP" "${args[@]}")"
printf '%s\n' "$OUT"

RATIO="$(printf '%s\n' "$OUT" | awk -F= '$1=="all_ratio_median"{print $2}' | tail -n1)"
if [[ -z "$RATIO" || "$RATIO" == "n/a" ]]; then
	echo "parity gate failed: missing all_ratio_median" >&2
	exit 3
fi

ALL_IMPLS="$(printf '%s\n' "$OUT" | awk -F= '$1=="all_go_impl_backends"{print $2}' | tail -n1)"
if [[ -z "$ALL_IMPLS" ]]; then
	echo "parity gate failed: missing all_go_impl_backends provenance" >&2
	exit 5
fi
IFS=',' read -r -a impls <<<"$ALL_IMPLS"
for impl in "${impls[@]}"; do
	[[ -z "$impl" ]] && continue
	if [[ "$TRAINER_BACKEND" == "direct" ]]; then
		if [[ "$impl" != "direct" && "$impl" != direct_* ]]; then
			echo "parity gate failed: expected all go_impl_backends in {direct,direct_*}, got $ALL_IMPLS" >&2
			exit 6
		fi
		continue
	fi
	if [[ "$impl" != "$TRAINER_BACKEND" ]]; then
		echo "parity gate failed: expected all go_impl_backends=$TRAINER_BACKEND, got $ALL_IMPLS" >&2
		exit 6
	fi
done

STRICT_OK="$(printf '%s\n' "$OUT" | awk -F= '$1=="all_strict_workload_pairing"{print $2}' | tail -n1)"
if [[ -z "$STRICT_OK" ]]; then
	echo "parity gate failed: missing all_strict_workload_pairing" >&2
	exit 9
fi
if [[ "$STRICT_OK" != "1" ]]; then
	echo "parity gate failed: strict workload pairing required (all_strict_workload_pairing=$STRICT_OK)" >&2
	exit 10
fi

LOSS_DELTA="$(printf '%s\n' "$OUT" | awk -F= '$1=="all_delta_loss_median"{print $2}' | tail -n1)"
if [[ -z "$LOSS_DELTA" || "$LOSS_DELTA" == "n/a" ]]; then
	echo "parity gate failed: missing all_delta_loss_median" >&2
	exit 7
fi
ABS_LOSS_DELTA="$(awk -v x="$LOSS_DELTA" 'BEGIN { if (x < 0) x = -x; printf "%.12f", x }')"
if ! awk -v d="$ABS_LOSS_DELTA" -v m="$MAX_LOSS_DELTA" 'BEGIN { exit !(d <= m) }'; then
	echo "parity gate failed: all_delta_loss_median(abs)=$ABS_LOSS_DELTA max=$MAX_LOSS_DELTA" >&2
	exit 8
fi

if awk -v r="$RATIO" -v t="$THRESHOLD" 'BEGIN { exit !(r <= t) }'; then
	echo "pure-go parity gate PASS: all_ratio_median=$RATIO threshold=$THRESHOLD all_delta_loss_median(abs)=$ABS_LOSS_DELTA max=$MAX_LOSS_DELTA"
	exit 0
fi

echo "pure-go parity gate FAIL: all_ratio_median=$RATIO threshold=$THRESHOLD" >&2
exit 4
