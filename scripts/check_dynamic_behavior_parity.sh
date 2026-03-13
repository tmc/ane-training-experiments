#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  check_dynamic_behavior_parity.sh [flags]

Runs C dynamic trainer and Go dynamic trainer with matching Stories config and
compares overlapping printed step losses.

Flags:
  --steps N           total steps to run (default: 31)
  --accum N           accumulation steps (default: 10)
  --lr X              learning rate (default: 3e-4)
  --seq N             sequence length for Go run (default: 256)
  --print-every N     Go print interval (default: 10)
  --model PATH        model path (default: ../assets/models/stories110M.bin)
  --data PATH         token data path (default: training/tinystories_data00.bin)
  --tolerance X       max allowed abs loss delta (default: 0.50)
  --timeout-sec N     timeout per C/Go run in seconds (default: 1200)
  --c-compact         use C vocab compaction mode (default uses --no-compact)
  --skip-build        skip C rebuild
  --keep-logs         keep output directory under /tmp
  -h, --help          show this help
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${script_dir}/.." && pwd)"

steps=31
accum=10
lr="3e-4"
seq=256
print_every=10
model="${root}/../assets/models/stories110M.bin"
data="${root}/training/tinystories_data00.bin"
tolerance="0.50"
timeout_sec=1200
c_no_compact=1
skip_build=0
keep_logs=0

while [[ $# -gt 0 ]]; do
	case "$1" in
	--steps)
		steps="$2"
		shift 2
		;;
	--accum)
		accum="$2"
		shift 2
		;;
	--lr)
		lr="$2"
		shift 2
		;;
	--seq)
		seq="$2"
		shift 2
		;;
	--print-every)
		print_every="$2"
		shift 2
		;;
	--model)
		model="$2"
		shift 2
		;;
	--data)
		data="$2"
		shift 2
		;;
	--tolerance)
		tolerance="$2"
		shift 2
		;;
	--timeout-sec)
		timeout_sec="$2"
		shift 2
		;;
	--c-compact)
		c_no_compact=0
		shift
		;;
	--skip-build)
		skip_build=1
		shift
		;;
	--keep-logs)
		keep_logs=1
		shift
		;;
	-h|--help)
		usage
		exit 0
		;;
	*)
		echo "error: unknown flag: $1" >&2
		exit 1
		;;
	esac
done

if [[ ! -f "$model" ]]; then
	echo "error: model not found: $model" >&2
	exit 1
fi
if [[ ! -f "$data" ]]; then
	echo "error: data not found: $data" >&2
	exit 1
fi

out_dir="$(mktemp -d /tmp/ane_behavior_parity_XXXXXX)"
if [[ "$keep_logs" -eq 0 ]]; then
	trap 'rm -rf "$out_dir"' EXIT
fi

c_log="$out_dir/c.log"
go_log="$out_dir/go.log"
c_loss="$out_dir/c.loss"
go_loss="$out_dir/go.loss"
cmp_loss="$out_dir/loss.tsv"

run_with_timeout() {
	if command -v timeout >/dev/null 2>&1; then
		timeout "$timeout_sec" "$@"
		return
	fi
	if command -v gtimeout >/dev/null 2>&1; then
		gtimeout "$timeout_sec" "$@"
		return
	fi
	perl -e 'alarm shift @ARGV; exec @ARGV' "$timeout_sec" "$@"
}

run_and_capture() {
	local log="$1"
	shift
	set +e
	run_with_timeout "$@" >"$log" 2>&1
	local status=$?
	set -e
	if [[ "$status" -eq 0 ]]; then
		return
	fi
	echo "error: command failed (status=$status): $*" >&2
	echo "log: $log" >&2
	tail -n 80 "$log" >&2 || true
	exit "$status"
}

extract_losses() {
	local in="$1"
	local out="$2"
	awk '
		/^step[[:space:]]+[0-9]+/ {
			step = $2
			loss = ""
			for (i = 1; i <= NF; i++) {
				if ($i ~ /^loss=/) {
					split($i, a, "=")
					loss = a[2]
					break
				}
			}
			if (loss != "") {
				printf "%d %.8f\n", step, loss + 0
			}
		}
	' "$in" >"$out"
}

echo "out_dir=$out_dir"
echo "config: model=$model data=$data steps=$steps accum=$accum lr=$lr seq=$seq"

if [[ "$skip_build" -eq 0 ]]; then
	echo "building C dynamic trainer (MODEL=stories110m, SEQ_OVERRIDE=$seq, ACCUM_STEPS_OVERRIDE=$accum)"
	make -B -C "${root}/training/training_dynamic" \
		train MODEL=stories110m SEQ_OVERRIDE="$seq" ACCUM_STEPS_OVERRIDE="$accum" >/dev/null
fi

echo "running C dynamic trainer"
extra_c_flags=""
if [[ "$c_no_compact" -eq 1 ]]; then
	extra_c_flags="--no-compact"
fi
run_and_capture "$c_log" bash -lc \
	"cd '$root/training/training_dynamic' && ./train --steps '$steps' --accum '$accum' --lr '$lr' --model '$model' --data '$data' $extra_c_flags"

echo "running Go dynamic trainer"
run_and_capture "$go_log" bash -lc \
	"cd '$root' && go run ./cmd/train-stories-dynamic-go --steps '$steps' --accum '$accum' --lr '$lr' --seq '$seq' --print-every '$print_every' --model '$model' --data '$data'"

extract_losses "$c_log" "$c_loss"
extract_losses "$go_log" "$go_loss"

if [[ ! -s "$c_loss" ]]; then
	echo "error: no C step-loss lines parsed" >&2
	tail -n 80 "$c_log" >&2 || true
	exit 1
fi
if [[ ! -s "$go_loss" ]]; then
	echo "error: no Go step-loss lines parsed" >&2
	tail -n 80 "$go_log" >&2 || true
	exit 1
fi

awk '
	NR == FNR { c[$1] = $2; next }
	($1 in c) {
		d = c[$1] - $2
		if (d < 0) d = -d
		printf "%d %.8f %.8f %.8f\n", $1, c[$1], $2, d
		n++
	}
	END {
		if (n == 0) {
			exit 3
		}
	}
' "$c_loss" "$go_loss" >"$cmp_loss" || {
	echo "error: no overlapping loss steps between C and Go outputs" >&2
	echo "C losses:" >&2
	cat "$c_loss" >&2
	echo "Go losses:" >&2
	cat "$go_loss" >&2
	exit 1
}

echo
echo "step   c_loss        go_loss       abs_delta"
awk '{ printf "%-6d %-12.6f %-12.6f %-12.6f\n", $1, $2, $3, $4 }' "$cmp_loss"

overlap="$(wc -l <"$cmp_loss" | tr -d ' ')"
mean_abs="$(awk '{ s += $4 } END { if (NR > 0) printf "%.8f", s / NR; else print "nan" }' "$cmp_loss")"
max_abs="$(awk 'BEGIN { m = -1 } { if ($4 > m) m = $4 } END { if (NR > 0) printf "%.8f", m; else print "nan" }' "$cmp_loss")"
max_step="$(awk 'BEGIN { m = -1; step = -1 } { if ($4 > m) { m = $4; step = $1 } } END { if (NR > 0) print step; else print "nan" }' "$cmp_loss")"

echo
echo "overlap_steps=$overlap mean_abs_delta=$mean_abs max_abs_delta=$max_abs max_step=$max_step tolerance=$tolerance"

awk -v max="$max_abs" -v tol="$tolerance" 'BEGIN { exit(max > tol ? 1 : 0) }' || {
	echo "parity=FAIL (max_abs_delta > tolerance)" >&2
	echo "logs: $out_dir" >&2
	exit 2
}

echo "parity=PASS"
