#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  c_dynamic_to_benchfmt.sh [--cpu N] [--no-suffix] [log_file...]

Reads C dynamic trainer logs and emits Go benchfmt-compatible lines for
benchstat. If no file is passed, reads one log from stdin.

Expected log snippets:
  Total steps:  31
  Compile:      6232ms (one-time, 53.1%)
  Train time:   5534ms (276.7ms/step)
  Wall time:    12.3s

Output benchmarks:
  BenchmarkDynamicStartupCompile-N
  BenchmarkDynamicTrainStep-N
  BenchmarkDynamicWallStep-N
  BenchmarkDynamicUpdateOverheadStep-N
EOF
}

cpu_suffix=1
no_suffix=0
while [[ $# -gt 0 ]]; do
	case "$1" in
	--cpu)
		cpu_suffix="$2"
		shift 2
		;;
	--no-suffix)
		no_suffix=1
		shift
		;;
	--help|-h)
		usage
		exit 0
		;;
	--)
		shift
		break
		;;
	*)
		break
		;;
	esac
done

if [[ "$no_suffix" -eq 0 ]] && ! [[ "$cpu_suffix" =~ ^[1-9][0-9]*$ ]]; then
	echo "error: --cpu must be a positive integer" >&2
	exit 1
fi

parse_log() {
	local src="$1"
	local steps=""
	local compile_ms=""
	local step_ms=""
	local wall_s=""

	while IFS= read -r line; do
		if [[ "$line" == *"Total steps:"* ]]; then
			local v="$line"
			v="${v#*Total steps:}"
			v="${v#"${v%%[![:space:]]*}"}"
			v="${v%%[!0-9]*}"
			steps="$v"
		fi
		if [[ "$line" == *"Compile:"* && "$line" == *"ms"* ]]; then
			local v="$line"
			v="${v#*Compile:}"
			v="${v#"${v%%[![:space:]]*}"}"
			v="${v%%ms*}"
			compile_ms="$v"
		fi
		if [[ "$line" == *"Train time:"* && "$line" == *"ms/step"* ]]; then
			local v="$line"
			v="${v#*\(}"
			v="${v%%ms/step*}"
			step_ms="$v"
		fi
		if [[ "$line" == *"Wall time:"* && "$line" == *"s"* ]]; then
			local v="$line"
			v="${v#*Wall time:}"
			v="${v#"${v%%[![:space:]]*}"}"
			v="${v%%s*}"
			wall_s="$v"
		fi
	done <"$src"

	if [[ -z "$steps" ]]; then
		echo "error: could not parse total steps from $src" >&2
		return 1
	fi
	if [[ -z "$compile_ms" ]]; then
		echo "error: could not parse compile ms from $src" >&2
		return 1
	fi
	if [[ -z "$step_ms" ]]; then
		echo "error: could not parse train-step ms from $src" >&2
		return 1
	fi
	if [[ -z "$wall_s" ]]; then
		echo "error: could not parse wall seconds from $src" >&2
		return 1
	fi
	if ! [[ "$steps" =~ ^[1-9][0-9]*$ ]]; then
		echo "error: parsed invalid total steps from $src: $steps" >&2
		return 1
	fi

	local compile_ns
	local step_ns
	local wall_step_ns
	local overhead_step_ns
	compile_ns="$(awk -v v="$compile_ms" 'BEGIN { printf("%.0f", v * 1000000.0) }')"
	step_ns="$(awk -v v="$step_ms" 'BEGIN { printf("%.0f", v * 1000000.0) }')"
	wall_step_ns="$(awk -v wall="$wall_s" -v n="$steps" 'BEGIN { printf("%.0f", wall * 1000000000.0 / n) }')"
	overhead_step_ns="$(awk -v wall="$wall_s" -v train_step="$step_ms" -v n="$steps" 'BEGIN {
		over = wall * 1000000000.0 / n - train_step * 1000000.0;
		if (over < 0) over = 0;
		printf("%.0f", over)
	}')"

	local compile_name="BenchmarkDynamicStartupCompile"
	local step_name="BenchmarkDynamicTrainStep"
	local wall_step_name="BenchmarkDynamicWallStep"
	local overhead_step_name="BenchmarkDynamicUpdateOverheadStep"
	if [[ "$no_suffix" -eq 0 ]]; then
		compile_name="${compile_name}-${cpu_suffix}"
		step_name="${step_name}-${cpu_suffix}"
		wall_step_name="${wall_step_name}-${cpu_suffix}"
		overhead_step_name="${overhead_step_name}-${cpu_suffix}"
	fi

	printf '%s\t1\t%s ns/op\n' "$compile_name" "$compile_ns"
	printf '%s\t1\t%s ns/op\n' "$step_name" "$step_ns"
	printf '%s\t1\t%s ns/op\n' "$wall_step_name" "$wall_step_ns"
	printf '%s\t1\t%s ns/op\n' "$overhead_step_name" "$overhead_step_ns"
}

if [[ "$#" -eq 0 ]]; then
	tmp="$(mktemp)"
	trap 'rm -f "$tmp"' EXIT
	cat >"$tmp"
	parse_log "$tmp"
	exit 0
fi

for log_file in "$@"; do
	if [[ ! -f "$log_file" ]]; then
		echo "error: log file not found: $log_file" >&2
		exit 1
	fi
	parse_log "$log_file"
done
