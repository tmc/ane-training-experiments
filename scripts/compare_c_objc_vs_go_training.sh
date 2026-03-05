#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STEPS=100
LR="3e-4"
DATA="$ROOT/training/tinystories_data00.bin"
DEFAULT_STORIES_MODEL="/Volumes/tmc/go/src/github.com/assets/models/stories110M.bin"
DEFAULT_GO_ANE_MODEL="$DEFAULT_STORIES_MODEL"
C_MODEL="$DEFAULT_STORIES_MODEL"
GO_MODEL=""
GO_BACKEND="ane"
C_MODE="ane"
FULL_BIN="$ROOT/training/train_large_ane"
DYNAMIC_BIN="$ROOT/training/training_dynamic/train"
DYNAMIC_ACCUM=20
FULL_ACCUM=0
VECLIB_THREADS=0
DW_CONCURRENCY=0
PROFILE="default"
SEQ_OVERRIDE=""
OUT_DIR="/tmp/ane_compare"
SKIP_BUILD=0
NO_ANE_EXTRAS=0
ANE_CLS_BWD=0
ALLOW_MISMATCH=0
WARMUP_STEPS=0
RUN_ORDER="c-go"

usage() {
	cat <<'EOF'
Usage:
  compare_c_objc_vs_go_training.sh [flags]

Flags:
  --steps N             Steps for both runs (default: 100)
  --lr F                Learning rate for both runs (default: 3e-4)
  --data PATH           TinyStories uint16 data path
  --c-model PATH        stories110M .bin path for C/ObjC run
  --go-model PATH       model path for Go run (.mlmodelc or .bin for ane, .bin for ane-dynamic/full/cpu)
  --go-backend MODE     Go backend: ane|ane-dynamic|full|cpu (default: ane)
  --c-mode MODE         C mode: ane|ane-dynamic|cpu (default: ane)
  --full-bin PATH       full trainer binary path (default: ./training/train_large_ane)
  --dynamic-bin PATH    dynamic trainer binary path (default: ./training/training_dynamic/train)
  --dynamic-accum N     accumulation steps for ane-dynamic mode (default: 20)
  --full-accum N        accumulation steps for full C/ObjC trainer (default: trainer default)
  --veclib-threads N    set VECLIB_MAXIMUM_THREADS for full C/ObjC trainer paths (default: process default)
  --dw-concurrency N    set dW async task concurrency for full C/ObjC trainer paths (default: trainer default)
  --profile NAME        preset workload profile: default|high-util
  --seq-override N      rebuild trainer binaries with SEQ=N for higher work per dispatch
  --allow-mismatch      allow non-equivalent workload pairings (default: false)
  --warmup-steps N      ignore first N overlapped steps in summary means (default: 0)
  --run-order ORDER     execution order: c-go|go-c (default: c-go)
  --out-dir DIR         Output directory for logs/csv (default: /tmp/ane_compare)
  --skip-build          Skip `make -C training train_large_ane`
  --no-ane-extras       Pass no-ane-extras to both C and Go
  --ane-cls-bwd         Enable ANE classifier backward (full/ane .bin paths)
  --help                Show this help

Environment overrides:
  C_TRAIN_CMD           Full shell command for C/ObjC training
  GO_TRAIN_CMD          Full shell command for Go training

Outputs:
  c_objc_<ts>.log
  go_<ts>.log
  c_steps_<ts>.csv
  go_steps_<ts>.csv
  delta_<ts>.csv
  summary_<ts>.txt
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--steps)
		STEPS="$2"
		shift 2
		;;
	--lr)
		LR="$2"
		shift 2
		;;
	--data)
		DATA="$2"
		shift 2
		;;
	--c-model)
		C_MODEL="$2"
		shift 2
		;;
	--go-model)
		GO_MODEL="$2"
		shift 2
		;;
	--go-backend)
		GO_BACKEND="$2"
		shift 2
		;;
		--c-mode)
			C_MODE="$2"
			shift 2
			;;
		--full-bin)
			FULL_BIN="$2"
			shift 2
			;;
		--dynamic-bin)
			DYNAMIC_BIN="$2"
			shift 2
			;;
		--dynamic-accum)
			DYNAMIC_ACCUM="$2"
			shift 2
			;;
		--full-accum)
			FULL_ACCUM="$2"
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
		--profile)
			PROFILE="$2"
			shift 2
			;;
		--seq-override)
			SEQ_OVERRIDE="$2"
			shift 2
			;;
	--out-dir)
		OUT_DIR="$2"
		shift 2
		;;
	--skip-build)
		SKIP_BUILD=1
		shift
		;;
	--no-ane-extras)
		NO_ANE_EXTRAS=1
		shift
		;;
	--ane-cls-bwd)
		ANE_CLS_BWD=1
		shift
		;;
	--help|-h)
		usage
		exit 0
		;;
		--allow-mismatch)
			ALLOW_MISMATCH=1
			shift
			;;
		--warmup-steps)
			WARMUP_STEPS="$2"
			shift 2
			;;
		--run-order)
			RUN_ORDER="$2"
			shift 2
			;;
	*)
		echo "unknown flag: $1" >&2
		usage >&2
		exit 2
		;;
	esac
done

if [[ "$PROFILE" != "default" && "$PROFILE" != "high-util" ]]; then
	echo "profile must be one of: default, high-util" >&2
	exit 1
fi
if [[ "$PROFILE" == "high-util" ]]; then
	if [[ "$FULL_ACCUM" -eq 0 ]]; then
		FULL_ACCUM=80
	fi
	if [[ -z "$SEQ_OVERRIDE" ]]; then
		SEQ_OVERRIDE=384
	fi
	if [[ "$VECLIB_THREADS" -eq 0 ]]; then
		VECLIB_THREADS=6
	fi
	if [[ "$DW_CONCURRENCY" -eq 0 ]]; then
		DW_CONCURRENCY=3
	fi
fi

if [[ -z "${GO_MODEL}" ]]; then
	case "$GO_BACKEND" in
	ane)
		GO_MODEL="$DEFAULT_GO_ANE_MODEL"
		;;
	ane-dynamic)
		GO_MODEL="$DEFAULT_STORIES_MODEL"
		;;
	full|cpu)
		GO_MODEL="$DEFAULT_STORIES_MODEL"
		;;
	esac
fi
GO_MODEL_LOWER="$(printf '%s' "$GO_MODEL" | tr '[:upper:]' '[:lower:]')"

if [[ ! -f "$DATA" ]]; then
	echo "data file not found: $DATA" >&2
	exit 1
fi
if [[ ! -f "$C_MODEL" ]]; then
	echo "c model file not found: $C_MODEL" >&2
	exit 1
fi
if [[ "$GO_BACKEND" != "full" && "$GO_BACKEND" != "ane" && "$GO_BACKEND" != "ane-dynamic" && "$GO_BACKEND" != "cpu" ]]; then
	echo "go backend must be 'full', 'ane', 'ane-dynamic', or 'cpu': $GO_BACKEND" >&2
	exit 1
fi
if ! [[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
	echo "warmup-steps must be a non-negative integer: $WARMUP_STEPS" >&2
	exit 1
fi
if [[ "$RUN_ORDER" != "c-go" && "$RUN_ORDER" != "go-c" ]]; then
	echo "run-order must be one of: c-go, go-c" >&2
	exit 1
fi
if ! [[ "$DYNAMIC_ACCUM" =~ ^[0-9]+$ ]] || [[ "$DYNAMIC_ACCUM" -lt 1 ]]; then
	echo "dynamic-accum must be a positive integer: $DYNAMIC_ACCUM" >&2
	exit 1
fi
if ! [[ "$FULL_ACCUM" =~ ^[0-9]+$ ]]; then
	echo "full-accum must be a non-negative integer: $FULL_ACCUM" >&2
	exit 1
fi
if ! [[ "$VECLIB_THREADS" =~ ^[0-9]+$ ]]; then
	echo "veclib-threads must be a non-negative integer: $VECLIB_THREADS" >&2
	exit 1
fi
if ! [[ "$DW_CONCURRENCY" =~ ^[0-9]+$ ]]; then
	echo "dw-concurrency must be a non-negative integer: $DW_CONCURRENCY" >&2
	exit 1
fi
if [[ -n "$SEQ_OVERRIDE" ]]; then
	if ! [[ "$SEQ_OVERRIDE" =~ ^[0-9]+$ ]] || [[ "$SEQ_OVERRIDE" -lt 1 ]]; then
		echo "seq-override must be a positive integer: $SEQ_OVERRIDE" >&2
		exit 1
	fi
fi
if [[ "$GO_BACKEND" == "full" && ! -f "$GO_MODEL" ]]; then
	echo "go model path is not a model file for full backend: $GO_MODEL" >&2
	exit 1
fi
if [[ "$C_MODE" != "ane" && "$C_MODE" != "ane-dynamic" && "$C_MODE" != "cpu" ]]; then
	echo "c mode must be 'ane', 'ane-dynamic', or 'cpu': $C_MODE" >&2
	exit 1
fi
if [[ "$GO_BACKEND" == "ane" ]]; then
	if [[ "$GO_MODEL_LOWER" == *.bin ]]; then
		if [[ ! -f "$GO_MODEL" ]]; then
			echo "go model path is not a model file for ane/.bin mode: $GO_MODEL" >&2
			exit 1
		fi
	elif [[ ! -d "$GO_MODEL" ]]; then
		echo "go model path is not a .mlmodelc directory: $GO_MODEL" >&2
		exit 1
	fi
fi
if [[ "$GO_BACKEND" == "ane-dynamic" && ! -f "$GO_MODEL" ]]; then
	echo "go model path is not a model file for ane-dynamic backend: $GO_MODEL" >&2
	exit 1
fi
if [[ "$GO_BACKEND" == "cpu" && ! -f "$GO_MODEL" ]]; then
	echo "go model path is not a model file for cpu backend: $GO_MODEL" >&2
	exit 1
fi
if [[ "$C_MODE" == "ane-dynamic" || "$GO_BACKEND" == "ane-dynamic" ]]; then
	if [[ ! -x "$DYNAMIC_BIN" && "$SKIP_BUILD" -eq 1 ]]; then
		echo "dynamic trainer binary is not executable: $DYNAMIC_BIN" >&2
		echo "hint: build with 'make -C training/training_dynamic train' or pass --dynamic-bin PATH" >&2
		exit 1
	fi
fi
if [[ "$C_MODE" == "ane" && ! -x "$FULL_BIN" && "$SKIP_BUILD" -eq 1 ]]; then
	echo "full trainer binary is not executable: $FULL_BIN" >&2
	echo "hint: build with 'make -C training train_large_ane' or pass --full-bin PATH" >&2
	exit 1
fi

strict_pair=0
backend_pair=0
if [[ "$C_MODE" == "cpu" && "$GO_BACKEND" == "cpu" ]]; then
	strict_pair=1
	backend_pair=1
fi
if [[ "$C_MODE" == "ane" && "$GO_BACKEND" == "full" ]]; then
	strict_pair=1
	backend_pair=1
fi
if [[ "$C_MODE" == "ane" && "$GO_BACKEND" == "ane" ]]; then
	backend_pair=1
	if [[ "$GO_MODEL_LOWER" == *.bin ]]; then
		strict_pair=1
	fi
fi
if [[ "$C_MODE" == "ane-dynamic" && "$GO_BACKEND" == "ane-dynamic" ]]; then
	strict_pair=1
	backend_pair=1
fi
if [[ "$backend_pair" -ne 1 && "$ALLOW_MISMATCH" -ne 1 ]]; then
	echo "requested pairing is not backend-aligned: c-mode=$C_MODE go-backend=$GO_BACKEND" >&2
	echo "use ane/ane, ane/full, or cpu/cpu; or pass --allow-mismatch to force" >&2
	exit 2
fi

C_FULL_BIN="$FULL_BIN"
C_DYNAMIC_BIN="$DYNAMIC_BIN"
GO_FULL_BIN="$FULL_BIN"
GO_DYNAMIC_BIN="$DYNAMIC_BIN"

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

C_LOG="$OUT_DIR/c_objc_${TS}.log"
GO_LOG="$OUT_DIR/go_${TS}.log"
C_CSV="$OUT_DIR/c_steps_${TS}.csv"
GO_CSV="$OUT_DIR/go_steps_${TS}.csv"
DELTA_CSV="$OUT_DIR/delta_${TS}.csv"
C_BATCH_CSV="$OUT_DIR/c_batches_${TS}.csv"
GO_BATCH_CSV="$OUT_DIR/go_batches_${TS}.csv"
BATCH_DELTA_CSV="$OUT_DIR/batch_delta_${TS}.csv"
SUMMARY_TXT="$OUT_DIR/summary_${TS}.txt"
C_CKPT="$OUT_DIR/c_ckpt_${TS}.bin"
GO_CKPT="$OUT_DIR/go_ckpt_${TS}.bin"
rm -f "$C_CKPT" "$GO_CKPT"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
	make -C "$ROOT/training" train_large_ane
	make -C "$ROOT/training/training_dynamic" train
	if [[ "$C_MODE" == "cpu" ]]; then
		make -C "$ROOT/training" train_large
	fi
fi
if [[ -n "$SEQ_OVERRIDE" ]]; then
	seq_full_bin="$ROOT/training/train_large_ane_seq${SEQ_OVERRIDE}"
	seq_dynamic_bin="$ROOT/training/training_dynamic/train_seq${SEQ_OVERRIDE}"
	if [[ "$SKIP_BUILD" -eq 0 || ! -x "$seq_full_bin" ]]; then
		make -B -C "$ROOT/training" train_large_ane SEQ_OVERRIDE="$SEQ_OVERRIDE" OUT="$(basename "$seq_full_bin")"
	fi
	if [[ "$SKIP_BUILD" -eq 0 || ! -x "$seq_dynamic_bin" ]]; then
		make -B -C "$ROOT/training/training_dynamic" train SEQ_OVERRIDE="$SEQ_OVERRIDE" OUT="$(basename "$seq_dynamic_bin")"
	fi
	C_FULL_BIN="$seq_full_bin"
	C_DYNAMIC_BIN="$seq_dynamic_bin"
	GO_FULL_BIN="$seq_full_bin"
	GO_DYNAMIC_BIN="$seq_dynamic_bin"
fi
if [[ "$C_MODE" == "ane" && ! -x "$C_FULL_BIN" ]]; then
	echo "full trainer binary is not executable after build: $C_FULL_BIN" >&2
	exit 1
fi
if [[ "$C_MODE" == "ane-dynamic" || "$GO_BACKEND" == "ane-dynamic" ]]; then
	if [[ ! -x "$C_DYNAMIC_BIN" ]]; then
		echo "dynamic trainer binary is not executable after build: $C_DYNAMIC_BIN" >&2
		exit 1
	fi
fi

binary_supports_flag() {
	local bin="$1"
	local flag="$2"
	if [[ ! -x "$bin" ]]; then
		return 1
	fi
	strings "$bin" 2>/dev/null | rg -F -q -- "$flag"
}

ensure_full_flag_support() {
	local missing=()
	if [[ "$VECLIB_THREADS" -gt 0 ]] && ! binary_supports_flag "$C_FULL_BIN" "--veclib-threads"; then
		missing+=("--veclib-threads")
	fi
	if [[ "$DW_CONCURRENCY" -gt 0 ]] && ! binary_supports_flag "$C_FULL_BIN" "--dw-concurrency"; then
		missing+=("--dw-concurrency")
	fi
	if [[ ${#missing[@]} -eq 0 ]]; then
		return
	fi
	if [[ -n "$SEQ_OVERRIDE" ]]; then
		echo "rebuilding seq override binary to pick up trainer flags: ${missing[*]}" >&2
		make -B -C "$ROOT/training" train_large_ane SEQ_OVERRIDE="$SEQ_OVERRIDE" OUT="$(basename "$C_FULL_BIN")"
		make -B -C "$ROOT/training/training_dynamic" train SEQ_OVERRIDE="$SEQ_OVERRIDE" OUT="$(basename "$C_DYNAMIC_BIN")"
		return
	fi
	echo "full trainer binary $C_FULL_BIN is missing support for: ${missing[*]}" >&2
	echo "rebuild training binaries (drop --skip-build) or remove unsupported flags" >&2
	exit 1
}

if [[ "$C_MODE" == "ane" || "$GO_BACKEND" == "ane" ]]; then
	ensure_full_flag_support
fi

common_extra=()
if [[ "$NO_ANE_EXTRAS" -eq 1 ]]; then
	common_extra+=(--no-ane-extras)
fi

c_flags=(--model "$C_MODEL" --data "$DATA" --steps "$STEPS" --lr "$LR" --ckpt "$C_CKPT")
if [[ "$FULL_ACCUM" -gt 0 ]]; then
	c_flags+=(--accum "$FULL_ACCUM")
fi
if [[ "$VECLIB_THREADS" -gt 0 ]]; then
	c_flags+=(--veclib-threads "$VECLIB_THREADS")
fi
if [[ "$DW_CONCURRENCY" -gt 0 ]]; then
	c_flags+=(--dw-concurrency "$DW_CONCURRENCY")
fi
if [[ "$NO_ANE_EXTRAS" -eq 1 ]]; then
	c_flags+=(--no-ane-extras)
fi
if [[ "$ANE_CLS_BWD" -eq 1 ]]; then
	c_flags+=(--ane-cls-bwd)
fi

run_c() {
	echo "running C/ObjC training..."
	set +e
	if [[ -n "${C_TRAIN_CMD:-}" ]]; then
		(
			cd "$ROOT"
			bash -lc "$C_TRAIN_CMD"
		) 2>&1 | tee "$C_LOG"
		C_RC=${PIPESTATUS[0]}
	else
		if [[ "$C_MODE" == "ane" ]]; then
			(
				cd "$ROOT"
				"$C_FULL_BIN" "${c_flags[@]}"
			) 2>&1 | tee "$C_LOG"
			C_RC=${PIPESTATUS[0]}
		elif [[ "$C_MODE" == "ane-dynamic" ]]; then
			(
				cd "$ROOT/training/training_dynamic"
				"$C_DYNAMIC_BIN" --model "$C_MODEL" --data "$DATA" --steps "$STEPS" --lr "$LR" --accum "$DYNAMIC_ACCUM" --ckpt "$C_CKPT"
			) 2>&1 | tee "$C_LOG"
			C_RC=${PIPESTATUS[0]}
		else
			(
				cd "$ROOT/training"
				./train_large --model "$C_MODEL" --steps "$STEPS" --lr "$LR"
			) 2>&1 | tee "$C_LOG"
			C_RC=${PIPESTATUS[0]}
		fi
	fi
	set -e
}

go_extra=()
if [[ "$NO_ANE_EXTRAS" -eq 1 ]]; then
	go_extra+=(-no-ane-extras)
fi
go_cmd=(go run ./cmd/train-stories-ane-go)
go_flags=()
if [[ "$C_MODE" == "cpu" && "$GO_BACKEND" == "cpu" ]]; then
	# Strict apples-to-apples CPU reference pairing with deterministic seed.
	go_cmd=(go run ./cmd/train-stories-go)
	go_flags=(
		-model "$GO_MODEL"
		-data "$DATA"
		-steps "$STEPS"
		-lr "$LR"
		-json=true
		-save-every 0
		-accum-steps "$STEPS"
		-seed 42
	)
else
	go_flags=(
		-backend "$GO_BACKEND"
		-model "$GO_MODEL"
		-data "$DATA"
		-ckpt "$GO_CKPT"
		-steps "$STEPS"
		-lr "$LR"
		-json=false
		-save-every 0
	)
	if [[ "$GO_BACKEND" == "ane-dynamic" ]]; then
		go_flags+=(-dynamic-bin "$GO_DYNAMIC_BIN" -accum-steps "$DYNAMIC_ACCUM")
	fi
	if [[ "$GO_BACKEND" == "full" || ( "$GO_BACKEND" == "ane" && "$GO_MODEL_LOWER" == *.bin ) ]]; then
		go_flags+=(-full-bin "$GO_FULL_BIN")
		if [[ "$FULL_ACCUM" -gt 0 ]]; then
			go_flags+=(-full-accum-steps "$FULL_ACCUM")
		fi
		if [[ "$VECLIB_THREADS" -gt 0 ]]; then
			go_flags+=(-veclib-threads "$VECLIB_THREADS")
		fi
		if [[ "$DW_CONCURRENCY" -gt 0 ]]; then
			go_flags+=(-dw-concurrency "$DW_CONCURRENCY")
		fi
	fi
	if [[ "$NO_ANE_EXTRAS" -eq 1 ]]; then
		go_flags+=(-no-ane-extras)
	fi
	if [[ "$ANE_CLS_BWD" -eq 1 && ( "$GO_BACKEND" == "full" || ( "$GO_BACKEND" == "ane" && "$GO_MODEL_LOWER" == *.bin ) ) ]]; then
		go_flags+=(-ane-cls-bwd)
	fi
fi

run_go() {
	echo "running Go training..."
	set +e
	if [[ -n "${GO_TRAIN_CMD:-}" ]]; then
		(
			cd "$ROOT"
			bash -lc "$GO_TRAIN_CMD"
		) 2>&1 | tee "$GO_LOG"
		GO_RC=${PIPESTATUS[0]}
	else
		(
			cd "$ROOT"
			"${go_cmd[@]}" "${go_flags[@]}"
		) 2>&1 | tee "$GO_LOG"
		GO_RC=${PIPESTATUS[0]}
	fi
	set -e
}

if [[ "$RUN_ORDER" == "c-go" ]]; then
	run_c
	run_go
else
	run_go
	run_c
fi

extract_steps() {
	local log="$1"
	local csv="$2"
	echo "step,loss,step_ms" >"$csv"
	awk '
	function jsonnum(key,     re, s) {
		re = "\"" key "\":[0-9.]+"
		if (match($0, re)) {
			s = substr($0, RSTART, RLENGTH)
			sub(/^"[^"]+":/, "", s)
			return s + 0
		}
		return ""
	}
	function set_step(step, loss, step_ms) {
		if (step == "" || loss == "" || step_ms == "") {
			return
		}
		if (!(step in seen) || source == 1) {
			seen[step] = 1
			losses[step] = loss
			stepms[step] = step_ms
		}
	}
	{
		step = ""
		loss = ""
		step_ms = ""
		batch_step_ms = ""
		source = 0

		if (index($0, "\"type\":\"step\"") > 0) {
			step = jsonnum("step")
			loss = jsonnum("loss")
			step_ms = 0
			x = jsonnum("t_ane")
			if (x != "") step_ms += x
			x = jsonnum("t_io")
			if (x != "") step_ms += x
			x = jsonnum("t_compile")
			if (x != "") step_ms += x
			x = jsonnum("t_cls")
			if (x != "") step_ms += x
			x = jsonnum("t_elem")
			if (x != "") step_ms += x
			x = jsonnum("t_rms")
			if (x != "") step_ms += x
			x = jsonnum("t_cblas_wait")
			if (x != "") step_ms += x
			source = 0
			set_step(step, loss, step_ms)
			next
		}

		for (i = 1; i <= NF; i++) {
			if ($i == "step" && i < NF) {
				step = $(i + 1)
			}
			if (index($i, "loss=") == 1) {
				loss = substr($i, 6)
			}
			if (index($i, "step_ms=") == 1) {
				step_ms = substr($i, 9)
			}
			if (index($i, "ms/step") > 0) {
				token = $i
				gsub(/^[()]+/, "", token)
				gsub(/ms\/step[)\]]*$/, "", token)
				if (token ~ /^[0-9.]+$/) {
					batch_step_ms = token
				}
			}
		}
		if (step != "") {
			last_step = step
		}
		if (loss != "") {
			last_loss = loss
		}
		if (step != "" && loss != "" && step_ms != "") {
			source = 1
			set_step(step, loss, step_ms)
			next
		}
		if (batch_step_ms != "" && last_step != "" && last_loss != "") {
			if (!(last_step in seen)) {
				set_step(last_step, last_loss, batch_step_ms)
			}
		}
	}
	END {
		for (s in seen) {
			printf "%s,%s,%s\n", s, losses[s], stepms[s]
		}
	}
	' "$log" | sort -t, -k1,1n >>"$csv"
}

extract_steps "$C_LOG" "$C_CSV"
extract_steps "$GO_LOG" "$GO_CSV"

extract_batches() {
	local log="$1"
	local csv="$2"
	echo "batch_idx,train_ms,step_ms" >"$csv"
	awk '
	function jsonnum(key,     re, s) {
		re = "\"" key "\":[0-9.]+"
		if (match($0, re)) {
			s = substr($0, RSTART, RLENGTH)
			sub(/^"[^"]+":/, "", s)
			return s + 0
		}
		return ""
	}
	{
		if (index($0, "\"type\":\"batch\"") > 0) {
			train_ms = jsonnum("train_ms")
			step_ms = jsonnum("ms_per_step")
			if (train_ms != "" && step_ms != "") {
				any_json = 1
				jidx = jsonnum("batch")
				if (jidx == "") {
					jcount++
					jidx = jcount
				}
				jseen[jidx] = 1
				jtrain[jidx] = train_ms
				jstep[jidx] = step_ms
			}
			next
		}

		train_ms = ""
		step_ms = ""
		for (i = 1; i <= NF; i++) {
			if (index($i, "train=") == 1) {
				token = substr($i, 7)
				gsub(/ms$/, "", token)
				if (token ~ /^[0-9.]+$/) {
					train_ms = token
				}
			}
			if (index($i, "ms/step") > 0) {
				token = $i
				gsub(/^[()]+/, "", token)
				gsub(/ms\/step[)\]]*$/, "", token)
				if (token ~ /^[0-9.]+$/) {
					step_ms = token
				}
			}
		}
		if (train_ms != "" && step_ms != "") {
			tcount++
			tseen[tcount] = 1
			ttrain[tcount] = train_ms
			tstep[tcount] = step_ms
		}
	}
	END {
		if (any_json || jcount > 0) {
			for (i in jseen) {
				printf "%d,%s,%s\n", i, jtrain[i], jstep[i]
			}
		} else {
			for (i in tseen) {
				printf "%d,%s,%s\n", i, ttrain[i], tstep[i]
			}
		}
	}
	' "$log" >>"$csv"
}

extract_batches "$C_LOG" "$C_BATCH_CSV"
extract_batches "$GO_LOG" "$GO_BATCH_CSV"

tmp_rows="$(mktemp)"
awk -F, '
NR==FNR {
	if (NR > 1) {
		c_loss[$1] = $2
		c_ms[$1] = $3
	}
	next
}
NR > 1 {
	if ($1 in c_loss) {
		d_loss = $2 - c_loss[$1]
		d_ms = $3 - c_ms[$1]
		printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", $1, c_loss[$1], $2, d_loss, c_ms[$1], $3, d_ms
	}
}
' "$C_CSV" "$GO_CSV" >"$tmp_rows"

{
	echo "step,c_loss,go_loss,delta_loss,c_step_ms,go_step_ms,delta_step_ms"
	cat "$tmp_rows"
} >"$DELTA_CSV"

tmp_batch_rows="$(mktemp)"
awk -F, '
NR==FNR {
	if (NR > 1) {
		c_train[$1] = $2
		c_step[$1] = $3
	}
	next
}
NR > 1 {
	if ($1 in c_step) {
		d_train = $2 - c_train[$1]
		d_step = $3 - c_step[$1]
		printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", $1, c_train[$1], $2, d_train, c_step[$1], $3, d_step
	}
}
' "$C_BATCH_CSV" "$GO_BATCH_CSV" >"$tmp_batch_rows"

{
	echo "batch_idx,c_train_ms,go_train_ms,delta_train_ms,c_step_ms,go_step_ms,delta_step_ms"
	cat "$tmp_batch_rows"
} >"$BATCH_DELTA_CSV"

detect_go_mode() {
	local log="$1"
	if grep -q "=== ANE Training: Stories110M (ANE-offloaded) ===" "$log"; then
		echo "ane_offloaded"
		return
	fi
	if grep -q "=== ANE Dynamic Training: Stories110M (12 layers) ===" "$log"; then
		echo "ane_dynamic"
		return
	fi
	if grep -q "=== ANE Stories Training (Go direct) ===" "$log"; then
		echo "ane_clientmodel"
		return
	fi
	if grep -q "=== ANE Training: Stories110M Go" "$log"; then
		echo "cpu_reference"
		return
	fi
	echo "unknown"
}

detect_c_mode() {
	local log="$1"
	if grep -q "=== ANE Training: Stories110M (ANE-offloaded) ===" "$log"; then
		echo "ane_offloaded"
		return
	fi
	if grep -q "=== ANE Dynamic Training: Stories110M (12 layers) ===" "$log"; then
		echo "ane_dynamic"
		return
	fi
	if grep -q "=== ANE Training: Stories110M (12 layers) ===" "$log"; then
		echo "cpu_reference"
		return
	fi
	echo "unknown"
}

extract_ane_util() {
	local log="$1"
	local line
	line="$(grep -E "ANE utilization:" "$log" | tail -n1 || true)"
	if [[ -z "$line" ]]; then
		echo "n/a"
		return
	fi
	echo "$line" | sed -E 's/.*ANE utilization:[[:space:]]*([0-9.]+)%.*/\1/'
}

extract_compile_pct() {
	local log="$1"
	local line
	line="$(grep -E "Compile time:" "$log" | tail -n1 || true)"
	if [[ -z "$line" ]]; then
		line="$(grep -E "^Compile:[[:space:]]+" "$log" | tail -n1 || true)"
	fi
	if [[ -z "$line" ]]; then
		echo "n/a"
		return
	fi
	echo "$line" | sed -E 's/.*\(([^)]*,[[:space:]]*)?([0-9]+(\.[0-9]+)?)%\).*/\2/'
}

GO_MODE="$(detect_go_mode "$GO_LOG")"
C_RUNTIME_MODE="$(detect_c_mode "$C_LOG")"
GO_ANE_UTIL="$(extract_ane_util "$GO_LOG")"
C_ANE_UTIL="$(extract_ane_util "$C_LOG")"
GO_COMPILE_PCT="$(extract_compile_pct "$GO_LOG")"
C_COMPILE_PCT="$(extract_compile_pct "$C_LOG")"

{
	echo "c_exit_code=$C_RC"
	echo "go_exit_code=$GO_RC"
	echo "go_backend_flag=$GO_BACKEND"
	echo "dynamic_accum=$DYNAMIC_ACCUM"
	echo "full_accum=$FULL_ACCUM"
	echo "profile=$PROFILE"
	echo "veclib_threads=$VECLIB_THREADS"
	echo "dw_concurrency=$DW_CONCURRENCY"
	echo "ane_cls_bwd=$ANE_CLS_BWD"
	echo "seq_override=${SEQ_OVERRIDE:-<none>}"
	echo "run_order=$RUN_ORDER"
	echo "c_model=$C_MODEL"
	echo "go_model=$GO_MODEL"
	echo "c_full_bin=$C_FULL_BIN"
	echo "go_full_bin=$GO_FULL_BIN"
	echo "c_dynamic_bin=$C_DYNAMIC_BIN"
	echo "go_dynamic_bin=$GO_DYNAMIC_BIN"
	echo "strict_workload_pairing=$strict_pair"
	echo "backend_pairing=$backend_pair"
	echo "allow_mismatch=$ALLOW_MISMATCH"
	echo "c_mode_flag=$C_MODE"
	echo "c_mode=$C_RUNTIME_MODE"
	echo "go_mode=$GO_MODE"
	echo "c_ane_util_pct=$C_ANE_UTIL"
	echo "go_ane_util_pct=$GO_ANE_UTIL"
	echo "c_compile_pct=$C_COMPILE_PCT"
	echo "go_compile_pct=$GO_COMPILE_PCT"
		awk -F, '
		BEGIN {
			n = 0
			sum_dl = 0
			sum_dm = 0
	}
	{
		n++
		sum_dl += $4
		sum_dm += $7
	}
	END {
		if (n == 0) {
			print "overlap_steps=0"
			print "avg_delta_loss(go-c)=n/a"
			print "avg_delta_step_ms(go-c)=n/a"
		} else {
			printf "overlap_steps=%d\n", n
			printf "avg_delta_loss(go-c)=%.6f\n", sum_dl / n
			printf "avg_delta_step_ms(go-c)=%.6f\n", sum_dm / n
		}
		}
		' "$tmp_rows"
		awk -F, -v warmup="$WARMUP_STEPS" '
		BEGIN {
			n = 0
			sum_dl = 0
			sum_dm = 0
			sum_c = 0
			sum_g = 0
		}
		{
			if (NR <= warmup) {
				next
			}
			n++
			sum_dl += $4
			sum_dm += $7
			sum_c += $5
			sum_g += $6
		}
		END {
			printf "warmup_steps=%d\n", warmup
			if (n == 0) {
				print "steady_overlap_steps=0"
				print "steady_avg_delta_loss(go-c)=n/a"
				print "steady_avg_delta_step_ms(go-c)=n/a"
				print "steady_go_vs_c_step_ratio=n/a"
			} else {
				printf "steady_overlap_steps=%d\n", n
				printf "steady_avg_delta_loss(go-c)=%.6f\n", sum_dl / n
				printf "steady_avg_delta_step_ms(go-c)=%.6f\n", sum_dm / n
				if (sum_c > 0) {
					printf "steady_go_vs_c_step_ratio=%.6f\n", sum_g / sum_c
				} else {
					print "steady_go_vs_c_step_ratio=n/a"
				}
			}
		}
		' "$tmp_rows"
	awk -F, '
	BEGIN {
		n = 0
		sum_train = 0
		sum_step = 0
	}
	{
		n++
		sum_train += $4
		sum_step += $7
	}
	END {
		if (n == 0) {
			print "batch_overlap=0"
			print "avg_batch_delta_train_ms(go-c)=n/a"
			print "avg_batch_delta_step_ms(go-c)=n/a"
		} else {
			printf "batch_overlap=%d\n", n
			printf "avg_batch_delta_train_ms(go-c)=%.6f\n", sum_train / n
			printf "avg_batch_delta_step_ms(go-c)=%.6f\n", sum_step / n
		}
	}
	' "$tmp_batch_rows"
	awk -F, '
	BEGIN {
		n = 0
		sum_c_step = 0
		sum_go_step = 0
	}
	{
		n++
		sum_c_step += $5
		sum_go_step += $6
	}
	END {
		if (n == 0) {
			print "avg_c_train_step_ms=n/a"
			print "avg_go_train_step_ms=n/a"
			print "go_vs_c_train_step_ratio=n/a"
		} else {
			c = sum_c_step / n
			g = sum_go_step / n
			printf "avg_c_train_step_ms=%.6f\n", c
			printf "avg_go_train_step_ms=%.6f\n", g
			if (c > 0) {
				printf "go_vs_c_train_step_ratio=%.6f\n", g / c
			} else {
				print "go_vs_c_train_step_ratio=n/a"
			}
		}
	}
	' "$([[ -s "$tmp_batch_rows" ]] && printf "%s" "$tmp_batch_rows" || printf "%s" "$tmp_rows")"
	echo "c_log=$C_LOG"
	echo "go_log=$GO_LOG"
	echo "c_csv=$C_CSV"
	echo "go_csv=$GO_CSV"
	echo "delta_csv=$DELTA_CSV"
	echo "c_batch_csv=$C_BATCH_CSV"
	echo "go_batch_csv=$GO_BATCH_CSV"
	echo "batch_delta_csv=$BATCH_DELTA_CSV"
	if [[ "$C_RUNTIME_MODE" == "ane_offloaded" && "$GO_MODE" == "cpu_reference" ]]; then
		echo "warning=go path is cpu_reference while c path is ane_offloaded; performance parity is not expected"
	fi
	if [[ "$C_RUNTIME_MODE" == "cpu_reference" && "$GO_MODE" == "ane_clientmodel" ]]; then
		echo "warning=go path is ane_clientmodel while c path is cpu_reference; performance parity is not expected"
	fi
	if [[ "$GO_BACKEND" == "ane" ]]; then
		if [[ "$GO_MODEL_LOWER" == *.bin ]]; then
			if [[ "$GO_MODE" != "ane_offloaded" ]]; then
				echo "warning=go backend flag was ane(.bin) but runtime mode was $GO_MODE"
			fi
		elif [[ "$GO_MODE" != "ane_clientmodel" ]]; then
			echo "warning=go backend flag was ane(.mlmodelc) but runtime mode was $GO_MODE"
		fi
	fi
	if [[ "$GO_BACKEND" == "ane-dynamic" && "$GO_MODE" != "ane_dynamic" ]]; then
		echo "warning=go backend flag was ane-dynamic but runtime mode was $GO_MODE"
	fi
	if [[ "$GO_BACKEND" == "cpu" && "$GO_MODE" != "cpu_reference" ]]; then
		echo "warning=go backend flag was cpu but runtime mode was $GO_MODE"
	fi
	if [[ "$C_MODE" == "ane" && "$C_RUNTIME_MODE" != "ane_offloaded" ]]; then
		echo "warning=c mode flag was ane but runtime mode was $C_RUNTIME_MODE"
	fi
	if [[ "$C_MODE" == "ane-dynamic" && "$C_RUNTIME_MODE" != "ane_dynamic" ]]; then
		echo "warning=c mode flag was ane-dynamic but runtime mode was $C_RUNTIME_MODE"
	fi
	if [[ "$C_MODE" == "cpu" && "$C_RUNTIME_MODE" != "cpu_reference" ]]; then
		echo "warning=c mode flag was cpu but runtime mode was $C_RUNTIME_MODE"
	fi
	if [[ "$C_MODE" == "ane" && ("$GO_BACKEND" == "ane" || "$GO_BACKEND" == "full") && "$GO_MODEL" != "$C_MODEL" ]]; then
		echo "warning=model mismatch between c and go runs; metrics are not directly comparable"
	fi
	if [[ "$C_MODE" == "ane-dynamic" && "$GO_BACKEND" == "ane-dynamic" && "$GO_MODEL" != "$C_MODEL" ]]; then
		echo "warning=model mismatch between c and go dynamic runs; metrics are not directly comparable"
	fi
	if [[ "$C_RUNTIME_MODE" == "cpu_reference" && "$GO_MODE" == "cpu_reference" ]]; then
		echo "warning=workload mismatch remains: c run is full train_large 12-layer path; go run is simplified stories cpu reference"
	fi
	if [[ "$strict_pair" -ne 1 ]]; then
		echo "warning=pairing is backend-aligned but not strict workload-equivalent; compare train-phase trends, not absolute latency"
	fi
} >"$SUMMARY_TXT"

rm -f "$tmp_rows"
rm -f "$tmp_batch_rows"

if command -v rg >/dev/null 2>&1; then
	grep_cmd=(rg)
else
	grep_cmd=(grep -E)
fi

echo
echo "==== Summary ===="
cat "$SUMMARY_TXT"
echo
echo "==== Last 10 step lines (C/ObjC) ===="
"${grep_cmd[@]}" "step " "$C_LOG" | tail -n 10 || true
echo
echo "==== Last 10 step lines (Go) ===="
"${grep_cmd[@]}" "^step " "$GO_LOG" | tail -n 10 || true

if [[ "$C_RC" -ne 0 || "$GO_RC" -ne 0 ]]; then
	echo
	echo "one or more runs failed (c_rc=$C_RC go_rc=$GO_RC)" >&2
	exit 1
fi
