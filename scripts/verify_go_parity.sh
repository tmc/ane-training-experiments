#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-$ROOT/training/tinystories_data00.bin}"
GO_ANE_MODEL="${GO_ANE_MODEL:-/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc}"
GO_CPU_MODEL="${GO_CPU_MODEL:-/Volumes/tmc/go/src/github.com/assets/models/stories110M.bin}"
RUN_C="${RUN_C:-1}"

if [[ ! -f "$DATA" ]]; then
  echo "missing data file: $DATA" >&2
  exit 1
fi
if [[ ! -d "$GO_ANE_MODEL" ]]; then
  echo "missing go ane model (.mlmodelc): $GO_ANE_MODEL" >&2
  exit 1
fi
if [[ ! -f "$GO_CPU_MODEL" ]]; then
  echo "missing go cpu model (.bin): $GO_CPU_MODEL" >&2
  exit 1
fi

cd "$ROOT"

echo "[1/6] go test ./..."
go test ./... -count=1 >/tmp/verify_go_parity_gotest.log

echo "[2/6] go full trainer smoke (backend=full)"
FULL_OUT="$(go run ./cmd/train-stories-ane-go \
  -backend=full \
  -model "$GO_CPU_MODEL" \
  -data "$DATA" \
  -steps 1 \
  -json=false \
  -save-every 0)"
echo "$FULL_OUT" >/tmp/verify_go_parity_full.log
if ! grep -Eq "step [0-9]+\\s+loss=" /tmp/verify_go_parity_full.log; then
  echo "full trainer smoke failed: no step line" >&2
  exit 1
fi

echo "[3/6] go ANE trainer smoke (backend=ane)"
ANE_OUT="$(go run ./cmd/train-stories-ane-go \
  -backend=ane \
  -model "$GO_ANE_MODEL" \
  -data "$DATA" \
  -steps 1 \
  -json=false \
  -save-every 0)"
echo "$ANE_OUT" >/tmp/verify_go_parity_ane.log
if ! grep -Eq "step [0-9]+ loss=" /tmp/verify_go_parity_ane.log; then
  echo "ANE trainer smoke failed: no step line" >&2
  exit 1
fi

echo "[4/6] go CPU reference trainer smoke (backend=cpu)"
CPU_OUT="$(go run ./cmd/train-stories-ane-go \
  -backend=cpu \
  -model "$GO_CPU_MODEL" \
  -data "$DATA" \
  -steps 1 \
  -json=false \
  -save-every 0)"
echo "$CPU_OUT" >/tmp/verify_go_parity_cpu.log
if ! grep -Eq "step [0-9]+" /tmp/verify_go_parity_cpu.log; then
  echo "CPU trainer smoke failed: no step line" >&2
  exit 1
fi

echo "[5/6] go asymmetric pipeline smoke"
PIPE_OUT="$(go run ./cmd/asymmetric-pipeline-go \
  -model "$GO_ANE_MODEL" \
  -iters 1 \
  -json=false)"
echo "$PIPE_OUT" >/tmp/verify_go_parity_pipeline.log
if ! grep -q "avg_total_ms=" /tmp/verify_go_parity_pipeline.log; then
  echo "pipeline smoke failed: no avg_total_ms line" >&2
  exit 1
fi

C_SUMMARY="skipped"
if [[ "$RUN_C" == "1" ]]; then
  if [[ ! -x "$ROOT/training/train_large_ane" ]]; then
    echo "[6/6] building training/train_large_ane"
    make -C "$ROOT/training" train_large_ane >/tmp/verify_go_parity_make.log
  fi
  echo "[6/6] objc train_large_ane smoke (1 step)"
  C_OUT="$(cd "$ROOT/training" && ./train_large_ane \
    --model "$GO_CPU_MODEL" \
    --data "$DATA" \
    --steps 1)"
  echo "$C_OUT" >/tmp/verify_go_parity_c.log
  C_SUMMARY="$(grep -E "Avg train:" /tmp/verify_go_parity_c.log | tail -n1 | sed 's/^ *//')"
fi

echo
echo "== parity smoke summary =="
echo "go test: PASS"
echo "go full trainer: PASS"
echo "go ane trainer: PASS"
echo "go cpu trainer: PASS"
echo "go asymmetric pipeline: PASS"
echo "objc train_large_ane: $C_SUMMARY"
echo
echo "logs:"
echo "  /tmp/verify_go_parity_gotest.log"
echo "  /tmp/verify_go_parity_full.log"
echo "  /tmp/verify_go_parity_ane.log"
echo "  /tmp/verify_go_parity_cpu.log"
echo "  /tmp/verify_go_parity_pipeline.log"
if [[ "$RUN_C" == "1" ]]; then
  echo "  /tmp/verify_go_parity_c.log"
fi
