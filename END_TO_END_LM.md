# End-to-End LM Plan (Go + ANE)

This document defines a practical path to run a full language model loop (prompt -> decode -> text) in this repository, then progressively move the heavy math onto ANE.

## Current State

Implemented now:
- Weight loading from `stories110M.bin` and checkpoint v2.
- CPU decoder with KV cache + RoPE + autoregressive sampling.
- CLI: `cmd/stories-generate-go`.

What this gives you today:
- End-to-end generation works with the real model weights.
- No ANE required for correctness testing.

## Minimal E2E Run

```bash
go run ./cmd/stories-generate-go \
  -model /Volumes/tmc/go/src/github.com/assets/models/stories110M.bin \
  -prompt-ids 1 -max-new 32 -temperature 0.8
```

Optional decoded text (instead of token IDs):

```bash
go run ./cmd/stories-generate-go \
  -model /Volumes/tmc/go/src/github.com/assets/models/stories110M.bin \
  -tokenizer /path/to/tokenizer.bin \
  -prompt-ids 1 -max-new 32 -temperature 0.8
```

## Target Architecture

Per token at position `t`:
1. CPU control plane schedules layer execution and sampling.
2. ANE executes dense layer math (Q/K/V/O and FFN projections).
3. CPU performs KV cache updates, attention softmax/masking, and token sampling.
4. Repeat until EOS or max tokens.

This split keeps ANE focused on high arithmetic intensity and avoids frequent small control-flow calls.

## Phase Plan

Phase 0 (done):
- CPU reference decoder for correctness.
- Stable token-level API for stepping through generation.

Phase 1:
- Add per-layer linear op abstraction: `LinearOp` interface with CPU + ANE implementations.
- Start by offloading `Wq/Wk/Wv/Wo` only.
- Keep FFN + attention on CPU.

Phase 2:
- Offload FFN (`W1/W2/W3`) and fuse SiLU where possible.
- Reuse compiled kernels across all tokens with fixed `(rows, cols)`.

Phase 3:
- Batch decode (`N` sequences) with shared kernel shapes.
- Add async pipeline: CPU softmax/sampling for token `t` while ANE computes layer `L+1` for token `t`.

Phase 4:
- Add long-context strategy (windowed KV, optional quantized KV).
- Add quality/perf regression harness (perplexity + tok/s + power).

## Kernel Strategy

Recommended initial kernel inventory:
- `QKV`: 3 projections from same input (prefer fused kernel where supported).
- `O`: attention output projection.
- `FFN-up`: `W1` and `W3`.
- `FFN-down`: `W2`.

Compile once per shape and cache by:
- op type
- input channels
- output channels
- spatial/sequence bucket
- precision mode

## Data Layout and I/O

Use channel-first flat buffers matching current ANE surface conventions:
- vector activations: `[C, S]` flattened as `C*S`
- fp16 on ANE boundary, fp32 in CPU control plane

Minimize conversion churn:
- keep persistent fp16 work buffers per layer
- avoid reallocating IOSurfaces per token

## Correctness Gates

Before each phase promotion:
1. Deterministic test with fixed seed and prompt.
2. Token-by-token parity vs CPU reference for first `N` tokens.
3. Perplexity check over a fixed validation shard.

## Performance Gates

Track and publish per run:
- `tok/s`
- `ms/token`
- ANE kernel time vs CPU time
- ANE compile count and cache hit rate

Success criteria by milestone:
- Phase 1: no regression in token outputs for deterministic decode.
- Phase 2: >= 1.5x tok/s vs CPU baseline on same prompt length.
- Phase 3: >= 2.5x tok/s for batch decode (`batch>=4`).

## Immediate Next Tasks

1. Create `ane/stories/linear.go` interface and CPU implementation backed by existing mat-vec.
2. Add `ane/stories/linear_ane.go` using `ane/model` for one projection (`Wq`) end-to-end.
3. Integrate the linear interface into `Decoder.forwardOne`.
4. Add `cmd/stories-generate-go -engine cpu|ane|auto`.
5. Add regression test for deterministic token sequence (short, fixed prompt).
