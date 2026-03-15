# ANE Step-Latency Optimization

Autonomous optimization of Go Apple Neural Engine training step latency. Target: match C/ObjC trainer at ~89ms/step (M4 Max, seq=384).

## Setup

1. **Branch**: `git checkout -b autoresearch/<tag>` from main (tag = today's date, e.g. `mar15`).
2. **Read in-scope files**: See CLAUDE.md "Editable files" section for the full list.
3. **Verify benchmark**: `ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=1s -count=1 ./ane/storiesane/`
4. **Install benchstat**: `go install golang.org/x/perf/cmd/benchstat@latest`

## Metric

**`ns/op` from `BenchmarkDynamicTrainStep`** — lower is better.

## Loop

1. Baseline: `ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=3s -count=5 ./ane/storiesane/ | tee bench_before.txt`
2. Edit files.
3. Tests: `go test -run='Test[^E]' -count=1 ./ane/storiesane/ ./ane/dynamicmatmul/`
4. Commit: `git add <files> && git commit -m "<description>"`
5. Benchmark: `ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=3s -count=5 ./ane/storiesane/ | tee bench_after.txt`
6. Compare: `benchstat bench_before.txt bench_after.txt`
7. Keep if `ns/op` decreased significantly. Revert if not. Log to `results.tsv`.
8. Repeat.

## Current State

| Metric | Value |
|--------|-------|
| Go step latency | ~155ms (under load ~66) |
| C/ObjC baseline | ~89ms |
| ANE HW time | ~94ms (64% of wall) |
| dW GEMM (overlapped) | ~34ms |
| Final head | ~42ms |
| Gap to close | ~66ms |

## Architecture

See CLAUDE.md for full details. Key facts:
- TinyStories 15M: dim=64, hidden=128, heads=8, layers=12, vocab=32K
- Channel-first (CF) data layout, FP16 at IOSurface boundaries
- Hybrid ANE+CPU: ANE runs forward/backward kernels, CPU handles GEMM/loss/optimizer
- GCD dispatch for dW GEMM jobs (overlapped with ANE)
- ~100 ANE kernel evals per step (24 forward + 72 backward + 5 final head)

## Optimization Surfaces

See CLAUDE.md "What you can change" for details. Priority areas:
1. IOSurface lock/unlock reduction (kernel-level, ~20-50us each)
2. Forward/backward pipelining (overlap CPU/ANE work)
3. CGo crossing reduction (batch operations)
4. Vectorization (vDSP/vForce for remaining scalar loops)

## Constraints

- Stack arrays in CGo cause SIGBUS — use heap allocation
- dim=64 is too small for CGo vectorization inner loops
- IOSurface lock/unlock is kernel-level — ~20-50us per pair
- Keep `accel_other.go`, `io_patch_other.go`, `gemm_accel_stub.go` in sync
- All shared MIL kernels — RequestPool/SharedEvent APIs don't work yet (see xane_api_warts.md)
