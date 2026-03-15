# ANE — Go Apple Neural Engine Training

This is a Go library for training transformer models on Apple Neural Engine. The primary optimization target is training step latency — getting the Go training path to match the C/ObjC trainer's performance.

## Setup

To set up a new experiment session:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `ane/storiesane/accel_darwin.go` — Accelerate vDSP/vvexpf wrappers (CGo). Editable.
   - `ane/storiesane/accel_other.go` — pure Go fallbacks. Keep in sync with accel_darwin.go.
   - `ane/storiesane/iosurface_accel_darwin.go` — CGo IOSurface read/write/copy. Editable.
   - `ane/storiesane/train_full.go` — backward pass, dW job submission, optimizer. Editable.
   - `ane/storiesane/engine.go` — Engine struct, Open, Step, Prepare. Editable with care.
   - `ane/storiesane/forward_common.go` — CPU forward kernels. Editable.
   - `ane/storiesane/loss_accel_darwin.go` — parallelized cross-entropy loss. Editable.
   - `ane/storiesane/gemm_accel_darwin.go` — CBLAS GEMM wrappers. Editable.
   - `ane/dynamicmatmul/executor.go` — dynamic matmul ANE executor. Editable.
   - `ane/dynamicmatmul/io_patch_darwin.go` — CGo IOSurface for dynamic matmul. Editable.
   - `ane/model/eval_cgo_darwin.go` — CGo objc_msgSend for ANE eval. Editable.
4. **Verify test data**: Check `ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=1s -count=1 ./ane/storiesane/` runs.
5. **Install benchstat**: `go install golang.org/x/perf/cmd/benchstat@latest`
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment measures steady-state training step latency (ms/op) and memory allocations (B/op, allocs/op).

### Editable files

**Tier 1 — Hot path optimizations** (highest impact):
- `ane/storiesane/accel_darwin.go` — vectorized math (vDSP, vvexpf, Accelerate)
- `ane/storiesane/iosurface_accel_darwin.go` — IOSurface CGo wrappers
- `ane/storiesane/loss_accel_darwin.go` — cross-entropy loss parallelization
- `ane/storiesane/gemm_accel_darwin.go` — CBLAS GEMM wrappers
- `ane/dynamicmatmul/io_patch_darwin.go` — dynamic matmul IOSurface CGo
- `ane/model/eval_cgo_darwin.go` — ANE eval dispatch CGo

**Tier 2 — Training internals** (deeper changes):
- `ane/storiesane/train_full.go` — backward pass, dW job submission, optimizer
- `ane/storiesane/forward_common.go` — CPU forward kernels
- `ane/storiesane/engine.go` — Engine struct, step logic
- `ane/storiesane/grad_tasks_darwin.go` — dW GEMM worker pool

**Tier 3 — Architecture** (major changes, high risk):
- `ane/storiesane/offload_darwin.go` — ANE kernel offload
- `ane/storiesane/layer_darwin.go` — ANE layer forward dispatch
- `ane/storiesane/backward_darwin.go` — ANE backward dispatch
- `ane/storiesane/runtime.go` — ANE kernel compilation, weight refresh

### Read-only files
- `ane/storiesane/*_test.go` — test files
- `ane/model/model_darwin.go` — model kernel management
- `ane/model/shared_mil_darwin.go` — shared MIL compilation

### Keep in sync
- `ane/storiesane/accel_other.go` must have the same function signatures as `accel_darwin.go`
- `ane/dynamicmatmul/io_patch_other.go` must have the same function signatures as `io_patch_darwin.go`
- `ane/storiesane/gemm_accel_stub.go` must have the same function signatures as `gemm_accel_darwin.go`

**The goal is simple: get the lowest step latency (ms/op).** The C/ObjC trainer baseline is ~89ms/step at seq=384 on M4 Max.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better performance is a great outcome.

**The first run**: Your very first run should always be to establish the baseline.

## Benchmarking with benchstat

Use Go benchmarks + `benchstat` to measure the impact of each change with statistical rigor.

**Capture a baseline** (before changing anything):
```bash
ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=3s -count=5 ./ane/storiesane/ | tee bench_before.txt
```

**After editing**, capture the new results:
```bash
ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=3s -count=5 ./ane/storiesane/ | tee bench_after.txt
```

**Compare:**
```bash
benchstat bench_before.txt bench_after.txt
```

Key metrics:
- `ns/op` — **this is the metric you are optimizing** (lower is better)
- `B/op` — bytes allocated per step (lower is better)
- `allocs/op` — allocations per step (lower is better)

A change is worth keeping if `ns/op` decreased with statistical significance. If benchstat shows `~` (no significant difference), the change has no effect — discard it.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

```
commit	ns_op	B_op	allocs_op	status	description
```

1. git commit hash (short, 7 chars)
2. ns/op (e.g. 153000000)
3. B/op (e.g. 1402000)
4. allocs/op (e.g. 12444)
5. status: `keep`, `discard`, or `crash`
6. short text description

NOTE: do not commit `results.tsv` — leave it untracked by git.

## The experiment loop

LOOP FOREVER:

1. Capture baseline benchmarks: `ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=3s -count=5 ./ane/storiesane/ | tee bench_before.txt`
2. Edit files with an experimental idea.
3. Verify it compiles and tests pass: `go test -run='Test[^E]' -count=1 ./ane/storiesane/ ./ane/dynamicmatmul/`
4. git commit: `git add <specific files> && git commit -m "<description>"`
5. Run benchmarks: `ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=3s -count=5 ./ane/storiesane/ | tee bench_after.txt`
6. Compare: `benchstat bench_before.txt bench_after.txt`
7. If `ns/op` improved (decreased) with statistical significance:
   - Keep the git commit.
   - `mv bench_after.txt bench_before.txt` (new baseline).
   - Log results to `results.tsv`.
8. If `ns/op` is equal or worse:
   - `git revert HEAD` to undo.
   - Log results to `results.tsv` with status `discard`.
9. If the build or run crashed:
   - If easy to fix (typo, bad constant), fix and re-run.
   - If fundamentally broken, `git revert HEAD`, log `crash`, move on.
10. Go to step 2.

**Timeout**: Each benchmark run should take ~30 seconds. If a run exceeds 5 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep or away. You are autonomous. If you run out of ideas, think harder — re-read the editable files, profile with `-cpuprofile=/tmp/cpu.prof`, try combining approaches. The loop runs until the human interrupts you.

## What you can change

### Vectorization (accel_darwin.go, loss_accel_darwin.go)
- Replace scalar loops with vDSP/vForce calls
- Change chunk sizes for vvexpf
- Fuse multiple operations into single CGo calls
- Add new Accelerate wrappers

### IOSurface operations (iosurface_accel_darwin.go, io_patch_darwin.go)
- Batch multiple IOSurface writes into single lock/unlock cycles
- Fuse write + eval + read into single CGo calls
- Reduce total lock/unlock count per step

### GEMM (gemm_accel_darwin.go)
- Batch multiple CBLAS calls into single CGo crossings
- Fuse dW accumulation across layers
- Change worker pool size or scheduling

### Cross-entropy loss (loss_accel_darwin.go)
- Change parallelization strategy (fewer/more workers)
- Compact vocab (only compute softmax for active tokens)
- Transpose logits for better cache locality

### Backward pass (train_full.go)
- Reorder operations to overlap CPU/ANE work
- Reduce buffer copies between operations
- Batch gradient accumulation differently

### Forward path (forward_common.go, layer_darwin.go)
- Vectorize remaining scalar loops
- Reduce intermediate buffer allocations
- Fuse operations

### Important constraints
- **Stack arrays in CGo cause SIGBUS** — always use heap allocation (`make()`) for buffers passed to C
- **RMS norm inner loop (dim=64) is too small** for CGo vectorization — scalar loop is faster
- **IOSurface lock/unlock is kernel-level** — each call costs ~20-50µs regardless of data size
- **CGo crossing overhead is ~1µs** — only worth vectorizing if the compute savings exceed this
- **Keep accel_other.go, io_patch_other.go, gemm_accel_stub.go in sync** with their darwin counterparts

## Architecture

### Data layout
- Channel-first (CF): `tensor[channel * seq + token]`
- FP16 conversion at IOSurface boundaries for ANE kernels
- FP32 compute on CPU

### Training pipeline (per step)
1. Forward: embed → 12 × (RMS norm → QKV → attention → FFN) → final RMS norm
2. Final head: classifier forward (ANE dynamic matmul) → cross-entropy loss
3. Backward: reverse of forward, ANE hybrid backward for FFN/attention
4. dW GEMM: accumulated on worker pool, overlapped with RMS backward
5. Optimizer: Adam update (CGo), weight refresh to IOSurfaces

### Model (TinyStories 15M)
- Vocab: 32,000 (Llama2 BPE)
- Dim: 64
- Hidden: 128
- Heads: 8
- Layers: 12
- Default sequence length: 384

### Current performance (M4 Max, seq=384)
- Go trainer: ~152ms/step (under load), estimated ~120-130ms normal
- C/ObjC trainer: ~89ms/step
- Ratio: ~1.4x

### CPU profile breakdown
- cgocall: 61% (CBLAS GEMM + vDSP + IOSurface — useful work)
- softmaxStridedCEBatch: 28% (parallelized, ~10ms wall)
- ANE HW execution: 16%
- CBLAS GEMM (dW): 11%
- IOSurface lock/unlock: 15% (kernel sync, hard to reduce)
- purego objc.Send: 1.1% (metrics path only)

## Test commands

```bash
# Smoke test (correctness + race safety)
go test -run='Test[^E]' -count=1 ./ane/storiesane/

# Full test suite
go test -run='Test[^E]' -count=1 ./ane/storiesane/ ./ane/dynamicmatmul/ ./ane/model/

# Dynamic train step benchmark
ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=3s -count=5 ./ane/storiesane/

# CPU profile
ANE_BENCH=1 go test -run='^$' -bench=BenchmarkDynamicTrainStep -benchtime=3s -count=1 -cpuprofile=/tmp/cpu.prof ./ane/storiesane/
go tool pprof -top -cum /tmp/cpu.prof

# Hotspot benchmark
go test -run='^$' -bench=BenchmarkDirectGoCPUHotspots -benchtime=5s -count=5 ./ane/storiesane/
```
