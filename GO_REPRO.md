# Go Reproduction Status

This repository now has a direct-Go path for ANE eval, training control, and Metal->ANE event orchestration.

## Mapping

- `api_exploration.m` -> `cmd/api-exploration`
- `inmem_basic.m` -> `cmd/inmem-basic`
- `inmem_peak.m` -> `cmd/inmem-peak`
- `sram_probe.m` -> `cmd/sram-probe`
- `_ANEClient` SRAM path -> `cmd/client-sram-probe`
- Cached ANE linear primitive smoke/bench -> `cmd/linear-go`
- Baseline Go trainer -> `cmd/train-go`
- Stories text generation -> `cmd/stories-generate-go`
- Direct-Go stories trainer -> `cmd/train-stories-ane-go`
- Asymmetric shared-event pipeline -> `cmd/asymmetric-pipeline-go`

## Packages

- `ane`: runtime probe API
- `ane/runtime`: ANE/CoreML framework load + class discovery
- `ane/iosurface`: IOSurface wrappers
- `ane/model`: `_ANEInMemoryModel` path
- `ane/clientmodel`: `_ANEClient` + `_ANEModel` path
- `ane/mil`: MIL/weight blob generation
- `ane/forward`: channel-first eval primitive
- `ane/linear`: cached ANE linear executor
- `ane/storiestrainer`: direct-Go Stories training control API
- `ane/pipeline`: asymmetric wait+signal pipeline helpers
- `ane/stories`: CPU-side model/optimizer/token utilities

## Wrapper-First Surfaces (`tmc/apple/private`)

These Go paths now prefer generated wrappers over raw selector sends for the
fragile ANE lifecycle:

- `_ANEClient` acquisition and diagnostics via `appleneuralengine.GetANEClientClass()`
- `_ANEModel` open/load/map/eval/unload through typed methods on `ANEClient`
- `_ANERequest` creation via `GetANERequestClass().RequestWithInputs...`
- Virtual-client and restricted-access diagnostics via typed methods
- Espresso multibuffer probe via `x/espresso`

Direct `clientmodel` shared-events execution is supported in Go.

## Parity Matrix (Go vs ObjC)

- Baseline `_ANEClient` compile/load/map/eval: `PASS`
- Shared-event CPU signal/wait API: `PASS`
- ANE completion callback barrier (`setCompletionHandler` in bridge): `PASS`
- Asymmetric Metal->ANE style orchestration (CPU-signaled variant): `PASS`
- Direct-Go stories trainer control loop (step/save/load): `PASS`
- Full `train_large_ane` workload reproduction from Go CLI (`-backend=full`): `PASS`
- Apples-to-apples C ane-offloaded vs Go `-backend=full` (step loss parity): `PASS`
- Full `train_large_ane` kernel topology parity in pure Go implementation: `IN PROGRESS`
- `_ANEChainingRequest` production path: `XFAIL (documented non-goal)`
- ANE->Metal `FW_SIGNAL=1` hardware signal path: `XFAIL (timeout/hang risk)`

## Run

```bash
# Probe ANE + wrapper reachability
go run ./cmd/ane-bindings-survey -json=false

# Direct-Go Stories trainer control loop
go run ./cmd/train-stories-ane-go \
  -model /Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc \
  -data ./training/tinystories_data00.bin \
  -steps 10 -accum-steps 5 -save-every 0 -json=false

# Full Stories110M ANE-offloaded training via Go CLI entrypoint
# (uses C/ObjC full topology binary under the hood)
go run ./cmd/train-stories-ane-go \
  -backend full \
  -model /Volumes/tmc/go/src/github.com/assets/models/stories110M.bin \
  -data ./training/tinystories_data00.bin \
  -steps 10 -json=false

# Compile-heavy parity experiment (forces clientmodel recompile each step)
go run ./cmd/train-stories-ane-go \
  -model /Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc \
  -data ./training/tinystories_data00.bin \
  -steps 10 -accum-steps 5 -save-every 0 -json=false \
  -recompile-every-step=true

# Asymmetric event pipeline smoke
go run ./cmd/asymmetric-pipeline-go \
  -model /Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc \
  -iters 5 -json=false

# Existing model/client smokes
ANE_SMOKE=1 go test ./ane/model -run Smoke -v
ANE_SMOKE=1 go test ./ane/linear -run Smoke -v

# Full direct-Go parity smoke (+ ObjC baseline)
./scripts/verify_go_parity.sh

# Side-by-side C vs Go comparison (pick explicit C mode)
# Strict apples-to-apples mode (default now): CPU reference pair only
./scripts/compare_c_objc_vs_go_training.sh --c-mode cpu --go-backend cpu \
  --go-model /Volumes/tmc/go/src/github.com/assets/models/stories110M.bin --steps 10

# Strict full-workload reproduction pair: C ANE-offloaded vs Go full backend
./scripts/compare_c_objc_vs_go_training.sh --c-mode ane --go-backend full --steps 10

# Optional: force non-equivalent research comparisons (explicit opt-in)
./scripts/compare_c_objc_vs_go_training.sh --c-mode ane --go-backend ane --steps 10 --allow-mismatch
```

## Notes

- Physical macOS hosts should assume no usable virtual client unless diagnostics prove otherwise.
- Shared-event lifecycle experiments should stay subprocess-isolated.
- Reported fp16 throughput is measured throughput, not Apple rated mixed-precision TOPS.
- For host/compiler paths that reject some MIL graphs with `InvalidMILProgram`, `ane/clientmodel`
  now supports compile fallback profiles via:
  `ANE_COMPILE_FALLBACK_PROFILES="modelType:netPlist,modelType2:netPlist2"`.
  Use `<empty>` or `-` for empty fields, for example:
  `ANE_COMPILE_FALLBACK_PROFILES="<empty>:fallback.plist,kANEFModelMIL:<empty>"`.
