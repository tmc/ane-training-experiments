//go:build darwin

package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/maderix/ANE/ane/stories"
	"github.com/maderix/ANE/ane/storiesane"
	"github.com/maderix/ANE/ane/storiestrainer"
)

type stepJSON struct {
	Type            string  `json:"type"`
	Step            uint32  `json:"step"`
	Loss            float32 `json:"loss"`
	TANE            float64 `json:"t_ane"`
	TCPU            float64 `json:"t_cpu"`
	TIO             float64 `json:"t_io"`
	TCompile        float64 `json:"t_compile,omitempty"`
	TStartup        float64 `json:"t_startup_compile,omitempty"`
	TRefresh        float64 `json:"t_weight_refresh,omitempty"`
	TCLS            float64 `json:"t_cls"`
	TElem           float64 `json:"t_elem"`
	TRMS            float64 `json:"t_rms"`
	TCBLASWait      float64 `json:"t_cblas_wait"`
	Compiles        uint32  `json:"compiles"`
	RestartRequired bool    `json:"restart_required,omitempty"`
}

type batchJSON struct {
	Type      string  `json:"type"`
	Batch     uint32  `json:"batch"`
	CompileMS float64 `json:"compile_ms"`
	TrainMS   float64 `json:"train_ms"`
	MSPerStep float64 `json:"ms_per_step"`
}

type restartJSON struct {
	Type         string `json:"type"`
	Step         uint32 `json:"step"`
	Remaining    uint32 `json:"remaining_steps"`
	RestartCount int    `json:"restart_count"`
	MaxRestarts  int    `json:"max_restarts"`
}

const (
	cDefaultModelPath = "stories110M.bin"
	cDefaultDataPath  = "tinystories_data00.bin"
	defaultBackend    = "ane"
)

func main() {
	var (
		modelPath      = flag.String("model", "", "model path (default: stories110M.bin)")
		modelKey       = flag.String("model-key", "s", "_ANEModel key")
		dataPath       = flag.String("data", "", "TinyStories uint16 token data path (default: tinystories_data00.bin)")
		ckptPath       = flag.String("ckpt", "", "checkpoint path")
		resume         = flag.Bool("resume", false, "resume from checkpoint")
		steps          = flag.Uint("steps", 100, "training steps")
		learningRate   = flag.Float64("lr", 3e-4, "learning rate")
		noANEExtras    = flag.Bool("no-ane-extras", false, "disable ANE extras")
		aneClsBwd      = flag.Bool("ane-cls-bwd", false, "enable ANE classifier backward (full/ane with .bin model)")
		aneHybridBwd   = flag.Bool("ane-hybrid-bwd", false, "enable experimental ANE hybrid backward for .bin storiesane training")
		backend        = flag.String("backend", defaultBackend, "training backend: auto|ane|ane-dynamic|cpu|full")
		fullBin        = flag.String("full-bin", "", "path to full C/ObjC trainer binary (default: ./training/train_large_ane)")
		fullAccumSteps = flag.Uint("full-accum-steps", 0, "override full C/ObjC trainer accumulation steps (0 uses trainer default)")
		veclibThreads  = flag.Int("veclib-threads", 0, "set VECLIB_MAXIMUM_THREADS for full C/ObjC trainer (0 uses process default)")
		dwConcurrency  = flag.Int("dw-concurrency", 0, "set C/ObjC dW async task concurrency for full trainer (0 uses trainer default, currently 3)")
		dynamicBin     = flag.String("dynamic-bin", "", "path to dynamic C/ObjC trainer binary (default: ./training/training_dynamic/train)")
		seqOverride    = flag.Uint("seq-override", 0, "sequence length for direct ane .bin training; also builds and uses C/ObjC trainer binaries with compile-time SEQ override for full/ane-dynamic backends")
		dynamicANECls  = flag.Bool("dynamic-ane-cls", false, "enable ANE classifier path for ane-dynamic backend")
		dynamicClsTile = flag.Int("dynamic-ane-cls-tile", 2048, "classifier tile size for ane-dynamic backend when -dynamic-ane-cls is enabled")
		jsonOut        = flag.Bool("json", true, "emit JSON telemetry to stderr")
		accumSteps     = flag.Uint("accum-steps", 10, "steps per batch telemetry window")
		saveEvery      = flag.Uint("save-every", 10, "checkpoint every N steps (0 disables)")
		saveFinal      = flag.Bool("save-final", false, "write final checkpoint on exit")
		compileBudget  = flag.Uint("compile-budget", uint(storiestrainer.DefaultCompileBudget), "compile budget before restart_required")
		disableBudget  = flag.Bool("disable-compile-budget", false, "disable compile-budget restart signaling")
		autoRestart    = flag.Bool("auto-restart", true, "auto-restart via exec on compile-budget checkpoint boundary")
		maxRestarts    = flag.Int("max-restarts", 32, "max exec() restarts when auto-restart is enabled")
		trainerBackend = flag.String("trainer-backend", storiestrainer.BackendAuto, "storiestrainer backend for direct ane path: auto|bridge|direct")
		inputBytes     = flag.Uint("input-bytes", 4096, "input tensor bytes")
		outputBytes    = flag.Uint("output-bytes", 4096, "output tensor bytes")
		recompileEach  = flag.Bool("recompile-every-step", false, "recompile ANE kernel at each step (parity experiment)")
		diagnostics    = flag.Bool("diagnostics", false, "print model/client diagnostics at startup")
		parityMode     = flag.Bool("parity-mode", false, "enforce full train_large_ane parity profile (backend=full, seq=384, full-accum=80, veclib=6, dw=3)")
	)
	flag.Parse()
	model := defaultIfEmpty(*modelPath, cDefaultModelPath)
	data := defaultIfEmpty(*dataPath, cDefaultDataPath)

	if *backend != "auto" && *backend != "ane" && *backend != "ane-dynamic" && *backend != "cpu" && *backend != "full" {
		fatalf("backend must be one of: auto, ane, ane-dynamic, cpu, full")
	}
	if *parityMode && *backend == "cpu" {
		fatalf("parity-mode is incompatible with backend=cpu")
	}
	if *parityMode && *backend == "ane-dynamic" {
		fatalf("parity-mode is incompatible with backend=ane-dynamic")
	}
	selectedBackend := resolveSelectedBackend(*backend, model)
	if shouldAutoEnableANEHybridBackward(os.Args[1:], selectedBackend, model, *noANEExtras, *aneHybridBwd) {
		*aneHybridBwd = true
	}
	if *parityMode {
		selectedBackend = "full"
		if *seqOverride == 0 {
			*seqOverride = 384
		}
		if *fullAccumSteps == 0 {
			*fullAccumSteps = 80
		}
		if *veclibThreads == 0 {
			*veclibThreads = 6
		}
		if *dwConcurrency == 0 {
			*dwConcurrency = 3
		}
		if *noANEExtras {
			fatalf("parity-mode requires ANE extras enabled (remove -no-ane-extras)")
		}
	}
	runFullWorkload := selectedBackend == "full"
	runDynamicWorkload := selectedBackend == "ane-dynamic"
	directSequenceErr := probeDirectStoriesSequence(selectedBackend, model, *seqOverride)
	autoBridgeToFull := shouldAutoBridgeStoriesBinToFull(selectedBackend, *trainerBackend, model, directSequenceErr)
	if *inputBytes == 0 || *outputBytes == 0 {
		fatalf("input-bytes and output-bytes must be > 0")
	}
	if *trainerBackend != storiestrainer.BackendAuto && *trainerBackend != storiestrainer.BackendBridge && *trainerBackend != storiestrainer.BackendDirect {
		fatalf("trainer-backend must be one of: %s, %s, %s", storiestrainer.BackendAuto, storiestrainer.BackendBridge, storiestrainer.BackendDirect)
	}
	if *steps == 0 {
		fmt.Println("steps=0, nothing to run")
		return
	}
	if *maxRestarts < 0 {
		fatalf("max-restarts must be >= 0")
	}
	if *veclibThreads < 0 {
		fatalf("veclib-threads must be >= 0")
	}
	if *dwConcurrency < 0 {
		fatalf("dw-concurrency must be >= 0")
	}
	requiresCheckpoint := !(runFullWorkload || runDynamicWorkload)
	if *resume && requiresCheckpoint && strings.TrimSpace(*ckptPath) == "" {
		fatalf("resume requires -ckpt")
	}
	if *saveEvery > 0 && requiresCheckpoint && strings.TrimSpace(*ckptPath) == "" {
		fatalf("save-every requires -ckpt")
	}
	if *saveFinal && requiresCheckpoint && strings.TrimSpace(*ckptPath) == "" {
		fatalf("save-final requires -ckpt")
	}
	if selectedBackend == "ane" && isStoriesBinModel(model) && *veclibThreads > 0 {
		if err := os.Setenv("VECLIB_MAXIMUM_THREADS", strconv.Itoa(*veclibThreads)); err != nil {
			fatalf("set VECLIB_MAXIMUM_THREADS: %v", err)
		}
	}
	if err := validateDirectStoriesBackend(selectedBackend, *trainerBackend, model, *seqOverride, directSequenceErr); err != nil {
		fatalf("%v", err)
	}
	if autoBridgeToFull {
		fullAccum := *fullAccumSteps
		if fullAccum == 0 && *accumSteps > 0 {
			fullAccum = *accumSteps
		}
		bridgeBin := *fullBin
		bridgeSeq := int(*seqOverride)
		if strings.TrimSpace(bridgeBin) != "" && bridgeSeq > 0 {
			bridgeSeq = 0
		}
		fmt.Printf("go_impl_backend=full_c_exec(auto_seq_bridge)\n")
		fmt.Printf("auto_bridge_reason=direct_compile_unsupported requested_seq=%d\n", effectiveStoriesSequence(*seqOverride))
		fmt.Printf("auto_bridge_detail=%q\n", directSequenceErr.Error())
		if err := runFullTraining(fullRunOptions{
			binPath:       bridgeBin,
			seqOverride:   bridgeSeq,
			modelPath:     model,
			dataPath:      data,
			ckptPath:      *ckptPath,
			resume:        *resume,
			steps:         int(*steps),
			lr:            *learningRate,
			accumSteps:    int(fullAccum),
			veclibThreads: *veclibThreads,
			dwConcurrency: *dwConcurrency,
			noANEExtras:   *noANEExtras,
			aneClsBwd:     *aneClsBwd,
		}); err != nil {
			fatalf("ane auto bridge: %v", err)
		}
		return
	}
	if runFullWorkload {
		fmt.Printf("go_impl_backend=full_c_exec\n")
		if err := runFullTraining(fullRunOptions{
			binPath:       *fullBin,
			seqOverride:   int(*seqOverride),
			modelPath:     model,
			dataPath:      data,
			ckptPath:      *ckptPath,
			resume:        *resume,
			steps:         int(*steps),
			lr:            *learningRate,
			accumSteps:    int(*fullAccumSteps),
			veclibThreads: *veclibThreads,
			dwConcurrency: *dwConcurrency,
			noANEExtras:   *noANEExtras,
			aneClsBwd:     *aneClsBwd,
		}); err != nil {
			fatalf("full backend: %v", err)
		}
		return
	}
	if runDynamicWorkload {
		fmt.Printf("go_impl_backend=dynamic_c_exec\n")
		if err := runDynamicTraining(dynamicRunOptions{
			binPath:       *dynamicBin,
			seqOverride:   int(*seqOverride),
			modelPath:     model,
			dataPath:      data,
			ckptPath:      *ckptPath,
			resume:        *resume,
			steps:         int(*steps),
			lr:            *learningRate,
			accumSteps:    int(*accumSteps),
			noANEExtras:   *noANEExtras,
			aneClassifier: *dynamicANECls,
			aneClsTile:    *dynamicClsTile,
		}); err != nil {
			fatalf("dynamic backend: %v", err)
		}
		return
	}

	if selectedBackend == "cpu" {
		fmt.Printf("go_impl_backend=cpu\n")
		runCPUReference(cpuRunOptions{
			modelPath:  model,
			dataPath:   data,
			ckptPath:   *ckptPath,
			resume:     *resume,
			steps:      int(*steps),
			lr:         *learningRate,
			jsonOut:    *jsonOut,
			accumSteps: int(*accumSteps),
			saveEvery:  int(*saveEvery),
			saveFinal:  *saveFinal,
		})
		return
	}

	budget := uint32(*compileBudget)
	if *disableBudget {
		budget = 0
	}
	restartCount := restartCountFromEnv()

	trainer, err := storiestrainer.Open(storiestrainer.Options{
		ModelPath:            model,
		ModelKey:             *modelKey,
		DataPath:             data,
		InputBytes:           uint32(*inputBytes),
		OutputBytes:          uint32(*outputBytes),
		SequenceLength:       uint32(*seqOverride),
		AccumSteps:           uint32(*accumSteps),
		Steps:                uint32(*steps),
		LearningRate:         float32(*learningRate),
		DisableANEExtras:     *noANEExtras,
		HybridBackward:       *aneHybridBwd,
		GradTaskConcurrency:  *dwConcurrency,
		CompileBudget:        budget,
		DisableCompileBudget: *disableBudget,
		RecompileEachStep:    *recompileEach,
		QoS:                  storiestrainer.DefaultQoS,
		Backend:              *trainerBackend,
	})
	if err != nil {
		fatalf("open trainer: %v", err)
	}
	defer func() { _ = trainer.Close() }()

	if *resume {
		if err := trainer.LoadCheckpoint(*ckptPath); err != nil {
			fatalf("load checkpoint: %v", err)
		}
		fmt.Printf("[RESUMED from %s]\n", *ckptPath)
	}

	d := trainer.Diagnostics()
	fmt.Printf("=== ANE Stories Training (Go, backend=%s) ===\n", trainer.Backend())
	goImplBackend := effectiveGoImplBackend(selectedBackend, model, d)
	fmt.Printf("go_impl_backend=%s\n", goImplBackend)
	fmt.Printf("model=%s data=%s steps=%d lr=%.6f input_bytes=%d output_bytes=%d ane_extras=%v compile_budget=%d restart_count=%d auto_restart=%v\n",
		model, data, *steps, *learningRate, *inputBytes, *outputBytes, !*noANEExtras, budget, restartCount, *autoRestart)
	if isStoriesBinModel(model) && selectedBackend == "ane" {
		fmt.Printf("storiesane: layer_forward=%v final_head_offload=%v hybrid_backward_requested=%v hybrid_backward=%v\n",
			d.LayerForwardEnabled,
			d.FinalHeadOffloadEnabled,
			d.HybridBackwardRequested,
			d.HybridBackwardEnabled,
		)
	}
	if *diagnostics {
		if isStoriesBinModel(model) && selectedBackend == "ane" {
			fmt.Printf("diagnostics: backend=%s use_ane=%v layer_forward_requested=%v layer_forward_enabled=%v compiled_layers=%d layer_init_error=%q final_head_offload=%v hybrid_backward_requested=%v hybrid_backward=%v backward_init_error=%q offload=%q rms_fwd=%v cls_fwd=%v softmax=%v cls_bwd=%v rms_bwd=%v\n",
				d.Backend,
				d.UseANE,
				d.LayerForwardRequested,
				d.LayerForwardEnabled,
				d.CompiledLayers,
				d.LayerInitError,
				d.FinalHeadOffloadEnabled,
				d.HybridBackwardRequested,
				d.HybridBackwardEnabled,
				d.BackwardInitError,
				d.OffloadDiagnostics,
				d.HasRMSForward,
				d.HasClassifierForward,
				d.HasSoftmax,
				d.HasClassifierBackward,
				d.HasRMSBackward,
			)
		} else {
			fmt.Printf("diagnostics: backend=%s restricted_access=%v known=%v virtual_client=%v known=%v vc_class=%q model_qd=%d qd_known=%v program_class=%q program_qd=%d program_qd_known=%v async_inflight=%d async_known=%v requests_inflight=%d requests_known=%v\n",
				d.Backend,
				d.AllowRestrictedAccess, d.AllowRestrictedAccessKnown,
				d.IsVirtualClient, d.IsVirtualClientKnown, d.VirtualClientClass,
				d.ModelQueueDepth, d.ModelQueueDepthKnown,
				d.ProgramClass, d.ProgramQueueDepth, d.ProgramQueueDepthKnown,
				d.CurrentAsyncRequestsInFlight, d.CurrentAsyncRequestsKnown,
				d.RequestsInFlightCount, d.RequestsInFlightCountKnown,
			)
		}
	}

	var (
		batchTrainMS   float64
		batchCompileMS float64
		batchCount     uint32
		totalStepMS    float64
		totalCompileMS float64
		totalStartupMS float64
		totalRefreshMS float64
		totalANEEvalMS float64
		totalCPUWorkMS float64
		totalIOMS      float64
		totalFinalMS   float64
		totalEmbedMS   float64
		totalRMSDWMS   float64
		totalDWGEMMMS  float64
		totalDWWaitMS  float64
		totalAdamMS    float64
		stepsDone      uint32
	)
	started := time.Now()
	for i := uint32(0); i < uint32(*steps); i++ {
		st, err := trainer.Step()
		if err != nil {
			if strings.Contains(err.Error(), "trainer finished") {
				break
			}
			fatalf("step %d: %v", i+1, err)
		}

		stepMS := float64(st.StepDuration) / float64(time.Millisecond)
		compileMS := float64(st.CompileDuration) / float64(time.Millisecond)
		evalMS := float64(st.EvalDuration) / float64(time.Millisecond)
		cpuMS := float64(st.CPUWorkDuration) / float64(time.Millisecond)
		ioMS := float64(st.WriteDuration+st.ReadDuration) / float64(time.Millisecond)
		startupMS := float64(st.StartupCompileDuration) / float64(time.Millisecond)
		refreshMS := float64(st.WeightRefreshDuration) / float64(time.Millisecond)
		finalMS := float64(st.FinalHeadDuration) / float64(time.Millisecond)
		embedMS := float64(st.EmbedGradDuration) / float64(time.Millisecond)
		rmsDWMS := float64(st.RMSDWDuration) / float64(time.Millisecond)
		dwGEMMMS := float64(st.DWGEMMDuration) / float64(time.Millisecond)
		dwWaitMS := float64(st.DWWaitDuration) / float64(time.Millisecond)
		adamMS := float64(st.AdamDuration) / float64(time.Millisecond)
		stepOut := st.Step
		if stepOut > 0 {
			stepOut--
		}
		batchTrainMS += stepMS
		batchCompileMS += compileMS
		batchCount++
		totalStepMS += stepMS
		totalCompileMS += compileMS
		totalStartupMS += startupMS
		totalRefreshMS += refreshMS
		totalANEEvalMS += evalMS
		totalCPUWorkMS += cpuMS
		totalIOMS += ioMS
		totalFinalMS += finalMS
		totalEmbedMS += embedMS
		totalRMSDWMS += rmsDWMS
		totalDWGEMMMS += dwGEMMMS
		totalDWWaitMS += dwWaitMS
		totalAdamMS += adamMS
		stepsDone++
		fmt.Printf("step %d loss=%.6f step_ms=%.3f compile_ms=%.3f startup_compile_ms=%.3f refresh_ms=%.3f ane_eval_ms=%.3f cpu_ms=%.3f io_ms=%.3f compiles=%d restart_required=%v\n",
			stepOut, st.Loss, stepMS, compileMS, startupMS, refreshMS, evalMS, cpuMS, ioMS, st.Compiles, st.RestartRequired)
		if *diagnostics && isStoriesBinModel(model) && selectedBackend == "ane" {
			fmt.Printf("  cpu_diag final_head_ms=%.3f embed_ms=%.3f rms_dw_ms=%.3f dw_gemm_ms=%.3f dw_wait_ms=%.3f adam_ms=%.3f\n",
				finalMS, embedMS, rmsDWMS, dwGEMMMS, dwWaitMS, adamMS)
		}

		if *jsonOut {
			emitJSON(stepJSON{
				Type:            "step",
				Step:            stepOut,
				Loss:            st.Loss,
				TANE:            evalMS,
				TCPU:            cpuMS,
				TIO:             ioMS,
				TCompile:        compileMS,
				TStartup:        startupMS,
				TRefresh:        refreshMS,
				TCLS:            0,
				TElem:           0,
				TRMS:            0,
				TCBLASWait:      0,
				Compiles:        st.Compiles,
				RestartRequired: st.RestartRequired,
			})
		}

		if *saveEvery > 0 && st.Step%uint32(*saveEvery) == 0 {
			if err := trainer.SaveCheckpoint(*ckptPath); err != nil {
				fatalf("save checkpoint at step %d: %v", st.Step, err)
			}
		}

		if *accumSteps > 0 && st.Step%uint32(*accumSteps) == 0 {
			perStep := batchTrainMS / float64(batchCount)
			fmt.Printf("  [batch %d: compile=%.1fms train=%.1fms (%.1fms/step) compiles=%d]\n",
				batchCount, batchCompileMS, batchTrainMS, perStep, st.Compiles)
			if *jsonOut {
				emitJSON(batchJSON{
					Type:      "batch",
					Batch:     st.Step,
					CompileMS: batchCompileMS,
					TrainMS:   batchTrainMS,
					MSPerStep: perStep,
				})
			}
			batchTrainMS = 0
			batchCompileMS = 0
			batchCount = 0
		}

		if st.RestartRequired {
			fmt.Printf("[compile budget reached at step %d]\n", st.Step)
			if *autoRestart {
				if strings.TrimSpace(*ckptPath) == "" {
					fatalf("compile budget reached at step %d but auto-restart requires -ckpt", st.Step)
				}
				if *maxRestarts > 0 && restartCount >= *maxRestarts {
					fatalf("compile budget reached at step %d but max restarts reached (%d)", st.Step, *maxRestarts)
				}
				if err := trainer.SaveCheckpoint(*ckptPath); err != nil {
					fatalf("save checkpoint for restart at step %d: %v", st.Step, err)
				}
				remaining := uint32(*steps) - st.Step
				fmt.Printf("[exec restart requested step=%d remaining=%d restart_count=%d]\n", st.Step, remaining, restartCount+1)
				if *jsonOut {
					emitJSON(restartJSON{
						Type:         "restart",
						Step:         st.Step,
						Remaining:    remaining,
						RestartCount: restartCount + 1,
						MaxRestarts:  *maxRestarts,
					})
				}
				_ = trainer.Close()
				args := buildRestartArgs(
					os.Args[0],
					model,
					*modelKey,
					data,
					*ckptPath,
					selectedBackend,
					*steps,
					*learningRate,
					*noANEExtras,
					*aneClsBwd,
					*aneHybridBwd,
					*jsonOut,
					*accumSteps,
					*saveEvery,
					*saveFinal,
					*compileBudget,
					*disableBudget,
					*autoRestart,
					*maxRestarts,
					*inputBytes,
					*outputBytes,
					*recompileEach,
					*diagnostics,
					*parityMode,
					*trainerBackend,
					*fullBin,
					*fullAccumSteps,
					*veclibThreads,
					*dwConcurrency,
					*dynamicBin,
					*seqOverride,
					*dynamicANECls,
					*dynamicClsTile,
				)
				if err := restartSelf(args, restartCount+1); err != nil {
					fatalf("exec restart failed at step %d: %v", st.Step, err)
				}
			}
			fmt.Printf("[restart required at step %d; stopping]\n", st.Step)
			break
		}
	}

	if *saveFinal {
		if err := trainer.SaveCheckpoint(*ckptPath); err != nil {
			fatalf("save final checkpoint: %v", err)
		}
		fmt.Printf("saved checkpoint: %s\n", *ckptPath)
	}
	startupCompileMS := float64(trainer.StartupCompileDuration()) / float64(time.Millisecond)
	printDirectSummary(model, effectiveStoriesSequence(*seqOverride), trainer.Diagnostics(), stepsDone, totalStepMS, totalCompileMS, totalStartupMS, startupCompileMS, totalRefreshMS, totalANEEvalMS, totalCPUWorkMS, totalIOMS)
	if *diagnostics && isStoriesBinModel(model) && selectedBackend == "ane" && stepsDone > 0 {
		fmt.Printf("Diagnostics CPU avg: final_head=%.1f embed=%.1f rms_dw=%.1f dw_gemm=%.1f dw_wait=%.1f adam=%.1f ms/step\n",
			totalFinalMS/float64(stepsDone),
			totalEmbedMS/float64(stepsDone),
			totalRMSDWMS/float64(stepsDone),
			totalDWGEMMMS/float64(stepsDone),
			totalDWWaitMS/float64(stepsDone),
			totalAdamMS/float64(stepsDone),
		)
	}
	fmt.Printf("wall_ms=%.3f\n", float64(time.Since(started))/float64(time.Millisecond))
}

func emitJSON(v any) {
	enc := json.NewEncoder(os.Stderr)
	if err := enc.Encode(v); err != nil {
		fatalf("emit json: %v", err)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

func restartCountFromEnv() int {
	s := strings.TrimSpace(os.Getenv("ANE_TRAIN_RESTART_COUNT"))
	if s == "" {
		return 0
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return 0
	}
	return n
}

func restartSelf(args []string, restartCount int) error {
	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("resolve executable: %w", err)
	}
	env := append(os.Environ(), fmt.Sprintf("ANE_TRAIN_RESTART_COUNT=%d", restartCount))
	if err := syscall.Exec(exe, args, env); err != nil {
		return fmt.Errorf("syscall.Exec: %w", err)
	}
	return nil
}

func buildRestartArgs(
	bin, model, modelKey, data, ckpt, backend string,
	steps uint,
	lr float64,
	noANEExtras, aneClsBwd, aneHybridBwd, jsonOut bool,
	accumSteps, saveEvery uint,
	saveFinal bool,
	compileBudget uint,
	disableBudget, autoRestart bool,
	maxRestarts int,
	inputBytes, outputBytes uint,
	recompileEach, diagnostics, parityMode bool,
	trainerBackend string,
	fullBin string,
	fullAccumSteps uint,
	veclibThreads, dwConcurrency int,
	dynamicBin string,
	seqOverride uint,
	dynamicANECls bool,
	dynamicClsTile int,
) []string {
	args := []string{
		bin,
		"-model", model,
		"-model-key", modelKey,
		"-data", data,
		"-backend", backend,
		"-steps", strconv.FormatUint(uint64(steps), 10),
		"-lr", strconv.FormatFloat(lr, 'g', -1, 64),
		"-ckpt", ckpt,
		"-resume",
		"-accum-steps", strconv.FormatUint(uint64(accumSteps), 10),
		"-save-every", strconv.FormatUint(uint64(saveEvery), 10),
		"-input-bytes", strconv.FormatUint(uint64(inputBytes), 10),
		"-output-bytes", strconv.FormatUint(uint64(outputBytes), 10),
		"-compile-budget", strconv.FormatUint(uint64(compileBudget), 10),
		"-max-restarts", strconv.Itoa(maxRestarts),
	}
	if noANEExtras {
		args = append(args, "-no-ane-extras")
	}
	if aneClsBwd {
		args = append(args, "-ane-cls-bwd=true")
	}
	if aneHybridBwd {
		args = append(args, "-ane-hybrid-bwd=true")
	}
	if jsonOut {
		args = append(args, "-json=true")
	} else {
		args = append(args, "-json=false")
	}
	if saveFinal {
		args = append(args, "-save-final")
	}
	if disableBudget {
		args = append(args, "-disable-compile-budget")
	}
	if autoRestart {
		args = append(args, "-auto-restart=true")
	} else {
		args = append(args, "-auto-restart=false")
	}
	if recompileEach {
		args = append(args, "-recompile-every-step=true")
	}
	if diagnostics {
		args = append(args, "-diagnostics=true")
	}
	if trainerBackend != "" {
		args = append(args, "-trainer-backend", trainerBackend)
	}
	if parityMode {
		args = append(args, "-parity-mode=true")
	}
	if fullBin != "" {
		args = append(args, "-full-bin", fullBin)
	}
	if fullAccumSteps > 0 {
		args = append(args, "-full-accum-steps", strconv.FormatUint(uint64(fullAccumSteps), 10))
	}
	if veclibThreads > 0 {
		args = append(args, "-veclib-threads", strconv.Itoa(veclibThreads))
	}
	if dwConcurrency > 0 {
		args = append(args, "-dw-concurrency", strconv.Itoa(dwConcurrency))
	}
	if dynamicBin != "" {
		args = append(args, "-dynamic-bin", dynamicBin)
	}
	if seqOverride > 0 {
		args = append(args, "-seq-override", strconv.FormatUint(uint64(seqOverride), 10))
	}
	if dynamicANECls {
		args = append(args, "-dynamic-ane-cls=true")
	}
	if dynamicClsTile > 0 {
		args = append(args, "-dynamic-ane-cls-tile", strconv.Itoa(dynamicClsTile))
	}
	return args
}

func printDirectSummary(model string, seq int, d storiestrainer.Diagnostics, stepsDone uint32, totalStepMS, totalCompileMS, totalStepStartupMS, startupCompileMS, totalRefreshMS, totalANEEvalMS, totalCPUWorkMS, totalIOMS float64) {
	if !isStoriesBinModel(model) || stepsDone == 0 {
		return
	}
	compileTotalMS := startupCompileMS + totalCompileMS
	trainMS := totalStepMS - totalCompileMS
	if trainMS < 0 {
		trainMS = 0
	}
	avgTrainMS := 0.0
	if stepsDone > 0 {
		avgTrainMS = trainMS / float64(stepsDone)
	}
	compilePct := 0.0
	if totalStepMS+startupCompileMS > 0 {
		compilePct = 100.0 * compileTotalMS / (totalStepMS + startupCompileMS)
	}
	aneOccupancy := 0.0
	if trainMS > 0 {
		aneOccupancy = 100.0 * totalANEEvalMS / trainMS
	}
	aneFlopsPerStep := storiesANEFlopsPerStep(seq, d)
	aneTFLOPS := 0.0
	if trainMS > 0 && aneFlopsPerStep > 0 {
		aneTFLOPS = (aneFlopsPerStep * float64(stepsDone)) / (trainMS * 1e9)
	}
	fmt.Printf("Compile time:    %.0f ms (%.1f%%)\n", compileTotalMS, compilePct)
	fmt.Printf("Startup compile: %.0f ms\n", startupCompileMS+totalStepStartupMS)
	fmt.Printf("Weight refresh:  %.0f ms\n", totalRefreshMS)
	fmt.Printf("Train time:      %.0f ms total, %.1f ms/step\n", trainMS, avgTrainMS)
	fmt.Printf("ANE eval time:   %.0f ms total, %.1f ms/step\n", totalANEEvalMS, totalANEEvalMS/float64(stepsDone))
	fmt.Printf("CPU work time:   %.0f ms total, %.1f ms/step\n", totalCPUWorkMS, totalCPUWorkMS/float64(stepsDone))
	fmt.Printf("IO time:         %.0f ms total, %.1f ms/step\n", totalIOMS, totalIOMS/float64(stepsDone))
	fmt.Printf("ANE TFLOPS:      %.2f sustained\n", aneTFLOPS)
	fmt.Printf("ANE utilization: %.1f%% of 15.8 TFLOPS\n", 100.0*aneTFLOPS/15.8)
	fmt.Printf("ANE occupancy:   %.1f%% of train time\n", aneOccupancy)
}

func storiesANEFlopsPerStep(seq int, d storiestrainer.Diagnostics) float64 {
	s := float64(seq)
	fwdFlops := float64(stories.NLayers) * (4.0*2.0*float64(stories.Dim*stories.Dim)*s + 2.0*2.0*float64(stories.Dim*stories.Hidden)*s + 2.0*float64(stories.Hidden*stories.Dim)*s)
	headDim := stories.Dim / stories.Heads
	sdpaFlops := float64(stories.NLayers) * 2.0 * float64(stories.Heads) * 5.0 * s * s * float64(headDim)
	clsFlops := 2.0 * float64(stories.Vocab*stories.Dim) * s
	ane := 0.0
	if d.LayerForwardEnabled {
		ane += fwdFlops + sdpaFlops
	}
	if d.HybridBackwardEnabled {
		ane += fwdFlops
	}
	if d.HasClassifierForward {
		ane += clsFlops
	}
	if d.HasClassifierBackward {
		ane += clsFlops
	}
	return ane
}

func defaultIfEmpty(v, fallback string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return fallback
	}
	return v
}

func resolveSelectedBackend(requested, model string) string {
	if requested != "auto" {
		return requested
	}
	if isStoriesBinModel(model) {
		return "ane"
	}
	return "ane"
}

func isStoriesBinModel(model string) bool {
	return strings.HasSuffix(strings.ToLower(model), ".bin")
}

var probeDirectStoriesSequenceFunc = storiesane.ProbeDirectSequence

func shouldAutoEnableANEHybridBackward(args []string, selectedBackend, model string, noANEExtras, hybridBackward bool) bool {
	if hybridBackward || noANEExtras {
		return false
	}
	if selectedBackend != "ane" || !isStoriesBinModel(model) {
		return false
	}
	return !flagProvided(args, "ane-hybrid-bwd")
}

func effectiveStoriesSequence(seqOverride uint) int {
	if seqOverride == 0 {
		return stories.SeqDefault
	}
	return int(seqOverride)
}

func probeDirectStoriesSequence(selectedBackend, model string, seqOverride uint) error {
	if selectedBackend != "ane" || !isStoriesBinModel(model) {
		return nil
	}
	return probeDirectStoriesSequenceFunc(effectiveStoriesSequence(seqOverride))
}

func validateDirectStoriesBackend(selectedBackend, trainerBackend, model string, seqOverride uint, probeErr error) error {
	if selectedBackend != "ane" || trainerBackend != storiestrainer.BackendDirect || !isStoriesBinModel(model) || probeErr == nil {
		return nil
	}
	return fmt.Errorf("trainer-backend=%s does not support seq-override=%d on this host: %w; use -trainer-backend %s or -backend full", trainerBackend, effectiveStoriesSequence(seqOverride), probeErr, storiestrainer.BackendAuto)
}

func shouldAutoBridgeStoriesBinToFull(selectedBackend, trainerBackend, model string, probeErr error) bool {
	if selectedBackend != "ane" || trainerBackend != storiestrainer.BackendAuto {
		return false
	}
	if !isStoriesBinModel(model) {
		return false
	}
	return probeErr != nil
}

func flagProvided(args []string, name string) bool {
	prefix := "-" + name
	for _, arg := range args {
		if arg == prefix || strings.HasPrefix(arg, prefix+"=") {
			return true
		}
	}
	return false
}

func effectiveGoImplBackend(selectedBackend, model string, d storiestrainer.Diagnostics) string {
	switch selectedBackend {
	case "full":
		return "full_c_exec"
	case "ane-dynamic":
		return "dynamic_c_exec"
	case "cpu":
		return "cpu"
	case "ane":
		if isStoriesBinModel(model) {
			if d.HybridBackwardEnabled {
				return "direct_hybrid_bwd"
			}
			return "direct"
		}
		return "direct_modelc"
	default:
		return selectedBackend
	}
}

type cpuRunOptions struct {
	modelPath  string
	dataPath   string
	ckptPath   string
	resume     bool
	steps      int
	lr         float64
	jsonOut    bool
	accumSteps int
	saveEvery  int
	saveFinal  bool
}

type fullRunOptions struct {
	binPath       string
	seqOverride   int
	modelPath     string
	dataPath      string
	ckptPath      string
	resume        bool
	steps         int
	lr            float64
	accumSteps    int
	veclibThreads int
	dwConcurrency int
	noANEExtras   bool
	aneClsBwd     bool
}

type dynamicRunOptions struct {
	binPath       string
	seqOverride   int
	modelPath     string
	dataPath      string
	ckptPath      string
	resume        bool
	steps         int
	lr            float64
	accumSteps    int
	noANEExtras   bool
	aneClassifier bool
	aneClsTile    int
}

func runFullTraining(opts fullRunOptions) error {
	bin, err := resolveFullTrainerBinary(opts.binPath, opts.seqOverride)
	if err != nil {
		return err
	}
	args := []string{
		"--model", opts.modelPath,
		"--data", opts.dataPath,
		"--steps", strconv.Itoa(opts.steps),
		"--lr", strconv.FormatFloat(opts.lr, 'g', -1, 64),
	}
	if opts.accumSteps > 0 {
		args = append(args, "--accum", strconv.Itoa(opts.accumSteps))
	}
	if opts.veclibThreads > 0 {
		args = append(args, "--veclib-threads", strconv.Itoa(opts.veclibThreads))
	}
	if opts.dwConcurrency > 0 {
		args = append(args, "--dw-concurrency", strconv.Itoa(opts.dwConcurrency))
	}
	if opts.ckptPath != "" {
		args = append(args, "--ckpt", opts.ckptPath)
	}
	if opts.resume {
		args = append(args, "--resume")
	}
	if opts.noANEExtras {
		args = append(args, "--no-ane-extras")
	}
	if opts.aneClsBwd {
		args = append(args, "--ane-cls-bwd")
	}
	cmd := exec.Command(bin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("run %s: %w", bin, err)
	}
	return nil
}

func runDynamicTraining(opts dynamicRunOptions) error {
	if opts.noANEExtras {
		fmt.Fprintf(os.Stderr, "warning: -no-ane-extras is ignored for ane-dynamic backend\n")
	}
	bin, err := resolveDynamicTrainerBinary(opts.binPath, opts.seqOverride)
	if err != nil {
		return err
	}
	args := []string{
		"--model", opts.modelPath,
		"--data", opts.dataPath,
		"--steps", strconv.Itoa(opts.steps),
		"--lr", strconv.FormatFloat(opts.lr, 'g', -1, 64),
		"--accum", strconv.Itoa(opts.accumSteps),
	}
	if opts.aneClassifier {
		args = append(args, "--ane-cls")
		if opts.aneClsTile > 0 {
			args = append(args, "--ane-cls-tile", strconv.Itoa(opts.aneClsTile))
		}
	}
	if opts.ckptPath != "" {
		args = append(args, "--ckpt", opts.ckptPath)
	}
	if opts.resume {
		args = append(args, "--resume")
	}
	cmd := exec.Command(bin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("run %s: %w", bin, err)
	}
	return nil
}

func resolveFullTrainerBinary(binPath string, seqOverride int) (string, error) {
	if strings.TrimSpace(binPath) != "" {
		if seqOverride > 0 {
			return "", fmt.Errorf("-seq-override cannot be used with explicit -full-bin")
		}
		if _, err := os.Stat(binPath); err != nil {
			return "", fmt.Errorf("full trainer binary %q: %w", binPath, err)
		}
		return binPath, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("resolve cwd: %w", err)
	}
	defaultBin := filepath.Join(cwd, "training", "train_large_ane")
	if seqOverride > 0 {
		seqBin := filepath.Join(cwd, "training", fmt.Sprintf("train_large_ane_seq%d", seqOverride))
		if _, err := os.Stat(seqBin); err == nil {
			return seqBin, nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("stat %s: %w", seqBin, err)
		}
		makeCmd := exec.Command("make", "-B", "-C", "training", "train_large_ane",
			fmt.Sprintf("SEQ_OVERRIDE=%d", seqOverride),
			fmt.Sprintf("OUT=%s", filepath.Base(seqBin)),
		)
		makeCmd.Stdout = os.Stdout
		makeCmd.Stderr = os.Stderr
		makeCmd.Stdin = os.Stdin
		if err := makeCmd.Run(); err != nil {
			return "", fmt.Errorf("build full trainer seq override with make: %w", err)
		}
		if _, err := os.Stat(seqBin); err != nil {
			return "", fmt.Errorf("full trainer seq override binary not found after build: %w", err)
		}
		return seqBin, nil
	}
	if _, err := os.Stat(defaultBin); err == nil {
		return defaultBin, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("stat %s: %w", defaultBin, err)
	}
	makeCmd := exec.Command("make", "-C", "training", "train_large_ane")
	makeCmd.Stdout = os.Stdout
	makeCmd.Stderr = os.Stderr
	makeCmd.Stdin = os.Stdin
	if err := makeCmd.Run(); err != nil {
		return "", fmt.Errorf("build full trainer with make: %w", err)
	}
	if _, err := os.Stat(defaultBin); err != nil {
		return "", fmt.Errorf("full trainer binary not found after build: %w", err)
	}
	return defaultBin, nil
}

func resolveDynamicTrainerBinary(binPath string, seqOverride int) (string, error) {
	if strings.TrimSpace(binPath) != "" {
		if seqOverride > 0 {
			return "", fmt.Errorf("-seq-override cannot be used with explicit -dynamic-bin")
		}
		if _, err := os.Stat(binPath); err != nil {
			return "", fmt.Errorf("dynamic trainer binary %q: %w", binPath, err)
		}
		return binPath, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("resolve cwd: %w", err)
	}
	defaultBin := filepath.Join(cwd, "training", "training_dynamic", "train")
	if seqOverride > 0 {
		seqBin := filepath.Join(cwd, "training", "training_dynamic", fmt.Sprintf("train_seq%d", seqOverride))
		if _, err := os.Stat(seqBin); err == nil {
			return seqBin, nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("stat %s: %w", seqBin, err)
		}
		makeCmd := exec.Command("make", "-B", "-C", "training/training_dynamic", "train",
			fmt.Sprintf("SEQ_OVERRIDE=%d", seqOverride),
			fmt.Sprintf("OUT=%s", filepath.Base(seqBin)),
		)
		makeCmd.Stdout = os.Stdout
		makeCmd.Stderr = os.Stderr
		makeCmd.Stdin = os.Stdin
		if err := makeCmd.Run(); err != nil {
			return "", fmt.Errorf("build dynamic trainer seq override with make: %w", err)
		}
		if _, err := os.Stat(seqBin); err != nil {
			return "", fmt.Errorf("dynamic trainer seq override binary not found after build: %w", err)
		}
		return seqBin, nil
	}
	if _, err := os.Stat(defaultBin); err == nil {
		return defaultBin, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("stat %s: %w", defaultBin, err)
	}
	makeCmd := exec.Command("make", "-C", "training/training_dynamic", "train")
	makeCmd.Stdout = os.Stdout
	makeCmd.Stderr = os.Stderr
	makeCmd.Stdin = os.Stdin
	if err := makeCmd.Run(); err != nil {
		return "", fmt.Errorf("build dynamic trainer with make: %w", err)
	}
	if _, err := os.Stat(defaultBin); err != nil {
		return "", fmt.Errorf("dynamic trainer binary not found after build: %w", err)
	}
	return defaultBin, nil
}

func runCPUReference(opts cpuRunOptions) {
	toks, err := cpuLoadTokens(opts.dataPath)
	if err != nil {
		fatalf("cpu backend: load tokens: %v", err)
	}
	if len(toks) < stories.SeqDefault+1 {
		fatalf("cpu backend: not enough tokens: %d", len(toks))
	}

	mw := stories.NewModelWeights(stories.Vocab)
	opt := stories.NewOptimState(stories.Vocab)
	meta := stories.TrainMeta{TotalSteps: opts.steps, LR: float32(opts.lr)}

	if opts.resume {
		if strings.TrimSpace(opts.ckptPath) == "" {
			fatalf("cpu backend: resume requires -ckpt")
		}
		m, err := stories.LoadCheckpoint(opts.ckptPath, mw, opt)
		if err != nil {
			fatalf("cpu backend: resume load: %v", err)
		}
		meta = m
		if opts.steps > 0 {
			meta.TotalSteps = meta.Step + opts.steps
		}
		if opts.lr > 0 {
			meta.LR = float32(opts.lr)
		}
		fmt.Printf("[RESUMED step %d, loss=%.4f]\n", meta.Step, meta.Loss)
	} else {
		cfg, err := cpuPreloadOrRandom(mw, opts.modelPath)
		if err != nil {
			fatalf("cpu backend: init model: %v", err)
		}
		fmt.Printf("=== ANE Training: Stories110M Go (CPU reference path) ===\n")
		fmt.Printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n", stories.Dim, stories.Hidden, stories.Heads, stories.SeqDefault, stories.Vocab, stories.NLayers)
		fmt.Printf("Model config: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n", cfg.Dim, cfg.HiddenDim, cfg.NLayers, cfg.NHeads, cpuAbs32(cfg.VocabSize), cfg.SeqLen)
		fmt.Printf("Accum %d steps per telemetry | Adam LR=%.1e b1=%.1f b2=%.3f\n", opts.accumSteps, meta.LR, 0.9, 0.999)
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	vocab := len(mw.Embed) / stories.Dim
	x := make([]float32, stories.Dim*stories.SeqDefault)
	xNorm := make([]float32, stories.Dim*stories.SeqDefault)
	logits := make([]float32, vocab*stories.SeqDefault)
	dLogits := make([]float32, vocab*stories.SeqDefault)
	dy := make([]float32, stories.Dim*stories.SeqDefault)
	dx := make([]float32, stories.Dim*stories.SeqDefault)
	gRMS := make([]float32, stories.Dim)
	gEmbed := make([]float32, len(mw.Embed))

	var batchMs, batchCls, batchElem, batchRMS float64
	for step := meta.Step; step < meta.TotalSteps; step++ {
		pos := rng.Intn(len(toks) - stories.SeqDefault - 1)
		input := toks[pos : pos+stories.SeqDefault]
		target := toks[pos+1 : pos+1+stories.SeqDefault]
		for i := range gRMS {
			gRMS[i] = 0
		}
		for i := range gEmbed {
			gEmbed[i] = 0
		}

		t0 := time.Now()
		tRMS0 := time.Now()
		stories.EmbedLookup(x, mw.Embed, input, stories.Dim, stories.SeqDefault)
		stories.RMSNorm(xNorm, x, mw.RMSFinal, stories.Dim, stories.SeqDefault)
		tRMS := cpuMS(time.Since(tRMS0))

		tCls0 := time.Now()
		stories.MatMulVocabSeq(logits, mw.Embed, xNorm, vocab, stories.Dim, stories.SeqDefault)
		tCls := cpuMS(time.Since(tCls0))

		tElem0 := time.Now()
		loss := stories.CrossEntropyLoss(dLogits, logits, target, vocab, stories.SeqDefault)
		stories.MatMulEmbedT(dy, mw.Embed, dLogits, vocab, stories.Dim, stories.SeqDefault)
		stories.MatMulGradEmbed(gEmbed, dLogits, xNorm, vocab, stories.Dim, stories.SeqDefault)
		stories.RMSNormBackward(dx, gRMS, dy, x, mw.RMSFinal, stories.Dim, stories.SeqDefault)
		stories.EmbedBackward(gEmbed, dx, input, stories.Dim, stories.SeqDefault)
		tElem := cpuMS(time.Since(tElem0))

		meta.AdamT++
		stories.AdamUpdate(mw.RMSFinal, gRMS, &opt.RMSFinal, meta.AdamT, meta.LR, 0.9, 0.999, 1e-8)
		stories.AdamUpdate(mw.Embed, gEmbed, &opt.Embed, meta.AdamT, meta.LR, 0.9, 0.999, 1e-8)

		stepMS := cpuMS(time.Since(t0))
		batchMs += stepMS
		batchCls += tCls
		batchElem += tElem
		batchRMS += tRMS

		meta.Step = step + 1
		meta.Loss = loss
		meta.CumTrain += stepMS
		meta.CumSteps++

		fmt.Printf("step %d    loss=%.6f step_ms=%.3f cls_ms=%.3f elem_ms=%.3f rms_ms=%.3f\n", step, loss, stepMS, tCls, tElem, tRMS)
		if opts.jsonOut {
			emitJSON(stepJSON{
				Type:       "step",
				Step:       uint32(step),
				Loss:       loss,
				TCLS:       tCls,
				TElem:      tElem,
				TRMS:       tRMS,
				Compiles:   0,
				TANE:       0,
				TIO:        0,
				TCBLASWait: 0,
			})
		}

		if opts.saveEvery > 0 && strings.TrimSpace(opts.ckptPath) != "" && meta.Step%opts.saveEvery == 0 {
			if err := stories.SaveCheckpoint(opts.ckptPath, meta, mw, opt); err != nil {
				fatalf("cpu backend: save checkpoint at step %d: %v", meta.Step, err)
			}
		}
		if opts.accumSteps > 0 && meta.Step%opts.accumSteps == 0 {
			fmt.Printf("  [batch %d: compile=0ms train=%.1fms (%.1fms/step) compiles=0]\n", opts.accumSteps, batchMs, batchMs/float64(opts.accumSteps))
			fmt.Printf("    ane=0.0 io=0.0 cls=%.1f elem=%.1f rms=%.1f cblas_wait=0.0 ms/step\n",
				batchCls/float64(opts.accumSteps), batchElem/float64(opts.accumSteps), batchRMS/float64(opts.accumSteps))
			if opts.jsonOut {
				emitJSON(batchJSON{
					Type:      "batch",
					Batch:     uint32(meta.Step),
					CompileMS: 0,
					TrainMS:   batchMs,
					MSPerStep: batchMs / float64(opts.accumSteps),
				})
			}
			batchMs, batchCls, batchElem, batchRMS = 0, 0, 0, 0
		}
	}

	if opts.saveFinal && strings.TrimSpace(opts.ckptPath) != "" {
		if err := stories.SaveCheckpoint(opts.ckptPath, meta, mw, opt); err != nil {
			fatalf("cpu backend: save final checkpoint: %v", err)
		}
		fmt.Printf("saved checkpoint: %s\n", opts.ckptPath)
	}
}

func cpuPreloadOrRandom(mw *stories.ModelWeights, modelPath string) (stories.Llama2Config, error) {
	loaded, cfg, err := stories.LoadPretrained(modelPath)
	if err == nil {
		*mw = *loaded
		return cfg, nil
	}
	stories.RandomInit(mw, 42)
	return stories.Llama2Config{Dim: stories.Dim, HiddenDim: stories.Hidden, NLayers: stories.NLayers, NHeads: stories.Heads, VocabSize: stories.Vocab, SeqLen: stories.SeqDefault}, nil
}

func cpuLoadTokens(path string) ([]uint16, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%2 != 0 {
		return nil, fmt.Errorf("odd file size %d", len(b))
	}
	t := make([]uint16, len(b)/2)
	for i := range t {
		t[i] = binary.LittleEndian.Uint16(b[2*i:])
	}
	return t, nil
}

func cpuAbs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

func cpuMS(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
