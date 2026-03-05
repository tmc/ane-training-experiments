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
	"github.com/maderix/ANE/ane/storiestrainer"
)

type stepJSON struct {
	Type            string  `json:"type"`
	Step            uint32  `json:"step"`
	Loss            float32 `json:"loss"`
	TANE            float64 `json:"t_ane"`
	TIO             float64 `json:"t_io"`
	TCompile        float64 `json:"t_compile,omitempty"`
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
		backend        = flag.String("backend", defaultBackend, "training backend: auto|ane|ane-dynamic|cpu|full")
		fullBin        = flag.String("full-bin", "", "path to full C/ObjC trainer binary (default: ./training/train_large_ane)")
		fullAccumSteps = flag.Uint("full-accum-steps", 0, "override full C/ObjC trainer accumulation steps (0 uses trainer default)")
		veclibThreads  = flag.Int("veclib-threads", 0, "set VECLIB_MAXIMUM_THREADS for full C/ObjC trainer (0 uses process default)")
		dwConcurrency  = flag.Int("dw-concurrency", 0, "set C/ObjC dW async task concurrency for full trainer (0 uses trainer default, currently 3)")
		dynamicBin     = flag.String("dynamic-bin", "", "path to dynamic C/ObjC trainer binary (default: ./training/training_dynamic/train)")
		seqOverride    = flag.Uint("seq-override", 0, "build and use C/ObjC trainer binaries with compile-time SEQ override (full/ane-dynamic backends)")
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
		inputBytes     = flag.Uint("input-bytes", 4096, "input tensor bytes")
		outputBytes    = flag.Uint("output-bytes", 4096, "output tensor bytes")
		recompileEach  = flag.Bool("recompile-every-step", false, "recompile ANE kernel at each step (parity experiment)")
		diagnostics    = flag.Bool("diagnostics", false, "print model/client diagnostics at startup")
		allowExpDirect = flag.Bool("allow-experimental-ane-trainer", false, "allow direct Go ane trainer path (forward-only/experimental; not full train_large_ane parity)")
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
	selectedBackend := *backend
	if selectedBackend == "auto" {
		if strings.HasSuffix(strings.ToLower(model), ".bin") {
			selectedBackend = "full"
		} else {
			selectedBackend = "ane"
		}
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
	runFullWorkload := selectedBackend == "full" || (selectedBackend == "ane" && strings.HasSuffix(strings.ToLower(model), ".bin"))
	runDynamicWorkload := selectedBackend == "ane-dynamic"
	if selectedBackend == "ane" && !runFullWorkload && !*allowExpDirect {
		fatalf("backend=ane with non-.bin model uses the experimental direct Go trainer (not full train_large_ane parity); use -backend full or -backend ane-dynamic for full training, or pass -allow-experimental-ane-trainer=true")
	}
	if *inputBytes == 0 || *outputBytes == 0 {
		fatalf("input-bytes and output-bytes must be > 0")
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
	if runFullWorkload {
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
		Steps:                uint32(*steps),
		LearningRate:         float32(*learningRate),
		DisableANEExtras:     *noANEExtras,
		CompileBudget:        budget,
		DisableCompileBudget: *disableBudget,
		RecompileEachStep:    *recompileEach,
		QoS:                  storiestrainer.DefaultQoS,
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

	fmt.Println("=== ANE Stories Training (Go direct) ===")
	fmt.Printf("model=%s data=%s steps=%d lr=%.6f input_bytes=%d output_bytes=%d ane_extras=%v compile_budget=%d restart_count=%d auto_restart=%v\n",
		model, data, *steps, *learningRate, *inputBytes, *outputBytes, !*noANEExtras, budget, restartCount, *autoRestart)
	if *diagnostics {
		d := trainer.Diagnostics()
		fmt.Printf("diagnostics: restricted_access=%v known=%v virtual_client=%v known=%v vc_class=%q model_qd=%d qd_known=%v program_class=%q program_qd=%d program_qd_known=%v async_inflight=%d async_known=%v requests_inflight=%d requests_known=%v\n",
			d.AllowRestrictedAccess, d.AllowRestrictedAccessKnown,
			d.IsVirtualClient, d.IsVirtualClientKnown, d.VirtualClientClass,
			d.ModelQueueDepth, d.ModelQueueDepthKnown,
			d.ProgramClass, d.ProgramQueueDepth, d.ProgramQueueDepthKnown,
			d.CurrentAsyncRequestsInFlight, d.CurrentAsyncRequestsKnown,
			d.RequestsInFlightCount, d.RequestsInFlightCountKnown,
		)
	}

	var (
		batchTrainMS   float64
		batchCompileMS float64
		batchCount     uint32
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
		ioMS := float64(st.WriteDuration+st.ReadDuration) / float64(time.Millisecond)
		stepOut := st.Step
		if stepOut > 0 {
			stepOut--
		}
		batchTrainMS += stepMS
		batchCompileMS += compileMS
		batchCount++
		fmt.Printf("step %d loss=%.6f step_ms=%.3f compile_ms=%.3f ane_ms=%.3f io_ms=%.3f compiles=%d restart_required=%v\n",
			stepOut, st.Loss, stepMS, compileMS, evalMS, ioMS, st.Compiles, st.RestartRequired)

		if *jsonOut {
			tANE := float64(st.EvalDuration) / float64(time.Millisecond)
			tIO := float64(st.WriteDuration+st.ReadDuration) / float64(time.Millisecond)
			emitJSON(stepJSON{
				Type:            "step",
				Step:            stepOut,
				Loss:            st.Loss,
				TANE:            tANE,
				TIO:             tIO,
				TCompile:        compileMS,
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

func buildRestartArgs(bin, model, modelKey, data, ckpt, backend string, steps uint, lr float64, noANEExtras, jsonOut bool, accumSteps, saveEvery uint, saveFinal bool, compileBudget uint, disableBudget, autoRestart bool, maxRestarts int, inputBytes, outputBytes uint, recompileEach, diagnostics bool) []string {
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
	return args
}

func defaultIfEmpty(v, fallback string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return fallback
	}
	return v
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
		m, err := stories.LoadCheckpointV2(opts.ckptPath, mw, opt)
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
			if err := stories.SaveCheckpointV2(opts.ckptPath, meta, mw, opt); err != nil {
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
		if err := stories.SaveCheckpointV2(opts.ckptPath, meta, mw, opt); err != nil {
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
