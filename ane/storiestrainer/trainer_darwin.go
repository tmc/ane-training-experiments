//go:build darwin

package storiestrainer

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/maderix/ANE/ane/clientmodel"
	"github.com/maderix/ANE/ane/stories"
	"github.com/maderix/ANE/ane/storiesane"
)

const (
	storiesCkptMagic   = uint32(0x53545231) // "STR1"
	storiesCkptVersion = uint32(2)
)

type storiesCkptV1 struct {
	Magic         uint32
	Version       uint32
	Step          uint32
	TotalSteps    uint32
	TokenPos      uint64
	CompileBudget uint32
	ANEExtras     uint32
	LR            float32
	LastLoss      float32
}

type storiesCkptV2 struct {
	Magic         uint32
	Version       uint32
	Step          uint32
	TotalSteps    uint32
	TokenPos      uint64
	CompileBudget uint32
	ANEExtras     uint32
	LR            float32
	LastLoss      float32
	CumTrainMS    float64
	CumWallMS     float64
	CumSteps      uint32
	CumBatches    uint32
	AdamT         uint32
}

// Trainer wraps the current direct-Go Stories training runtime.
type Trainer struct {
	backend string

	// _ANEClient-backed .mlmodelc mode.
	k             *clientmodel.Kernel
	compileOpts   clientmodel.CompileOptions
	inputBuf      []float32
	inputBytes    []byte
	outputBytes   []byte
	outputCount   uint32
	compileBase   uint32
	recompileEach bool

	// Common training state.
	tokens        []uint16
	tokenPos      uint64
	step          uint32
	totalSteps    uint32
	compileBudget uint32
	aneExtras     bool
	lr            float32
	lastLoss      float32
	compiles      uint32
	cumTrainMS    float64
	cumWallMS     float64
	cumSteps      uint32
	cumBatches    uint32
	adamT         uint32
	started       time.Time

	// Pure-Go .bin mode.
	cpuMode bool
	engine  *storiesane.Engine
}

// Open creates a direct-Go stories trainer.
func Open(opts Options) (*Trainer, error) {
	norm, err := normalizeOptions(opts)
	if err != nil {
		return nil, err
	}
	switch norm.Backend {
	case BackendAuto, BackendDirect:
	case BackendBridge:
		return nil, fmt.Errorf("stories trainer open: backend %q is not supported in pure-go mode", BackendBridge)
	default:
		return nil, fmt.Errorf("stories trainer open: unsupported backend %q", norm.Backend)
	}

	tokens, err := loadTokens(norm.DataPath)
	if err != nil {
		return nil, fmt.Errorf("stories trainer open: load tokens: %w", err)
	}
	if strings.HasSuffix(strings.ToLower(norm.ModelPath), ".bin") {
		return openBinTrainer(norm, tokens)
	}
	return openModelCTrainer(norm, tokens)
}

func openModelCTrainer(norm Options, tokens []uint16) (*Trainer, error) {
	baseCompiles := clientmodel.CompileCount()
	compileOpts := clientmodel.CompileOptions{
		CompiledModelPath: norm.ModelPath,
		ModelKey:          norm.ModelKey,
		QoS:               norm.QoS,
		InputBytes:        []int{int(norm.InputBytes)},
		OutputBytes:       []int{int(norm.OutputBytes)},
	}
	k, err := clientmodel.Compile(compileOpts)
	if err != nil {
		return nil, fmt.Errorf("stories trainer open: compile client kernel: %w", err)
	}
	return &Trainer{
		backend:       BackendDirect,
		k:             k,
		compileOpts:   compileOpts,
		tokens:        tokens,
		totalSteps:    norm.Steps,
		compileBudget: norm.CompileBudget,
		aneExtras:     !norm.DisableANEExtras,
		lr:            norm.LearningRate,
		compiles:      clientmodel.CompileCount() - baseCompiles,
		compileBase:   baseCompiles,
		recompileEach: norm.RecompileEachStep,
		outputCount:   norm.OutputBytes / 4,
		inputBuf:      make([]float32, norm.InputBytes/4),
		inputBytes:    make([]byte, norm.InputBytes),
		outputBytes:   make([]byte, norm.OutputBytes),
		started:       time.Now(),
	}, nil
}

func openBinTrainer(norm Options, tokens []uint16) (*Trainer, error) {
	seq := int(norm.SequenceLength)
	if seq <= 0 {
		seq = stories.SeqDefault
	}
	engine, err := storiesane.Open(storiesane.Options{
		ModelPath:  norm.ModelPath,
		Tokens:     tokens,
		Seq:        seq,
		AccumSteps: int(norm.AccumSteps),
		LR:         norm.LearningRate,
		Seed:       42,
	})
	if err != nil {
		return nil, fmt.Errorf("stories trainer open: init storiesane engine: %w", err)
	}
	return &Trainer{
		backend:       BackendDirect,
		tokens:        tokens,
		totalSteps:    norm.Steps,
		compileBudget: norm.CompileBudget,
		aneExtras:     !norm.DisableANEExtras,
		lr:            norm.LearningRate,
		started:       time.Now(),
		cpuMode:       true,
		engine:        engine,
	}, nil
}

func normalizeOptions(opts Options) (Options, error) {
	if opts.ModelPath == "" {
		return opts, fmt.Errorf("stories trainer open: model path is empty")
	}
	if _, err := os.Stat(opts.ModelPath); err != nil {
		return opts, fmt.Errorf("stories trainer open: model path: %w", err)
	}
	if opts.DataPath == "" {
		return opts, fmt.Errorf("stories trainer open: data path is empty")
	}
	if _, err := os.Stat(opts.DataPath); err != nil {
		return opts, fmt.Errorf("stories trainer open: data path: %w", err)
	}
	if opts.ModelKey == "" {
		opts.ModelKey = "s"
	}
	if opts.Backend == "" {
		opts.Backend = BackendAuto
	}
	switch opts.Backend {
	case BackendAuto, BackendBridge, BackendDirect:
	default:
		return opts, fmt.Errorf("stories trainer open: backend must be %q, %q, or %q", BackendAuto, BackendBridge, BackendDirect)
	}
	if opts.InputBytes == 0 || opts.OutputBytes == 0 {
		return opts, fmt.Errorf("stories trainer open: input and output bytes must be > 0")
	}
	if opts.LearningRate <= 0 {
		opts.LearningRate = 3e-4
	}
	if opts.DisableCompileBudget {
		opts.CompileBudget = 0
	} else if opts.CompileBudget == 0 {
		opts.CompileBudget = DefaultCompileBudget
	}
	if opts.QoS == 0 {
		opts.QoS = DefaultQoS
	}
	return opts, nil
}

// Step runs one training step.
func (t *Trainer) Step() (StepStats, error) {
	if t == nil {
		return StepStats{}, fmt.Errorf("stories trainer step: trainer is closed")
	}
	if t.cpuMode {
		return t.stepCPU()
	}
	return t.stepModelC()
}

func (t *Trainer) stepModelC() (StepStats, error) {
	if t.k == nil {
		return StepStats{}, fmt.Errorf("stories trainer step: trainer is closed")
	}
	if t.totalSteps > 0 && t.step >= t.totalSteps {
		return StepStats{}, fmt.Errorf("stories trainer step: trainer finished")
	}

	start := time.Now()
	compileDur := time.Duration(0)
	if t.recompileEach {
		compileStart := time.Now()
		nextKernel, err := clientmodel.Compile(t.compileOpts)
		compileDur = time.Since(compileStart)
		if err != nil {
			return StepStats{}, fmt.Errorf("stories trainer step: recompile: %w", err)
		}
		if t.k != nil {
			t.k.Close()
		}
		t.k = nextKernel
	}

	t.fillModelCInput()
	encodeF32LE(t.inputBuf, t.inputBytes)

	writeStart := time.Now()
	if err := t.k.WriteInput(0, t.inputBytes); err != nil {
		return StepStats{}, fmt.Errorf("stories trainer step: write input: %w", err)
	}
	writeDur := time.Since(writeStart)

	evalStart := time.Now()
	if err := t.k.Eval(); err != nil {
		return StepStats{}, fmt.Errorf("stories trainer step: eval: %w", err)
	}
	evalDur := time.Since(evalStart)

	readStart := time.Now()
	if err := t.k.ReadOutput(0, t.outputBytes); err != nil {
		return StepStats{}, fmt.Errorf("stories trainer step: read output: %w", err)
	}
	readDur := time.Since(readStart)

	n := int(t.outputCount)
	if n > 1024 {
		n = 1024
	}
	if n > 0 {
		var sum float64
		for i := 0; i < n; i++ {
			f := math.Float32frombits(binary.LittleEndian.Uint32(t.outputBytes[i*4:]))
			sum += math.Abs(float64(f))
		}
		t.lastLoss = float32(sum / float64(n))
	} else {
		t.lastLoss = 0
	}

	t.step++
	t.compiles = clientmodel.CompileCount() - t.compileBase
	t.cumSteps++
	t.cumTrainMS += float64(time.Since(start)) / float64(time.Millisecond)
	t.cumWallMS = float64(time.Since(t.started)) / float64(time.Millisecond)

	totalDur := time.Since(start)
	restart := t.compileBudget > 0 && t.compiles >= t.compileBudget
	return StepStats{
		Step:            t.step,
		Loss:            t.lastLoss,
		StepDuration:    totalDur,
		CompileDuration: compileDur,
		WriteDuration:   writeDur,
		EvalDuration:    evalDur,
		ReadDuration:    readDur,
		Compiles:        t.compiles,
		RestartRequired: restart,
	}, nil
}

func (t *Trainer) stepCPU() (StepStats, error) {
	if t.engine == nil {
		return StepStats{}, fmt.Errorf("stories trainer step: trainer is closed")
	}
	if t.totalSteps > 0 && t.step >= t.totalSteps {
		return StepStats{}, fmt.Errorf("stories trainer step: trainer finished")
	}

	res, err := t.engine.Step()
	if err != nil {
		return StepStats{}, fmt.Errorf("stories trainer step: %w", err)
	}
	t.lastLoss = res.Loss
	t.step++
	if t.totalSteps > 0 && t.step >= t.totalSteps {
		if err := t.engine.Flush(); err != nil {
			return StepStats{}, fmt.Errorf("stories trainer step: flush pending: %w", err)
		}
	}
	st := t.engine.State()
	t.tokenPos = st.TokenPos
	t.cumSteps = st.CumSteps
	t.cumTrainMS = st.CumTrainMS
	t.cumWallMS = st.CumWallMS
	t.cumBatches = st.CumBatches
	t.adamT = st.AdamT

	return StepStats{
		Step:            t.step,
		Loss:            t.lastLoss,
		StepDuration:    res.StepDuration,
		CompileDuration: 0,
		WriteDuration:   0,
		EvalDuration:    res.StepDuration,
		ReadDuration:    0,
		Compiles:        0,
		RestartRequired: false,
	}, nil
}

// SaveCheckpoint stores trainer state.
func (t *Trainer) SaveCheckpoint(path string) error {
	if t == nil {
		return fmt.Errorf("stories trainer save checkpoint: trainer is closed")
	}
	if path == "" {
		return fmt.Errorf("stories trainer save checkpoint: path is empty")
	}
	if t.cpuMode {
		if t.engine == nil {
			return fmt.Errorf("stories trainer save checkpoint: trainer is closed")
		}
		return t.engine.SaveCheckpoint(path, stories.TrainMeta{
			Step:       int(t.step),
			TotalSteps: int(t.totalSteps),
		})
	}
	if t.k == nil {
		return fmt.Errorf("stories trainer save checkpoint: trainer is closed")
	}

	ckpt := storiesCkptV2{
		Magic:         storiesCkptMagic,
		Version:       storiesCkptVersion,
		Step:          t.step,
		TotalSteps:    t.totalSteps,
		TokenPos:      t.tokenPos,
		CompileBudget: t.compileBudget,
		ANEExtras:     boolToUint32(t.aneExtras),
		LR:            t.lr,
		LastLoss:      t.lastLoss,
		CumTrainMS:    t.cumTrainMS,
		CumWallMS:     t.cumWallMS,
		CumSteps:      t.cumSteps,
		CumBatches:    t.cumBatches,
		AdamT:         t.adamT,
	}
	if err := os.WriteFile(path, structToBytesV2(&ckpt), 0o644); err != nil {
		return fmt.Errorf("stories trainer save checkpoint: %w", err)
	}
	return nil
}

// LoadCheckpoint restores trainer state.
func (t *Trainer) LoadCheckpoint(path string) error {
	if t == nil {
		return fmt.Errorf("stories trainer load checkpoint: trainer is closed")
	}
	if path == "" {
		return fmt.Errorf("stories trainer load checkpoint: path is empty")
	}
	if t.cpuMode {
		if t.engine == nil {
			return fmt.Errorf("stories trainer load checkpoint: trainer is closed")
		}
		meta, err := t.engine.LoadCheckpoint(path)
		if err != nil {
			return fmt.Errorf("stories trainer load checkpoint: %w", err)
		}
		t.step = uint32(meta.Step)
		t.totalSteps = uint32(meta.TotalSteps)
		t.lr = meta.LR
		t.lastLoss = meta.Loss
		t.cumTrainMS = meta.CumTrain
		t.cumWallMS = meta.CumWall
		t.cumSteps = uint32(meta.CumSteps)
		t.cumBatches = uint32(meta.CumBatches)
		t.adamT = uint32(meta.AdamT)
		t.tokenPos = t.engine.State().TokenPos
		return nil
	}
	if t.k == nil {
		return fmt.Errorf("stories trainer load checkpoint: trainer is closed")
	}

	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("stories trainer load checkpoint: %w", err)
	}
	if len(b) < binary.Size(storiesCkptV1{}) {
		return fmt.Errorf("stories trainer load checkpoint: short checkpoint")
	}
	magic := binary.LittleEndian.Uint32(b[0:])
	ver := binary.LittleEndian.Uint32(b[4:])
	if magic != storiesCkptMagic {
		return fmt.Errorf("stories trainer load checkpoint: bad checkpoint header")
	}
	if ver == 1 {
		var ckpt storiesCkptV1
		if err := bytesToStructV1(b, &ckpt); err != nil {
			return fmt.Errorf("stories trainer load checkpoint: %w", err)
		}
		t.step = ckpt.Step
		t.totalSteps = ckpt.TotalSteps
		t.tokenPos = ckpt.TokenPos
		t.compileBudget = ckpt.CompileBudget
		t.aneExtras = ckpt.ANEExtras != 0
		t.lr = ckpt.LR
		t.lastLoss = ckpt.LastLoss
		return nil
	}
	if ver != storiesCkptVersion {
		return fmt.Errorf("stories trainer load checkpoint: bad checkpoint header")
	}
	if len(b) < binary.Size(storiesCkptV2{}) {
		return fmt.Errorf("stories trainer load checkpoint: short checkpoint")
	}
	var ckpt storiesCkptV2
	if err := bytesToStructV2(b, &ckpt); err != nil {
		return fmt.Errorf("stories trainer load checkpoint: %w", err)
	}
	t.step = ckpt.Step
	t.totalSteps = ckpt.TotalSteps
	t.tokenPos = ckpt.TokenPos
	t.compileBudget = ckpt.CompileBudget
	t.aneExtras = ckpt.ANEExtras != 0
	t.lr = ckpt.LR
	t.lastLoss = ckpt.LastLoss
	t.cumTrainMS = ckpt.CumTrainMS
	t.cumWallMS = ckpt.CumWallMS
	t.cumSteps = ckpt.CumSteps
	t.cumBatches = ckpt.CumBatches
	t.adamT = ckpt.AdamT
	return nil
}

// Close releases trainer resources.
func (t *Trainer) Close() error {
	if t == nil {
		return nil
	}
	if t.k != nil {
		t.k.Close()
		t.k = nil
	}
	if t.engine != nil {
		t.engine.Close()
		t.engine = nil
	}
	return nil
}

// Diagnostics returns best-effort runtime diagnostics for the backing model/client.
func (t *Trainer) Diagnostics() Diagnostics {
	if t == nil {
		return Diagnostics{}
	}
	if t.k == nil {
		return Diagnostics{Backend: t.backend}
	}
	d := t.k.Diagnostics()
	return Diagnostics{
		Backend:                      BackendDirect,
		HasVirtualClient:             d.HasVirtualClient,
		VirtualClientClass:           d.VirtualClientClass,
		AllowRestrictedAccess:        d.AllowRestrictedAccess,
		AllowRestrictedAccessKnown:   d.AllowRestrictedAccessKnown,
		IsVirtualClient:              d.IsVirtualClient,
		IsVirtualClientKnown:         d.IsVirtualClientKnown,
		ModelQueueDepth:              d.ModelQueueDepth,
		ModelQueueDepthKnown:         d.ModelQueueDepthKnown,
		ProgramClass:                 d.ProgramClass,
		ProgramQueueDepth:            d.ProgramQueueDepth,
		ProgramQueueDepthKnown:       d.ProgramQueueDepthKnown,
		CurrentAsyncRequestsInFlight: d.CurrentAsyncRequestsInFlight,
		CurrentAsyncRequestsKnown:    d.CurrentAsyncRequestsInFlightOK,
		RequestsInFlightCount:        d.RequestsInFlightCount,
		RequestsInFlightCountKnown:   d.RequestsInFlightCountKnown,
	}
}

// Backend reports which implementation is active.
func (t *Trainer) Backend() string {
	if t == nil {
		return ""
	}
	if t.backend != "" {
		return t.backend
	}
	if t.k == nil && !t.cpuMode {
		return ""
	}
	return BackendDirect
}

func loadTokens(path string) ([]uint16, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 2 {
		return nil, fmt.Errorf("token file too small")
	}
	n := len(b) / 2
	out := make([]uint16, n)
	for i := 0; i < n; i++ {
		out[i] = binary.LittleEndian.Uint16(b[i*2:])
	}
	return out, nil
}

func (t *Trainer) fillModelCInput() {
	if len(t.inputBuf) == 0 {
		return
	}
	if len(t.tokens) == 0 {
		for i := range t.inputBuf {
			t.inputBuf[i] = float32((uint64(i)+uint64(t.step))%1024) / 1024.0
		}
		return
	}
	pos := t.tokenPos
	for i := range t.inputBuf {
		tok := t.tokens[pos%uint64(len(t.tokens))]
		t.inputBuf[i] = float32(tok%32000) / 32000.0
		pos++
	}
	t.tokenPos = pos % uint64(len(t.tokens))
}

func encodeF32LE(src []float32, dst []byte) {
	for i, v := range src {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(v))
	}
}

func boolToUint32(v bool) uint32 {
	if v {
		return 1
	}
	return 0
}

func structToBytesV2(v *storiesCkptV2) []byte {
	b := make([]byte, binary.Size(*v))
	binary.LittleEndian.PutUint32(b[0:], v.Magic)
	binary.LittleEndian.PutUint32(b[4:], v.Version)
	binary.LittleEndian.PutUint32(b[8:], v.Step)
	binary.LittleEndian.PutUint32(b[12:], v.TotalSteps)
	binary.LittleEndian.PutUint64(b[16:], v.TokenPos)
	binary.LittleEndian.PutUint32(b[24:], v.CompileBudget)
	binary.LittleEndian.PutUint32(b[28:], v.ANEExtras)
	binary.LittleEndian.PutUint32(b[32:], math.Float32bits(v.LR))
	binary.LittleEndian.PutUint32(b[36:], math.Float32bits(v.LastLoss))
	binary.LittleEndian.PutUint64(b[40:], math.Float64bits(v.CumTrainMS))
	binary.LittleEndian.PutUint64(b[48:], math.Float64bits(v.CumWallMS))
	binary.LittleEndian.PutUint32(b[56:], v.CumSteps)
	binary.LittleEndian.PutUint32(b[60:], v.CumBatches)
	binary.LittleEndian.PutUint32(b[64:], v.AdamT)
	return b
}

func bytesToStructV1(b []byte, v *storiesCkptV1) error {
	if len(b) < binary.Size(*v) {
		return fmt.Errorf("short data")
	}
	v.Magic = binary.LittleEndian.Uint32(b[0:])
	v.Version = binary.LittleEndian.Uint32(b[4:])
	v.Step = binary.LittleEndian.Uint32(b[8:])
	v.TotalSteps = binary.LittleEndian.Uint32(b[12:])
	v.TokenPos = binary.LittleEndian.Uint64(b[16:])
	v.CompileBudget = binary.LittleEndian.Uint32(b[24:])
	v.ANEExtras = binary.LittleEndian.Uint32(b[28:])
	v.LR = math.Float32frombits(binary.LittleEndian.Uint32(b[32:]))
	v.LastLoss = math.Float32frombits(binary.LittleEndian.Uint32(b[36:]))
	return nil
}

func bytesToStructV2(b []byte, v *storiesCkptV2) error {
	if len(b) < binary.Size(*v) {
		return fmt.Errorf("short data")
	}
	v.Magic = binary.LittleEndian.Uint32(b[0:])
	v.Version = binary.LittleEndian.Uint32(b[4:])
	v.Step = binary.LittleEndian.Uint32(b[8:])
	v.TotalSteps = binary.LittleEndian.Uint32(b[12:])
	v.TokenPos = binary.LittleEndian.Uint64(b[16:])
	v.CompileBudget = binary.LittleEndian.Uint32(b[24:])
	v.ANEExtras = binary.LittleEndian.Uint32(b[28:])
	v.LR = math.Float32frombits(binary.LittleEndian.Uint32(b[32:]))
	v.LastLoss = math.Float32frombits(binary.LittleEndian.Uint32(b[36:]))
	v.CumTrainMS = math.Float64frombits(binary.LittleEndian.Uint64(b[40:]))
	v.CumWallMS = math.Float64frombits(binary.LittleEndian.Uint64(b[48:]))
	v.CumSteps = binary.LittleEndian.Uint32(b[56:])
	v.CumBatches = binary.LittleEndian.Uint32(b[60:])
	v.AdamT = binary.LittleEndian.Uint32(b[64:])
	return nil
}
