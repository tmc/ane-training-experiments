//go:build darwin

package storiestrainer

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/maderix/ANE/ane/bridge"
	"github.com/maderix/ANE/ane/clientmodel"
)

const (
	storiesCkptMagic   = uint32(0x53545231) // "STR1"
	storiesCkptVersion = uint32(1)
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

// Trainer wraps a direct-Go _ANEClient kernel and Stories training state.
type Trainer struct {
	bridgeRuntime *bridge.Runtime
	bridgeTrainer *bridge.StoriesTrainer
	backend       string

	k           *clientmodel.Kernel
	compileOpts clientmodel.CompileOptions

	tokens        []uint16
	tokenPos      uint64
	step          uint32
	totalSteps    uint32
	compileBudget uint32
	aneExtras     bool
	lr            float32
	lastLoss      float32
	compiles      uint32
	compileBase   uint32
	recompileEach bool

	inputCount  uint32
	outputCount uint32
	inputBuf    []float32
	inputBytes  []byte
	outputBytes []byte
}

// Open creates a direct-Go stories trainer over clientmodel.
func Open(opts Options) (*Trainer, error) {
	norm, err := normalizeOptions(opts)
	if err != nil {
		return nil, err
	}
	if bt, err := openBridgeTrainer(norm); err == nil {
		return bt, nil
	}
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

	tokens, err := loadTokens(norm.DataPath)
	if err != nil {
		if k != nil {
			k.Close()
		}
		return nil, fmt.Errorf("stories trainer open: load tokens: %w", err)
	}

	t := &Trainer{
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
		inputCount:    norm.InputBytes / 4,
		outputCount:   norm.OutputBytes / 4,
		inputBuf:      make([]float32, norm.InputBytes/4),
		inputBytes:    make([]byte, norm.InputBytes),
		outputBytes:   make([]byte, norm.OutputBytes),
	}
	return t, nil
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
	if t != nil && t.bridgeTrainer != nil {
		st, err := t.bridgeTrainer.Step()
		if err != nil {
			return StepStats{}, err
		}
		d := time.Duration(st.StepMS * float64(time.Millisecond))
		return StepStats{
			Step:            st.Step,
			Loss:            st.Loss,
			StepDuration:    d,
			CompileDuration: 0,
			WriteDuration:   0,
			EvalDuration:    d,
			ReadDuration:    0,
			Compiles:        st.Compiles,
			RestartRequired: st.RestartRequired,
		}, nil
	}
	if t == nil || t.k == nil {
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
	t.fillInput()
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

// SaveCheckpoint stores trainer state.
func (t *Trainer) SaveCheckpoint(path string) error {
	if t != nil && t.bridgeTrainer != nil {
		return t.bridgeTrainer.SaveCheckpoint(path)
	}
	if t == nil || t.k == nil {
		return fmt.Errorf("stories trainer save checkpoint: trainer is closed")
	}
	if path == "" {
		return fmt.Errorf("stories trainer save checkpoint: path is empty")
	}
	ckpt := storiesCkptV1{
		Magic:         storiesCkptMagic,
		Version:       storiesCkptVersion,
		Step:          t.step,
		TotalSteps:    t.totalSteps,
		TokenPos:      t.tokenPos,
		CompileBudget: t.compileBudget,
		ANEExtras:     boolToUint32(t.aneExtras),
		LR:            t.lr,
		LastLoss:      t.lastLoss,
	}
	b := structToBytes(&ckpt)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("stories trainer save checkpoint: %w", err)
	}
	return nil
}

// LoadCheckpoint restores trainer state.
func (t *Trainer) LoadCheckpoint(path string) error {
	if t != nil && t.bridgeTrainer != nil {
		return t.bridgeTrainer.LoadCheckpoint(path)
	}
	if t == nil || t.k == nil {
		return fmt.Errorf("stories trainer load checkpoint: trainer is closed")
	}
	if path == "" {
		return fmt.Errorf("stories trainer load checkpoint: path is empty")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("stories trainer load checkpoint: %w", err)
	}
	if len(b) < binary.Size(storiesCkptV1{}) {
		return fmt.Errorf("stories trainer load checkpoint: short checkpoint")
	}
	var ckpt storiesCkptV1
	if err := bytesToStruct(b, &ckpt); err != nil {
		return fmt.Errorf("stories trainer load checkpoint: %w", err)
	}
	if ckpt.Magic != storiesCkptMagic || ckpt.Version != storiesCkptVersion {
		return fmt.Errorf("stories trainer load checkpoint: bad checkpoint header")
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

// Close releases trainer resources.
func (t *Trainer) Close() error {
	if t == nil {
		return nil
	}
	if t.bridgeTrainer != nil {
		_ = t.bridgeTrainer.Close()
		t.bridgeTrainer = nil
	}
	t.bridgeRuntime = nil
	if t.k != nil {
		t.k.Close()
		t.k = nil
	}
	return nil
}

// Diagnostics returns best-effort runtime diagnostics for the backing model/client.
func (t *Trainer) Diagnostics() Diagnostics {
	if t == nil {
		return Diagnostics{}
	}
	if t.bridgeTrainer != nil {
		return Diagnostics{Backend: BackendBridge}
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

// Backend reports which implementation is active: bridge or direct.
func (t *Trainer) Backend() string {
	if t == nil {
		return ""
	}
	if t.backend != "" {
		return t.backend
	}
	if t.bridgeTrainer != nil {
		return BackendBridge
	}
	if t.k == nil {
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

func openBridgeTrainer(opts Options) (*Trainer, error) {
	rt, err := bridge.Load(bridge.LoadOptions{})
	if err != nil {
		return nil, err
	}
	bt, err := rt.OpenStoriesTrainer(bridge.StoriesTrainerOptions{
		ModelPath:     opts.ModelPath,
		ModelKey:      opts.ModelKey,
		DataPath:      opts.DataPath,
		InputBytes:    opts.InputBytes,
		OutputBytes:   opts.OutputBytes,
		TotalSteps:    opts.Steps,
		LR:            opts.LearningRate,
		ANEExtras:     !opts.DisableANEExtras,
		CompileBudget: opts.CompileBudget,
	})
	if err != nil {
		return nil, err
	}
	return &Trainer{
		bridgeRuntime: rt,
		bridgeTrainer: bt,
		backend:       BackendBridge,
	}, nil
}

func (t *Trainer) fillInput() {
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

func structToBytes(v *storiesCkptV1) []byte {
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
	return b
}

func bytesToStruct(b []byte, v *storiesCkptV1) error {
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
