package storiesane

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/maderix/ANE/ane/stories"
)

// Options configures a pure-Go Stories training engine over .bin weights.
type Options struct {
	ModelPath  string
	Tokens     []uint16
	Seq        int
	AccumSteps int
	LR         float32
	Seed       int64
}

// State captures resumable engine state.
type State struct {
	TokenPos     uint64
	LastLoss     float32
	CumTrainMS   float64
	CumWallMS    float64
	CumSteps     uint32
	CumBatches   uint32
	AdamT        uint32
	PendingSteps uint32
}

// StepResult reports one training step.
type StepResult struct {
	Loss         float32
	StepDuration time.Duration
}

// Engine runs the current pure-Go Stories training loop.
//
// NOTE: This is a baseline implementation. It does not yet mirror train_large_ane
// topology or ANE-offloaded multi-layer execution.
type Engine struct {
	mw  *stories.ModelWeights
	opt *stories.OptimState

	tokens     []uint16
	seq        int
	accumSteps int
	lr         float32
	seed       int64
	rng        uint64
	state      State
	start      time.Time

	x           []float32
	xNorm       []float32
	logits      []float32
	dLogits     []float32
	dy          []float32
	dx          []float32
	gRMS        []float32
	gEmbed      []float32
	accumGRMS   []float32
	accumGEmbed []float32
}

const (
	drand48Mul  = uint64(0x5DEECE66D)
	drand48Add  = uint64(0xB)
	drand48Mask = uint64((1 << 48) - 1)
)

// Open constructs an engine with pretrained .bin weights and token data.
func Open(opts Options) (*Engine, error) {
	if opts.ModelPath == "" {
		return nil, fmt.Errorf("storiesane open: model path is empty")
	}
	if len(opts.Tokens) == 0 {
		return nil, fmt.Errorf("storiesane open: tokens are empty")
	}
	seq := opts.Seq
	if seq <= 0 {
		seq = stories.SeqDefault
	}
	if len(opts.Tokens) < seq+1 {
		return nil, fmt.Errorf("storiesane open: not enough tokens for seq=%d", seq)
	}
	if opts.LR <= 0 {
		opts.LR = 3e-4
	}
	if opts.AccumSteps <= 0 {
		opts.AccumSteps = 1
	}
	if opts.Seed == 0 {
		opts.Seed = 42
	}

	mw := stories.NewModelWeights(stories.Vocab)
	opt := stories.NewOptimState(stories.Vocab)
	loaded, _, err := stories.LoadPretrained(opts.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("storiesane open: load pretrained model: %w", err)
	}
	*mw = *loaded

	return &Engine{
		mw:          mw,
		opt:         opt,
		tokens:      opts.Tokens,
		seq:         seq,
		accumSteps:  opts.AccumSteps,
		lr:          opts.LR,
		seed:        opts.Seed,
		rng:         drand48Seed(opts.Seed),
		start:       time.Now(),
		x:           make([]float32, stories.Dim*seq),
		xNorm:       make([]float32, stories.Dim*seq),
		logits:      make([]float32, stories.Vocab*seq),
		dLogits:     make([]float32, stories.Vocab*seq),
		dy:          make([]float32, stories.Dim*seq),
		dx:          make([]float32, stories.Dim*seq),
		gRMS:        make([]float32, stories.Dim),
		gEmbed:      make([]float32, len(mw.Embed)),
		accumGRMS:   make([]float32, stories.Dim),
		accumGEmbed: make([]float32, len(mw.Embed)),
	}, nil
}

// Step runs one engine step.
func (e *Engine) Step() (StepResult, error) {
	if e == nil || e.mw == nil || e.opt == nil {
		return StepResult{}, fmt.Errorf("storiesane step: engine is closed")
	}
	if len(e.tokens) < e.seq+1 {
		return StepResult{}, fmt.Errorf("storiesane step: not enough tokens")
	}

	t0 := time.Now()
	limit := uint64(len(e.tokens) - e.seq - 1)
	pos := uint64(0)
	if limit > 0 {
		pos = uint64(e.nextFloat64() * float64(limit))
		if pos >= limit {
			pos = limit - 1
		}
	}
	input := e.tokens[pos : pos+uint64(e.seq)]
	target := e.tokens[pos+1 : pos+1+uint64(e.seq)]
	e.state.TokenPos = pos + uint64(e.seq)

	for i := range e.gRMS {
		e.gRMS[i] = 0
	}
	for i := range e.gEmbed {
		e.gEmbed[i] = 0
	}

	stories.EmbedLookup(e.x, e.mw.Embed, input, stories.Dim, e.seq)
	stories.RMSNorm(e.xNorm, e.x, e.mw.RMSFinal, stories.Dim, e.seq)
	stories.MatMulVocabSeq(e.logits, e.mw.Embed, e.xNorm, stories.Vocab, stories.Dim, e.seq)
	loss := stories.CrossEntropyLoss(e.dLogits, e.logits, target, stories.Vocab, e.seq)
	stories.MatMulEmbedT(e.dy, e.mw.Embed, e.dLogits, stories.Vocab, stories.Dim, e.seq)
	stories.MatMulGradEmbed(e.gEmbed, e.dLogits, e.xNorm, stories.Vocab, stories.Dim, e.seq)
	stories.RMSNormBackward(e.dx, e.gRMS, e.dy, e.x, e.mw.RMSFinal, stories.Dim, e.seq)
	stories.EmbedBackward(e.gEmbed, e.dx, input, stories.Dim, e.seq)

	for i, v := range e.gRMS {
		e.accumGRMS[i] += v
	}
	for i, v := range e.gEmbed {
		e.accumGEmbed[i] += v
	}
	e.state.PendingSteps++
	if int(e.state.PendingSteps) >= e.accumSteps {
		e.flushPending()
	}

	dur := time.Since(t0)
	e.state.LastLoss = loss
	e.state.CumSteps++
	e.state.CumTrainMS += float64(dur) / float64(time.Millisecond)
	e.state.CumWallMS = float64(time.Since(e.start)) / float64(time.Millisecond)

	return StepResult{Loss: loss, StepDuration: dur}, nil
}

// Flush applies any pending accumulated gradients.
func (e *Engine) Flush() error {
	if e == nil || e.mw == nil || e.opt == nil {
		return fmt.Errorf("storiesane flush: engine is closed")
	}
	e.flushPending()
	return nil
}

// State returns a copy of current engine state.
func (e *Engine) State() State {
	if e == nil {
		return State{}
	}
	return e.state
}

// LoadState restores engine state counters.
func (e *Engine) LoadState(s State) error {
	if e == nil || e.mw == nil || e.opt == nil {
		return fmt.Errorf("storiesane load state: engine is closed")
	}
	e.state = s
	return nil
}

// SaveCheckpoint persists the current model, optimizer, and training metadata.
func (e *Engine) SaveCheckpoint(path string, meta stories.TrainMeta) error {
	if e == nil || e.mw == nil || e.opt == nil {
		return fmt.Errorf("storiesane save checkpoint: engine is closed")
	}
	meta.LR = e.lr
	meta.Loss = e.state.LastLoss
	meta.CumTrain = e.state.CumTrainMS
	meta.CumWall = e.state.CumWallMS
	meta.CumSteps = int(e.state.CumSteps)
	meta.CumBatches = int(e.state.CumBatches)
	meta.AdamT = int(e.state.AdamT)
	if err := stories.SaveCheckpointV2(path, meta, e.mw, e.opt); err != nil {
		return err
	}
	return e.appendTrailer(path)
}

// LoadCheckpoint restores the current model, optimizer, and training metadata.
func (e *Engine) LoadCheckpoint(path string) (stories.TrainMeta, error) {
	if e == nil || e.mw == nil || e.opt == nil {
		return stories.TrainMeta{}, fmt.Errorf("storiesane load checkpoint: engine is closed")
	}
	meta, err := stories.LoadCheckpointV2(path, e.mw, e.opt)
	if err != nil {
		return stories.TrainMeta{}, err
	}
	e.lr = meta.LR
	e.state = State{
		LastLoss:   meta.Loss,
		CumTrainMS: meta.CumTrain,
		CumWallMS:  meta.CumWall,
		CumSteps:   uint32(meta.CumSteps),
		CumBatches: uint32(meta.CumBatches),
		AdamT:      uint32(meta.AdamT),
	}
	clear(e.accumGRMS)
	clear(e.accumGEmbed)
	if err := e.loadTrailer(path); err != nil {
		return stories.TrainMeta{}, err
	}
	e.rng = drand48Seed(e.seed + int64(meta.Step))
	e.start = time.Now().Add(-time.Duration(meta.CumWall * float64(time.Millisecond)))
	return meta, nil
}

// Close releases engine resources.
func (e *Engine) Close() {
	if e == nil {
		return
	}
	e.mw = nil
	e.opt = nil
	e.tokens = nil
	e.x = nil
	e.xNorm = nil
	e.logits = nil
	e.dLogits = nil
	e.dy = nil
	e.dx = nil
	e.gRMS = nil
	e.gEmbed = nil
	e.accumGRMS = nil
	e.accumGEmbed = nil
}

func (e *Engine) flushPending() {
	if e.state.PendingSteps == 0 {
		return
	}
	scale := float32(1.0 / float64(e.state.PendingSteps))
	for i := range e.accumGRMS {
		e.accumGRMS[i] *= scale
	}
	for i := range e.accumGEmbed {
		e.accumGEmbed[i] *= scale
	}
	e.state.AdamT++
	stories.AdamUpdate(e.mw.RMSFinal, e.accumGRMS, &e.opt.RMSFinal, int(e.state.AdamT), e.lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(e.mw.Embed, e.accumGEmbed, &e.opt.Embed, int(e.state.AdamT), e.lr, 0.9, 0.999, 1e-8)
	clear(e.accumGRMS)
	clear(e.accumGEmbed)
	e.state.PendingSteps = 0
	e.state.CumBatches++
}

const (
	trailerMagic   = "SANE"
	trailerVersion = uint32(1)
)

func (e *Engine) appendTrailer(path string) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return fmt.Errorf("storiesane append trailer: %w", err)
	}
	defer f.Close()
	if _, err := f.Write([]byte(trailerMagic)); err != nil {
		return fmt.Errorf("storiesane append trailer: write magic: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, trailerVersion); err != nil {
		return fmt.Errorf("storiesane append trailer: write version: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, e.state.TokenPos); err != nil {
		return fmt.Errorf("storiesane append trailer: write token pos: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, e.state.PendingSteps); err != nil {
		return fmt.Errorf("storiesane append trailer: write pending steps: %w", err)
	}
	if err := writeF32Slice(f, e.accumGRMS); err != nil {
		return fmt.Errorf("storiesane append trailer: write accum rms: %w", err)
	}
	if err := writeF32Slice(f, e.accumGEmbed); err != nil {
		return fmt.Errorf("storiesane append trailer: write accum embed: %w", err)
	}
	return nil
}

func (e *Engine) loadTrailer(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("storiesane load trailer: %w", err)
	}
	defer f.Close()
	if _, err := f.Seek(int64(storiesCheckpointSize()), io.SeekStart); err != nil {
		return fmt.Errorf("storiesane load trailer: seek: %w", err)
	}
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return nil
		}
		return fmt.Errorf("storiesane load trailer: read magic: %w", err)
	}
	if string(magic[:]) != trailerMagic {
		return nil
	}
	var ver uint32
	if err := binary.Read(f, binary.LittleEndian, &ver); err != nil {
		return fmt.Errorf("storiesane load trailer: read version: %w", err)
	}
	if ver != trailerVersion {
		return fmt.Errorf("storiesane load trailer: unsupported version %d", ver)
	}
	if err := binary.Read(f, binary.LittleEndian, &e.state.TokenPos); err != nil {
		return fmt.Errorf("storiesane load trailer: read token pos: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &e.state.PendingSteps); err != nil {
		return fmt.Errorf("storiesane load trailer: read pending steps: %w", err)
	}
	if err := readF32Slice(f, e.accumGRMS); err != nil {
		return fmt.Errorf("storiesane load trailer: read accum rms: %w", err)
	}
	if err := readF32Slice(f, e.accumGEmbed); err != nil {
		return fmt.Errorf("storiesane load trailer: read accum embed: %w", err)
	}
	return nil
}

func storiesCheckpointSize() int {
	const headerBytes = 96
	layerWeights := stories.NLayers * (stories.WQSize*3 + stories.WOSize + stories.W1Size + stories.W2Size + stories.W3Size + stories.Dim*2)
	layerOpt := stories.NLayers * (stories.WQSize*6 + stories.WOSize*2 + stories.W1Size*2 + stories.W2Size*2 + stories.W3Size*2 + stories.Dim*4)
	final := stories.Dim * 3
	embed := stories.Vocab * stories.Dim * 3
	return headerBytes + 4*(layerWeights+layerOpt+final+embed)
}

func writeF32Slice(w io.Writer, vals []float32) error {
	for _, v := range vals {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	return nil
}

func readF32Slice(r io.Reader, vals []float32) error {
	for i := range vals {
		if err := binary.Read(r, binary.LittleEndian, &vals[i]); err != nil {
			return err
		}
	}
	return nil
}

func drand48Seed(seed int64) uint64 {
	return ((uint64(seed) & 0xffffffff) << 16) | 0x330E
}

func (e *Engine) nextFloat64() float64 {
	e.rng = (drand48Mul*e.rng + drand48Add) & drand48Mask
	return float64(e.rng) / float64(uint64(1)<<48)
}
