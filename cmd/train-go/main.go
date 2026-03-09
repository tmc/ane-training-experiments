package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/maderix/ANE/ane/linear"
)

func main() {
	var (
		dataPath    = flag.String("data", "/Volumes/tmc/go/src/github.com/maderix/ANE/training/tinystories_data00.bin", "path to uint16 token data")
		steps       = flag.Int("steps", 2000, "training steps")
		batchSize   = flag.Int("batch", 64, "batch size")
		vocabSize   = flag.Int("vocab", 2048, "effective vocab")
		mapMode     = flag.String("map", "hash", "token mapping mode: hash or strict")
		engine      = flag.String("engine", "auto", "logits engine: auto, cpu, ane")
		lr          = flag.Float64("lr", 0.05, "learning rate")
		seed        = flag.Int64("seed", time.Now().UnixNano(), "rng seed")
		loadPath    = flag.String("load", "", "optional checkpoint path to load")
		savePath    = flag.String("save", "", "optional checkpoint path to save")
		saveEvery   = flag.Int("save-every", 0, "save checkpoint every N steps (0 disables periodic saves)")
		bench       = flag.Bool("bench", false, "benchmark engines and exit")
		benchList   = flag.String("bench-engines", "cpu,ane", "comma-separated engine list for benchmark")
		jsonOut     = flag.Bool("json", false, "emit C-style JSON telemetry to stderr")
		accumSteps  = flag.Int("accum-steps", 10, "steps per batch telemetry window")
		maxCompiles = flag.Int("max-compiles", 100, "compile budget before restart (ane only, <=0 disables)")
		autoRestart = flag.Bool("auto-restart", false, "restart process when compile budget reached (ane only)")
		ckptPath    = flag.String("ckpt", "/tmp/train-go.ckpt", "checkpoint path for restart flow")
		resume      = flag.Bool("resume", false, "resume mode for restart flow")
	)
	flag.Parse()

	tokens, err := loadTokens(*dataPath)
	if err != nil {
		fatalf("load tokens: %v", err)
	}
	if len(tokens) < 2 {
		fatalf("need at least 2 tokens, got %d", len(tokens))
	}
	if *vocabSize <= 1 {
		fatalf("vocab must be > 1")
	}
	if *mapMode != "hash" && *mapMode != "strict" {
		fatalf("invalid -map=%q (want hash or strict)", *mapMode)
	}
	if *engine != "auto" && *engine != "cpu" && *engine != "ane" {
		fatalf("invalid -engine=%q (want auto, cpu, ane)", *engine)
	}
	if *saveEvery < 0 {
		fatalf("save-every must be >= 0")
	}
	if *accumSteps < 1 {
		fatalf("accum-steps must be >= 1")
	}
	maxID := maxToken(tokens)

	rng := rand.New(rand.NewSource(*seed))
	model := newBigramModel(*vocabSize, rng)
	if *resume && *loadPath == "" {
		*loadPath = *ckptPath
	}
	if *loadPath != "" {
		ckpt, err := loadCheckpoint(*loadPath)
		if err != nil {
			fatalf("load checkpoint: %v", err)
		}
		if ckpt.Vocab != *vocabSize {
			fatalf("load checkpoint: vocab mismatch: checkpoint=%d flag=%d", ckpt.Vocab, *vocabSize)
		}
		copy(model.w, ckpt.Weights)
		fmt.Printf("loaded checkpoint: %s (step=%d avg_loss=%.4f)\n", *loadPath, ckpt.Step, ckpt.AvgLoss)
	}
	if *bench {
		if err := runBenchmarks(tokens, model.w, *steps, *batchSize, *vocabSize, *mapMode == "hash", *lr, *seed, *benchList); err != nil {
			fatalf("benchmark: %v", err)
		}
		return
	}

	logitsEngine, err := newLogitsEngine(*engine, *vocabSize, *batchSize)
	if err != nil {
		fatalf("init engine: %v", err)
	}
	defer func() { _ = logitsEngine.Close() }()

	fmt.Printf("train-go bigram\n")
	fmt.Printf("tokens=%d max_token=%d vocab=%d map=%s engine=%s steps=%d batch=%d lr=%.4f\n",
		len(tokens), maxID, *vocabSize, *mapMode, logitsEngine.Name(), *steps, *batchSize, *lr)
	if *autoRestart && (logitsEngine.Name() == "ane") {
		fmt.Printf("Accum %d steps per recompile | compile budget=%d\n", *accumSteps, *maxCompiles)
	}

	res, err := runTrainingStories(storiesRunOpts{
		tokens:       tokens,
		model:        model,
		logitsEngine: logitsEngine,
		steps:        *steps,
		batch:        *batchSize,
		lr:           *lr,
		hashMap:      *mapMode == "hash",
		rng:          rng,
		savePath:     *savePath,
		saveEvery:    *saveEvery,
		accumSteps:   *accumSteps,
		jsonOut:      *jsonOut,
		maxCompiles:  *maxCompiles,
	})
	if err != nil {
		fatalf("training: %v", err)
	}
	if *savePath != "" {
		if err := saveCheckpoint(*savePath, *vocabSize, *steps, res.AvgLoss, model.w); err != nil {
			fatalf("save final checkpoint: %v", err)
		}
		fmt.Printf("saved checkpoint: %s\n", *savePath)
	}
	if *autoRestart && (logitsEngine.Name() == "ane") && *maxCompiles > 0 && res.Compiles >= *maxCompiles && res.StepsDone < *steps {
		if err := saveCheckpoint(*ckptPath, *vocabSize, res.StepsDone, res.AvgLoss, model.w); err != nil {
			fatalf("save restart checkpoint: %v", err)
		}
		remaining := *steps - res.StepsDone
		fmt.Printf("[exec() restart step %d, %d compiles, loss=%.4f]\n", res.StepsDone, res.Compiles, res.AvgLoss)
		if err := reexecWithResume(*ckptPath, remaining); err != nil {
			fatalf("restart: %v", err)
		}
	}
}

type trainResult struct {
	AvgLoss      float64
	Used         int
	StepsDone    int
	TotalForward time.Duration
	TotalCompile time.Duration
	TotalEval    time.Duration
	TotalHWEval  time.Duration
	TotalUpdate  time.Duration
	Wall         time.Duration
	Compiles     int
}

func runTraining(tokens []uint16, m *bigramModel, logitsEngine logitsProvider, steps, batch int, lr float64, hashMap bool, rng *rand.Rand, savePath string, saveEvery int, logProgress bool) (trainResult, error) {
	var res trainResult
	avgLoss := 0.0
	haveAvg := false
	start := time.Now()
	for step := 1; step <= steps; step++ {
		loss, n, stats, err := m.trainStep(tokens, batch, lr, hashMap, rng, logitsEngine)
		if err != nil {
			return res, fmt.Errorf("step %d: %w", step, err)
		}
		if n == 0 {
			return res, fmt.Errorf("no usable samples for vocab=%d in strict mode; try -map hash or larger -vocab", m.vocab)
		}
		if !haveAvg {
			avgLoss = loss
			haveAvg = true
		} else {
			avgLoss = 0.98*avgLoss + 0.02*loss
		}
		res.Used += n
		res.StepsDone = step
		res.TotalForward += stats.Forward
		res.TotalCompile += stats.Compile
		res.TotalEval += stats.Eval
		res.TotalHWEval += stats.HWEval
		res.TotalUpdate += stats.Update
		res.Compiles += stats.Compiles
		if savePath != "" && saveEvery > 0 && step%saveEvery == 0 {
			if err := saveCheckpoint(savePath, m.vocab, step, avgLoss, m.w); err != nil {
				return res, fmt.Errorf("save checkpoint at step %d: %w", step, err)
			}
		}
		if logProgress && (step == 1 || step%100 == 0 || step == steps) {
			ppl := math.Exp(avgLoss)
			fmt.Printf("step=%d loss=%.4f avg_loss=%.4f ppl=%.2f used=%d t_forward=%s t_compile=%s t_eval=%s t_update=%s\n",
				step, loss, avgLoss, ppl, res.Used,
				res.TotalForward.Round(time.Millisecond), res.TotalCompile.Round(time.Millisecond),
				res.TotalEval.Round(time.Millisecond), res.TotalUpdate.Round(time.Millisecond))
		}
	}
	res.AvgLoss = avgLoss
	res.Wall = time.Since(start)
	return res, nil
}

type storiesRunOpts struct {
	tokens       []uint16
	model        *bigramModel
	logitsEngine logitsProvider
	steps        int
	batch        int
	lr           float64
	hashMap      bool
	rng          *rand.Rand
	savePath     string
	saveEvery    int
	accumSteps   int
	jsonOut      bool
	maxCompiles  int
}

func runTrainingStories(opts storiesRunOpts) (trainResult, error) {
	var res trainResult
	avgLoss := 0.0
	haveAvg := false
	start := time.Now()
	var (
		batchCompile time.Duration
		batchTrain   time.Duration
		batchSteps   int
	)
	for step := 1; step <= opts.steps; step++ {
		stepStart := time.Now()
		loss, n, stats, err := opts.model.trainStep(opts.tokens, opts.batch, opts.lr, opts.hashMap, opts.rng, opts.logitsEngine)
		if err != nil {
			return res, fmt.Errorf("step %d: %w", step, err)
		}
		if n == 0 {
			return res, fmt.Errorf("no usable samples for vocab=%d in strict mode; try -map hash or larger -vocab", opts.model.vocab)
		}
		if !haveAvg {
			avgLoss = loss
			haveAvg = true
		} else {
			avgLoss = 0.98*avgLoss + 0.02*loss
		}
		res.Used += n
		res.StepsDone = step
		res.TotalForward += stats.Forward
		res.TotalCompile += stats.Compile
		res.TotalEval += stats.Eval
		res.TotalHWEval += stats.HWEval
		res.TotalUpdate += stats.Update
		res.Compiles += stats.Compiles
		batchCompile += stats.Compile
		batchTrain += time.Since(stepStart)
		batchSteps++

		if opts.savePath != "" && opts.saveEvery > 0 && step%opts.saveEvery == 0 {
			if err := saveCheckpoint(opts.savePath, opts.model.vocab, step, avgLoss, opts.model.w); err != nil {
				return res, fmt.Errorf("save checkpoint at step %d: %w", step, err)
			}
		}

		if step == 1 || step%100 == 0 || step == opts.steps {
			ppl := math.Exp(avgLoss)
			fmt.Printf("step=%d loss=%.4f avg_loss=%.4f ppl=%.2f used=%d t_forward=%s t_compile=%s t_eval=%s t_update=%s\n",
				step, loss, avgLoss, ppl, res.Used,
				res.TotalForward.Round(time.Millisecond), res.TotalCompile.Round(time.Millisecond),
				res.TotalEval.Round(time.Millisecond), res.TotalUpdate.Round(time.Millisecond))
		}
		if opts.jsonOut {
			fmt.Fprintf(os.Stderr, "{\"type\":\"step\",\"step\":%d,\"loss\":%.6f,\"t_ane\":%.3f,\"t_io\":0.000,\"t_cls\":0.000,\"t_elem\":%.3f,\"t_rms\":0.000,\"t_cblas_wait\":0.000,\"compiles\":%d}\n",
				step-1, loss, ms(stats.HWEval), ms(stats.Update), res.Compiles)
		}
		if opts.accumSteps > 0 && step%opts.accumSteps == 0 {
			cms := ms(batchCompile)
			tms := ms(batchTrain)
			fmt.Printf("  [batch %d: compile=%.0fms train=%.1fms (%.1fms/step) compiles=%d]\n",
				batchSteps, cms, tms, tms/float64(batchSteps), res.Compiles)
			fmt.Printf("    ane=%.1f io=0.0 cls=0.0 elem=%.1f rms=0.0 cblas_wait=0.0 ms/step\n",
				ms(res.TotalHWEval)/float64(step), ms(res.TotalUpdate)/float64(step))
			if opts.jsonOut {
				fmt.Fprintf(os.Stderr, "{\"type\":\"batch\",\"batch\":%d,\"compile_ms\":%.1f,\"train_ms\":%.1f,\"ms_per_step\":%.1f}\n",
					step, cms, tms, tms/float64(batchSteps))
			}
			batchCompile = 0
			batchTrain = 0
			batchSteps = 0
		}
		if opts.maxCompiles > 0 && res.Compiles >= opts.maxCompiles {
			break
		}
	}
	res.AvgLoss = avgLoss
	res.Wall = time.Since(start)
	return res, nil
}

func runBenchmarks(tokens []uint16, baseW []float32, steps, batch, vocab int, hashMap bool, lr float64, seed int64, benchList string) error {
	engines := parseBenchEngines(benchList)
	if len(engines) == 0 {
		return fmt.Errorf("empty bench engine list")
	}
	fmt.Printf("benchmark: steps=%d batch=%d vocab=%d map=%s engines=%s\n", steps, batch, vocab, mapString(hashMap), strings.Join(engines, ","))
	for _, engine := range engines {
		w := append([]float32(nil), baseW...)
		m := newBigramModelFromWeights(vocab, w)
		rng := rand.New(rand.NewSource(seed))
		logitsEngine, err := newLogitsEngine(engine, vocab, batch)
		if err != nil {
			fmt.Printf("bench engine=%s status=init_error err=%v\n", engine, err)
			continue
		}
		res, err := runTraining(tokens, m, logitsEngine, steps, batch, lr, hashMap, rng, "", 0, false)
		_ = logitsEngine.Close()
		if err != nil {
			fmt.Printf("bench engine=%s status=run_error err=%v\n", engine, err)
			continue
		}
		tokPerSec := float64(res.Used) / res.Wall.Seconds()
		stepPerSec := float64(steps) / res.Wall.Seconds()
		fmt.Printf("bench engine=%s status=ok tokens=%d wall=%s tok/s=%.1f step/s=%.2f avg_loss=%.4f t_forward=%s t_compile=%s t_eval=%s t_update=%s compile/step=%s\n",
			engine, res.Used, res.Wall.Round(time.Millisecond), tokPerSec, stepPerSec, res.AvgLoss,
			res.TotalForward.Round(time.Millisecond), res.TotalCompile.Round(time.Millisecond),
			res.TotalEval.Round(time.Millisecond), res.TotalUpdate.Round(time.Millisecond),
			(res.TotalCompile / time.Duration(steps)).Round(time.Millisecond))
		if res.TotalHWEval > 0 {
			fmt.Printf("bench engine=%s hw_eval=%s hw_eval/step=%s\n",
				engine, res.TotalHWEval.Round(time.Millisecond), (res.TotalHWEval / time.Duration(steps)).Round(time.Microsecond))
		}
	}
	return nil
}

func parseBenchEngines(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		out = append(out, p)
	}
	return out
}

func mapString(hashMap bool) string {
	if hashMap {
		return "hash"
	}
	return "strict"
}

type logitsProvider interface {
	Name() string
	LogitsInto(dst, w []float32, xs []int, vocab int) (engineStats, error)
	Close() error
}

type weightAwareLogitsProvider interface {
	WeightsUpdated(w []float32, rows []int, vocab int) error
}

func newLogitsEngine(name string, vocab, batch int) (logitsProvider, error) {
	switch name {
	case "cpu":
		return cpuLogitsProvider{}, nil
	case "ane":
		return newANELogitsProvider(vocab, batch)
	case "auto":
		p, err := newANELogitsProvider(vocab, batch)
		if err == nil {
			return p, nil
		}
		fmt.Printf("warning: ANE unavailable, falling back to CPU logits: %v\n", err)
		return cpuLogitsProvider{}, nil
	default:
		return nil, fmt.Errorf("unknown engine %q", name)
	}
}

type cpuLogitsProvider struct{}

func (cpuLogitsProvider) Name() string { return "cpu" }
func (cpuLogitsProvider) Close() error { return nil }

func (cpuLogitsProvider) LogitsInto(dst, w []float32, xs []int, vocab int) (engineStats, error) {
	start := time.Now()
	if len(dst) != len(xs)*vocab {
		return engineStats{}, fmt.Errorf("cpu logits: output len=%d want=%d", len(dst), len(xs)*vocab)
	}
	for i, x := range xs {
		copy(dst[i*vocab:(i+1)*vocab], w[x*vocab:(x+1)*vocab])
	}
	return engineStats{Eval: time.Since(start)}, nil
}

type aneLogitsProvider struct {
	exDynamic   *linear.DynamicExecutor
	vocab       int
	spatial     int
	weightsIO   []float32
	weightsSeen bool
	compileInit time.Duration
	compileSeen bool
}

func newANELogitsProvider(vocab, batch int) (*aneLogitsProvider, error) {
	spatial := roundUp32(batch)
	p := &aneLogitsProvider{
		vocab:   vocab,
		spatial: spatial,
	}

	dyn := linear.NewDynamic(linear.Options{})
	start := time.Now()
	if err := dyn.Prepare(spatial, vocab, vocab); err != nil {
		dyn.Close()
		return nil, fmt.Errorf("ane logits: prepare dynamic executor: %w", err)
	}
	p.exDynamic = dyn
	p.compileInit = time.Since(start)
	return p, nil
}

func (p *aneLogitsProvider) Name() string { return "ane" }

func (p *aneLogitsProvider) Close() error {
	if p.exDynamic != nil {
		p.exDynamic.Close()
	}
	return nil
}

func (p *aneLogitsProvider) LogitsInto(dst, w []float32, xs []int, vocab int) (engineStats, error) {
	var est engineStats
	forwardStart := time.Now()
	if vocab != p.vocab {
		return est, fmt.Errorf("ane logits: vocab mismatch have=%d want=%d", p.vocab, vocab)
	}
	if len(xs) > p.spatial {
		return est, fmt.Errorf("ane logits: batch=%d exceeds spatial=%d", len(xs), p.spatial)
	}
	if len(dst) != len(xs)*vocab {
		return est, fmt.Errorf("ane logits: output len=%d want=%d", len(dst), len(xs)*vocab)
	}
	evalStart := time.Now()
	if !p.weightsSeen || len(p.weightsIO) != len(w) {
		p.weightsIO = growFloat32(p.weightsIO, len(w))
		transposeWeightsRowMajorOIToIO(p.weightsIO, w, p.vocab, p.vocab)
		if err := p.exDynamic.PrimeWeightsIO(p.spatial, p.vocab, p.vocab, p.weightsIO); err != nil {
			return est, fmt.Errorf("ane logits: prime dynamic weights: %w", err)
		}
		p.weightsSeen = true
	}
	lst, err := p.exDynamic.LinearOneHotIOIntoWithStats(context.Background(), dst, xs, p.spatial, p.vocab, p.vocab)
	if err != nil {
		return est, fmt.Errorf("ane logits: dynamic linear: %w", err)
	}
	est.Eval = time.Since(evalStart)
	est.HWEval = time.Duration(lst.HWExecutionNS) * time.Nanosecond
	if !p.compileSeen {
		est.Compile = p.compileInit
		est.Compiles = 1
		p.compileSeen = true
	}

	est.Forward = time.Since(forwardStart)
	return est, nil
}

func (p *aneLogitsProvider) WeightsUpdated(w []float32, rows []int, vocab int) error {
	if p == nil || p.exDynamic == nil || !p.weightsSeen || vocab != p.vocab {
		return nil
	}
	updateWeightsRowMajorRowsToIO(p.weightsIO, w, rows, vocab)
	if err := p.exDynamic.UpdateWeightsIORows(p.spatial, p.vocab, p.vocab, p.weightsIO, rows); err != nil {
		return fmt.Errorf("ane logits: update dynamic weights: %w", err)
	}
	return nil
}

type engineStats struct {
	Forward  time.Duration
	Compile  time.Duration
	Eval     time.Duration
	HWEval   time.Duration
	Compiles int
}

type stepStats struct {
	Forward  time.Duration
	Compile  time.Duration
	Eval     time.Duration
	HWEval   time.Duration
	Update   time.Duration
	Compiles int
}

type bigramModel struct {
	vocab  int
	w      []float32 // row-major [vocab][vocab]
	p      []float32 // softmax buffer
	g      []float32 // grad buffer
	logits []float32
	xs     []int
	ys     []int
}

func newBigramModel(vocab int, rng *rand.Rand) *bigramModel {
	w := make([]float32, vocab*vocab)
	for i := range w {
		w[i] = float32((rng.Float64()*2 - 1) * 0.01)
	}
	return &bigramModel{vocab: vocab, w: w, p: make([]float32, vocab), g: make([]float32, vocab)}
}

func newBigramModelFromWeights(vocab int, w []float32) *bigramModel {
	return &bigramModel{vocab: vocab, w: w, p: make([]float32, vocab), g: make([]float32, vocab)}
}

func (m *bigramModel) trainStep(tokens []uint16, batch int, lr float64, hashMap bool, rng *rand.Rand, logits logitsProvider) (float64, int, stepStats, error) {
	var stats stepStats
	totalLoss := 0.0
	xs, ys := sampleBatchInto(m.xs[:0], m.ys[:0], tokens, batch, hashMap, m.vocab, rng)
	m.xs, m.ys = xs, ys
	if len(xs) == 0 {
		return 0, 0, stats, nil
	}
	m.logits = growFloat32(m.logits, len(xs)*m.vocab)
	est, err := logits.LogitsInto(m.logits, m.w, xs, m.vocab)
	if err != nil {
		return 0, 0, stats, err
	}
	stats.Forward = est.Forward
	stats.Compile = est.Compile
	stats.Eval = est.Eval
	stats.HWEval = est.HWEval
	stats.Compiles = est.Compiles

	updateStart := time.Now()
	touchedRows := make([]int, 0, len(xs))
	for i := range xs {
		y := ys[i]
		softmaxInto(m.p, m.logits[i*m.vocab:(i+1)*m.vocab])
		p := m.p[y]
		if p < 1e-12 {
			p = 1e-12
		}
		totalLoss += -math.Log(float64(p))

		copy(m.g, m.p)
		m.g[y] -= 1
		rowID := xs[i]
		row := m.w[rowID*m.vocab : (rowID+1)*m.vocab]
		for j := 0; j < m.vocab; j++ {
			row[j] -= float32(lr) * m.g[j]
		}
		touchedRows = appendUniqueInt(touchedRows, rowID)
	}
	if updater, ok := logits.(weightAwareLogitsProvider); ok {
		if err := updater.WeightsUpdated(m.w, touchedRows, m.vocab); err != nil {
			return 0, 0, stats, err
		}
	}
	stats.Update = time.Since(updateStart)
	return totalLoss / float64(len(xs)), len(xs), stats, nil
}

func sampleBatchInto(xs, ys []int, tokens []uint16, batch int, hashMap bool, vocab int, rng *rand.Rand) ([]int, []int) {
	maxTries := batch * 64
	for tries := 0; tries < maxTries && len(xs) < batch; tries++ {
		pos := rng.Intn(len(tokens) - 1)
		x := int(tokens[pos])
		y := int(tokens[pos+1])
		if hashMap {
			x %= vocab
			y %= vocab
		} else if x >= vocab || y >= vocab {
			continue
		}
		xs = append(xs, x)
		ys = append(ys, y)
	}
	return xs, ys
}

func growFloat32(buf []float32, n int) []float32 {
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

func appendUniqueInt(dst []int, v int) []int {
	for _, x := range dst {
		if x == v {
			return dst
		}
	}
	return append(dst, v)
}

func transposeWeightsRowMajorOIToIO(dst, src []float32, inDim, outDim int) {
	for out := 0; out < outDim; out++ {
		row := src[out*inDim : (out+1)*inDim]
		for in := 0; in < inDim; in++ {
			dst[in*outDim+out] = row[in]
		}
	}
}

func updateWeightsRowMajorRowsToIO(dst, src []float32, rows []int, vocab int) {
	for _, rowID := range rows {
		row := src[rowID*vocab : (rowID+1)*vocab]
		for in := 0; in < vocab; in++ {
			dst[in*vocab+rowID] = row[in]
		}
	}
}

func softmaxInto(dst, logits []float32) {
	mx := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > mx {
			mx = logits[i]
		}
	}
	sum := 0.0
	for i := range logits {
		e := float32(math.Exp(float64(logits[i] - mx)))
		dst[i] = e
		sum += float64(e)
	}
	inv := float32(1.0 / sum)
	for i := range dst {
		dst[i] *= inv
	}
}

func loadTokens(path string) ([]uint16, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%2 != 0 {
		return nil, fmt.Errorf("odd file size %d", len(b))
	}
	toks := make([]uint16, len(b)/2)
	for i := 0; i < len(toks); i++ {
		toks[i] = binary.LittleEndian.Uint16(b[2*i:])
	}
	return toks, nil
}

func maxToken(tokens []uint16) uint16 {
	var mx uint16
	for _, t := range tokens {
		if t > mx {
			mx = t
		}
	}
	return mx
}

const ckptMagic = "ANEBG01\n"

type checkpoint struct {
	Vocab   int
	Step    int
	AvgLoss float64
	Weights []float32
}

func loadCheckpoint(path string) (*checkpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic := make([]byte, len(ckptMagic))
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic) != ckptMagic {
		return nil, fmt.Errorf("bad checkpoint magic")
	}
	var vocab uint32
	var step uint64
	var avgLoss float64
	if err := binary.Read(f, binary.LittleEndian, &vocab); err != nil {
		return nil, fmt.Errorf("read vocab: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &step); err != nil {
		return nil, fmt.Errorf("read step: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &avgLoss); err != nil {
		return nil, fmt.Errorf("read avg_loss: %w", err)
	}
	weights := make([]float32, int(vocab)*int(vocab))
	for i := range weights {
		if err := binary.Read(f, binary.LittleEndian, &weights[i]); err != nil {
			return nil, fmt.Errorf("read weights[%d]: %w", i, err)
		}
	}
	return &checkpoint{
		Vocab:   int(vocab),
		Step:    int(step),
		AvgLoss: avgLoss,
		Weights: weights,
	}, nil
}

func saveCheckpoint(path string, vocab, step int, avgLoss float64, weights []float32) error {
	if len(weights) != vocab*vocab {
		return fmt.Errorf("checkpoint: weight count=%d want=%d", len(weights), vocab*vocab)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("mkdir checkpoint dir: %w", err)
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := f.Write([]byte(ckptMagic)); err != nil {
		_ = f.Close()
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(vocab)); err != nil {
		_ = f.Close()
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(step)); err != nil {
		_ = f.Close()
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, avgLoss); err != nil {
		_ = f.Close()
		return err
	}
	for i := range weights {
		if err := binary.Write(f, binary.LittleEndian, weights[i]); err != nil {
			_ = f.Close()
			return err
		}
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	return nil
}

func roundUp32(v int) int {
	if v < 32 {
		return 32
	}
	return (v + 31) &^ 31
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func reexecWithResume(ckptPath string, remaining int) error {
	args := append([]string{}, os.Args[1:]...)
	args = append(args, "-resume", "-load", ckptPath, "-steps", fmt.Sprintf("%d", remaining))
	cmd := exec.Command(os.Args[0], args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
