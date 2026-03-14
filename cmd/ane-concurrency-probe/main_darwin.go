//go:build darwin

package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	xane "github.com/tmc/apple/x/ane"
	xanetelemetry "github.com/tmc/apple/x/ane/telemetry"
)

type stats struct {
	Count int
	Mean  float64
	P50   float64
	P95   float64
	P99   float64
	Max   float64
}

type runMode int

const (
	modeEval runMode = iota
	modeBidirectional
)

func main() {
	var (
		modelPath   = flag.String("compiled", "/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc", "path to compiled .mlmodelc")
		modelKey    = flag.String("model-key", "s", "model key")
		inputBytes  = flag.Int("input-bytes", 4096, "single input tensor logical bytes")
		outputBytes = flag.Int("output-bytes", 4096, "single output tensor logical bytes")
		workersRaw  = flag.String("workers", "1,2,4,8", "comma-separated worker depths")
		iters       = flag.Int("iters", 200, "eval iterations per worker")
		sampleMS    = flag.Int("sample-ms", 2, "in-flight sampling period in ms")
		qos         = flag.Uint("qos", 21, "ANE QoS")
		modeRaw     = flag.String("mode", "eval", "run mode: eval|bidir")
	)
	flag.Parse()

	if *inputBytes <= 0 || *outputBytes <= 0 || *iters <= 0 || *sampleMS <= 0 {
		log.Fatalf("invalid flags: input/output/iters/sample must be > 0")
	}
	mode, err := parseMode(*modeRaw)
	if err != nil {
		log.Fatal(err)
	}
	workerDepths, err := parseWorkerList(*workersRaw)
	if err != nil {
		log.Fatalf("parse workers: %v", err)
	}

	rt, err := xane.Open()
	if err != nil {
		log.Fatalf("open runtime: %v", err)
	}
	defer rt.Close()

	fmt.Printf("model=%s input_bytes=%d output_bytes=%d iters=%d sample_ms=%d mode=%s qos=%d\n", *modelPath, *inputBytes, *outputBytes, *iters, *sampleMS, *modeRaw, *qos)
	fmt.Printf("runtime: x/ane compile_count=%d\n", rt.CompileCount())
	fmt.Println("depth,queue_depth,mean_ms,p50_ms,p95_ms,p99_ms,max_ms,evals_per_sec,max_inflight,avg_inflight")

	for _, depth := range workerDepths {
		var (
			diag        xanetelemetry.Diagnostics
			flat        []float64
			runDur      time.Duration
			inflightMax int64
			avgInflight float64
		)
		switch mode {
		case modeEval:
			diag, flat, runDur, inflightMax, avgInflight, err = runEvalDepth(rt, *modelPath, *modelKey, uint32(*qos), *inputBytes, *outputBytes, depth, *iters, *sampleMS)
		case modeBidirectional:
			diag, flat, runDur, inflightMax, avgInflight, err = runBidirectionalDepth(rt, *modelPath, *modelKey, uint32(*qos), *inputBytes, *outputBytes, depth, *iters, *sampleMS)
		default:
			err = fmt.Errorf("unsupported mode %q", *modeRaw)
		}
		if err != nil {
			log.Printf("depth=%d run failed: %v", depth, err)
			continue
		}

		st := summarize(flat)
		eps := 0.0
		if runDur > 0 {
			eps = float64(len(flat)) / runDur.Seconds()
		}
		queueDepth := 0
		if diag.ModelQueueDepthKnown {
			queueDepth = diag.ModelQueueDepth
		}
		fmt.Printf(
			"%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%d,%.3f\n",
			depth,
			queueDepth,
			st.Mean, st.P50, st.P95, st.P99, st.Max,
			eps,
			inflightMax,
			avgInflight,
		)
	}
}

func runEvalDepth(rt *xane.Runtime, modelPath, modelKey string, qos uint32, inputBytes, outputBytes, depth, iters, sampleMS int) (xanetelemetry.Diagnostics, []float64, time.Duration, int64, float64, error) {
	k, err := compileKernel(rt, modelPath, modelKey, qos, inputBytes, outputBytes)
	if err != nil {
		return xanetelemetry.Diagnostics{}, nil, 0, 0, 0, err
	}
	defer k.Close()
	diag := xanetelemetry.ProbeDiagnostics(k)

	input := patternedInput(k.InputAllocSize(0))
	if err := k.WriteInput(0, input); err != nil {
		return xanetelemetry.Diagnostics{}, nil, 0, 0, 0, fmt.Errorf("stage input: %w", err)
	}

	pool, err := xane.NewRequestPool(k, depth)
	if err != nil {
		return xanetelemetry.Diagnostics{}, nil, 0, 0, 0, fmt.Errorf("request pool: %w", err)
	}
	defer pool.Close()

	var inflight atomic.Int64
	stopSampler := startInflightSampler(&inflight, sampleMS)

	allLat := make([][]float64, depth)
	var runWG sync.WaitGroup
	runWG.Add(depth)
	start := make(chan struct{})
	runStart := time.Now()
	for i := 0; i < depth; i++ {
		idx := i
		go func() {
			defer runWG.Done()
			local := make([]float64, 0, iters)
			<-start
			for j := 0; j < iters; j++ {
				req := pool.Acquire()
				inflight.Add(1)
				t0 := time.Now()
				err := req.Eval()
				ms := msSince(t0)
				inflight.Add(-1)
				req.Release()
				if err != nil {
					log.Printf("depth=%d worker=%d iter=%d eval error: %v", depth, idx, j, err)
					return
				}
				local = append(local, ms)
			}
			allLat[idx] = local
		}()
	}
	close(start)
	runWG.Wait()
	runDur := time.Since(runStart)
	maxInflight, avg := stopSampler()

	return diag, flatten(allLat), runDur, maxInflight, avg, nil
}

func runBidirectionalDepth(rt *xane.Runtime, modelPath, modelKey string, qos uint32, inputBytes, outputBytes, depth, iters, sampleMS int) (xanetelemetry.Diagnostics, []float64, time.Duration, int64, float64, error) {
	ks := make([]*xane.Kernel, 0, depth)
	defer closeKernels(ks)
	for i := 0; i < depth; i++ {
		k, err := compileKernel(rt, modelPath, modelKey, qos, inputBytes, outputBytes)
		if err != nil {
			return xanetelemetry.Diagnostics{}, nil, 0, 0, 0, err
		}
		if err := k.WriteInput(0, patternedInput(k.InputAllocSize(0))); err != nil {
			k.Close()
			return xanetelemetry.Diagnostics{}, nil, 0, 0, 0, fmt.Errorf("stage input: %w", err)
		}
		ks = append(ks, k)
	}
	diag := xanetelemetry.ProbeDiagnostics(ks[0])

	waitEvents := make([]*xane.SharedEvent, depth)
	signalEvents := make([]*xane.SharedEvent, depth)
	defer closeEvents(waitEvents)
	defer closeEvents(signalEvents)
	for i := 0; i < depth; i++ {
		var err error
		waitEvents[i], err = xane.NewSharedEvent()
		if err != nil {
			return xanetelemetry.Diagnostics{}, nil, 0, 0, 0, fmt.Errorf("create wait event: %w", err)
		}
		signalEvents[i], err = xane.NewSharedEvent()
		if err != nil {
			return xanetelemetry.Diagnostics{}, nil, 0, 0, 0, fmt.Errorf("create signal event: %w", err)
		}
	}

	var inflight atomic.Int64
	stopSampler := startInflightSampler(&inflight, sampleMS)

	allLat := make([][]float64, depth)
	var runWG sync.WaitGroup
	runWG.Add(depth)
	start := make(chan struct{})
	runStart := time.Now()
	for i := 0; i < depth; i++ {
		idx := i
		go func() {
			defer runWG.Done()
			local := make([]float64, 0, iters)
			<-start
			for j := 0; j < iters; j++ {
				v := uint64(j + 1)
				waitEvents[idx].Signal(v)
				inflight.Add(1)
				t0 := time.Now()
				err := ks[idx].EvalBidirectional(
					waitEvents[idx].Port(), v,
					signalEvents[idx].Port(), v,
					xane.SharedEventEvalOptions{
						DisableIOFencesUseSharedEvents: true,
						EnableFWToFWSignal:             false,
					},
				)
				ms := msSince(t0)
				inflight.Add(-1)
				if err != nil {
					log.Printf("depth=%d worker=%d iter=%d bidir eval error: %v", depth, idx, j, err)
					return
				}
				if ok := signalEvents[idx].Wait(v, 250*time.Millisecond); !ok {
					log.Printf("depth=%d worker=%d iter=%d signal wait timed out", depth, idx, j)
					return
				}
				local = append(local, ms)
			}
			allLat[idx] = local
		}()
	}
	close(start)
	runWG.Wait()
	runDur := time.Since(runStart)
	maxInflight, avg := stopSampler()

	return diag, flatten(allLat), runDur, maxInflight, avg, nil
}

func compileKernel(rt *xane.Runtime, modelPath, modelKey string, qos uint32, inputBytes, outputBytes int) (*xane.Kernel, error) {
	k, err := rt.Compile(xane.CompileOptions{
		ModelType:   xane.ModelTypePackage,
		PackagePath: modelPath,
		ModelKey:    modelKey,
		QoS:         qos,
	})
	if err != nil {
		return nil, err
	}
	if k.NumInputs() != 1 || k.NumOutputs() != 1 {
		k.Close()
		return nil, fmt.Errorf("compiled model reported %d inputs and %d outputs; want 1 input and 1 output", k.NumInputs(), k.NumOutputs())
	}
	if got := k.InputLayout(0).LogicalBytes(); got != inputBytes {
		k.Close()
		return nil, fmt.Errorf("input logical bytes = %d, want %d", got, inputBytes)
	}
	if got := k.OutputLayout(0).LogicalBytes(); got != outputBytes {
		k.Close()
		return nil, fmt.Errorf("output logical bytes = %d, want %d", got, outputBytes)
	}
	return k, nil
}

func patternedInput(n int) []byte {
	buf := make([]byte, n)
	for i := range buf {
		buf[i] = byte((i % 251) + 1)
	}
	return buf
}

func startInflightSampler(inflight *atomic.Int64, sampleMS int) func() (int64, float64) {
	var inflightMax atomic.Int64
	var inflightSum atomic.Int64
	var inflightSamples atomic.Int64
	stop := make(chan struct{})
	var samplerWG sync.WaitGroup
	samplerWG.Add(1)
	go func() {
		defer samplerWG.Done()
		tk := time.NewTicker(time.Duration(sampleMS) * time.Millisecond)
		defer tk.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tk.C:
				s := inflight.Load()
				for {
					m := inflightMax.Load()
					if s <= m || inflightMax.CompareAndSwap(m, s) {
						break
					}
				}
				inflightSum.Add(s)
				inflightSamples.Add(1)
			}
		}
	}()
	return func() (int64, float64) {
		close(stop)
		samplerWG.Wait()
		n := inflightSamples.Load()
		avg := 0.0
		if n > 0 {
			avg = float64(inflightSum.Load()) / float64(n)
		}
		return inflightMax.Load(), avg
	}
}

func closeKernels(ks []*xane.Kernel) {
	for _, k := range ks {
		if k != nil {
			_ = k.Close()
		}
	}
}

func closeEvents(es []*xane.SharedEvent) {
	for _, e := range es {
		if e != nil {
			_ = e.Close()
		}
	}
}

func parseMode(raw string) (runMode, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "eval":
		return modeEval, nil
	case "bidir":
		return modeBidirectional, nil
	default:
		return modeEval, fmt.Errorf("invalid mode %q (want eval|bidir)", raw)
	}
}

func parseWorkerList(raw string) ([]int, error) {
	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	seen := map[int]bool{}
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("invalid worker depth %q", p)
		}
		if !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	sort.Ints(out)
	if len(out) == 0 {
		return nil, fmt.Errorf("no worker depths provided")
	}
	return out, nil
}

func msSince(t time.Time) float64 {
	return float64(time.Since(t).Microseconds()) / 1000.0
}

func flatten(v [][]float64) []float64 {
	n := 0
	for _, x := range v {
		n += len(x)
	}
	out := make([]float64, 0, n)
	for _, x := range v {
		out = append(out, x...)
	}
	return out
}

func summarize(v []float64) stats {
	if len(v) == 0 {
		return stats{}
	}
	x := append([]float64(nil), v...)
	sort.Float64s(x)
	sum := 0.0
	for _, n := range x {
		sum += n
	}
	return stats{
		Count: len(x),
		Mean:  sum / float64(len(x)),
		P50:   percentile(x, 0.50),
		P95:   percentile(x, 0.95),
		P99:   percentile(x, 0.99),
		Max:   x[len(x)-1],
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}
	pos := p * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return sorted[lo]
	}
	frac := pos - float64(lo)
	return sorted[lo] + (sorted[hi]-sorted[lo])*frac
}
