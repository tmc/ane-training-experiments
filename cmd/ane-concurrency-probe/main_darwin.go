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

	"github.com/maderix/ANE/ane/clientmodel"
	"github.com/maderix/ANE/internal/clientkernel"
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
		inputBytes  = flag.Int("input-bytes", 4096, "single input tensor bytes")
		outputBytes = flag.Int("output-bytes", 4096, "single output tensor bytes")
		workersRaw  = flag.String("workers", "1,2,4,8", "comma-separated worker depths")
		iters       = flag.Int("iters", 200, "eval iterations per worker")
		sampleMS    = flag.Int("sample-ms", 2, "diagnostics sampling period in ms")
		forceNew    = flag.Bool("force-client-new", false, "force dedicated _ANEClient per kernel")
		preferPriv  = flag.Bool("prefer-private-client", false, "prefer _ANEClient.sharedPrivateConnection")
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

	fmt.Printf("model=%s input_bytes=%d output_bytes=%d iters=%d sample_ms=%d mode=%s\n", *modelPath, *inputBytes, *outputBytes, *iters, *sampleMS, *modeRaw)
	fmt.Printf("force_client_new=%v prefer_private_client=%v\n", *forceNew, *preferPriv)
	fmt.Println("depth,queue_depth,program_queue_depth,mean_ms,p50_ms,p95_ms,p99_ms,max_ms,evals_per_sec,max_inflight,avg_inflight")

	for _, depth := range workerDepths {
		ks, err := openKernels(depth, clientkernel.EvalOptions{
			ModelPath:         *modelPath,
			ModelKey:          "s",
			ForceNewClient:    *forceNew,
			PreferPrivateConn: *preferPriv,
			InputBytes:        uint32(*inputBytes),
			OutputBytes:       uint32(*outputBytes),
		})
		if err != nil {
			log.Printf("depth=%d open failed: %v", depth, err)
			continue
		}

		input := make([]byte, *inputBytes)
		for i := range input {
			input[i] = byte((i % 251) + 1)
		}
		for _, k := range ks {
			if err := k.WriteInput(0, input); err != nil {
				closeAll(ks)
				log.Fatalf("depth=%d write input: %v", depth, err)
			}
		}

		d0 := ks[0].Diagnostics()
		waitEvents := make([]*clientmodel.SharedEvent, depth)
		signalEvents := make([]*clientmodel.SharedEvent, depth)
		if mode == modeBidirectional {
			for i := 0; i < depth; i++ {
				waitEvents[i], err = clientmodel.NewSharedEvent()
				if err != nil {
					closeAll(ks)
					closeEvents(waitEvents)
					closeEvents(signalEvents)
					log.Fatalf("depth=%d create wait event: %v", depth, err)
				}
				signalEvents[i], err = clientmodel.NewSharedEvent()
				if err != nil {
					closeAll(ks)
					closeEvents(waitEvents)
					closeEvents(signalEvents)
					log.Fatalf("depth=%d create signal event: %v", depth, err)
				}
			}
		}
		var inflightMax int64
		var inflightSum int64
		var inflightSamples int64
		stopSampler := make(chan struct{})
		var samplerWG sync.WaitGroup
		samplerWG.Add(1)
		go func() {
			defer samplerWG.Done()
			tk := time.NewTicker(time.Duration(*sampleMS) * time.Millisecond)
			defer tk.Stop()
			for {
				select {
				case <-stopSampler:
					return
				case <-tk.C:
					var s int64
					for _, k := range ks {
						di := k.Diagnostics()
						if di.CurrentAsyncRequestsInFlightOK {
							s += di.CurrentAsyncRequestsInFlight
						}
					}
					for {
						m := atomic.LoadInt64(&inflightMax)
						if s <= m || atomic.CompareAndSwapInt64(&inflightMax, m, s) {
							break
						}
					}
					atomic.AddInt64(&inflightSum, s)
					atomic.AddInt64(&inflightSamples, 1)
				}
			}
		}()

		allLat := make([][]float64, depth)
		var runWG sync.WaitGroup
		runWG.Add(depth)
		start := make(chan struct{})
		runStart := time.Now()
		for i := 0; i < depth; i++ {
			idx := i
			go func() {
				defer runWG.Done()
				local := make([]float64, 0, *iters)
				<-start
				for j := 0; j < *iters; j++ {
					t0 := time.Now()
					switch mode {
					case modeEval:
						if err := ks[idx].Eval(); err != nil {
							log.Printf("depth=%d worker=%d iter=%d eval error: %v", depth, idx, j, err)
							return
						}
					case modeBidirectional:
						v := uint64(j + 1)
						if err := waitEvents[idx].Signal(v); err != nil {
							log.Printf("depth=%d worker=%d iter=%d wait signal error: %v", depth, idx, j, err)
							return
						}
						if err := ks[idx].EvalBidirectional(
							waitEvents[idx].Port(), v,
							signalEvents[idx].Port(), v,
							clientmodel.SharedEventEvalOptions{
								DisableIOFencesUseSharedEvents: true,
								EnableFWToFWSignal:             false,
							},
						); err != nil {
							log.Printf("depth=%d worker=%d iter=%d bidir eval error: %v", depth, idx, j, err)
							return
						}
						ok, err := signalEvents[idx].Wait(v, 250*time.Millisecond)
						if err != nil || !ok {
							log.Printf("depth=%d worker=%d iter=%d signal wait error: ok=%v err=%v", depth, idx, j, ok, err)
							return
						}
					default:
						log.Printf("depth=%d worker=%d unsupported mode", depth, idx)
						return
					}
					local = append(local, msSince(t0))
				}
				allLat[idx] = local
			}()
		}
		close(start)
		runWG.Wait()
		runDur := time.Since(runStart)
		close(stopSampler)
		samplerWG.Wait()

		flat := flatten(allLat)
		st := summarize(flat)
		eps := 0.0
		if runDur > 0 {
			eps = float64(len(flat)) / runDur.Seconds()
		}
		avgInflight := 0.0
		if n := atomic.LoadInt64(&inflightSamples); n > 0 {
			avgInflight = float64(atomic.LoadInt64(&inflightSum)) / float64(n)
		}

		fmt.Printf(
			"%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%d,%.3f\n",
			depth,
			d0.ModelQueueDepth,
			d0.ProgramQueueDepth,
			st.Mean, st.P50, st.P95, st.P99, st.Max,
			eps,
			atomic.LoadInt64(&inflightMax),
			avgInflight,
		)

		closeEvents(waitEvents)
		closeEvents(signalEvents)
		closeAll(ks)
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

func openKernels(n int, opts clientkernel.EvalOptions) ([]*clientmodel.Kernel, error) {
	ks := make([]*clientmodel.Kernel, 0, n)
	for i := 0; i < n; i++ {
		k, err := clientkernel.Compile(opts)
		if err != nil {
			closeAll(ks)
			return nil, err
		}
		ks = append(ks, k)
	}
	return ks, nil
}

func closeAll(ks []*clientmodel.Kernel) {
	for _, k := range ks {
		if k != nil {
			k.Close()
		}
	}
}

func closeEvents(es []*clientmodel.SharedEvent) {
	for _, e := range es {
		if e != nil {
			_ = e.Close()
		}
	}
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
	i := int(math.Floor(pos))
	f := pos - float64(i)
	if i+1 >= len(sorted) {
		return sorted[i]
	}
	return sorted[i]*(1-f) + sorted[i+1]*f
}
