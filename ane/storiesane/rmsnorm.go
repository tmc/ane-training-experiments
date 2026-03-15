package storiesane

import (
	"math"
	"runtime"
	"sync"
)

// rmsNormGradPool holds pre-allocated shard buffers to avoid per-call
// allocations in rmsNormGradWeightsWithRRMS. The flat buffer is sized
// for maxWorkers × dim floats and reused across calls.
var rmsNormGradPool struct {
	mu   sync.Mutex
	buf  []float32
	maxW int
}

func rmsNormGradWeights(dw, dy, x, w []float32, d, s int) {
	rmsNormGradWeightsWithRRMS(dw, dy, x, nil, d, s)
}

func rmsNormGradWeightsWithRRMS(dw, dy, x, rrms []float32, d, s int) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || s < workers*4 {
		rmsNormGradWeightsRange(dw, dy, x, rrms, d, s, 0, s)
		return
	}
	if workers > s {
		workers = s
	}
	chunk := (s + workers - 1) / workers

	// Use pooled shard buffer to avoid per-call allocations.
	rmsNormGradPool.mu.Lock()
	need := workers * d
	if len(rmsNormGradPool.buf) < need {
		rmsNormGradPool.buf = make([]float32, need)
		rmsNormGradPool.maxW = workers
	}
	buf := rmsNormGradPool.buf[:need]
	clear(buf)
	rmsNormGradPool.mu.Unlock()

	var wg sync.WaitGroup
	active := 0
	for worker := 0; worker < workers; worker++ {
		start := worker * chunk
		if start >= s {
			break
		}
		end := start + chunk
		if end > s {
			end = s
		}
		shard := buf[worker*d : (worker+1)*d]
		active++
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			rmsNormGradWeightsRange(shard, dy, x, rrms, d, s, start, end)
		}(start, end)
	}
	wg.Wait()
	for w := 0; w < active; w++ {
		shard := buf[w*d : (w+1)*d]
		for i := range dw {
			dw[i] += shard[i]
		}
	}
}

// rmsNormBackwardPooled computes both dx and dw using pooled shard buffers,
// avoiding per-call allocations from stories.RMSNormBackward.
func rmsNormBackwardPooled(dx, dw, dy, x, w []float32, d, s int) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || s < workers*4 {
		rmsNormBackwardRange(dx, dw, dy, x, w, d, s, 0, s)
		return
	}
	if workers > s {
		workers = s
	}
	chunk := (s + workers - 1) / workers

	rmsNormGradPool.mu.Lock()
	need := workers * d
	if len(rmsNormGradPool.buf) < need {
		rmsNormGradPool.buf = make([]float32, need)
		rmsNormGradPool.maxW = workers
	}
	buf := rmsNormGradPool.buf[:need]
	clear(buf)
	rmsNormGradPool.mu.Unlock()

	var wg sync.WaitGroup
	active := 0
	for worker := 0; worker < workers; worker++ {
		start := worker * chunk
		if start >= s {
			break
		}
		end := start + chunk
		if end > s {
			end = s
		}
		shard := buf[worker*d : (worker+1)*d]
		active++
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			rmsNormBackwardRange(dx, shard, dy, x, w, d, s, start, end)
		}(start, end)
	}
	wg.Wait()
	for w := 0; w < active; w++ {
		shard := buf[w*d : (w+1)*d]
		for i := range dw {
			dw[i] += shard[i]
		}
	}
}

func rmsNormBackwardRange(dx, dw, dy, x, w []float32, d, s, start, end int) {
	for t := start; t < end; t++ {
		sum := 0.0
		for i := 0; i < d; i++ {
			v := float64(x[i*s+t])
			sum += v * v
		}
		rrms := 1.0 / math.Sqrt(sum/float64(d)+1e-5)
		rrms2InvD := (rrms * rrms) / float64(d)
		dot := 0.0
		for i := 0; i < d; i++ {
			dot += float64(dy[i*s+t] * x[i*s+t] * w[i])
		}
		for i := 0; i < d; i++ {
			v := float64(dy[i*s+t]) - float64(x[i*s+t])*dot*rrms2InvD
			dx[i*s+t] = float32(v * rrms * float64(w[i]))
			dw[i] += float32(float64(dy[i*s+t]*x[i*s+t]) * rrms)
		}
	}
}

func rmsNormGradWeightsRange(dw, dy, x, rrms []float32, d, s, start, end int) {
	for t := start; t < end; t++ {
		scale := 0.0
		if len(rrms) > t {
			scale = float64(rrms[t])
		} else {
			sum := 0.0
			for i := 0; i < d; i++ {
				v := float64(x[i*s+t])
				sum += v * v
			}
			scale = 1.0 / math.Sqrt(sum/float64(d)+1e-5)
		}
		for i := 0; i < d; i++ {
			dw[i] += float32(float64(dy[i*s+t]*x[i*s+t]) * scale)
		}
	}
}
