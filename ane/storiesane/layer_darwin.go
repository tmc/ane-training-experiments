//go:build darwin

package storiesane

import (
	"fmt"

	"github.com/maderix/ANE/ane/mil"
	"github.com/maderix/ANE/ane/model"
	"github.com/maderix/ANE/ane/stories"
)

type layerForwardWeights struct {
	RMSAtt []float32
	Wq     []float32
	Wk     []float32
	Wv     []float32
	Wo     []float32
	RMSFFN []float32
	W1     []float32
	W2     []float32
	W3     []float32
}

type layerForward struct {
	dim    int
	hidden int
	heads  int
	seq    int

	att *model.Kernel
	ffn *model.Kernel

	attOut []float32
	ffnOut []float32
	x2     []float32
}

func compileStoriesLayerForward(layer stories.LayerWeights, seq int) (*layerForward, error) {
	return compileLayerForward(stories.Dim, stories.Hidden, stories.Heads, seq, layerForwardWeights{
		RMSAtt: layer.RMSAtt,
		Wq:     layer.Wq,
		Wk:     layer.Wk,
		Wv:     layer.Wv,
		Wo:     layer.Wo,
		RMSFFN: layer.RMSFFN,
		W1:     layer.W1,
		W2:     layer.W2,
		W3:     layer.W3,
	})
}

func compileLayerForward(dim, hidden, heads, seq int, w layerForwardWeights) (_ *layerForward, err error) {
	if dim <= 0 || hidden <= 0 || heads <= 0 || seq <= 0 {
		return nil, fmt.Errorf("compile layer forward: invalid shape dim=%d hidden=%d heads=%d seq=%d", dim, hidden, heads, seq)
	}
	if dim%heads != 0 {
		return nil, fmt.Errorf("compile layer forward: dim=%d is not divisible by heads=%d", dim, heads)
	}
	if err := validateLayerWeights(dim, hidden, w); err != nil {
		return nil, err
	}
	rmsAttBlob, err := mil.BuildVectorWeightBlob(w.RMSAtt)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: rms_att blob: %w", err)
	}
	wqBlob, err := mil.BuildWeightBlob(w.Wq, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: wq blob: %w", err)
	}
	wkBlob, err := mil.BuildWeightBlob(w.Wk, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: wk blob: %w", err)
	}
	wvBlob, err := mil.BuildWeightBlob(w.Wv, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: wv blob: %w", err)
	}
	woBlob, err := mil.BuildWeightBlob(w.Wo, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: wo blob: %w", err)
	}
	maskBlob, err := mil.BuildCausalMaskBlob(seq)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: mask blob: %w", err)
	}
	rmsFFNBlob, err := mil.BuildVectorWeightBlob(w.RMSFFN)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: rms_ffn blob: %w", err)
	}
	w1Blob, err := mil.BuildWeightBlob(w.W1, hidden, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: w1 blob: %w", err)
	}
	w2Blob, err := mil.BuildWeightBlob(w.W2, dim, hidden)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: w2 blob: %w", err)
	}
	w3Blob, err := mil.BuildWeightBlob(w.W3, hidden, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: w3 blob: %w", err)
	}

	att, err := compileMultiFP16Kernel(
		mil.GenSDPAForwardTaps(dim, heads, seq),
		[]model.WeightFile{
			{Path: "@model_path/weights/rms1.bin", Blob: rmsAttBlob},
			{Path: "@model_path/weights/wq.bin", Blob: wqBlob},
			{Path: "@model_path/weights/wk.bin", Blob: wkBlob},
			{Path: "@model_path/weights/wv.bin", Blob: wvBlob},
			{Path: "@model_path/weights/wo.bin", Blob: woBlob},
			{Path: "@model_path/weights/mask.bin", Blob: maskBlob},
		},
		dim,
		6*dim,
		seq,
	)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: attention: %w", err)
	}
	defer func() {
		if err != nil {
			closeKernel(att)
		}
	}()

	ffn, err := compileMultiFP16Kernel(
		mil.GenFFNForwardTaps(dim, hidden, seq),
		[]model.WeightFile{
			{Path: "@model_path/weights/rms2.bin", Blob: rmsFFNBlob},
			{Path: "@model_path/weights/w1.bin", Blob: w1Blob},
			{Path: "@model_path/weights/w2.bin", Blob: w2Blob},
			{Path: "@model_path/weights/w3.bin", Blob: w3Blob},
		},
		dim,
		2*dim+3*hidden,
		seq,
	)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: ffn: %w", err)
	}

	return &layerForward{
		dim:    dim,
		hidden: hidden,
		heads:  heads,
		seq:    seq,
		att:    att,
		ffn:    ffn,
		attOut: make([]float32, 6*dim*seq),
		ffnOut: make([]float32, (2*dim+3*hidden)*seq),
		x2:     make([]float32, dim*seq),
	}, nil
}

func (lf *layerForward) close() {
	if lf == nil {
		return
	}
	closeKernel(lf.att)
	closeKernel(lf.ffn)
	lf.att = nil
	lf.ffn = nil
	lf.attOut = nil
	lf.ffnOut = nil
	lf.x2 = nil
}

func (lf *layerForward) run(out, x []float32) error {
	return lf.runWithTaps(out, x, nil)
}

func (lf *layerForward) runWithTaps(out, x []float32, cache *layerCache) error {
	if lf == nil || lf.att == nil || lf.ffn == nil {
		return fmt.Errorf("run layer forward: layer is closed")
	}
	want := lf.dim * lf.seq
	if len(x) != want {
		return fmt.Errorf("run layer forward: input len=%d want=%d", len(x), want)
	}
	if len(out) != want {
		return fmt.Errorf("run layer forward: output len=%d want=%d", len(out), want)
	}

	if err := lf.att.WriteInputFP16(0, x); err != nil {
		return fmt.Errorf("run layer forward: write attention input: %w", err)
	}
	if err := lf.att.Eval(); err != nil {
		return fmt.Errorf("run layer forward: eval attention: %w", err)
	}
	if err := lf.att.ReadOutputFP16(0, lf.attOut); err != nil {
		return fmt.Errorf("run layer forward: read attention output: %w", err)
	}
	attMain := lf.attOut[:want]
	for i := range lf.x2 {
		lf.x2[i] = x[i] + attMain[i]
	}

	if err := lf.ffn.WriteInputFP16(0, lf.x2); err != nil {
		return fmt.Errorf("run layer forward: write ffn input: %w", err)
	}
	if err := lf.ffn.Eval(); err != nil {
		return fmt.Errorf("run layer forward: eval ffn: %w", err)
	}
	if err := lf.ffn.ReadOutputFP16(0, lf.ffnOut); err != nil {
		return fmt.Errorf("run layer forward: read ffn output: %w", err)
	}
	ffnMain := lf.ffnOut[:want]
	for i := range out {
		out[i] = lf.x2[i] + ffnMain[i]
	}
	if cache != nil {
		copy(cache.xNorm, lf.attOut[5*want:6*want])
		copy(cache.q, lf.attOut[want:2*want])
		copy(cache.k, lf.attOut[2*want:3*want])
		copy(cache.v, lf.attOut[3*want:4*want])
		copy(cache.attOut, lf.attOut[4*want:5*want])
		copy(cache.x2, lf.x2)
		hiddenSpan := lf.hidden * lf.seq
		copy(cache.h1, lf.ffnOut[want:want+hiddenSpan])
		copy(cache.h3, lf.ffnOut[want+hiddenSpan:want+2*hiddenSpan])
		copy(cache.gate, lf.ffnOut[want+2*hiddenSpan:want+3*hiddenSpan])
		copy(cache.x2Norm, lf.ffnOut[want+3*hiddenSpan:want+3*hiddenSpan+want])
	}
	return nil
}

func validateLayerWeights(dim, hidden int, w layerForwardWeights) error {
	check := func(name string, got, want int) error {
		if got != want {
			return fmt.Errorf("compile layer forward: %s len=%d want=%d", name, got, want)
		}
		return nil
	}
	if err := check("rms_att", len(w.RMSAtt), dim); err != nil {
		return err
	}
	if err := check("wq", len(w.Wq), dim*dim); err != nil {
		return err
	}
	if err := check("wk", len(w.Wk), dim*dim); err != nil {
		return err
	}
	if err := check("wv", len(w.Wv), dim*dim); err != nil {
		return err
	}
	if err := check("wo", len(w.Wo), dim*dim); err != nil {
		return err
	}
	if err := check("rms_ffn", len(w.RMSFFN), dim); err != nil {
		return err
	}
	if err := check("w1", len(w.W1), hidden*dim); err != nil {
		return err
	}
	if err := check("w2", len(w.W2), dim*hidden); err != nil {
		return err
	}
	if err := check("w3", len(w.W3), hidden*dim); err != nil {
		return err
	}
	return nil
}

func compileMultiFP16Kernel(milText string, weights []model.WeightFile, inCh, outCh, seq int) (*model.Kernel, error) {
	return model.Compile(model.CompileOptions{
		MILText:     milText,
		WeightFiles: weights,
	})
}
