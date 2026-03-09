//go:build darwin

package storiesane

import (
	"fmt"
	"strings"

	"github.com/maderix/ANE/ane/dynamicmatmul"
	"github.com/maderix/ANE/ane/mil"
	"github.com/maderix/ANE/ane/model"
	"github.com/maderix/ANE/ane/stories"
)

var (
	classifierForwardTileCandidates  = []int{2048, 1024, 512}
	classifierBackwardTileCandidates = []int{2048, 1024}
	classifierDynamicTileCandidates  = []int{4096, 2048, 1024, 512}
)

type classifierTile struct {
	start  int
	size   int
	kernel *model.Kernel
}

type classifierDynamicTile struct {
	start int
	size  int
	exec  *dynamicmatmul.Executor
}

type offload struct {
	rmsFwd    *model.Kernel
	clsFwdDyn []classifierDynamicTile
	clsFwd    *model.Kernel
	clsFwdTil []classifierTile
	softmax   *model.Kernel
	clsBwdDyn []classifierDynamicTile
	clsBwd    *model.Kernel
	clsBwdTil []classifierTile
	rmsBwd    *model.Kernel
	rmsBwdIn  []float32
	clsBwdTmp []float32

	clsFwdTile int
	clsBwdTile int
	diag       []string
}

func newOffload(mw *stories.ModelWeights, seq int, useANE bool) *offload {
	return newOffloadWithState(mw, seq, useANE, nil, nil, nil)
}

func refreshOffload(prev *offload, mw *stories.ModelWeights, seq int, useANE bool) *offload {
	var softmax *model.Kernel
	var clsFwdDyn []classifierDynamicTile
	var clsBwdDyn []classifierDynamicTile
	if prev != nil {
		softmax = prev.softmax
		clsFwdDyn = prev.clsFwdDyn
		clsBwdDyn = prev.clsBwdDyn
		prev.softmax = nil
		prev.clsFwdDyn = nil
		prev.clsBwdDyn = nil
		prev.close()
	}
	return newOffloadWithState(mw, seq, useANE, softmax, clsFwdDyn, clsBwdDyn)
}

func newOffloadWithState(mw *stories.ModelWeights, seq int, useANE bool, softmax *model.Kernel, clsFwdDyn, clsBwdDyn []classifierDynamicTile) *offload {
	if !useANE || mw == nil || seq <= 0 {
		closeKernel(softmax)
		closeDynamicClassifierTiles(clsFwdDyn)
		closeDynamicClassifierTiles(clsBwdDyn)
		return nil
	}
	o := &offload{
		rmsBwdIn: make([]float32, 2*stories.Dim*seq),
		softmax:  softmax,
	}
	var err error

	if blob, err := mil.BuildVectorWeightBlob(mw.RMSFinal); err == nil {
		o.rmsFwd, err = compileFP16Kernel(
			mil.GenFinalRMSNorm(stories.Dim, seq),
			"@model_path/weights/rms_w.bin",
			blob,
			stories.Dim,
			stories.Dim,
			seq,
		)
		if err != nil {
			o.notef("rms forward compile failed: %v", err)
		}
		o.rmsBwd, err = compileFP16Kernel(
			mil.GenRMSNormBackward(stories.Dim, seq),
			"@model_path/weights/rms_w.bin",
			blob,
			2*stories.Dim,
			stories.Dim,
			seq,
		)
		if err != nil {
			o.notef("rms backward compile failed: %v", err)
		}
	} else {
		o.notef("rms weight blob failed: %v", err)
	}
	o.clsFwdDyn, o.clsFwdTile, err = prepareDynamicClassifierForward(clsFwdDyn, mw.Embed, seq)
	if err != nil {
		o.notef("classifier forward dynamic compile failed: %v", err)
	}
	if len(o.clsFwdDyn) > 0 && o.clsFwdTile > 0 {
		o.notef("classifier forward using dynamic tile size %d", o.clsFwdTile)
	}
	if len(o.clsFwdDyn) == 0 {
		if blob, err := mil.BuildWeightBlob(mw.Embed, stories.Vocab, stories.Dim); err == nil {
			o.clsFwd, err = compileFP16Kernel(
				mil.GenClassifierForward(stories.Dim, stories.Vocab, seq),
				"@model_path/weights/embed.bin",
				blob,
				stories.Dim,
				stories.Vocab,
				seq,
			)
			if err != nil {
				o.notef("classifier forward full compile failed: %v", err)
			}
		} else {
			o.notef("classifier forward weight blob failed: %v", err)
		}
		if o.clsFwd == nil {
			o.clsFwdTil, o.clsFwdTile, err = compileClassifierForwardTiles(mw.Embed, seq)
			if err != nil {
				o.notef("classifier forward tiled compile failed: %v", err)
			} else if o.clsFwdTile > 0 {
				o.notef("classifier forward using tile size %d", o.clsFwdTile)
			}
		}
	}
	if o.softmax == nil {
		o.softmax, err = compileSoftmaxKernel(stories.Vocab, seq)
		if err != nil {
			o.notef("softmax compile failed: %v", err)
		}
	}
	o.clsBwdDyn, o.clsBwdTile, err = prepareDynamicClassifierBackward(clsBwdDyn, mw.Embed, seq)
	if err != nil {
		o.notef("classifier backward dynamic compile failed: %v", err)
	}
	if len(o.clsBwdDyn) > 0 && o.clsBwdTile > 0 {
		o.notef("classifier backward using dynamic vocab tile size %d", o.clsBwdTile)
	}
	if len(o.clsBwdDyn) == 0 {
		if blob, err := mil.BuildTransposedWeightBlob(mw.Embed, stories.Vocab, stories.Dim); err == nil {
			o.clsBwd, err = compileFP16Kernel(
				mil.GenClassifierBackward(stories.Dim, stories.Vocab, seq),
				"@model_path/weights/embed_t.bin",
				blob,
				stories.Vocab,
				stories.Dim,
				seq,
			)
			if err != nil {
				o.notef("classifier backward full compile failed: %v", err)
			}
		} else {
			o.notef("classifier backward weight blob failed: %v", err)
		}
		if o.clsBwd == nil {
			o.clsBwdTil, o.clsBwdTile, err = compileClassifierBackwardTiles(mw.Embed, seq)
			if err != nil {
				o.notef("classifier backward tiled compile failed: %v", err)
			} else if o.clsBwdTile > 0 {
				o.notef("classifier backward using tile size %d", o.clsBwdTile)
			}
		}
	}
	if len(o.clsBwdDyn) > 0 || len(o.clsBwdTil) > 0 {
		o.clsBwdTmp = make([]float32, stories.Dim*seq)
	}

	if !o.hasRMSForward() && !o.hasClassifierForward() && !o.hasSoftmax() && !o.hasClassifierBackward() && !o.hasRMSBackward() {
		o.close()
		return nil
	}
	return o
}

func (o *offload) notef(format string, args ...any) {
	if o == nil {
		return
	}
	o.diag = append(o.diag, fmt.Sprintf(format, args...))
}

func (o *offload) diagnosticSummary() string {
	if o == nil || len(o.diag) == 0 {
		return ""
	}
	return strings.Join(o.diag, "; ")
}

func compileFP16Kernel(milText, weightPath string, weightBlob []byte, inCh, outCh, seq int) (*model.Kernel, error) {
	k, err := model.Compile(model.CompileOptions{
		MILText:    milText,
		WeightBlob: weightBlob,
		WeightPath: weightPath,
	})
	if err != nil {
		return nil, err
	}
	return k, nil
}

func compileSoftmaxKernel(vocab, seq int) (*model.Kernel, error) {
	return compileFP16Kernel(
		mil.GenSoftmaxVocab(vocab, seq),
		"",
		nil,
		vocab,
		vocab,
		seq,
	)
}

func (o *offload) close() {
	if o == nil {
		return
	}
	closeKernel(o.rmsFwd)
	closeDynamicClassifierTiles(o.clsFwdDyn)
	closeKernel(o.clsFwd)
	closeClassifierTiles(o.clsFwdTil)
	closeKernel(o.softmax)
	closeDynamicClassifierTiles(o.clsBwdDyn)
	closeKernel(o.clsBwd)
	closeClassifierTiles(o.clsBwdTil)
	closeKernel(o.rmsBwd)
	o.rmsFwd = nil
	o.clsFwdDyn = nil
	o.clsFwd = nil
	o.clsFwdTil = nil
	o.softmax = nil
	o.clsBwdDyn = nil
	o.clsBwd = nil
	o.clsBwdTil = nil
	o.rmsBwd = nil
	o.rmsBwdIn = nil
	o.clsBwdTmp = nil
}

func closeKernel(k *model.Kernel) {
	if k != nil {
		k.Close()
	}
}

func closeClassifierTiles(tiles []classifierTile) {
	for i := range tiles {
		closeKernel(tiles[i].kernel)
	}
}

func closeDynamicClassifierTiles(tiles []classifierDynamicTile) {
	for i := range tiles {
		if tiles[i].exec != nil {
			tiles[i].exec.Close()
		}
	}
}

func (o *offload) hasRMSForward() bool {
	return o != nil && o.rmsFwd != nil
}

func (o *offload) hasClassifierForward() bool {
	return o != nil && (len(o.clsFwdDyn) > 0 || o.clsFwd != nil || len(o.clsFwdTil) > 0)
}

func (o *offload) hasSoftmax() bool {
	return o != nil && o.softmax != nil
}

func (o *offload) hasClassifierBackward() bool {
	return o != nil && (len(o.clsBwdDyn) > 0 || o.clsBwd != nil || len(o.clsBwdTil) > 0)
}

func (o *offload) hasRMSBackward() bool {
	return o != nil && o.rmsBwd != nil
}

func (o *offload) disableRMSForward() {
	disableKernel(&o.rmsFwd)
}

func (o *offload) disableClassifierForward() {
	closeDynamicClassifierTiles(o.clsFwdDyn)
	o.clsFwdDyn = nil
	disableKernel(&o.clsFwd)
	closeClassifierTiles(o.clsFwdTil)
	o.clsFwdTil = nil
}

func (o *offload) disableSoftmax() {
	disableKernel(&o.softmax)
}

func (o *offload) disableClassifierBackward() {
	closeDynamicClassifierTiles(o.clsBwdDyn)
	o.clsBwdDyn = nil
	disableKernel(&o.clsBwd)
	closeClassifierTiles(o.clsBwdTil)
	o.clsBwdTil = nil
	o.clsBwdTmp = nil
}

func (o *offload) disableRMSBackward() {
	disableKernel(&o.rmsBwd)
}

func disableKernel(k **model.Kernel) {
	if *k != nil {
		(*k).Close()
		*k = nil
	}
}

func (o *offload) runRMSForward(out, x []float32) error {
	if err := o.rmsFwd.WriteInputFP16(0, x); err != nil {
		return err
	}
	if err := o.rmsFwd.Eval(); err != nil {
		return err
	}
	return o.rmsFwd.ReadOutputFP16(0, out)
}

func (o *offload) runClassifierForward(logits, xNorm []float32) error {
	if len(o.clsFwdDyn) > 0 {
		seq := len(xNorm) / stories.Dim
		for _, tile := range o.clsFwdDyn {
			dst := logits[tile.start*seq : (tile.start+tile.size)*seq]
			if _, err := tile.exec.EvalCFIOInto(dst, xNorm); err != nil {
				return err
			}
		}
		return nil
	}
	if o.clsFwd != nil {
		if err := o.clsFwd.WriteInputFP16(0, xNorm); err != nil {
			return err
		}
		if err := o.clsFwd.Eval(); err != nil {
			return err
		}
		return o.clsFwd.ReadOutputFP16(0, logits)
	}
	for _, tile := range o.clsFwdTil {
		if err := tile.kernel.WriteInputFP16(0, xNorm); err != nil {
			return err
		}
		if err := tile.kernel.Eval(); err != nil {
			return err
		}
		seq := len(xNorm) / stories.Dim
		dst := logits[tile.start*seq : (tile.start+tile.size)*seq]
		if err := tile.kernel.ReadOutputFP16(0, dst); err != nil {
			return err
		}
	}
	return nil
}

func (o *offload) runClassifierSoftmax(probs, xNorm []float32) error {
	if o.softmax == nil {
		return fmt.Errorf("softmax kernel is unavailable")
	}
	if len(o.clsFwdDyn) > 0 {
		for _, tile := range o.clsFwdDyn {
			if _, err := tile.exec.EvalCF(xNorm); err != nil {
				return err
			}
			if err := tile.exec.CopyOutputToInput(o.softmax, 0, tile.start); err != nil {
				return err
			}
		}
	} else if o.clsFwd != nil {
		if err := o.clsFwd.WriteInputFP16(0, xNorm); err != nil {
			return err
		}
		if err := o.clsFwd.Eval(); err != nil {
			return err
		}
		if err := model.CopyOutputChannelsToInput(o.softmax, 0, 0, o.clsFwd, 0, 0, stories.Vocab); err != nil {
			return err
		}
	} else {
		for _, tile := range o.clsFwdTil {
			if err := tile.kernel.WriteInputFP16(0, xNorm); err != nil {
				return err
			}
			if err := tile.kernel.Eval(); err != nil {
				return err
			}
			if err := model.CopyOutputChannelsToInput(o.softmax, 0, tile.start, tile.kernel, 0, 0, tile.size); err != nil {
				return err
			}
		}
	}
	if err := o.softmax.Eval(); err != nil {
		return err
	}
	return o.softmax.ReadOutputFP16(0, probs)
}

func (o *offload) runSoftmax(probs []float32) error {
	if err := o.softmax.WriteInputFP16(0, probs); err != nil {
		return err
	}
	if err := o.softmax.Eval(); err != nil {
		return err
	}
	return o.softmax.ReadOutputFP16(0, probs)
}

func (o *offload) runClassifierBackward(dy, dLogits []float32) error {
	if len(o.clsBwdDyn) > 0 {
		for i := range dy {
			dy[i] = 0
		}
		seq := len(dy) / stories.Dim
		for _, tile := range o.clsBwdDyn {
			src := dLogits[tile.start*seq : (tile.start+tile.size)*seq]
			if _, err := tile.exec.EvalCFIOInto(o.clsBwdTmp, src); err != nil {
				return err
			}
			for i, v := range o.clsBwdTmp {
				dy[i] += v
			}
		}
		return nil
	}
	if o.clsBwd != nil {
		if err := o.clsBwd.WriteInputFP16(0, dLogits); err != nil {
			return err
		}
		if err := o.clsBwd.Eval(); err != nil {
			return err
		}
		return o.clsBwd.ReadOutputFP16(0, dy)
	}
	for i := range dy {
		dy[i] = 0
	}
	seq := len(dy) / stories.Dim
	for _, tile := range o.clsBwdTil {
		src := dLogits[tile.start*seq : (tile.start+tile.size)*seq]
		if err := tile.kernel.WriteInputFP16(0, src); err != nil {
			return err
		}
		if err := tile.kernel.Eval(); err != nil {
			return err
		}
		if err := tile.kernel.ReadOutputFP16(0, o.clsBwdTmp); err != nil {
			return err
		}
		for i, v := range o.clsBwdTmp {
			dy[i] += v
		}
	}
	return nil
}

func (o *offload) runRMSBackward(dx, dy, x []float32) error {
	copy(o.rmsBwdIn, dy)
	copy(o.rmsBwdIn[len(dy):], x)
	if err := o.rmsBwd.WriteInputFP16(0, o.rmsBwdIn); err != nil {
		return err
	}
	if err := o.rmsBwd.Eval(); err != nil {
		return err
	}
	return o.rmsBwd.ReadOutputFP16(0, dx)
}

func prepareDynamicClassifierForward(prev []classifierDynamicTile, embed []float32, seq int) ([]classifierDynamicTile, int, error) {
	vocab := len(embed) / stories.Dim
	if vocab <= 0 || vocab*stories.Dim != len(embed) {
		closeDynamicClassifierTiles(prev)
		return nil, 0, fmt.Errorf("embed size=%d is not divisible by dim=%d", len(embed), stories.Dim)
	}
	if len(prev) > 0 {
		if err := primeDynamicClassifierForward(prev, embed); err == nil {
			return prev, prev[0].size, nil
		}
		closeDynamicClassifierTiles(prev)
	}

	var errs []string
	for _, tile := range classifierDynamicTileCandidates {
		if tile <= 0 {
			continue
		}
		if tile > vocab {
			tile = vocab
		}
		tiles, err := compileDynamicClassifierForwardAtSize(embed, vocab, seq, tile)
		if err == nil {
			return tiles, tile, nil
		}
		errs = append(errs, fmt.Sprintf("tile=%d: %v", tile, err))
	}
	if len(errs) == 0 {
		return nil, 0, fmt.Errorf("no dynamic classifier forward tile candidates")
	}
	return nil, 0, fmt.Errorf("%s", strings.Join(errs, "; "))
}

func prepareDynamicClassifierBackward(prev []classifierDynamicTile, embed []float32, seq int) ([]classifierDynamicTile, int, error) {
	vocab := len(embed) / stories.Dim
	if vocab <= 0 || vocab*stories.Dim != len(embed) {
		closeDynamicClassifierTiles(prev)
		return nil, 0, fmt.Errorf("embed size=%d is not divisible by dim=%d", len(embed), stories.Dim)
	}
	if len(prev) > 0 {
		if err := primeDynamicClassifierBackward(prev, embed); err == nil {
			return prev, prev[0].size, nil
		}
		closeDynamicClassifierTiles(prev)
	}

	var errs []string
	for _, tile := range classifierDynamicTileCandidates {
		if tile <= 0 {
			continue
		}
		if tile > vocab {
			tile = vocab
		}
		tiles, err := compileDynamicClassifierBackwardAtSize(embed, vocab, seq, tile)
		if err == nil {
			return tiles, tile, nil
		}
		errs = append(errs, fmt.Sprintf("tile=%d: %v", tile, err))
	}
	if len(errs) == 0 {
		return nil, 0, fmt.Errorf("no dynamic classifier backward tile candidates")
	}
	return nil, 0, fmt.Errorf("%s", strings.Join(errs, "; "))
}

func compileDynamicClassifierForwardAtSize(embed []float32, vocab, seq, tile int) ([]classifierDynamicTile, error) {
	ranges := classifierTileRanges(vocab, tile)
	tiles := make([]classifierDynamicTile, 0, len(ranges))
	scratch := make([]float32, stories.Dim*tile)
	for _, r := range ranges {
		ex, err := dynamicmatmul.New(seq, stories.Dim, r.size, dynamicmatmul.Options{TileOut: r.size})
		if err != nil {
			closeDynamicClassifierTiles(tiles)
			return nil, err
		}
		wIO := scratch[:stories.Dim*r.size]
		transposeClassifierForwardTile(wIO, embed, r.start, r.size)
		if err := ex.PrimeWeightsIO(wIO); err != nil {
			ex.Close()
			closeDynamicClassifierTiles(tiles)
			return nil, err
		}
		tiles = append(tiles, classifierDynamicTile{start: r.start, size: r.size, exec: ex})
	}
	return tiles, nil
}

func compileDynamicClassifierBackwardAtSize(embed []float32, vocab, seq, tile int) ([]classifierDynamicTile, error) {
	ranges := classifierTileRanges(vocab, tile)
	tiles := make([]classifierDynamicTile, 0, len(ranges))
	for _, r := range ranges {
		ex, err := dynamicmatmul.New(seq, r.size, stories.Dim, dynamicmatmul.Options{TileOut: stories.Dim})
		if err != nil {
			closeDynamicClassifierTiles(tiles)
			return nil, err
		}
		weights := embed[r.start*stories.Dim : (r.start+r.size)*stories.Dim]
		if err := ex.PrimeWeightsIO(weights); err != nil {
			ex.Close()
			closeDynamicClassifierTiles(tiles)
			return nil, err
		}
		tiles = append(tiles, classifierDynamicTile{start: r.start, size: r.size, exec: ex})
	}
	return tiles, nil
}

func primeDynamicClassifierForward(tiles []classifierDynamicTile, embed []float32) error {
	if len(tiles) == 0 {
		return nil
	}
	maxTile := 0
	for _, tile := range tiles {
		if tile.size > maxTile {
			maxTile = tile.size
		}
	}
	scratch := make([]float32, stories.Dim*maxTile)
	for _, tile := range tiles {
		wIO := scratch[:stories.Dim*tile.size]
		transposeClassifierForwardTile(wIO, embed, tile.start, tile.size)
		if err := tile.exec.PrimeWeightsIO(wIO); err != nil {
			return err
		}
	}
	return nil
}

func primeDynamicClassifierBackward(tiles []classifierDynamicTile, embed []float32) error {
	for _, tile := range tiles {
		weights := embed[tile.start*stories.Dim : (tile.start+tile.size)*stories.Dim]
		if err := tile.exec.PrimeWeightsIO(weights); err != nil {
			return err
		}
	}
	return nil
}

func transposeClassifierForwardTile(dst, embed []float32, start, size int) {
	for d := 0; d < stories.Dim; d++ {
		row := dst[d*size : (d+1)*size]
		for i := 0; i < size; i++ {
			row[i] = embed[(start+i)*stories.Dim+d]
		}
	}
}

func compileClassifierForwardTiles(embed []float32, seq int) ([]classifierTile, int, error) {
	return compileClassifierTiles(
		embed,
		seq,
		classifierForwardTileCandidates,
		"@model_path/weights/embed.bin",
		func(weights []float32, outCh int) ([]byte, error) {
			return mil.BuildWeightBlob(weights, outCh, stories.Dim)
		},
		func(outCh int) string {
			return mil.GenClassifierForward(stories.Dim, outCh, seq)
		},
		func(outCh int) (int, int) {
			return stories.Dim, outCh
		},
	)
}

func compileClassifierBackwardTiles(embed []float32, seq int) ([]classifierTile, int, error) {
	return compileClassifierTiles(
		embed,
		seq,
		classifierBackwardTileCandidates,
		"@model_path/weights/embed_t.bin",
		func(weights []float32, outCh int) ([]byte, error) {
			return mil.BuildTransposedWeightBlob(weights, outCh, stories.Dim)
		},
		func(outCh int) string {
			return mil.GenClassifierBackward(stories.Dim, outCh, seq)
		},
		func(outCh int) (int, int) {
			return outCh, stories.Dim
		},
	)
}

func compileClassifierTiles(
	embed []float32,
	seq int,
	tileCandidates []int,
	weightPath string,
	buildBlob func([]float32, int) ([]byte, error),
	genMIL func(int) string,
	ioShape func(int) (int, int),
) ([]classifierTile, int, error) {
	vocab := len(embed) / stories.Dim
	if vocab <= 0 || vocab*stories.Dim != len(embed) {
		return nil, 0, fmt.Errorf("embed size=%d is not divisible by dim=%d", len(embed), stories.Dim)
	}
	var errs []string
	for _, tile := range tileCandidates {
		if tile <= 0 {
			continue
		}
		if tile > vocab {
			tile = vocab
		}
		tiles, err := compileClassifierTilesAtSize(embed, vocab, seq, tile, weightPath, buildBlob, genMIL, ioShape)
		if err == nil {
			return tiles, tile, nil
		}
		errs = append(errs, fmt.Sprintf("tile=%d: %v", tile, err))
	}
	if len(errs) == 0 {
		return nil, 0, fmt.Errorf("no classifier tile candidates")
	}
	return nil, 0, fmt.Errorf("%s", strings.Join(errs, "; "))
}

func compileClassifierTilesAtSize(
	embed []float32,
	vocab int,
	seq int,
	tile int,
	weightPath string,
	buildBlob func([]float32, int) ([]byte, error),
	genMIL func(int) string,
	ioShape func(int) (int, int),
) ([]classifierTile, error) {
	ranges := classifierTileRanges(vocab, tile)
	tiles := make([]classifierTile, 0, len(ranges))
	for _, r := range ranges {
		weights := embed[r.start*stories.Dim : (r.start+r.size)*stories.Dim]
		blob, err := buildBlob(weights, r.size)
		if err != nil {
			closeClassifierTiles(tiles)
			return nil, err
		}
		inCh, outCh := ioShape(r.size)
		k, err := compileFP16Kernel(
			genMIL(r.size),
			weightPath,
			blob,
			inCh,
			outCh,
			seq,
		)
		if err != nil {
			closeClassifierTiles(tiles)
			return nil, err
		}
		tiles = append(tiles, classifierTile{start: r.start, size: r.size, kernel: k})
	}
	return tiles, nil
}
