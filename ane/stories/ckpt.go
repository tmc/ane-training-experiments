package stories

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const (
	ckptMagic   int32 = 0x424C5A54
	ckptVersion int32 = 2
)

type TrainMeta struct {
	Step       int
	TotalSteps int
	LR         float32
	Loss       float32
	CumCompile float64
	CumTrain   float64
	CumWall    float64
	CumSteps   int
	CumBatches int
	AdamT      int
}

type OptimState struct {
	RMSFinal AdamState
	Embed    AdamState
}

func NewOptimState(vocab int) *OptimState {
	return &OptimState{
		RMSFinal: NewAdamState(Dim),
		Embed:    NewAdamState(vocab * Dim),
	}
}

func SaveCheckpointV2(path string, meta TrainMeta, mw *ModelWeights, opt *OptimState) error {
	if len(mw.Layers) != NLayers {
		return fmt.Errorf("layers=%d want=%d", len(mw.Layers), NLayers)
	}
	vocab := len(mw.Embed) / Dim
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer f.Close()

	writeI32 := func(v int32) error { return binary.Write(f, binary.LittleEndian, v) }
	writeF32 := func(v float32) error { return binary.Write(f, binary.LittleEndian, v) }
	writeF64 := func(v float64) error { return binary.Write(f, binary.LittleEndian, v) }

	if err := writeI32(ckptMagic); err != nil {
		return err
	}
	if err := writeI32(ckptVersion); err != nil {
		return err
	}
	if err := writeI32(int32(meta.Step)); err != nil {
		return err
	}
	if err := writeI32(int32(meta.TotalSteps)); err != nil {
		return err
	}
	if err := writeI32(NLayers); err != nil {
		return err
	}
	if err := writeI32(Vocab); err != nil {
		return err
	}
	if err := writeI32(Dim); err != nil {
		return err
	}
	if err := writeI32(Hidden); err != nil {
		return err
	}
	if err := writeI32(Heads); err != nil {
		return err
	}
	if err := writeI32(SeqDefault); err != nil {
		return err
	}
	if err := writeF32(meta.LR); err != nil {
		return err
	}
	if err := writeF32(meta.Loss); err != nil {
		return err
	}
	if err := writeF64(meta.CumCompile); err != nil {
		return err
	}
	if err := writeF64(meta.CumTrain); err != nil {
		return err
	}
	if err := writeF64(meta.CumWall); err != nil {
		return err
	}
	if err := writeI32(int32(meta.CumSteps)); err != nil {
		return err
	}
	if err := writeI32(int32(meta.CumBatches)); err != nil {
		return err
	}
	if err := writeI32(int32(meta.AdamT)); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if err := writeI32(0); err != nil {
			return err
		}
	}

	for i := range mw.Layers {
		l := &mw.Layers[i]
		if err := writeF32s(f, l.Wq); err != nil {
			return err
		}
		if err := writeF32s(f, l.Wk); err != nil {
			return err
		}
		if err := writeF32s(f, l.Wv); err != nil {
			return err
		}
		if err := writeF32s(f, l.Wo); err != nil {
			return err
		}
		if err := writeF32s(f, l.W1); err != nil {
			return err
		}
		if err := writeF32s(f, l.W2); err != nil {
			return err
		}
		if err := writeF32s(f, l.W3); err != nil {
			return err
		}
		if err := writeF32s(f, l.RMSAtt); err != nil {
			return err
		}
		if err := writeF32s(f, l.RMSFFN); err != nil {
			return err
		}
	}
	for range mw.Layers {
		if err := writeZerosF32(f, WQSize); err != nil {
			return err
		}
		if err := writeZerosF32(f, WQSize); err != nil {
			return err
		}
		if err := writeZerosF32(f, WQSize); err != nil {
			return err
		}
		if err := writeZerosF32(f, WQSize); err != nil {
			return err
		}
		if err := writeZerosF32(f, WQSize); err != nil {
			return err
		}
		if err := writeZerosF32(f, WQSize); err != nil {
			return err
		}
		if err := writeZerosF32(f, WOSize); err != nil {
			return err
		}
		if err := writeZerosF32(f, WOSize); err != nil {
			return err
		}
		if err := writeZerosF32(f, W1Size); err != nil {
			return err
		}
		if err := writeZerosF32(f, W1Size); err != nil {
			return err
		}
		if err := writeZerosF32(f, W2Size); err != nil {
			return err
		}
		if err := writeZerosF32(f, W2Size); err != nil {
			return err
		}
		if err := writeZerosF32(f, W3Size); err != nil {
			return err
		}
		if err := writeZerosF32(f, W3Size); err != nil {
			return err
		}
		if err := writeZerosF32(f, Dim); err != nil {
			return err
		}
		if err := writeZerosF32(f, Dim); err != nil {
			return err
		}
		if err := writeZerosF32(f, Dim); err != nil {
			return err
		}
		if err := writeZerosF32(f, Dim); err != nil {
			return err
		}
	}

	if err := writeF32s(f, mw.RMSFinal); err != nil {
		return err
	}
	if err := writeF32s(f, opt.RMSFinal.M); err != nil {
		return err
	}
	if err := writeF32s(f, opt.RMSFinal.V); err != nil {
		return err
	}
	if err := writeF32s(f, mw.Embed); err != nil {
		return err
	}
	if err := writeF32s(f, opt.Embed.M); err != nil {
		return err
	}
	if err := writeF32s(f, opt.Embed.V); err != nil {
		return err
	}

	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	_ = vocab
	return nil
}

func LoadCheckpointV2(path string, mw *ModelWeights, opt *OptimState) (TrainMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return TrainMeta{}, err
	}
	defer f.Close()

	readI32 := func() (int32, error) {
		var v int32
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	}
	readF32 := func() (float32, error) {
		var v float32
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	}
	readF64 := func() (float64, error) {
		var v float64
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	}

	magic, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	ver, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	if magic != ckptMagic || ver != ckptVersion {
		return TrainMeta{}, fmt.Errorf("bad checkpoint header magic=%x version=%d", magic, ver)
	}
	step, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	totalSteps, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	nLayers, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	vocabSize, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	dim, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	hidden, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	heads, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	_, err = readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	if nLayers != NLayers || vocabSize != Vocab || dim != Dim || hidden != Hidden || heads != Heads {
		return TrainMeta{}, fmt.Errorf("checkpoint config mismatch")
	}
	lr, err := readF32()
	if err != nil {
		return TrainMeta{}, err
	}
	loss, err := readF32()
	if err != nil {
		return TrainMeta{}, err
	}
	cumCompile, err := readF64()
	if err != nil {
		return TrainMeta{}, err
	}
	cumTrain, err := readF64()
	if err != nil {
		return TrainMeta{}, err
	}
	cumWall, err := readF64()
	if err != nil {
		return TrainMeta{}, err
	}
	cumSteps, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	cumBatches, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	adamT, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	for i := 0; i < 3; i++ {
		if _, err := readI32(); err != nil {
			return TrainMeta{}, err
		}
	}

	for i := range mw.Layers {
		l := &mw.Layers[i]
		if err := readF32s(f, l.Wq); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.Wk); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.Wv); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.Wo); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.W1); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.W2); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.W3); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.RMSAtt); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.RMSFFN); err != nil {
			return TrainMeta{}, err
		}
	}
	for range mw.Layers {
		if err := skipF32(f, WQSize*2+WQSize*2+WQSize*2+WOSize*2+W1Size*2+W2Size*2+W3Size*2+Dim*2+Dim*2); err != nil {
			return TrainMeta{}, err
		}
	}
	if err := readF32s(f, mw.RMSFinal); err != nil {
		return TrainMeta{}, err
	}
	if err := readF32s(f, opt.RMSFinal.M); err != nil {
		return TrainMeta{}, err
	}
	if err := readF32s(f, opt.RMSFinal.V); err != nil {
		return TrainMeta{}, err
	}
	if err := readF32s(f, mw.Embed); err != nil {
		return TrainMeta{}, err
	}
	if err := readF32s(f, opt.Embed.M); err != nil {
		return TrainMeta{}, err
	}
	if err := readF32s(f, opt.Embed.V); err != nil {
		return TrainMeta{}, err
	}

	return TrainMeta{
		Step:       int(step),
		TotalSteps: int(totalSteps),
		LR:         lr,
		Loss:       loss,
		CumCompile: cumCompile,
		CumTrain:   cumTrain,
		CumWall:    cumWall,
		CumSteps:   int(cumSteps),
		CumBatches: int(cumBatches),
		AdamT:      int(adamT),
	}, nil
}

func writeF32s(w io.Writer, vals []float32) error {
	for i := range vals {
		if err := binary.Write(w, binary.LittleEndian, vals[i]); err != nil {
			return err
		}
	}
	return nil
}

func writeZerosF32(w io.Writer, n int) error {
	zeros := make([]byte, 4096)
	total := int64(n * 4)
	for total > 0 {
		chunk := int64(len(zeros))
		if total < chunk {
			chunk = total
		}
		if _, err := w.Write(zeros[:chunk]); err != nil {
			return err
		}
		total -= chunk
	}
	return nil
}

func skipF32(r io.Reader, n int) error {
	_, err := io.CopyN(io.Discard, r, int64(n*4))
	return err
}
