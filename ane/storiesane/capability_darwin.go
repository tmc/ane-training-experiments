//go:build darwin

package storiesane

import (
	"fmt"
	"sync"

	"github.com/maderix/ANE/ane/mil"
	"github.com/maderix/ANE/ane/model"
	"github.com/maderix/ANE/ane/stories"
)

var (
	directSequenceProbeMu    sync.Mutex
	directSequenceProbeCache = make(map[int]string)

	probeDirectSequenceCompileFunc = compileDirectSequenceProbe
)

// ProbeDirectSequence reports whether the direct-Go in-memory ANE compile path
// can compile a minimal stories-shaped kernel for seq on this host.
func ProbeDirectSequence(seq int) error {
	if seq <= 0 {
		seq = stories.SeqDefault
	}

	directSequenceProbeMu.Lock()
	if msg, ok := directSequenceProbeCache[seq]; ok {
		directSequenceProbeMu.Unlock()
		if msg == "" {
			return nil
		}
		return fmt.Errorf("%s", msg)
	}
	directSequenceProbeMu.Unlock()

	err := probeDirectSequenceCompileFunc(seq)
	msg := ""
	if err != nil {
		msg = err.Error()
	}

	directSequenceProbeMu.Lock()
	directSequenceProbeCache[seq] = msg
	directSequenceProbeMu.Unlock()

	if msg == "" {
		return nil
	}
	return fmt.Errorf("%s", msg)
}

func compileDirectSequenceProbe(seq int) error {
	blob, err := mil.BuildWeightBlob(make([]float32, stories.Dim*stories.Dim), stories.Dim, stories.Dim)
	if err != nil {
		return fmt.Errorf("build probe weights: %w", err)
	}
	k, err := model.Compile(model.CompileOptions{
		MILText: mil.GenConvFP16(stories.Dim, stories.Dim, seq),
		WeightFiles: []model.WeightFile{{
			Path: "@model_path/weights/weight.bin",
			Blob: blob,
		}},
	})
	if err != nil {
		return fmt.Errorf("compile probe kernel: %w", err)
	}
	defer k.Close()
	return nil
}
