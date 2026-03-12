//go:build darwin

package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/maderix/ANE/ane/storiestrainer"
)

func TestSmokeDirectStoriesTrainerSequences(t *testing.T) {
	if os.Getenv("ANE_SMOKE") != "1" {
		t.Skip("set ANE_SMOKE=1 to run direct stories trainer smoke test")
	}

	root := repoRootSmoke(t)
	modelPath := firstExistingSmoke(
		filepath.Join(root, "..", "..", "assets", "models", "stories110M.bin"),
		"/Volumes/tmc/go/src/github.com/assets/models/stories110M.bin",
	)
	if modelPath == "" {
		t.Skip("stories110M.bin not found")
	}
	dataPath := firstExistingSmoke(
		filepath.Join(root, "training", "tinystories_data00.bin"),
		filepath.Join(root, "tinystories_data00.bin"),
	)
	if dataPath == "" {
		t.Skip("tinystories_data00.bin not found")
	}

	for _, seq := range []uint{256, 384} {
		t.Run("seq_"+itoaSmoke(int(seq)), func(t *testing.T) {
			probeErr := probeDirectStoriesSequence("ane", modelPath, seq)
			if probeErr != nil {
				t.Fatalf("probeDirectStoriesSequence(%d): %v", seq, probeErr)
			}
			if shouldAutoBridgeStoriesBinToFull("ane", storiestrainer.BackendAuto, modelPath, probeErr) {
				t.Fatalf("shouldAutoBridgeStoriesBinToFull(seq=%d)=true want false", seq)
			}

			trainer, err := storiestrainer.Open(storiestrainer.Options{
				ModelPath:      modelPath,
				DataPath:       dataPath,
				SequenceLength: uint32(seq),
				AccumSteps:     1,
				Steps:          1,
				LearningRate:   3e-4,
				HybridBackward: true,
				Backend:        storiestrainer.BackendDirect,
			})
			if err != nil {
				t.Fatalf("storiestrainer.Open(seq=%d): %v", seq, err)
			}
			defer trainer.Close()

			d := trainer.Diagnostics()
			if !d.LayerForwardEnabled {
				t.Fatalf("LayerForwardEnabled=false seq=%d diagnostics=%+v", seq, d)
			}
			if !d.FinalHeadOffloadEnabled {
				t.Fatalf("FinalHeadOffloadEnabled=false seq=%d diagnostics=%+v", seq, d)
			}
			if !d.HasRMSBackward {
				t.Fatalf("HasRMSBackward=false seq=%d diagnostics=%+v", seq, d)
			}
		})
	}
}

func repoRootSmoke(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

func firstExistingSmoke(paths ...string) string {
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

func itoaSmoke(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
