//go:build darwin

package storiestrainer

import (
	"math"
	"os"
	"testing"

	"github.com/maderix/ANE/ane/clientmodel"
)

func TestNormalizeOptions(t *testing.T) {
	_, err := normalizeOptions(Options{})
	if err == nil {
		t.Fatalf("normalizeOptions empty opts succeeded; want error")
	}

	tmp := t.TempDir()
	modelPath := tmp + "/model.mlmodelc"
	dataPath := tmp + "/data.bin"
	if err := os.WriteFile(modelPath, []byte("m"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := os.WriteFile(dataPath, []byte("d"), 0o644); err != nil {
		t.Fatalf("write data: %v", err)
	}

	opts, err := normalizeOptions(Options{
		ModelPath:   modelPath,
		DataPath:    dataPath,
		InputBytes:  4096,
		OutputBytes: 4096,
	})
	if err != nil {
		t.Fatalf("normalizeOptions: %v", err)
	}
	if opts.ModelKey != "s" {
		t.Fatalf("ModelKey=%q want s", opts.ModelKey)
	}
	if opts.LearningRate <= 0 {
		t.Fatalf("LearningRate=%v want >0", opts.LearningRate)
	}
	if opts.CompileBudget != DefaultCompileBudget {
		t.Fatalf("CompileBudget=%d want %d", opts.CompileBudget, DefaultCompileBudget)
	}
	if opts.QoS != DefaultQoS {
		t.Fatalf("QoS=%d want %d", opts.QoS, DefaultQoS)
	}

	opts, err = normalizeOptions(Options{
		ModelPath:            modelPath,
		DataPath:             dataPath,
		InputBytes:           4096,
		OutputBytes:          4096,
		DisableCompileBudget: true,
	})
	if err != nil {
		t.Fatalf("normalizeOptions(disable budget): %v", err)
	}
	if opts.CompileBudget != 0 {
		t.Fatalf("CompileBudget=%d want 0 when disabled", opts.CompileBudget)
	}

	opts, err = normalizeOptions(Options{
		ModelPath:   modelPath,
		DataPath:    dataPath,
		InputBytes:  4096,
		OutputBytes: 4096,
	})
	if err != nil {
		t.Fatalf("normalizeOptions(default backend): %v", err)
	}
	if opts.Backend != BackendAuto {
		t.Fatalf("Backend=%q want %q", opts.Backend, BackendAuto)
	}

	if _, err := normalizeOptions(Options{
		ModelPath:   modelPath,
		DataPath:    dataPath,
		InputBytes:  4096,
		OutputBytes: 4096,
		Backend:     "bad",
	}); err == nil {
		t.Fatalf("normalizeOptions invalid backend succeeded; want error")
	}
}

func TestClosedTrainerErrors(t *testing.T) {
	var tr Trainer
	if got := tr.Backend(); got != "" {
		t.Fatalf("Backend()=%q want empty for zero-value trainer", got)
	}
	if _, err := tr.Step(); err == nil {
		t.Fatalf("Step on closed trainer succeeded; want error")
	}
	if err := tr.SaveCheckpoint(""); err == nil {
		t.Fatalf("SaveCheckpoint empty path succeeded; want error")
	}
	if err := tr.LoadCheckpoint(""); err == nil {
		t.Fatalf("LoadCheckpoint empty path succeeded; want error")
	}
	if err := tr.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestBackendExplicit(t *testing.T) {
	tr := &Trainer{backend: BackendDirect}
	if got := tr.Backend(); got != BackendDirect {
		t.Fatalf("Backend()=%q want %q", got, BackendDirect)
	}
}

func TestOpenBridgeBackendUnsupported(t *testing.T) {
	tmp := t.TempDir()
	modelPath := tmp + "/model.mlmodelc"
	dataPath := tmp + "/data.bin"
	if err := os.WriteFile(modelPath, []byte("m"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := os.WriteFile(dataPath, []byte("d"), 0o644); err != nil {
		t.Fatalf("write data: %v", err)
	}
	_, err := Open(Options{
		ModelPath:   modelPath,
		DataPath:    dataPath,
		InputBytes:  4096,
		OutputBytes: 4096,
		Backend:     BackendBridge,
	})
	if err == nil {
		t.Fatalf("Open with backend=%q succeeded; want error", BackendBridge)
	}
}

func TestCheckpointRoundTrip(t *testing.T) {
	tr := &Trainer{
		k:             &clientmodel.Kernel{},
		step:          7,
		totalSteps:    123,
		tokenPos:      42,
		compileBudget: 99,
		aneExtras:     true,
		lr:            1e-3,
		lastLoss:      0.875,
	}
	p := t.TempDir() + "/ckpt.bin"
	if err := tr.SaveCheckpoint(p); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}
	other := &Trainer{k: &clientmodel.Kernel{}}
	if err := other.LoadCheckpoint(p); err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}
	if other.step != tr.step || other.totalSteps != tr.totalSteps || other.tokenPos != tr.tokenPos {
		t.Fatalf("loaded counters mismatch: got step=%d total=%d pos=%d", other.step, other.totalSteps, other.tokenPos)
	}
	if other.compileBudget != tr.compileBudget || other.aneExtras != tr.aneExtras {
		t.Fatalf("loaded flags mismatch: got budget=%d extras=%v", other.compileBudget, other.aneExtras)
	}
	if math.Abs(float64(other.lr-tr.lr)) > 1e-8 {
		t.Fatalf("loaded lr mismatch: got=%v want=%v", other.lr, tr.lr)
	}
	if math.Abs(float64(other.lastLoss-tr.lastLoss)) > 1e-8 {
		t.Fatalf("loaded loss mismatch: got=%v want=%v", other.lastLoss, tr.lastLoss)
	}
}
