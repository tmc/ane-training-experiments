//go:build darwin

package storiesane

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/maderix/ANE/ane/stories"
)

func TestSmokeANEStepMatchesCPU(t *testing.T) {
	if os.Getenv("ANE_SMOKE") != "1" {
		t.Skip("set ANE_SMOKE=1 to run storiesane ANE smoke test")
	}

	root := repoRoot(t)
	ckptPath := firstExisting(
		filepath.Join(root, "ane_stories110M_ckpt.bin"),
		filepath.Join(root, "training", "ane_stories110M_ckpt.bin"),
	)
	if ckptPath == "" {
		t.Skip("stories checkpoint not found")
	}
	dataPath := firstExisting(
		filepath.Join(root, "training", "tinystories_data00.bin"),
		filepath.Join(root, "tinystories_data00.bin"),
	)
	if dataPath == "" {
		t.Skip("stories token data not found")
	}

	const seq = 8
	tokens, err := readTokenPrefix(dataPath, seq+1+32)
	if err != nil {
		t.Fatalf("read tokens: %v", err)
	}
	modelPath := filepath.Join(t.TempDir(), "stories110M.bin")
	if err := writePretrainedFromCheckpoint(modelPath, ckptPath); err != nil {
		t.Fatalf("convert checkpoint: %v", err)
	}

	cpu, err := Open(Options{
		ModelPath:  modelPath,
		Tokens:     tokens,
		Seq:        seq,
		AccumSteps: 1,
		LR:         3e-4,
		Seed:       123,
		UseANE:     false,
	})
	if err != nil {
		t.Fatalf("Open(cpu): %v", err)
	}
	defer cpu.Close()

	ane, err := Open(Options{
		ModelPath:  modelPath,
		Tokens:     tokens,
		Seq:        seq,
		AccumSteps: 1,
		LR:         3e-4,
		Seed:       123,
		UseANE:     true,
	})
	if err != nil {
		t.Fatalf("Open(ane): %v", err)
	}
	defer ane.Close()
	if ane.off == nil {
		t.Skip("ANE offload unavailable")
	}
	if diag := ane.off.diagnosticSummary(); diag != "" {
		t.Logf("offload: %s", diag)
	}
	if !ane.off.hasClassifierForward() || !ane.off.hasSoftmax() {
		t.Skipf("classifier/softmax offload unavailable: %s", ane.off.diagnosticSummary())
	}

	cpuStep, err := cpu.Step()
	if err != nil {
		t.Fatalf("cpu Step: %v", err)
	}
	aneStep, err := ane.Step()
	if err != nil {
		t.Fatalf("ane Step: %v", err)
	}
	if math.IsNaN(float64(aneStep.Loss)) || math.IsInf(float64(aneStep.Loss), 0) {
		t.Fatalf("ane loss is not finite: %v", aneStep.Loss)
	}
	if diff := math.Abs(float64(cpuStep.Loss - aneStep.Loss)); diff > 0.1 {
		t.Fatalf("loss mismatch cpu=%v ane=%v diff=%v", cpuStep.Loss, aneStep.Loss, diff)
	}
	t.Logf("cpu loss=%0.6f ane loss=%0.6f diff=%0.6f", cpuStep.Loss, aneStep.Loss, cpuStep.Loss-aneStep.Loss)
}

func TestSmokeOffloadKernelsAvailable(t *testing.T) {
	if os.Getenv("ANE_SMOKE") != "1" {
		t.Skip("set ANE_SMOKE=1 to run storiesane ANE offload smoke test")
	}
	mw := &stories.ModelWeights{
		RMSFinal: make([]float32, stories.Dim),
		Embed:    make([]float32, stories.Vocab*stories.Dim),
	}
	for i := range mw.RMSFinal {
		mw.RMSFinal[i] = 1
	}
	for i := 0; i < len(mw.Embed); i += stories.Dim + 1 {
		mw.Embed[i] = 1
	}

	off := newOffload(mw, 8, true, false)
	if off == nil {
		t.Skip("ANE offload unavailable")
	}
	defer off.close()
	if diag := off.diagnosticSummary(); diag != "" {
		t.Logf("offload: %s", diag)
	}
	if !off.hasClassifierForward() {
		t.Fatalf("classifier forward unavailable: %s", off.diagnosticSummary())
	}
	if !off.hasSoftmax() {
		t.Fatalf("softmax unavailable: %s", off.diagnosticSummary())
	}
}

func TestSmokeEvalLogitsANEMatchesCPU(t *testing.T) {
	if os.Getenv("ANE_SMOKE") != "1" {
		t.Skip("set ANE_SMOKE=1 to run storiesane ANE logits smoke test")
	}

	root := repoRoot(t)
	ckptPath := firstExisting(
		filepath.Join(root, "ane_stories110M_ckpt.bin"),
		filepath.Join(root, "training", "ane_stories110M_ckpt.bin"),
	)
	if ckptPath == "" {
		t.Skip("stories checkpoint not found")
	}
	dataPath := firstExisting(
		filepath.Join(root, "training", "tinystories_data00.bin"),
		filepath.Join(root, "tinystories_data00.bin"),
	)
	if dataPath == "" {
		t.Skip("stories token data not found")
	}

	const seq = 8
	tokens, err := readTokenPrefix(dataPath, seq+1+32)
	if err != nil {
		t.Fatalf("read tokens: %v", err)
	}
	modelPath := filepath.Join(t.TempDir(), "stories110M.bin")
	if err := writePretrainedFromCheckpoint(modelPath, ckptPath); err != nil {
		t.Fatalf("convert checkpoint: %v", err)
	}

	cpu, err := Open(Options{
		ModelPath:  modelPath,
		Tokens:     tokens,
		Seq:        seq,
		AccumSteps: 1,
		LR:         3e-4,
		Seed:       123,
		UseANE:     false,
	})
	if err != nil {
		t.Fatalf("Open(cpu): %v", err)
	}
	defer cpu.Close()

	ane, err := Open(Options{
		ModelPath:  modelPath,
		Tokens:     tokens,
		Seq:        seq,
		AccumSteps: 1,
		LR:         3e-4,
		Seed:       123,
		UseANE:     true,
	})
	if err != nil {
		t.Fatalf("Open(ane): %v", err)
	}
	defer ane.Close()

	input := tokens[:seq]
	want, err := cpu.EvalLogits(input)
	if err != nil {
		t.Fatalf("cpu EvalLogits: %v", err)
	}
	got, err := ane.EvalLogits(input)
	if err != nil {
		t.Fatalf("ane EvalLogits: %v", err)
	}
	if ane.layerInitErr != nil {
		t.Fatalf("ANE layer init failed: %v", ane.layerInitErr)
	}
	if len(ane.layers) != stories.NLayers {
		t.Fatalf("ANE layers=%d want=%d", len(ane.layers), stories.NLayers)
	}

	maxDiff := 0.0
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 0.25 {
		t.Fatalf("max logits diff=%v want <= 0.25", maxDiff)
	}
	t.Logf("max logits diff=%0.6f", maxDiff)
}

func TestSmokeANEHybridBackwardStepMatchesCPU(t *testing.T) {
	if os.Getenv("ANE_SMOKE") != "1" {
		t.Skip("set ANE_SMOKE=1 to run storiesane hybrid backward smoke test")
	}

	root := repoRoot(t)
	ckptPath := firstExisting(
		filepath.Join(root, "ane_stories110M_ckpt.bin"),
		filepath.Join(root, "training", "ane_stories110M_ckpt.bin"),
	)
	if ckptPath == "" {
		t.Skip("stories checkpoint not found")
	}
	dataPath := firstExisting(
		filepath.Join(root, "training", "tinystories_data00.bin"),
		filepath.Join(root, "tinystories_data00.bin"),
	)
	if dataPath == "" {
		t.Skip("stories token data not found")
	}

	const seq = 8
	tokens, err := readTokenPrefix(dataPath, seq+1+32)
	if err != nil {
		t.Fatalf("read tokens: %v", err)
	}
	modelPath := filepath.Join(t.TempDir(), "stories110M.bin")
	if err := writePretrainedFromCheckpoint(modelPath, ckptPath); err != nil {
		t.Fatalf("convert checkpoint: %v", err)
	}

	cpu, err := Open(Options{
		ModelPath:  modelPath,
		Tokens:     tokens,
		Seq:        seq,
		AccumSteps: 1,
		LR:         3e-4,
		Seed:       123,
		UseANE:     false,
	})
	if err != nil {
		t.Fatalf("Open(cpu): %v", err)
	}
	defer cpu.Close()

	ane, err := Open(Options{
		ModelPath:      modelPath,
		Tokens:         tokens,
		Seq:            seq,
		AccumSteps:     1,
		LR:             3e-4,
		Seed:           123,
		UseANE:         true,
		HybridBackward: true,
	})
	if err != nil {
		t.Fatalf("Open(ane hybrid): %v", err)
	}
	defer ane.Close()

	if ane.off == nil {
		t.Skip("ANE offload unavailable")
	}
	if !ane.off.hasClassifierForward() || !ane.off.hasSoftmax() {
		t.Skipf("classifier/softmax offload unavailable: %s", ane.off.diagnosticSummary())
	}
	if err := ane.ensureBackward(); err != nil {
		t.Skipf("hybrid backward unavailable: %v", err)
	}
	d := ane.Diagnostics()
	if !d.HybridBackwardRequested || !d.HybridBackwardEnabled {
		t.Fatalf("hybrid diagnostics not enabled before step: %+v", d)
	}

	cpuStep, err := cpu.Step()
	if err != nil {
		t.Fatalf("cpu Step: %v", err)
	}
	aneStep, err := ane.Step()
	if err != nil {
		t.Fatalf("ane hybrid Step: %v", err)
	}
	if math.IsNaN(float64(aneStep.Loss)) || math.IsInf(float64(aneStep.Loss), 0) {
		t.Fatalf("ane hybrid loss is not finite: %v", aneStep.Loss)
	}
	if diff := math.Abs(float64(cpuStep.Loss - aneStep.Loss)); diff > 0.1 {
		t.Fatalf("hybrid loss mismatch cpu=%v ane=%v diff=%v", cpuStep.Loss, aneStep.Loss, diff)
	}
	t.Logf("cpu loss=%0.6f ane hybrid loss=%0.6f diff=%0.6f", cpuStep.Loss, aneStep.Loss, cpuStep.Loss-aneStep.Loss)
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

func firstExisting(paths ...string) string {
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

func readTokenPrefix(path string, count int) ([]uint16, error) {
	if count <= 0 {
		return nil, fmt.Errorf("count=%d must be > 0", count)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	out := make([]uint16, count)
	if err := binary.Read(f, binary.LittleEndian, out); err != nil {
		return nil, err
	}
	return out, nil
}

func writePretrainedFromCheckpoint(dstPath, ckptPath string) error {
	src, err := os.Open(ckptPath)
	if err != nil {
		return err
	}
	defer src.Close()

	dst, err := os.Create(dstPath)
	if err != nil {
		return err
	}
	defer dst.Close()

	cfg := stories.Llama2Config{
		Dim:       stories.Dim,
		HiddenDim: stories.Hidden,
		NLayers:   stories.NLayers,
		NHeads:    stories.Heads,
		NKVHeads:  stories.Heads,
		VocabSize: stories.Vocab,
		SeqLen:    stories.SeqDefault,
	}
	if err := binary.Write(dst, binary.LittleEndian, &cfg); err != nil {
		return err
	}

	if err := copyCheckpointSection(dst, src, checkpointEmbedOffset(), bytesOf(stories.Vocab*stories.Dim)); err != nil {
		return fmt.Errorf("copy embed: %w", err)
	}
	for layer := 0; layer < stories.NLayers; layer++ {
		if err := copyLayerChunk(dst, src, layer, layerOffsetRMSAtt, bytesOf(stories.Dim)); err != nil {
			return fmt.Errorf("copy rms_att[%d]: %w", layer, err)
		}
	}
	for layer := 0; layer < stories.NLayers; layer++ {
		if err := copyLayerChunk(dst, src, layer, layerOffsetWq, bytesOf(stories.WQSize)); err != nil {
			return fmt.Errorf("copy wq[%d]: %w", layer, err)
		}
	}
	for layer := 0; layer < stories.NLayers; layer++ {
		if err := copyLayerChunk(dst, src, layer, layerOffsetWk, bytesOf(stories.WQSize)); err != nil {
			return fmt.Errorf("copy wk[%d]: %w", layer, err)
		}
	}
	for layer := 0; layer < stories.NLayers; layer++ {
		if err := copyLayerChunk(dst, src, layer, layerOffsetWv, bytesOf(stories.WQSize)); err != nil {
			return fmt.Errorf("copy wv[%d]: %w", layer, err)
		}
	}
	for layer := 0; layer < stories.NLayers; layer++ {
		if err := copyLayerChunk(dst, src, layer, layerOffsetWo, bytesOf(stories.WOSize)); err != nil {
			return fmt.Errorf("copy wo[%d]: %w", layer, err)
		}
	}
	for layer := 0; layer < stories.NLayers; layer++ {
		if err := copyLayerChunk(dst, src, layer, layerOffsetRMSFFN, bytesOf(stories.Dim)); err != nil {
			return fmt.Errorf("copy rms_ffn[%d]: %w", layer, err)
		}
	}
	for layer := 0; layer < stories.NLayers; layer++ {
		if err := copyLayerChunk(dst, src, layer, layerOffsetW1, bytesOf(stories.W1Size)); err != nil {
			return fmt.Errorf("copy w1[%d]: %w", layer, err)
		}
	}
	for layer := 0; layer < stories.NLayers; layer++ {
		if err := copyLayerChunk(dst, src, layer, layerOffsetW2, bytesOf(stories.W2Size)); err != nil {
			return fmt.Errorf("copy w2[%d]: %w", layer, err)
		}
	}
	for layer := 0; layer < stories.NLayers; layer++ {
		if err := copyLayerChunk(dst, src, layer, layerOffsetW3, bytesOf(stories.W3Size)); err != nil {
			return fmt.Errorf("copy w3[%d]: %w", layer, err)
		}
	}
	if err := copyCheckpointSection(dst, src, checkpointRMSFinalOffset(), bytesOf(stories.Dim)); err != nil {
		return fmt.Errorf("copy rms_final: %w", err)
	}

	if err := dst.Sync(); err != nil {
		return err
	}
	return dst.Close()
}

func copyLayerChunk(dst io.Writer, src *os.File, layer int, relOffset, n int64) error {
	return copyCheckpointSection(dst, src, checkpointLayerOffset(layer)+relOffset, n)
}

func copyCheckpointSection(dst io.Writer, src *os.File, off, n int64) error {
	if _, err := src.Seek(off, io.SeekStart); err != nil {
		return err
	}
	written, err := io.CopyN(dst, src, n)
	if err != nil {
		return err
	}
	if written != n {
		return fmt.Errorf("copied %d bytes, want %d", written, n)
	}
	return nil
}

func bytesOf(count int32) int64 {
	return int64(count) * 4
}

const checkpointHeaderBytes int64 = 96

func checkpointLayerOffset(layer int) int64 {
	return checkpointHeaderBytes + int64(layer)*checkpointLayerBytes()
}

func checkpointLayerBytes() int64 {
	return bytesOf(3*stories.WQSize + stories.WOSize + stories.W1Size + stories.W2Size + stories.W3Size + 2*stories.Dim)
}

func checkpointZeroBlockOffset() int64 {
	return checkpointHeaderBytes + int64(stories.NLayers)*checkpointLayerBytes()
}

func checkpointZeroBlockBytes() int64 {
	perLayer := bytesOf(6*stories.WQSize + 2*stories.WOSize + 2*stories.W1Size + 2*stories.W2Size + 2*stories.W3Size + 4*stories.Dim)
	return int64(stories.NLayers) * perLayer
}

func checkpointRMSFinalOffset() int64 {
	return checkpointZeroBlockOffset() + checkpointZeroBlockBytes()
}

func checkpointEmbedOffset() int64 {
	return checkpointRMSFinalOffset() + bytesOf(3*stories.Dim)
}

var (
	layerOffsetWq     int64
	layerOffsetWk     = layerOffsetWq + bytesOf(stories.WQSize)
	layerOffsetWv     = layerOffsetWk + bytesOf(stories.WQSize)
	layerOffsetWo     = layerOffsetWv + bytesOf(stories.WQSize)
	layerOffsetW1     = layerOffsetWo + bytesOf(stories.WOSize)
	layerOffsetW2     = layerOffsetW1 + bytesOf(stories.W1Size)
	layerOffsetW3     = layerOffsetW2 + bytesOf(stories.W2Size)
	layerOffsetRMSAtt = layerOffsetW3 + bytesOf(stories.W3Size)
	layerOffsetRMSFFN = layerOffsetRMSAtt + bytesOf(stories.Dim)
)
