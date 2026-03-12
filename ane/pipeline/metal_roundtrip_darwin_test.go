//go:build darwin

package pipeline

import (
	"bytes"
	"encoding/binary"
	"fmt"
	xane "github.com/tmc/apple/x/ane"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestSmokeZeroCopyMetalToANESignal(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}
	out := &bytes.Buffer{}
	logOut := io.Writer(out)
	if os.Getenv("ANE_SMOKE_CLEANUP") != "" {
		logOut = io.MultiWriter(out, os.Stdout)
	}
	err := runZeroCopyMetalToANESignal(logOut)
	if err != nil {
		t.Fatalf("smoke failed: %v\n%s", err, out.Bytes())
	}
	if !bytes.Contains(out.Bytes(), []byte("ANE output matched oracle")) {
		t.Fatalf("smoke missing success marker\n%s", out.Bytes())
	}
	t.Logf("smoke ok:\n%s", out.Bytes())
}

func runZeroCopyMetalToANESignal(w io.Writer) error {
	const (
		timeout = 2 * time.Second
	)
	logf := func(format string, args ...any) {
		fmt.Fprintf(w, format+"\n", args...)
	}
	modelPath, err := zeroCopyModelPath()
	if err != nil {
		return err
	}

	rt, err := xane.Open()
	if err != nil {
		return fmt.Errorf("open runtime: %w", err)
	}

	md, err := xane.OpenMetal()
	if err != nil {
		return fmt.Errorf("open metal: %w", err)
	}

	k, err := rt.Compile(xane.CompileOptions{
		ModelType:   xane.ModelTypePackage,
		PackagePath: modelPath,
		ModelKey:    "s",
	})
	if err != nil {
		return fmt.Errorf("compile package kernel: %w", err)
	}

	cq := md.Device().NewCommandQueue()
	if cq == nil || cq.GetID() == 0 {
		return fmt.Errorf("new command queue returned nil")
	}

	outputAllocBytes := k.OutputAllocSize(0)
	if err := baselineWriteInput(k); err != nil {
		return fmt.Errorf("baseline write input: %w", err)
	}
	want := make([]byte, outputAllocBytes)
	if err := k.Eval(); err != nil {
		return fmt.Errorf("baseline Eval: %w", err)
	}
	if err := k.ReadOutput(0, want); err != nil {
		return fmt.Errorf("baseline ReadOutput: %w", err)
	}

	mtlWait, aneWait, err := md.NewMetalSharedEvent()
	if err != nil {
		return fmt.Errorf("new wait event: %w", err)
	}
	gpuToANE := mtlWait.SignaledValue() + 1
	logf("shared events gpuToANE=%d", gpuToANE)

	evalErrc := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		evalErrc <- k.EvalWithWait(
			aneWait,
			gpuToANE,
			xane.SharedEventEvalOptions{
				DisableIOFencesUseSharedEvents: true,
				EnableFWToFWSignal:             true,
			},
		)
		time.Sleep(200 * time.Millisecond)
	}()
	time.Sleep(200 * time.Microsecond)
	logf("ANE eval enqueued")
	cbIn := cq.CommandBuffer()
	if cbIn == nil || cbIn.GetID() == 0 {
		return fmt.Errorf("new input command buffer returned nil")
	}
	cbIn.EncodeSignalEventValue(mtlWait, gpuToANE)
	cbIn.Commit()
	logf("GPU signal committed")
	cbIn.WaitUntilCompleted()
	logf("GPU input copy completed")
	if err := <-evalErrc; err != nil {
		return fmt.Errorf("EvalWithWait: %w", err)
	}
	logf("ANE eval completed")
	logf("reading ANE output")
	got, err := readRawOutput(k, outputAllocBytes)
	if err != nil {
		return err
	}
	logf("read ANE output")
	if !bytes.Equal(got, want) {
		for i := range want {
			if got[i] != want[i] {
				return fmt.Errorf("output[%d] byte = 0x%02x, want 0x%02x", i, got[i], want[i])
			}
		}
		return fmt.Errorf("output mismatch")
	}
	logf("ANE output matched oracle")
	logf("closing wait event")
	_ = aneWait.Close()
	logf("closed wait event")
	logf("closing kernel")
	_ = k.Close()
	logf("closed kernel")
	logf("closing metal device")
	_ = md.Close()
	logf("closed metal device")
	logf("closing runtime")
	_ = rt.Close()
	logf("closed runtime")
	return nil
}

func readRawOutput(k *xane.Kernel, n int) ([]byte, error) {
	got := make([]byte, n)
	if err := k.ReadOutput(0, got); err != nil {
		return nil, fmt.Errorf("ReadOutput: %w", err)
	}
	return got, nil
}

func zeroCopyInput(n int) []byte {
	buf := make([]byte, n)
	if n%4 != 0 {
		for i := range buf {
			buf[i] = byte(i*29 + 7)
		}
		return buf
	}
	for i := 0; i < n/4; i++ {
		v := float32((i%17)+1) * 0.25
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func baselineWriteInput(k *xane.Kernel) error {
	layout := k.InputLayout(0)
	switch layout.ElemSize {
	case 2:
		data := make([]float32, layout.LogicalElements())
		for i := range data {
			data[i] = float32((i%17)+1) * 0.25
		}
		return k.WriteInputFP16(0, data)
	case 4:
		data := make([]float32, layout.LogicalElements())
		for i := range data {
			data[i] = float32((i%17)+1) * 0.25
		}
		return k.WriteInputF32(0, data)
	default:
		buf := zeroCopyInput(k.InputAllocSize(0))
		return k.WriteInput(0, buf)
	}
}

func zeroCopyModelPath() (string, error) {
	if p := os.Getenv("ANE_CHAIN_MODEL_PATH"); p != "" {
		return p, nil
	}
	p := "/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc"
	if _, err := os.Stat(p); err == nil {
		return p, nil
	}
	alt := "/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add.mlmodelc"
	if _, err := os.Stat(alt); err == nil {
		return alt, nil
	}
	return "", fmt.Errorf("set ANE_CHAIN_MODEL_PATH to a compiled chaining model (.mlmodelc); checked %s and %s", filepath.Clean(p), filepath.Clean(alt))
}
