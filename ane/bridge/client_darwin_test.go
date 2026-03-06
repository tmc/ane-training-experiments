//go:build darwin

package bridge

import (
	"math"
	"os"
	"testing"
	"unsafe"
)

func TestOpenClientValidation(t *testing.T) {
	var rt Runtime
	if _, err := rt.OpenClient("", "s", 4, 4); err == nil {
		t.Fatalf("OpenClient with empty path succeeded; want error")
	}
	if _, err := rt.OpenClient("/tmp/m.mlmodelc", "s", 0, 4); err == nil {
		t.Fatalf("OpenClient with zero input bytes succeeded; want error")
	}
}

func TestClientReadWriteEval(t *testing.T) {
	var (
		wroteCount int32
		closed     bool
	)
	rt := &Runtime{
		open: func(string, string, uintptr, uintptr) uintptr { return 0x42 },
		close: func(uintptr) {
			closed = true
		},
		eval: func(uintptr) bool { return true },
		writeInput: func(_ uintptr, _ unsafe.Pointer, count int32) {
			wroteCount = count
		},
		readOutput: func(_ uintptr, ptr unsafe.Pointer, count int32) {
			dst := unsafe.Slice((*float32)(ptr), int(count))
			for i := range dst {
				dst[i] = float32(i + 1)
			}
		},
	}
	c, err := rt.OpenClient("/tmp/model.mlmodelc", "s", 16, 16)
	if err != nil {
		t.Fatalf("OpenClient: %v", err)
	}
	if err := c.WriteInputF32([]float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("WriteInputF32: %v", err)
	}
	if err := c.WriteInputF32([]float32{1, 2, 3}); err == nil {
		t.Fatalf("WriteInputF32 partial input succeeded; want error")
	}
	if wroteCount != 4 {
		t.Fatalf("WriteInputF32 count=%d want 4", wroteCount)
	}
	if err := c.Eval(); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	got := make([]float32, 4)
	if err := c.ReadOutputF32(got); err != nil {
		t.Fatalf("ReadOutputF32: %v", err)
	}
	if err := c.ReadOutputF32(make([]float32, 3)); err == nil {
		t.Fatalf("ReadOutputF32 partial output succeeded; want error")
	}
	for i, v := range got {
		want := float32(i + 1)
		if v != want {
			t.Fatalf("ReadOutputF32[%d]=%v want %v", i, v, want)
		}
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !closed {
		t.Fatalf("Close did not call runtime close")
	}
}

func TestSharedEventLifecycle(t *testing.T) {
	var released uintptr
	rt := &Runtime{
		createSharedEvent: func() uintptr { return 0x99 },
		sharedEventPort:   func(uintptr) uint32 { return 7 },
		releaseObjc: func(obj uintptr) {
			released = obj
		},
	}
	ev, err := rt.NewSharedEvent()
	if err != nil {
		t.Fatalf("NewSharedEvent: %v", err)
	}
	if ev.Port != 7 {
		t.Fatalf("NewSharedEvent port=%d want 7", ev.Port)
	}
	if err := ev.Close(); err != nil {
		t.Fatalf("SharedEvent.Close: %v", err)
	}
	if released != 0x99 {
		t.Fatalf("release object=%#x want %#x", released, uintptr(0x99))
	}
}

func TestEvalWithSignalAndBidirectionalErrors(t *testing.T) {
	rt := &Runtime{
		open: func(string, string, uintptr, uintptr) uintptr { return 0x11 },
		close: func(uintptr) {
		},
		evalWithSignalEvent: func(uintptr, unsafe.Pointer, uint32, unsafe.Pointer, uint32, uint32, uint64) int32 {
			return 0
		},
		evalBidirectional: func(uintptr, unsafe.Pointer, uint32, unsafe.Pointer, uint32, uint32, uint64, uint32, uint64) int32 {
			return 0
		},
	}
	c, err := rt.OpenClient("/tmp/model.mlmodelc", "s", 16, 16)
	if err != nil {
		t.Fatalf("OpenClient: %v", err)
	}
	in := []float32{1, 2, 3, 4}
	out := make([]float32, 4)
	if err := c.EvalWithSignalEvent(0, 1, in, out); err == nil {
		t.Fatalf("EvalWithSignalEvent with zero port succeeded; want error")
	}
	if err := c.EvalWithSignalEvent(9, 1, in, out); err != nil {
		t.Fatalf("EvalWithSignalEvent: %v", err)
	}
	if err := c.EvalWithSignalEvent(9, 1, append(in, 5), out); err == nil {
		t.Fatalf("EvalWithSignalEvent with oversized input succeeded; want error")
	}
	if err := c.EvalBidirectional(0, 1, 9, 1, in, out); err == nil {
		t.Fatalf("EvalBidirectional with zero wait port succeeded; want error")
	}
	if err := c.EvalBidirectional(8, 1, 9, 2, in, out); err != nil {
		t.Fatalf("EvalBidirectional: %v", err)
	}
	if err := c.EvalBidirectional(8, 1, 9, 2, in, append(out, 9)); err == nil {
		t.Fatalf("EvalBidirectional with oversized output succeeded; want error")
	}
}

func TestF32ByteRoundTrip(t *testing.T) {
	src := []float32{1.25, -3.5, float32(math.SmallestNonzeroFloat32)}
	b := F32ToBytes(src)
	dst := BytesToF32(b)
	if len(dst) != len(src) {
		t.Fatalf("len(dst)=%d want %d", len(dst), len(src))
	}
	for i := range src {
		if dst[i] != src[i] {
			t.Fatalf("dst[%d]=%v want %v", i, dst[i], src[i])
		}
	}
}

func TestBridgeSmokeIntegration(t *testing.T) {
	if os.Getenv("ANE_BRIDGE_SMOKE") != "1" {
		t.Skip("set ANE_BRIDGE_SMOKE=1 to run bridge integration smoke")
	}
	model := os.Getenv("ANE_BRIDGE_MODEL")
	if model == "" {
		model = "/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc"
	}
	rt, err := Load(LoadOptions{})
	if err != nil {
		t.Fatalf("Load runtime: %v", err)
	}
	client, err := rt.OpenClient(model, "s", 4096, 4096)
	if err != nil {
		t.Fatalf("OpenClient: %v", err)
	}
	defer func() { _ = client.Close() }()

	in := make([]float32, 1024)
	for i := range in {
		in[i] = float32(i + 1)
	}
	out := make([]float32, 1024)
	if err := client.WriteInputF32(in); err != nil {
		t.Fatalf("WriteInputF32: %v", err)
	}
	if err := client.Eval(); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if err := client.ReadOutputF32(out); err != nil {
		t.Fatalf("ReadOutputF32: %v", err)
	}
	if out[0] == 0 && out[1] == 0 && out[2] == 0 {
		t.Fatalf("output appears zeroed: [%v %v %v]", out[0], out[1], out[2])
	}
}

func TestBridgeBidirectionalIntegration(t *testing.T) {
	if os.Getenv("ANE_BRIDGE_SMOKE") != "1" {
		t.Skip("set ANE_BRIDGE_SMOKE=1 to run bridge integration smoke")
	}
	model := os.Getenv("ANE_BRIDGE_MODEL")
	if model == "" {
		model = "/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc"
	}
	rt, err := Load(LoadOptions{})
	if err != nil {
		t.Fatalf("Load runtime: %v", err)
	}
	client, err := rt.OpenClient(model, "s", 4096, 4096)
	if err != nil {
		t.Fatalf("OpenClient: %v", err)
	}
	defer func() { _ = client.Close() }()

	waitEvent, err := rt.NewSharedEvent()
	if err != nil {
		t.Fatalf("NewSharedEvent(wait): %v", err)
	}
	defer func() { _ = waitEvent.Close() }()
	signalEvent, err := rt.NewSharedEvent()
	if err != nil {
		t.Fatalf("NewSharedEvent(signal): %v", err)
	}
	defer func() { _ = signalEvent.Close() }()

	in := make([]float32, 1024)
	for i := range in {
		in[i] = float32(i + 1)
	}
	out := make([]float32, 1024)
	if err := rt.SignalEventCPU(waitEvent.Port, 1); err != nil {
		t.Fatalf("SignalEventCPU(wait): %v", err)
	}
	if err := client.EvalBidirectional(waitEvent.Port, 1, signalEvent.Port, 1, in, out); err != nil {
		t.Fatalf("EvalBidirectional: %v", err)
	}
	ok, err := rt.WaitEventCPU(signalEvent.Port, 1, 5000)
	if err != nil {
		t.Fatalf("WaitEventCPU(signal): %v", err)
	}
	if !ok {
		t.Fatalf("WaitEventCPU(signal) timed out")
	}
	if out[0] == 0 && out[1] == 0 && out[2] == 0 {
		t.Fatalf("output appears zeroed: [%v %v %v]", out[0], out[1], out[2])
	}
}
