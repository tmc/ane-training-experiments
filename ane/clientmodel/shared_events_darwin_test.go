//go:build darwin

package clientmodel

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/apple/objc"
)

func TestSharedEventOptionsDictionary(t *testing.T) {
	opts := sharedEventOptions(SharedEventEvalOptions{
		DisableIOFencesUseSharedEvents: true,
		EnableFWToFWSignal:             false,
	})
	if opts == 0 {
		t.Fatalf("sharedEventOptions returned nil")
	}
	if got := objc.Send[uint64](opts, objc.Sel("count")); got != 2 {
		t.Fatalf("sharedEventOptions count=%d want=2", got)
	}
}

func TestSmokeEvalBidirectionalSharedEvents(t *testing.T) {
	if os.Getenv("ANE_SMOKE_SHARED") != "1" {
		t.Skip("set ANE_SMOKE_SHARED=1 to run shared-events smoke test")
	}
	if os.Getenv("ANE_SMOKE_SHARED_CHILD") != "1" {
		cmd := exec.Command(os.Args[0], "-test.run", "^TestSmokeEvalBidirectionalSharedEvents$")
		cmd.Env = append(os.Environ(), "ANE_SMOKE_SHARED=1", "ANE_SMOKE_SHARED_CHILD=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("shared-events child failed: %v\n%s", err, out)
		}
		return
	}

	k, candidate := compileSharedSmokeKernel(t)
	defer k.Close()

	waitEvent, err := NewSharedEvent()
	if err != nil {
		t.Fatalf("create wait event failed: %v", err)
	}
	defer func() { _ = waitEvent.Close() }()

	signalEvent, err := NewSharedEvent()
	if err != nil {
		t.Fatalf("create signal event failed: %v", err)
	}
	defer func() { _ = signalEvent.Close() }()

	in := make([]byte, candidate.inputBytes)
	for i := range in {
		in[i] = byte((i % 251) + 1)
	}
	if err := k.WriteInput(0, in); err != nil {
		t.Fatalf("WriteInput: %v", err)
	}

	const waitValue = uint64(1)
	const signalValue = uint64(1)
	if err := waitEvent.Signal(waitValue); err != nil {
		t.Fatalf("wait event signal: %v", err)
	}

	if err := k.EvalBidirectional(
		waitEvent.Port(),
		waitValue,
		signalEvent.Port(),
		signalValue,
		SharedEventEvalOptions{
			DisableIOFencesUseSharedEvents: true,
			EnableFWToFWSignal:             false,
		},
	); err != nil {
		t.Fatalf("EvalBidirectional: %v", err)
	}

	if ok, err := signalEvent.Wait(signalValue, 5*time.Second); err != nil || !ok {
		if err != nil {
			t.Fatalf("signal event wait error: %v", err)
		}
		t.Fatalf("signal event wait timed out")
	}

	out := make([]byte, candidate.outputBytes)
	if err := k.ReadOutput(0, out); err != nil {
		t.Fatalf("ReadOutput: %v", err)
	}
	f := bytesToFloat32(out)
	if len(f) == 0 {
		t.Fatalf("empty output")
	}
	for i, v := range f {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d]=%v", i, v)
		}
	}
}

type smokeCandidate struct {
	path        string
	inputBytes  int
	outputBytes int
}

func compileSharedSmokeKernel(t *testing.T) (*Kernel, smokeCandidate) {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatalf("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	candidates := []smokeCandidate{
		{
			path:        os.Getenv("ANE_SMOKE_SHARED_MODEL"),
			inputBytes:  envInt("ANE_SMOKE_SHARED_INPUT_BYTES", 4096),
			outputBytes: envInt("ANE_SMOKE_SHARED_OUTPUT_BYTES", 4096),
		},
		{path: filepath.Join(root, "testdata", "ffn", "tiny_coreml_ffn.mlmodelc"), inputBytes: 64 * 4, outputBytes: 64 * 4},
		{path: filepath.Join(root, "testdata", "ffn", "probe_nn", "draft_probe_nn.mlmodelc"), inputBytes: 768 * 4, outputBytes: 768 * 4},
		{path: "/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc", inputBytes: 4096, outputBytes: 4096},
	}

	for _, c := range candidates {
		if c.path == "" {
			continue
		}
		if _, err := os.Stat(c.path); err != nil {
			continue
		}
		k, err := Compile(CompileOptions{
			CompiledModelPath: c.path,
			ModelKey:          "s",
			InputBytes:        []int{c.inputBytes},
			OutputBytes:       []int{c.outputBytes},
		})
		if err == nil {
			return k, c
		}
		t.Logf("candidate compile failed (%s): %v", c.path, err)
	}
	t.Fatalf("no shared-events smoke model compiled successfully")
	return nil, smokeCandidate{}
}

func envInt(name string, def int) int {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return def
	}
	return n
}
