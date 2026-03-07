//go:build darwin

package clientmodel

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestSmokeCompileAndEvalProbeNN(t *testing.T) {
	if os.Getenv("ANE_SMOKE") != "1" {
		t.Skip("set ANE_SMOKE=1 to run _ANEClient smoke test")
	}
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatalf("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	type candidate struct {
		path       string
		inputBytes int
		outBytes   int
	}
	candidates := []candidate{
		{path: filepath.Join(root, "testdata", "ffn", "tiny_coreml_ffn.mlmodelc"), inputBytes: 64 * 4, outBytes: 64 * 4},
		{path: filepath.Join(root, "testdata", "ffn", "probe_nn", "draft_probe_nn.mlmodelc"), inputBytes: 768 * 4, outBytes: 768 * 4},
	}

	var k *Kernel
	var err error
	var used candidate
	for _, c := range candidates {
		if _, statErr := os.Stat(c.path); statErr != nil {
			continue
		}
		k, err = Compile(CompileOptions{
			CompiledModelPath: c.path,
			ModelKey:          "s",
			InputBytes:        []int{c.inputBytes},
			OutputBytes:       []int{c.outBytes},
		})
		if err == nil {
			used = c
			break
		}
		t.Logf("candidate compile failed (%s): %v", c.path, err)
	}
	if k == nil {
		t.Fatalf("no smoke model compiled successfully")
	}
	defer k.Close()

	count := used.inputBytes / 4
	in := make([]byte, count*4)
	if err := k.WriteInput(0, in); err != nil {
		t.Fatalf("WriteInput: %v", err)
	}
	if err := k.Eval(); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	out := make([]byte, used.outBytes)
	if err := k.ReadOutput(0, out); err != nil {
		t.Fatalf("ReadOutput: %v", err)
	}
	f := bytesToFloat32(out)
	if len(f) != used.outBytes/4 {
		t.Fatalf("output len=%d want=%d", len(f), used.outBytes/4)
	}
	for i, v := range f {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d]=%v", i, v)
		}
	}
}

func bytesToFloat32(b []byte) []float32 {
	if len(b)%4 != 0 {
		return nil
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		u := uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24
		out[i] = math.Float32frombits(u)
	}
	return out
}
