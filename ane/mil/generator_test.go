package mil

import (
	"strings"
	"testing"
)

func TestGenConvContainsExpectedOps(t *testing.T) {
	mil := GenConv(16, 32, 8)
	for _, s := range []string{"func main<ios18>", "conv(", "BLOBFILE(", "cast_out"} {
		if !strings.Contains(mil, s) {
			t.Fatalf("GenConv missing %q", s)
		}
	}
}

func TestBuildWeightBlob(t *testing.T) {
	weights := make([]float32, 6)
	for i := range weights {
		weights[i] = float32(i) + 0.5
	}
	blob, err := BuildWeightBlob(weights, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(blob) != 64+64+len(weights)*2 {
		t.Fatalf("blob len=%d", len(blob))
	}
	if blob[0] != 0x01 || blob[4] != 0x02 {
		t.Fatalf("invalid header magic")
	}
}
