package mil

import (
	"strings"
	"testing"
)

func TestGenDynamicMatmulContainsExpectedOps(t *testing.T) {
	mil := GenDynamicMatmul(64, 32, 16)
	for _, want := range []string{
		"slice_by_size(",
		"reshape(",
		"transpose(",
		"matmul(",
		"cast(dtype = to16",
		"cast(dtype = to32",
	} {
		if !strings.Contains(mil, want) {
			t.Fatalf("GenDynamicMatmul missing %q", want)
		}
	}
}
