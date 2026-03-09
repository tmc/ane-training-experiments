package mil

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func TestGenConvContainsExpectedOps(t *testing.T) {
	mil := GenConv(16, 32, 8)
	for _, s := range []string{"conv(", "weight.bin"} {
		if !strings.Contains(mil, s) {
			t.Fatalf("GenConv missing %q", s)
		}
	}
}

func TestExtraGeneratorsContainExpectedOps(t *testing.T) {
	tests := []struct {
		name string
		mil  string
		want []string
	}{
		{
			name: "classifier forward",
			mil:  GenClassifierForward(768, 32000, 256),
			want: []string{"conv(", "embed.bin", "cls"},
		},
		{
			name: "classifier backward",
			mil:  GenClassifierBackward(768, 32000, 256),
			want: []string{"matmul(", "embed_t.bin", "reshape("},
		},
		{
			name: "softmax vocab",
			mil:  GenSoftmaxVocab(32000, 256),
			want: []string{"softmax(", "axis=ax"},
		},
		{
			name: "final rmsnorm",
			mil:  GenFinalRMSNorm(768, 256),
			want: []string{"reduce_sum(", "pow(", "rms_w.bin"},
		},
		{
			name: "rmsnorm backward",
			mil:  GenRMSNormBackward(768, 256),
			want: []string{"slice_by_size(", "reduce_sum(", "sub(", "rms_w.bin"},
		},
		{
			name: "ffn forward",
			mil:  GenFFNForward(768, 2048, 256),
			want: []string{"w1.bin", "w2.bin", "w3.bin", "sigmoid(", "mul(", "conv("},
		},
		{
			name: "ffn forward taps",
			mil:  GenFFNForwardTaps(768, 2048, 256),
			want: []string{"concat(", "rms2.bin", "w1.bin", "w2.bin", "w3.bin", "sigmoid(", "xn"},
		},
		{
			name: "ffn backward",
			mil:  GenFFNBackward(768, 2048, 256),
			want: []string{"slice_by_size(", "w2t.bin", "w1t.bin", "w3t.bin", "sub(", "add(", "concat("},
		},
		{
			name: "sdpa forward taps",
			mil:  GenSDPAForwardTaps(768, 12, 256),
			want: []string{"rms1.bin", "wq.bin", "wk.bin", "wv.bin", "wo.bin", "mask.bin", "softmax(", "concat("},
		},
		{
			name: "qkv backward",
			mil:  GenQKVBackward(768, 12, 256),
			want: []string{"slice_by_size(", "wqt.bin", "wkt.bin", "wvt.bin", "conv(", "add("},
		},
		{
			name: "sdpa backward1",
			mil:  GenSDPABackward1(768, 12, 256),
			want: []string{"wot.bin", "mask.bin", "softmax(", "matmul(", "concat("},
		},
		{
			name: "sdpa backward2",
			mil:  GenSDPABackward2(768, 12, 256),
			want: []string{"reduce_sum(", "sub(", "matmul(", "transpose(", "concat("},
		},
	}
	for _, tc := range tests {
		for _, s := range tc.want {
			if !strings.Contains(tc.mil, s) {
				t.Fatalf("%s missing %q", tc.name, s)
			}
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
	if bytes.Equal(blob[:64], make([]byte, 64)) {
		t.Fatalf("header is all zero")
	}
	if bytes.Equal(blob[128:], make([]byte, len(weights)*2)) {
		t.Fatalf("payload is all zero")
	}
}

func TestBuildWeightBlobWrongSize(t *testing.T) {
	if _, err := BuildWeightBlob([]float32{1, 2, 3}, 2, 2); err == nil {
		t.Fatal("BuildWeightBlob accepted wrong size")
	}
}

func TestBuildFP16Blob(t *testing.T) {
	blob, err := BuildFP16Blob([]float32{1, -2})
	if err != nil {
		t.Fatal(err)
	}
	if len(blob) != 64+64+4 {
		t.Fatalf("blob len=%d", len(blob))
	}
	got := []uint16{
		uint16(blob[128]) | uint16(blob[129])<<8,
		uint16(blob[130]) | uint16(blob[131])<<8,
	}
	want := []uint16{Float32ToFP16(1), Float32ToFP16(-2)}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("payload[%d]=0x%x want 0x%x", i, got[i], w)
		}
	}
}

func TestBuildCausalMaskBlob(t *testing.T) {
	blob, err := BuildCausalMaskBlob(2)
	if err != nil {
		t.Fatal(err)
	}
	got := []uint16{
		uint16(blob[128]) | uint16(blob[129])<<8,
		uint16(blob[130]) | uint16(blob[131])<<8,
		uint16(blob[132]) | uint16(blob[133])<<8,
		uint16(blob[134]) | uint16(blob[135])<<8,
	}
	want := []uint16{
		Float32ToFP16(0), Float32ToFP16(-65504),
		Float32ToFP16(0), Float32ToFP16(0),
	}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("mask[%d]=0x%x want 0x%x", i, got[i], w)
		}
	}
}

func TestBuildIdentityWeightBlob(t *testing.T) {
	blob, err := BuildIdentityWeightBlob(4)
	if err != nil {
		t.Fatal(err)
	}
	if len(blob) != 64+64+4*4*2 {
		t.Fatalf("blob len=%d", len(blob))
	}
}

func TestBuildVectorWeightBlob(t *testing.T) {
	blob, err := BuildVectorWeightBlob([]float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if len(blob) != 64+64+4*2 {
		t.Fatalf("blob len=%d", len(blob))
	}
}

func TestBuildTransposedWeightBlob(t *testing.T) {
	blob, err := BuildTransposedWeightBlob([]float32{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	got := []uint16{
		uint16(blob[128]) | uint16(blob[129])<<8,
		uint16(blob[130]) | uint16(blob[131])<<8,
		uint16(blob[132]) | uint16(blob[133])<<8,
		uint16(blob[134]) | uint16(blob[135])<<8,
		uint16(blob[136]) | uint16(blob[137])<<8,
		uint16(blob[138]) | uint16(blob[139])<<8,
	}
	want := []uint16{
		Float32ToFP16(1), Float32ToFP16(4),
		Float32ToFP16(2), Float32ToFP16(5),
		Float32ToFP16(3), Float32ToFP16(6),
	}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("payload[%d]=0x%x want 0x%x", i, got[i], w)
		}
	}
}

func TestFP16RoundTrip(t *testing.T) {
	tests := []float32{0, 1, -1, 0.5, 10, 123.5}
	for _, in := range tests {
		got := FP16ToFloat32(Float32ToFP16(in))
		if diff := math.Abs(float64(in - got)); diff > 0.05 {
			t.Fatalf("round-trip(%v)=%v diff=%v", in, got, diff)
		}
	}
}
