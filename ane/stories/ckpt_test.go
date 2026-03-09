package stories

import (
	"bytes"
	"testing"
)

func TestLayerOptimStateRoundTrip(t *testing.T) {
	in := zeroLayerOptimState()
	fill := func(v []float32, start float32) {
		for i := range v {
			v[i] = start + float32(i)/10
		}
	}
	fill(in.Wq.M, 0.1)
	fill(in.Wq.V, 1.1)
	fill(in.Wk.M, 2.1)
	fill(in.Wk.V, 3.1)
	fill(in.Wv.M, 4.1)
	fill(in.Wv.V, 5.1)
	fill(in.Wo.M, 6.1)
	fill(in.Wo.V, 7.1)
	fill(in.W1.M, 8.1)
	fill(in.W1.V, 9.1)
	fill(in.W2.M, 10.1)
	fill(in.W2.V, 11.1)
	fill(in.W3.M, 12.1)
	fill(in.W3.V, 13.1)
	fill(in.RMSAtt.M, 14.1)
	fill(in.RMSAtt.V, 15.1)
	fill(in.RMSFFN.M, 16.1)
	fill(in.RMSFFN.V, 17.1)

	var buf bytes.Buffer
	if err := writeLayerOptimState(&buf, in); err != nil {
		t.Fatalf("writeLayerOptimState: %v", err)
	}

	out := zeroLayerOptimState()
	if err := readLayerOptimState(&buf, &out); err != nil {
		t.Fatalf("readLayerOptimState: %v", err)
	}

	for _, tc := range []struct {
		name string
		got  []float32
		want []float32
	}{
		{"wq.m", out.Wq.M, in.Wq.M},
		{"wq.v", out.Wq.V, in.Wq.V},
		{"wk.m", out.Wk.M, in.Wk.M},
		{"wk.v", out.Wk.V, in.Wk.V},
		{"wv.m", out.Wv.M, in.Wv.M},
		{"wv.v", out.Wv.V, in.Wv.V},
		{"wo.m", out.Wo.M, in.Wo.M},
		{"wo.v", out.Wo.V, in.Wo.V},
		{"w1.m", out.W1.M, in.W1.M},
		{"w1.v", out.W1.V, in.W1.V},
		{"w2.m", out.W2.M, in.W2.M},
		{"w2.v", out.W2.V, in.W2.V},
		{"w3.m", out.W3.M, in.W3.M},
		{"w3.v", out.W3.V, in.W3.V},
		{"rmsatt.m", out.RMSAtt.M, in.RMSAtt.M},
		{"rmsatt.v", out.RMSAtt.V, in.RMSAtt.V},
		{"rmsffn.m", out.RMSFFN.M, in.RMSFFN.M},
		{"rmsffn.v", out.RMSFFN.V, in.RMSFFN.V},
	} {
		for i, want := range tc.want {
			if tc.got[i] != want {
				t.Fatalf("%s[%d]=%v want %v", tc.name, i, tc.got[i], want)
			}
		}
	}
}
