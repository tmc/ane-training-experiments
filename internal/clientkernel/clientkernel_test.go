package clientkernel

import "testing"

func TestValidate(t *testing.T) {
	tests := []struct {
		name string
		opts EvalOptions
	}{
		{name: "missing model", opts: EvalOptions{InputBytes: 1, OutputBytes: 1}},
		{name: "missing input", opts: EvalOptions{ModelPath: "/tmp/model.mlmodelc", OutputBytes: 1}},
		{name: "missing output", opts: EvalOptions{ModelPath: "/tmp/model.mlmodelc", InputBytes: 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Validate(tt.opts); err == nil {
				t.Fatalf("Validate(%+v) succeeded; want error", tt.opts)
			}
		})
	}
}

func TestWithDefaults(t *testing.T) {
	opts := WithDefaults(EvalOptions{})
	if opts.ModelKey != defaultModelKey {
		t.Fatalf("ModelKey=%q want %q", opts.ModelKey, defaultModelKey)
	}

	opts = WithDefaults(EvalOptions{ModelKey: "custom"})
	if opts.ModelKey != "custom" {
		t.Fatalf("ModelKey=%q want custom", opts.ModelKey)
	}
}
