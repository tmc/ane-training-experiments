//go:build darwin

package clientmodel

import (
	"reflect"
	"testing"

	"github.com/tmc/apple/objc"
)

func TestWithDefaults(t *testing.T) {
	opts := withDefaults(CompileOptions{})
	if opts.ModelKey != defaultModelKey {
		t.Fatalf("ModelKey=%q want=%q", opts.ModelKey, defaultModelKey)
	}
	if opts.QoS != defaultQoS {
		t.Fatalf("QoS=%d want=%d", opts.QoS, defaultQoS)
	}
	if opts.ModelType != "" {
		t.Fatalf("ModelType=%q want empty", opts.ModelType)
	}
	if opts.NetPlistFilename != "" {
		t.Fatalf("NetPlistFilename=%q want empty", opts.NetPlistFilename)
	}
}

func TestCompileOptionsDictionary(t *testing.T) {
	empty := compileOptionsDictionary("", "")
	if empty == 0 {
		t.Fatalf("compileOptionsDictionary returned nil")
	}
	if got := objc.Send[uint64](empty, objc.Sel("count")); got != 0 {
		t.Fatalf("empty options count=%d want=0", got)
	}

	full := compileOptionsDictionary("kANEFModelMIL", "model.mil")
	if full == 0 {
		t.Fatalf("compileOptionsDictionary full returned nil")
	}
	if got := objc.Send[uint64](full, objc.Sel("count")); got != 2 {
		t.Fatalf("full options count=%d want=2", got)
	}
}

func TestKernelDiagnosticsNil(t *testing.T) {
	var k *Kernel

	if k.HasVirtualClient() {
		t.Fatalf("HasVirtualClient on nil kernel = true, want false")
	}
	if got, ok := k.IsVirtualClient(); got || ok {
		t.Fatalf("IsVirtualClient on nil kernel = (%v,%v), want (false,false)", got, ok)
	}
	if got, ok := k.QueueDepth(); got != 0 || ok {
		t.Fatalf("QueueDepth on nil kernel = (%d,%v), want (0,false)", got, ok)
	}
	if ref, err := k.InputSurfaceRef(0); err == nil || ref != 0 {
		t.Fatalf("InputSurfaceRef on nil kernel = (%#x,%v), want error", ref, err)
	}
	if ref, err := k.OutputSurfaceRef(0); err == nil || ref != 0 {
		t.Fatalf("OutputSurfaceRef on nil kernel = (%#x,%v), want error", ref, err)
	}
	if code, ok := k.VirtualClientConnect(); code != 0 || ok {
		t.Fatalf("VirtualClientConnect on nil kernel = (%d,%v), want (0,false)", code, ok)
	}
	if k.SupportsCompletionEventEval() {
		t.Fatalf("SupportsCompletionEventEval on nil kernel = true, want false")
	}
	d := k.Diagnostics()
	if d.HasVirtualClient || d.VirtualClientConnectKnown || d.SupportsCompletionEventEval || d.IsVirtualClientKnown || d.ModelQueueDepthKnown || d.ProgramQueueDepthKnown || d.CurrentAsyncRequestsInFlightOK || d.RequestsInFlightCountKnown {
		t.Fatalf("Diagnostics on nil kernel reported unexpected known fields: %+v", d)
	}
}

func TestSetRequestCompletionHandlerValidation(t *testing.T) {
	done := make(chan completionResult, 1)
	if _, err := setRequestCompletionHandler(0, done); err == nil {
		t.Fatalf("setRequestCompletionHandler(0) error=nil, want error")
	}
}

func TestCompileCountReset(t *testing.T) {
	ResetCompileCount()
	if got := CompileCount(); got != 0 {
		t.Fatalf("CompileCount after reset = %d, want 0", got)
	}
}

func TestParseCompileFallbackProfiles(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []compileProfile
	}{
		{name: "empty", in: "", want: nil},
		{name: "single modeltype", in: "kANEFModelMIL", want: []compileProfile{{modelType: "kANEFModelMIL"}}},
		{
			name: "pairs and empty normalization",
			in:   "kANEFModelMIL:ane.plist, <empty>:fallback.plist, -:<empty>, kANEFModelMIL:ane.plist",
			want: []compileProfile{
				{modelType: "kANEFModelMIL", netPlist: "ane.plist"},
				{modelType: "", netPlist: "fallback.plist"},
				{modelType: "", netPlist: ""},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseCompileFallbackProfiles(tt.in)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("parseCompileFallbackProfiles(%q)=%v want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestIsInvalidMILCompileErr(t *testing.T) {
	if isInvalidMILCompileErr(nil) {
		t.Fatalf("nil error detected as InvalidMILProgram")
	}
	if !isInvalidMILCompileErr("compileModel failed: InvalidMILProgram") {
		t.Fatalf("InvalidMILProgram string not detected")
	}
	if !isInvalidMILCompileErr("invalidmilprogram") {
		t.Fatalf("case-insensitive InvalidMILProgram string not detected")
	}
	if isInvalidMILCompileErr("compileModel failed: unknown error") {
		t.Fatalf("non InvalidMILProgram string incorrectly detected")
	}
}

func TestBuildCompileFallbackProfiles(t *testing.T) {
	t.Setenv("ANE_COMPILE_FALLBACK_PROFILES", "")
	if got := buildCompileFallbackProfiles(CompileOptions{}); got != nil {
		t.Fatalf("buildCompileFallbackProfiles(empty)=%v want nil", got)
	}

	t.Setenv("ANE_COMPILE_FALLBACK_PROFILES", "kANEFModelMIL:fallback.plist")
	got := buildCompileFallbackProfiles(CompileOptions{ModelType: "kANEFModelMIL", NetPlistFilename: "base.plist"})
	want := []compileProfile{
		{},
		{modelType: "kANEFModelMIL", netPlist: "fallback.plist"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildCompileFallbackProfiles(non-empty primary)=%v want %v", got, want)
	}

	t.Setenv("ANE_COMPILE_FALLBACK_PROFILES", "kANEFModelMIL:base.plist")
	got = buildCompileFallbackProfiles(CompileOptions{ModelType: "kANEFModelMIL", NetPlistFilename: "base.plist"})
	want = []compileProfile{{}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildCompileFallbackProfiles(skip duplicate primary)=%v want %v", got, want)
	}
}
