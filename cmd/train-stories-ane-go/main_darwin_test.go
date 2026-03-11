//go:build darwin

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/maderix/ANE/ane/storiestrainer"
)

func TestRestartCountFromEnv(t *testing.T) {
	t.Setenv("ANE_TRAIN_RESTART_COUNT", "")
	if got := restartCountFromEnv(); got != 0 {
		t.Fatalf("restartCountFromEnv()=%d want 0", got)
	}
	t.Setenv("ANE_TRAIN_RESTART_COUNT", "7")
	if got := restartCountFromEnv(); got != 7 {
		t.Fatalf("restartCountFromEnv()=%d want 7", got)
	}
	t.Setenv("ANE_TRAIN_RESTART_COUNT", "-4")
	if got := restartCountFromEnv(); got != 0 {
		t.Fatalf("restartCountFromEnv()=%d want 0 for negative input", got)
	}
	t.Setenv("ANE_TRAIN_RESTART_COUNT", "bad")
	if got := restartCountFromEnv(); got != 0 {
		t.Fatalf("restartCountFromEnv()=%d want 0 for invalid input", got)
	}
}

func TestBuildRestartArgs(t *testing.T) {
	args := buildRestartArgs(
		"/tmp/train-stories-ane-go",
		"/tmp/model.mlmodelc",
		"s",
		"/tmp/data.bin",
		"/tmp/ckpt.bin",
		"ane",
		100,
		3e-4,
		true,
		true,
		true,
		false,
		10,
		5,
		true,
		100,
		false,
		true,
		8,
		4096,
		8192,
		true,
		true,
		true,
		"direct",
		"/tmp/full-train",
		80,
		6,
		3,
		"/tmp/dyn-train",
		384,
		true,
		2048,
	)
	joined := strings.Join(args, " ")
	mustContain := []string{
		"-resume",
		"-model /tmp/model.mlmodelc",
		"-model-key s",
		"-data /tmp/data.bin",
		"-ckpt /tmp/ckpt.bin",
		"-backend ane",
		"-steps 100",
		"-max-restarts 8",
		"-input-bytes 4096",
		"-output-bytes 8192",
		"-no-ane-extras",
		"-save-final",
		"-ane-cls-bwd=true",
		"-ane-hybrid-bwd=true",
		"-json=false",
		"-auto-restart=true",
		"-recompile-every-step=true",
		"-diagnostics=true",
		"-trainer-backend direct",
		"-parity-mode=true",
		"-full-bin /tmp/full-train",
		"-full-accum-steps 80",
		"-veclib-threads 6",
		"-dw-concurrency 3",
		"-dynamic-bin /tmp/dyn-train",
		"-seq-override 384",
		"-dynamic-ane-cls=true",
		"-dynamic-ane-cls-tile 2048",
	}
	for _, token := range mustContain {
		if !strings.Contains(joined, token) {
			t.Fatalf("args missing %q in %q", token, joined)
		}
	}
}

func TestRestartSelfResolveExecutable(t *testing.T) {
	// Smoke-test that restartSelf can at least resolve executable.
	// We intentionally do not attempt syscall.Exec in tests.
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	if exe == "" {
		t.Fatal("os.Executable returned empty path")
	}
}

func TestDefaultIfEmpty(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		fallback string
		want     string
	}{
		{name: "value", input: "x", fallback: "y", want: "x"},
		{name: "empty", input: "", fallback: "y", want: "y"},
		{name: "spaces", input: "   ", fallback: "y", want: "y"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := defaultIfEmpty(tt.input, tt.fallback); got != tt.want {
				t.Fatalf("defaultIfEmpty(%q,%q)=%q want %q", tt.input, tt.fallback, got, tt.want)
			}
		})
	}
}

func TestDefaultBackendIsANE(t *testing.T) {
	if defaultBackend != "ane" {
		t.Fatalf("defaultBackend=%q want %q", defaultBackend, "ane")
	}
}

func TestIsStoriesBinModel(t *testing.T) {
	for _, tt := range []struct {
		path string
		want bool
	}{
		{path: "stories110M.bin", want: true},
		{path: "STORIES110M.BIN", want: true},
		{path: "model.mlmodelc", want: false},
	} {
		if got := isStoriesBinModel(tt.path); got != tt.want {
			t.Fatalf("isStoriesBinModel(%q)=%v want %v", tt.path, got, tt.want)
		}
	}
}

func TestFlagProvided(t *testing.T) {
	args := []string{
		"-backend", "ane",
		"-ane-hybrid-bwd=false",
		"-steps=2",
	}
	if !flagProvided(args, "ane-hybrid-bwd") {
		t.Fatal("flagProvided(ane-hybrid-bwd)=false want true")
	}
	if !flagProvided(args, "steps") {
		t.Fatal("flagProvided(steps)=false want true")
	}
	if flagProvided(args, "missing") {
		t.Fatal("flagProvided(missing)=true want false")
	}
}

func TestShouldAutoEnableANEHybridBackward(t *testing.T) {
	tests := []struct {
		name            string
		args            []string
		selectedBackend string
		model           string
		noANEExtras     bool
		hybridBackward  bool
		want            bool
	}{
		{
			name:            "auto enable direct bin",
			args:            []string{"-backend", "ane"},
			selectedBackend: "ane",
			model:           "stories110M.bin",
			want:            true,
		},
		{
			name:            "explicit hybrid flag disables auto",
			args:            []string{"-backend", "ane", "-ane-hybrid-bwd=false"},
			selectedBackend: "ane",
			model:           "stories110M.bin",
			want:            false,
		},
		{
			name:            "already enabled",
			args:            []string{"-backend", "ane"},
			selectedBackend: "ane",
			model:           "stories110M.bin",
			hybridBackward:  true,
			want:            false,
		},
		{
			name:            "extras disabled",
			args:            []string{"-backend", "ane"},
			selectedBackend: "ane",
			model:           "stories110M.bin",
			noANEExtras:     true,
			want:            false,
		},
		{
			name:            "wrapper backend",
			args:            []string{"-backend", "full"},
			selectedBackend: "full",
			model:           "stories110M.bin",
			want:            false,
		},
		{
			name:            "modelc path",
			args:            []string{"-backend", "ane"},
			selectedBackend: "ane",
			model:           "model.mlmodelc",
			want:            false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldAutoEnableANEHybridBackward(tt.args, tt.selectedBackend, tt.model, tt.noANEExtras, tt.hybridBackward)
			if got != tt.want {
				t.Fatalf("shouldAutoEnableANEHybridBackward()=%v want %v", got, tt.want)
			}
		})
	}
}

func TestShouldAutoBridgeStoriesBinToFull(t *testing.T) {
	tests := []struct {
		name            string
		selectedBackend string
		trainerBackend  string
		model           string
		probeErr        error
		want            bool
	}{
		{
			name:            "auto bridge unsupported direct compile",
			selectedBackend: "ane",
			trainerBackend:  storiestrainer.BackendAuto,
			model:           "stories110M.bin",
			probeErr:        fmt.Errorf("compile failed"),
			want:            true,
		},
		{
			name:            "direct backend stays direct",
			selectedBackend: "ane",
			trainerBackend:  storiestrainer.BackendDirect,
			model:           "stories110M.bin",
			probeErr:        fmt.Errorf("compile failed"),
			want:            false,
		},
		{
			name:            "supported direct compile does not bridge",
			selectedBackend: "ane",
			trainerBackend:  storiestrainer.BackendAuto,
			model:           "stories110M.bin",
			want:            false,
		},
		{
			name:            "modelc does not bridge",
			selectedBackend: "ane",
			trainerBackend:  storiestrainer.BackendAuto,
			model:           "model.mlmodelc",
			probeErr:        fmt.Errorf("compile failed"),
			want:            false,
		},
		{
			name:            "full backend does not bridge",
			selectedBackend: "full",
			trainerBackend:  storiestrainer.BackendAuto,
			model:           "stories110M.bin",
			probeErr:        fmt.Errorf("compile failed"),
			want:            false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldAutoBridgeStoriesBinToFull(tt.selectedBackend, tt.trainerBackend, tt.model, tt.probeErr)
			if got != tt.want {
				t.Fatalf("shouldAutoBridgeStoriesBinToFull()=%v want %v", got, tt.want)
			}
		})
	}
}

func TestProbeDirectStoriesSequence(t *testing.T) {
	prev := probeDirectStoriesSequenceFunc
	probeDirectStoriesSequenceFunc = func(seq int) error {
		if seq == 384 {
			return fmt.Errorf("compile failed")
		}
		return nil
	}
	defer func() {
		probeDirectStoriesSequenceFunc = prev
	}()

	if err := probeDirectStoriesSequence("ane", "stories110M.bin", 256); err != nil {
		t.Fatalf("probeDirectStoriesSequence(default support): %v", err)
	}
	if err := probeDirectStoriesSequence("ane", "stories110M.bin", 384); err == nil {
		t.Fatal("probeDirectStoriesSequence(384)=nil want error")
	}
	if err := probeDirectStoriesSequence("full", "stories110M.bin", 384); err != nil {
		t.Fatalf("probeDirectStoriesSequence(non-ane): %v", err)
	}
	if err := probeDirectStoriesSequence("ane", "model.mlmodelc", 384); err != nil {
		t.Fatalf("probeDirectStoriesSequence(modelc): %v", err)
	}
}

func TestValidateDirectStoriesBackend(t *testing.T) {
	err := validateDirectStoriesBackend("ane", storiestrainer.BackendDirect, "stories110M.bin", 384, fmt.Errorf("compile failed"))
	if err == nil {
		t.Fatal("validateDirectStoriesBackend()=nil want error")
	}
	if !strings.Contains(err.Error(), "trainer-backend=direct") {
		t.Fatalf("validateDirectStoriesBackend()=%q missing direct backend context", err)
	}
	if err := validateDirectStoriesBackend("ane", storiestrainer.BackendAuto, "stories110M.bin", 384, fmt.Errorf("compile failed")); err != nil {
		t.Fatalf("validateDirectStoriesBackend(auto): %v", err)
	}
}

func TestResolveSelectedBackend(t *testing.T) {
	if got := resolveSelectedBackend("auto", "stories110M.bin"); got != "ane" {
		t.Fatalf("resolveSelectedBackend(auto, .bin)=%q want ane", got)
	}
	if got := resolveSelectedBackend("auto", "model.mlmodelc"); got != "ane" {
		t.Fatalf("resolveSelectedBackend(auto, .mlmodelc)=%q want ane", got)
	}
	if got := resolveSelectedBackend("full", "stories110M.bin"); got != "full" {
		t.Fatalf("resolveSelectedBackend(full)=%q want full", got)
	}
}

func TestEffectiveGoImplBackend(t *testing.T) {
	if got := effectiveGoImplBackend("ane", "stories110M.bin", storiestrainer.Diagnostics{}); got != "storiesane" {
		t.Fatalf("effectiveGoImplBackend(bin)=%q want storiesane", got)
	}
	if got := effectiveGoImplBackend("ane", "stories110M.bin", storiestrainer.Diagnostics{HybridBackwardEnabled: true}); got != "storiesane+hybrid-bwd" {
		t.Fatalf("effectiveGoImplBackend(bin hybrid)=%q want storiesane+hybrid-bwd", got)
	}
	if got := effectiveGoImplBackend("ane", "model.mlmodelc", storiestrainer.Diagnostics{}); got != "direct_modelc" {
		t.Fatalf("effectiveGoImplBackend(modelc)=%q want direct_modelc", got)
	}
}

func TestRunFullTrainingExplicitBinary(t *testing.T) {
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "full.args")
	bin := filepath.Join(dir, "full_runner.sh")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" >" + argsFile + "\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatalf("write script: %v", err)
	}
	if err := runFullTraining(fullRunOptions{
		binPath:       bin,
		modelPath:     "/tmp/model.bin",
		dataPath:      "/tmp/data.bin",
		ckptPath:      "/tmp/ckpt.bin",
		resume:        true,
		steps:         20,
		lr:            3e-4,
		accumSteps:    80,
		veclibThreads: 6,
		dwConcurrency: 3,
		noANEExtras:   true,
		aneClsBwd:     true,
	}); err != nil {
		t.Fatalf("runFullTraining: %v", err)
	}
	b, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("read args file: %v", err)
	}
	got := string(b)
	for _, want := range []string{
		"--model",
		"/tmp/model.bin",
		"--data",
		"/tmp/data.bin",
		"--steps",
		"20",
		"--accum",
		"80",
		"--veclib-threads",
		"6",
		"--dw-concurrency",
		"3",
		"--resume",
		"--no-ane-extras",
		"--ane-cls-bwd",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("full args missing %q in %q", want, got)
		}
	}
}

func TestRunDynamicTrainingExplicitBinary(t *testing.T) {
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "dyn.args")
	bin := filepath.Join(dir, "dyn_runner.sh")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" >" + argsFile + "\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatalf("write script: %v", err)
	}
	if err := runDynamicTraining(dynamicRunOptions{
		binPath:       bin,
		modelPath:     "/tmp/model.bin",
		dataPath:      "/tmp/data.bin",
		ckptPath:      "/tmp/ckpt.bin",
		resume:        true,
		steps:         12,
		lr:            1e-4,
		accumSteps:    20,
		aneClassifier: true,
		aneClsTile:    2048,
	}); err != nil {
		t.Fatalf("runDynamicTraining: %v", err)
	}
	b, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("read args file: %v", err)
	}
	got := string(b)
	for _, want := range []string{
		"--model",
		"/tmp/model.bin",
		"--data",
		"/tmp/data.bin",
		"--steps",
		"12",
		"--accum",
		"20",
		"--ane-cls",
		"--ane-cls-tile",
		"2048",
		"--resume",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("dynamic args missing %q in %q", want, got)
		}
	}
}

func TestResolveBinaryValidation(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "tool")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatalf("write tool: %v", err)
	}

	got, err := resolveFullTrainerBinary(bin, 0)
	if err != nil {
		t.Fatalf("resolveFullTrainerBinary: %v", err)
	}
	if got != bin {
		t.Fatalf("resolveFullTrainerBinary=%q want %q", got, bin)
	}

	if _, err := resolveFullTrainerBinary(bin, 256); err == nil {
		t.Fatalf("resolveFullTrainerBinary expected error for seq override with explicit bin")
	}

	got, err = resolveDynamicTrainerBinary(bin, 0)
	if err != nil {
		t.Fatalf("resolveDynamicTrainerBinary: %v", err)
	}
	if got != bin {
		t.Fatalf("resolveDynamicTrainerBinary=%q want %q", got, bin)
	}

	if _, err := resolveDynamicTrainerBinary(bin, 256); err == nil {
		t.Fatalf("resolveDynamicTrainerBinary expected error for seq override with explicit bin")
	}
}
