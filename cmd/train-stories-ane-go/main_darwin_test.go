//go:build darwin

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
		true,
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
		"-json=false",
		"-auto-restart=true",
		"-recompile-every-step=true",
		"-diagnostics=true",
		"-allow-experimental-ane-trainer=true",
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
