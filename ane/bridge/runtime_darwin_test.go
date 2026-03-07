//go:build darwin

package bridge

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveLibraryPathExplicit(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, defaultBridgeLibraryName)
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatalf("write bridge dylib: %v", err)
	}
	got, err := resolveLibraryPath(LoadOptions{LibraryPath: path})
	if err != nil {
		t.Fatalf("resolve library path: %v", err)
	}
	want, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("filepath.Abs: %v", err)
	}
	if got != want {
		t.Fatalf("resolve library path = %q, want %q", got, want)
	}
}

func TestResolveLibraryPathEnv(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, defaultBridgeLibraryName)
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatalf("write bridge dylib: %v", err)
	}

	old := os.Getenv("ANE_BRIDGE_DYLIB")
	t.Cleanup(func() { _ = os.Setenv("ANE_BRIDGE_DYLIB", old) })
	if err := os.Setenv("ANE_BRIDGE_DYLIB", path); err != nil {
		t.Fatalf("set ANE_BRIDGE_DYLIB: %v", err)
	}

	got, err := resolveLibraryPath(LoadOptions{})
	if err != nil {
		t.Fatalf("resolve library path: %v", err)
	}
	want, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("filepath.Abs: %v", err)
	}
	if got != want {
		t.Fatalf("resolve library path = %q, want %q", got, want)
	}
}

func TestSignalEventCPUZeroPort(t *testing.T) {
	t.Parallel()
	var rt Runtime
	if err := rt.SignalEventCPU(0, 1); err == nil {
		t.Fatalf("SignalEventCPU(0, 1) succeeded; want error")
	}
}

func TestWaitEventCPUZeroPort(t *testing.T) {
	t.Parallel()
	var rt Runtime
	ok, err := rt.WaitEventCPU(0, 1, 1)
	if err == nil {
		t.Fatalf("WaitEventCPU(0, 1, 1) err=nil, ok=%v; want error", ok)
	}
}
