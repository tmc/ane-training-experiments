//go:build darwin

package model

import "testing"

func TestSharedMILProgramReleaseRetainsCacheEntry(t *testing.T) {
	sharedMILCache.mu.Lock()
	savedM := sharedMILCache.m
	savedP := sharedMILCache.p
	sharedMILCache.m = make(map[string]*sharedMILProgram)
	sharedMILCache.p = make(map[string]*sharedMILPending)
	sharedMILCache.mu.Unlock()
	t.Cleanup(func() {
		sharedMILCache.mu.Lock()
		sharedMILCache.m = savedM
		sharedMILCache.p = savedP
		sharedMILCache.mu.Unlock()
	})

	prog := &sharedMILProgram{key: "k", refs: 1}

	sharedMILCache.mu.Lock()
	sharedMILCache.m[prog.key] = prog
	sharedMILCache.mu.Unlock()

	prog.release()

	sharedMILCache.mu.Lock()
	defer sharedMILCache.mu.Unlock()
	if prog.refs != 0 {
		t.Fatalf("refs=%d, want 0", prog.refs)
	}
	if sharedMILCache.m[prog.key] != prog {
		t.Fatal("shared MIL program was evicted from cache")
	}
}
