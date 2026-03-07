//go:build !darwin

package bridge

import (
	"fmt"
	"unsafe"
)

// LoadOptions configures bridge dylib resolution.
type LoadOptions struct {
	LibraryPath string
}

// Runtime is unavailable on non-darwin platforms.
type Runtime struct{}

// Load always fails on non-darwin platforms.
func Load(LoadOptions) (*Runtime, error) {
	return nil, fmt.Errorf("ane bridge runtime is only supported on darwin")
}

func (r *Runtime) openClientHandle(string, string, uint32, uint32) uintptr { return 0 }
func (r *Runtime) closeClientHandle(uintptr)                               {}
func (r *Runtime) evalClientHandle(uintptr) bool                           { return false }
func (r *Runtime) writeClientInput(uintptr, unsafe.Pointer, int32)         {}
func (r *Runtime) readClientOutput(uintptr, unsafe.Pointer, int32)         {}
func (r *Runtime) createSharedEventObject() uintptr                        { return 0 }
func (r *Runtime) sharedEventObjectPort(uintptr) uint32                    { return 0 }
func (r *Runtime) releaseObjcObject(uintptr)                               {}
func (r *Runtime) evalWithSignalEventHandle(uintptr, unsafe.Pointer, uint32, unsafe.Pointer, uint32, uint32, uint64) int32 {
	return -1
}
func (r *Runtime) evalBidirectionalHandle(uintptr, unsafe.Pointer, uint32, unsafe.Pointer, uint32, uint32, uint64, uint32, uint64) int32 {
	return -1
}
func (r *Runtime) hasStoriesTrainer() bool { return false }
func (r *Runtime) storiesOpenHandle(string, string, string, uint32, uint32, uint32, float32, bool, uint32) uintptr {
	return 0
}
func (r *Runtime) storiesStepHandle(uintptr, unsafe.Pointer) int32 { return -1 }
func (r *Runtime) storiesSaveCheckpoint(uintptr, string) int32     { return -1 }
func (r *Runtime) storiesLoadCheckpoint(uintptr, string) int32     { return -1 }
func (r *Runtime) storiesCloseHandle(uintptr)                      {}
func (r *Runtime) storiesLastErrString() string                    { return "" }

// HasSignalEventCPU reports whether CPU event signaling is available.
func (r *Runtime) HasSignalEventCPU() bool { return false }

// HasWaitEventCPU reports whether CPU event waiting is available.
func (r *Runtime) HasWaitEventCPU() bool { return false }

// SignalEventCPU is unsupported on non-darwin platforms.
func (r *Runtime) SignalEventCPU(uint32, uint64) error {
	return fmt.Errorf("signal event: ane bridge runtime is only supported on darwin")
}

// WaitEventCPU is unsupported on non-darwin platforms.
func (r *Runtime) WaitEventCPU(uint32, uint64, uint32) (bool, error) {
	return false, fmt.Errorf("wait event: ane bridge runtime is only supported on darwin")
}
