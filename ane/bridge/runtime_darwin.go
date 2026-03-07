//go:build darwin

package bridge

import (
	"fmt"
	"os"
	"path/filepath"
	goruntime "runtime"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/apple/iosurface"
)

const defaultBridgeLibraryName = "libane_bridge.dylib"

// LoadOptions configures bridge dylib resolution.
type LoadOptions struct {
	// LibraryPath forces an explicit dylib path when non-empty.
	LibraryPath string
}

// Runtime holds bound bridge entry points.
type Runtime struct {
	lib uintptr

	init       func() int32
	open       func(string, string, uintptr, uintptr) uintptr
	close      func(uintptr)
	writeInput func(uintptr, unsafe.Pointer, int32)
	readOutput func(uintptr, unsafe.Pointer, int32)
	eval       func(uintptr) bool

	createSharedEvent func() uintptr
	sharedEventPort   func(uintptr) uint32
	releaseObjc       func(uintptr)

	evalWithSignalEvent func(uintptr, unsafe.Pointer, uint32, unsafe.Pointer, uint32, uint32, uint64) int32
	evalBidirectional   func(uintptr, unsafe.Pointer, uint32, unsafe.Pointer, uint32, uint32, uint64, uint32, uint64) int32
	signalEventCPU      func(uint32, uint64) int32
	waitEventCPU        func(uint32, uint64, uint32) int32

	storiesOpen      func(string, string, string, uintptr, uintptr, uint32, float32, bool, uint32) uintptr
	storiesStep      func(uintptr, unsafe.Pointer) int32
	storiesSaveCkpt  func(uintptr, string) int32
	storiesLoadCkpt  func(uintptr, string) int32
	storiesClose     func(uintptr)
	storiesLastError func(unsafe.Pointer, uintptr) int32
}

// Load resolves and opens libane_bridge.dylib, then binds supported symbols.
func Load(opts LoadOptions) (*Runtime, error) {
	path, err := resolveLibraryPath(opts)
	if err != nil {
		return nil, err
	}
	lib, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("load bridge dylib %q: %w", path, err)
	}

	rt := &Runtime{lib: lib}
	if err := bindRequired(rt, lib); err != nil {
		return nil, err
	}
	bindOptional(rt, lib)

	if rc := rt.init(); rc != 0 {
		return nil, fmt.Errorf("ane_bridge_init failed: rc=%d", rc)
	}
	return rt, nil
}

func bindRequired(rt *Runtime, lib uintptr) error {
	for _, symbol := range []string{
		"ane_bridge_init",
		"ane_bridge_client_open",
		"ane_bridge_client_close",
		"ane_bridge_client_write_input",
		"ane_bridge_client_read_output",
		"ane_bridge_client_eval",
	} {
		if _, err := purego.Dlsym(lib, symbol); err != nil {
			return fmt.Errorf("bridge symbol %q not found", symbol)
		}
	}
	purego.RegisterLibFunc(&rt.init, lib, "ane_bridge_init")
	purego.RegisterLibFunc(&rt.open, lib, "ane_bridge_client_open")
	purego.RegisterLibFunc(&rt.close, lib, "ane_bridge_client_close")
	purego.RegisterLibFunc(&rt.writeInput, lib, "ane_bridge_client_write_input")
	purego.RegisterLibFunc(&rt.readOutput, lib, "ane_bridge_client_read_output")
	purego.RegisterLibFunc(&rt.eval, lib, "ane_bridge_client_eval")
	return nil
}

func bindOptional(rt *Runtime, lib uintptr) {
	registerOptional(lib, "ane_bridge_create_shared_event", &rt.createSharedEvent)
	registerOptional(lib, "ane_bridge_shared_event_port", &rt.sharedEventPort)
	registerOptional(lib, "ane_bridge_release_objc", &rt.releaseObjc)
	registerOptional(lib, "ane_bridge_eval_with_signal_event", &rt.evalWithSignalEvent)
	registerOptional(lib, "ane_bridge_eval_bidirectional", &rt.evalBidirectional)
	registerOptional(lib, "ane_bridge_signal_event_cpu", &rt.signalEventCPU)
	registerOptional(lib, "ane_bridge_wait_event_cpu", &rt.waitEventCPU)
	registerOptional(lib, "ane_bridge_stories_open", &rt.storiesOpen)
	registerOptional(lib, "ane_bridge_stories_step", &rt.storiesStep)
	registerOptional(lib, "ane_bridge_stories_save_checkpoint", &rt.storiesSaveCkpt)
	registerOptional(lib, "ane_bridge_stories_load_checkpoint", &rt.storiesLoadCkpt)
	registerOptional(lib, "ane_bridge_stories_close", &rt.storiesClose)
	registerOptional(lib, "ane_bridge_stories_last_error", &rt.storiesLastError)
}

func registerOptional[T any](lib uintptr, name string, target *T) {
	if _, err := purego.Dlsym(lib, name); err == nil {
		purego.RegisterLibFunc(target, lib, name)
	}
}

func resolveLibraryPath(opts LoadOptions) (string, error) {
	if opts.LibraryPath != "" {
		p, err := filepath.Abs(opts.LibraryPath)
		if err != nil {
			return "", fmt.Errorf("resolve bridge path %q: %w", opts.LibraryPath, err)
		}
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("bridge dylib not found at %q: %w", p, err)
		}
		return p, nil
	}

	candidates := make([]string, 0, 6)
	if env := os.Getenv("ANE_BRIDGE_DYLIB"); env != "" {
		candidates = append(candidates, env)
	}
	if repoPath, ok := repoBridgePath(); ok {
		candidates = append(candidates, repoPath)
	}
	candidates = append(candidates,
		filepath.Join("bridge", defaultBridgeLibraryName),
		filepath.Join("..", "bridge", defaultBridgeLibraryName),
		defaultBridgeLibraryName,
	)
	for _, c := range candidates {
		p, err := filepath.Abs(c)
		if err != nil {
			continue
		}
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf(
		"bridge dylib not found; set ANE_BRIDGE_DYLIB or place %s under ./bridge or ../bridge",
		defaultBridgeLibraryName,
	)
}

func repoBridgePath() (string, bool) {
	_, file, _, ok := goruntime.Caller(0)
	if !ok || file == "" {
		return "", false
	}
	repoRoot := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	return filepath.Join(repoRoot, "bridge", defaultBridgeLibraryName), true
}

// OpenClient opens a model-backed bridge client handle.
func (r *Runtime) openClientHandle(modelPath, modelKey string, inputBytes, outputBytes uint32) uintptr {
	if r == nil || r.open == nil {
		return 0
	}
	return r.open(modelPath, modelKey, uintptr(inputBytes), uintptr(outputBytes))
}

// CloseClient closes a bridge client handle.
func (r *Runtime) closeClientHandle(handle uintptr) {
	if r == nil || r.close == nil || handle == 0 {
		return
	}
	r.close(handle)
}

// EvalClient runs baseline evaluation with no shared events.
func (r *Runtime) evalClientHandle(handle uintptr) bool {
	if r == nil || r.eval == nil || handle == 0 {
		return false
	}
	return r.eval(handle)
}

func (r *Runtime) writeClientInput(handle uintptr, ptr unsafe.Pointer, count int32) {
	if r == nil || r.writeInput == nil || handle == 0 || ptr == nil || count < 0 {
		return
	}
	r.writeInput(handle, ptr, count)
}

func (r *Runtime) readClientOutput(handle uintptr, ptr unsafe.Pointer, count int32) {
	if r == nil || r.readOutput == nil || handle == 0 || ptr == nil || count < 0 {
		return
	}
	r.readOutput(handle, ptr, count)
}

func (r *Runtime) createSharedEventObject() uintptr {
	if r == nil || r.createSharedEvent == nil {
		return 0
	}
	return r.createSharedEvent()
}

func (r *Runtime) sharedEventObjectPort(obj uintptr) uint32 {
	if r == nil || r.sharedEventPort == nil || obj == 0 {
		return 0
	}
	return r.sharedEventPort(obj)
}

func (r *Runtime) releaseObjcObject(obj uintptr) {
	if r == nil || r.releaseObjc == nil || obj == 0 {
		return
	}
	r.releaseObjc(obj)
}

func (r *Runtime) evalWithSignalEventHandle(handle uintptr, inputPtr unsafe.Pointer, inputCount uint32, outputPtr unsafe.Pointer, outputCount uint32, signalPort uint32, signalValue uint64) int32 {
	if r == nil || r.evalWithSignalEvent == nil || handle == 0 {
		return -1
	}
	return r.evalWithSignalEvent(handle, inputPtr, inputCount, outputPtr, outputCount, signalPort, signalValue)
}

func (r *Runtime) evalBidirectionalHandle(handle uintptr, inputPtr unsafe.Pointer, inputCount uint32, outputPtr unsafe.Pointer, outputCount uint32, waitPort uint32, waitValue uint64, signalPort uint32, signalValue uint64) int32 {
	if r == nil || r.evalBidirectional == nil || handle == 0 {
		return -1
	}
	return r.evalBidirectional(handle, inputPtr, inputCount, outputPtr, outputCount, waitPort, waitValue, signalPort, signalValue)
}

func (r *Runtime) hasStoriesTrainer() bool {
	return r != nil &&
		r.storiesOpen != nil &&
		r.storiesStep != nil &&
		r.storiesSaveCkpt != nil &&
		r.storiesLoadCkpt != nil &&
		r.storiesClose != nil
}

func (r *Runtime) storiesOpenHandle(modelPath, modelKey, dataPath string, inputBytes, outputBytes uint32, totalSteps uint32, lr float32, aneExtras bool, compileBudget uint32) uintptr {
	if !r.hasStoriesTrainer() {
		return 0
	}
	return r.storiesOpen(modelPath, modelKey, dataPath, uintptr(inputBytes), uintptr(outputBytes), totalSteps, lr, aneExtras, compileBudget)
}

func (r *Runtime) storiesStepHandle(handle uintptr, statsPtr unsafe.Pointer) int32 {
	if r == nil || r.storiesStep == nil || handle == 0 {
		return -1
	}
	return r.storiesStep(handle, statsPtr)
}

func (r *Runtime) storiesSaveCheckpoint(handle uintptr, path string) int32 {
	if r == nil || r.storiesSaveCkpt == nil || handle == 0 {
		return -1
	}
	return r.storiesSaveCkpt(handle, path)
}

func (r *Runtime) storiesLoadCheckpoint(handle uintptr, path string) int32 {
	if r == nil || r.storiesLoadCkpt == nil || handle == 0 {
		return -1
	}
	return r.storiesLoadCkpt(handle, path)
}

func (r *Runtime) storiesCloseHandle(handle uintptr) {
	if r == nil || r.storiesClose == nil || handle == 0 {
		return
	}
	r.storiesClose(handle)
}

func (r *Runtime) storiesLastErrString() string {
	if r == nil || r.storiesLastError == nil {
		return ""
	}
	buf := make([]byte, 512)
	rc := r.storiesLastError(unsafe.Pointer(&buf[0]), uintptr(len(buf)))
	if rc <= 0 {
		return ""
	}
	n := 0
	for n < len(buf) && buf[n] != 0 {
		n++
	}
	return string(buf[:n])
}

// HasSignalEventCPU reports whether ane_bridge_signal_event_cpu is available.
func (r *Runtime) HasSignalEventCPU() bool {
	return r != nil && r.signalEventCPU != nil
}

// HasWaitEventCPU reports whether ane_bridge_wait_event_cpu is available.
func (r *Runtime) HasWaitEventCPU() bool {
	return r != nil && r.waitEventCPU != nil
}

// SignalEventCPU increments an IOSurfaceSharedEvent signaled value.
//
// It uses bridge C entry points when available, otherwise falls back to the
// iosurface Go binding.
func (r *Runtime) SignalEventCPU(eventPort uint32, value uint64) error {
	if eventPort == 0 {
		return fmt.Errorf("signal event: event port is zero")
	}
	if r != nil && r.signalEventCPU != nil {
		if rc := r.signalEventCPU(eventPort, value); rc != 0 {
			return fmt.Errorf("signal event: ane_bridge_signal_event_cpu rc=%d", rc)
		}
		return nil
	}
	ev := iosurface.NewIOSurfaceSharedEventWithMachPort(eventPort)
	if ev.GetID() == 0 {
		return fmt.Errorf("signal event: failed to bind shared event for port %d", eventPort)
	}
	defer ev.Release()
	ev.SetSignaledValue(value)
	return nil
}

// WaitEventCPU waits for IOSurfaceSharedEvent.signaledValue to reach value.
//
// It uses bridge C entry points when available, otherwise falls back to the
// iosurface Go binding wait API.
func (r *Runtime) WaitEventCPU(eventPort uint32, value uint64, timeoutMS uint32) (bool, error) {
	if eventPort == 0 {
		return false, fmt.Errorf("wait event: event port is zero")
	}
	if r != nil && r.waitEventCPU != nil {
		rc := r.waitEventCPU(eventPort, value, timeoutMS)
		switch rc {
		case 0:
			return true, nil
		case 1:
			return false, nil
		default:
			return false, fmt.Errorf("wait event: ane_bridge_wait_event_cpu rc=%d", rc)
		}
	}
	ev := iosurface.NewIOSurfaceSharedEventWithMachPort(eventPort)
	if ev.GetID() == 0 {
		return false, fmt.Errorf("wait event: failed to bind shared event for port %d", eventPort)
	}
	defer ev.Release()
	return ev.WaitUntilSignaledValueTimeoutMS(value, uint64(timeoutMS)), nil
}
