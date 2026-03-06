package bridge

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// Client wraps a daemon-backed ANE bridge handle.
type Client struct {
	rt          *Runtime
	handle      uintptr
	inputBytes  uint32
	outputBytes uint32
}

// SharedEvent wraps a retained IOSurfaceSharedEvent object and its Mach port.
type SharedEvent struct {
	Port uint32

	rt  *Runtime
	obj uintptr
}

// StoriesTrainerOptions controls bridge-backed trainer initialization.
type StoriesTrainerOptions struct {
	ModelPath     string
	ModelKey      string
	DataPath      string
	InputBytes    uint32
	OutputBytes   uint32
	TotalSteps    uint32
	LR            float32
	ANEExtras     bool
	CompileBudget uint32
}

// StoriesStepStats mirrors ANEStoriesStepStats from the bridge C API.
type StoriesStepStats struct {
	Step            uint32
	Loss            float32
	StepMS          float64
	Compiles        uint32
	RestartRequired bool
}

// StoriesTrainer wraps a bridge C trainer handle.
type StoriesTrainer struct {
	rt     *Runtime
	handle uintptr
}

// OpenClient opens a typed ANE bridge client.
func (r *Runtime) OpenClient(modelPath, modelKey string, inputBytes, outputBytes uint32) (*Client, error) {
	if r == nil {
		return nil, fmt.Errorf("open client: runtime is nil")
	}
	if modelPath == "" {
		return nil, fmt.Errorf("open client: model path is empty")
	}
	if inputBytes == 0 || outputBytes == 0 {
		return nil, fmt.Errorf("open client: input and output bytes must be > 0")
	}
	handle := r.openClientHandle(modelPath, modelKey, inputBytes, outputBytes)
	if handle == 0 {
		return nil, fmt.Errorf("open client: bridge returned nil handle")
	}
	return &Client{
		rt:          r,
		handle:      handle,
		inputBytes:  inputBytes,
		outputBytes: outputBytes,
	}, nil
}

// Close releases the underlying bridge client handle.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	if c.handle != 0 && c.rt != nil {
		c.rt.closeClientHandle(c.handle)
	}
	c.handle = 0
	return nil
}

// Eval runs a baseline ANE eval with no shared events.
func (c *Client) Eval() error {
	if c == nil || c.rt == nil || c.handle == 0 {
		return fmt.Errorf("eval: client is closed")
	}
	if ok := c.rt.evalClientHandle(c.handle); !ok {
		return fmt.Errorf("eval: bridge eval failed")
	}
	return nil
}

// WriteInputF32 writes float32 data to the mapped input surface.
func (c *Client) WriteInputF32(src []float32) error {
	if c == nil || c.rt == nil || c.handle == 0 {
		return fmt.Errorf("write input: client is closed")
	}
	if len(src) == 0 {
		return fmt.Errorf("write input: got 0 bytes, want %d", c.inputBytes)
	}
	n := uint32(len(src) * 4)
	if n != c.inputBytes {
		return fmt.Errorf("write input: got %d bytes, want %d", n, c.inputBytes)
	}
	c.rt.writeClientInput(c.handle, unsafe.Pointer(&src[0]), int32(len(src)))
	return nil
}

// ReadOutputF32 reads float32 data from the mapped output surface.
func (c *Client) ReadOutputF32(dst []float32) error {
	if c == nil || c.rt == nil || c.handle == 0 {
		return fmt.Errorf("read output: client is closed")
	}
	if len(dst) == 0 {
		return fmt.Errorf("read output: got 0 bytes, want %d", c.outputBytes)
	}
	n := uint32(len(dst) * 4)
	if n != c.outputBytes {
		return fmt.Errorf("read output: got %d bytes, want %d", n, c.outputBytes)
	}
	c.rt.readClientOutput(c.handle, unsafe.Pointer(&dst[0]), int32(len(dst)))
	return nil
}

// EvalWithSignalEvent executes with ANE->Metal signal semantics (FW_SIGNAL=0 path).
func (c *Client) EvalWithSignalEvent(signalPort uint32, signalValue uint64, input []float32, output []float32) error {
	if c == nil || c.rt == nil || c.handle == 0 {
		return fmt.Errorf("eval with signal event: client is closed")
	}
	if signalPort == 0 {
		return fmt.Errorf("eval with signal event: signal port is zero")
	}
	if uint32(len(input))*4 > c.inputBytes {
		return fmt.Errorf("eval with signal event: input is %d bytes, want <= %d", len(input)*4, c.inputBytes)
	}
	if uint32(len(output))*4 > c.outputBytes {
		return fmt.Errorf("eval with signal event: output is %d bytes, want <= %d", len(output)*4, c.outputBytes)
	}
	var inPtr unsafe.Pointer
	if len(input) > 0 {
		inPtr = unsafe.Pointer(&input[0])
	}
	var outPtr unsafe.Pointer
	if len(output) > 0 {
		outPtr = unsafe.Pointer(&output[0])
	}
	rc := c.rt.evalWithSignalEventHandle(
		c.handle,
		inPtr,
		uint32(len(input)),
		outPtr,
		uint32(len(output)),
		signalPort,
		signalValue,
	)
	if rc != 0 {
		return fmt.Errorf("eval with signal event: bridge rc=%d", rc)
	}
	return nil
}

// EvalBidirectional executes with wait+signal shared events in one request.
func (c *Client) EvalBidirectional(waitPort uint32, waitValue uint64, signalPort uint32, signalValue uint64, input []float32, output []float32) error {
	if c == nil || c.rt == nil || c.handle == 0 {
		return fmt.Errorf("eval bidirectional: client is closed")
	}
	if waitPort == 0 || signalPort == 0 {
		return fmt.Errorf("eval bidirectional: wait and signal ports must be non-zero")
	}
	if uint32(len(input))*4 > c.inputBytes {
		return fmt.Errorf("eval bidirectional: input is %d bytes, want <= %d", len(input)*4, c.inputBytes)
	}
	if uint32(len(output))*4 > c.outputBytes {
		return fmt.Errorf("eval bidirectional: output is %d bytes, want <= %d", len(output)*4, c.outputBytes)
	}
	var inPtr unsafe.Pointer
	if len(input) > 0 {
		inPtr = unsafe.Pointer(&input[0])
	}
	var outPtr unsafe.Pointer
	if len(output) > 0 {
		outPtr = unsafe.Pointer(&output[0])
	}
	rc := c.rt.evalBidirectionalHandle(
		c.handle,
		inPtr,
		uint32(len(input)),
		outPtr,
		uint32(len(output)),
		waitPort,
		waitValue,
		signalPort,
		signalValue,
	)
	if rc != 0 {
		return fmt.Errorf("eval bidirectional: bridge rc=%d", rc)
	}
	return nil
}

// NewSharedEvent allocates a retained IOSurfaceSharedEvent and returns its port.
func (r *Runtime) NewSharedEvent() (*SharedEvent, error) {
	if r == nil {
		return nil, fmt.Errorf("new shared event: runtime is nil")
	}
	obj := r.createSharedEventObject()
	if obj == 0 {
		return nil, fmt.Errorf("new shared event: bridge returned nil object")
	}
	port := r.sharedEventObjectPort(obj)
	if port == 0 {
		r.releaseObjcObject(obj)
		return nil, fmt.Errorf("new shared event: event port is zero")
	}
	return &SharedEvent{
		Port: port,
		rt:   r,
		obj:  obj,
	}, nil
}

// Close releases the retained Objective-C shared event object.
func (e *SharedEvent) Close() error {
	if e == nil {
		return nil
	}
	if e.obj != 0 && e.rt != nil {
		e.rt.releaseObjcObject(e.obj)
	}
	e.obj = 0
	e.Port = 0
	return nil
}

// OpenStoriesTrainer opens a bridge-backed trainer handle.
func (r *Runtime) OpenStoriesTrainer(opts StoriesTrainerOptions) (*StoriesTrainer, error) {
	if r == nil {
		return nil, fmt.Errorf("open stories trainer: runtime is nil")
	}
	if !r.hasStoriesTrainer() {
		return nil, fmt.Errorf("open stories trainer: stories trainer symbols are unavailable")
	}
	if opts.ModelPath == "" {
		return nil, fmt.Errorf("open stories trainer: model path is empty")
	}
	if opts.InputBytes == 0 || opts.OutputBytes == 0 {
		return nil, fmt.Errorf("open stories trainer: input and output bytes must be > 0")
	}
	handle := r.storiesOpenHandle(
		opts.ModelPath,
		opts.ModelKey,
		opts.DataPath,
		opts.InputBytes,
		opts.OutputBytes,
		opts.TotalSteps,
		opts.LR,
		opts.ANEExtras,
		opts.CompileBudget,
	)
	if handle == 0 {
		msg := r.storiesLastErrString()
		if msg == "" {
			msg = "bridge returned nil handle"
		}
		return nil, fmt.Errorf("open stories trainer: %s", msg)
	}
	return &StoriesTrainer{rt: r, handle: handle}, nil
}

// Close releases the underlying stories trainer handle.
func (t *StoriesTrainer) Close() error {
	if t == nil {
		return nil
	}
	if t.rt != nil && t.handle != 0 {
		t.rt.storiesCloseHandle(t.handle)
	}
	t.handle = 0
	return nil
}

// Step advances one trainer step.
func (t *StoriesTrainer) Step() (StoriesStepStats, error) {
	var st StoriesStepStats
	if t == nil || t.rt == nil || t.handle == 0 {
		return st, fmt.Errorf("stories step: trainer is closed")
	}
	var raw struct {
		Step            uint32
		Loss            float32
		StepMS          float64
		Compiles        uint32
		RestartRequired uint32
	}
	rc := t.rt.storiesStepHandle(t.handle, unsafe.Pointer(&raw))
	if rc != 0 {
		msg := t.rt.storiesLastErrString()
		if msg == "" {
			if rc == 1 {
				msg = "trainer finished"
			} else {
				msg = fmt.Sprintf("bridge rc=%d", rc)
			}
		}
		return st, fmt.Errorf("stories step: %s", msg)
	}
	st = StoriesStepStats{
		Step:            raw.Step,
		Loss:            raw.Loss,
		StepMS:          raw.StepMS,
		Compiles:        raw.Compiles,
		RestartRequired: raw.RestartRequired != 0,
	}
	return st, nil
}

// SaveCheckpoint stores trainer state.
func (t *StoriesTrainer) SaveCheckpoint(path string) error {
	if t == nil || t.rt == nil || t.handle == 0 {
		return fmt.Errorf("save checkpoint: trainer is closed")
	}
	if path == "" {
		return fmt.Errorf("save checkpoint: path is empty")
	}
	rc := t.rt.storiesSaveCheckpoint(t.handle, path)
	if rc != 0 {
		msg := t.rt.storiesLastErrString()
		if msg == "" {
			msg = fmt.Sprintf("bridge rc=%d", rc)
		}
		return fmt.Errorf("save checkpoint: %s", msg)
	}
	return nil
}

// LoadCheckpoint restores trainer state.
func (t *StoriesTrainer) LoadCheckpoint(path string) error {
	if t == nil || t.rt == nil || t.handle == 0 {
		return fmt.Errorf("load checkpoint: trainer is closed")
	}
	if path == "" {
		return fmt.Errorf("load checkpoint: path is empty")
	}
	rc := t.rt.storiesLoadCheckpoint(t.handle, path)
	if rc != 0 {
		msg := t.rt.storiesLastErrString()
		if msg == "" {
			msg = fmt.Sprintf("bridge rc=%d", rc)
		}
		return fmt.Errorf("load checkpoint: %s", msg)
	}
	return nil
}

// F32ToBytes converts float32 values to little-endian byte storage.
func F32ToBytes(src []float32) []byte {
	out := make([]byte, len(src)*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// BytesToF32 converts little-endian bytes into float32 values.
func BytesToF32(src []byte) []float32 {
	if len(src)%4 != 0 {
		return nil
	}
	out := make([]float32, len(src)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
	return out
}
