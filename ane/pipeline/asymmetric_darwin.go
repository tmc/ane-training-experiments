//go:build darwin

// Package pipeline provides supported Metal->ANE asymmetric synchronization helpers.
package pipeline

import (
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"github.com/maderix/ANE/ane/clientmodel"
)

// Options configures an asymmetric pipeline runner.
type Options struct {
	ModelPath     string
	ModelKey      string
	InputBytes    uint32
	OutputBytes   uint32
	WaitTimeoutMS uint32
}

// Runner manages one model client and two shared events.
type Runner struct {
	k           *clientmodel.Kernel
	waitEvent   *clientmodel.SharedEvent
	signalEvent *clientmodel.SharedEvent
	timeoutMS   uint64
	inputBytes  []byte
	outputBytes []byte
}

// Open creates a new asymmetric pipeline runner.
func Open(opts Options) (*Runner, error) {
	if opts.ModelPath == "" {
		return nil, fmt.Errorf("pipeline open: model path is empty")
	}
	if opts.InputBytes == 0 || opts.OutputBytes == 0 {
		return nil, fmt.Errorf("pipeline open: input and output bytes must be > 0")
	}
	if opts.ModelKey == "" {
		opts.ModelKey = "s"
	}
	if opts.WaitTimeoutMS == 0 {
		opts.WaitTimeoutMS = 5000
	}

	k, err := clientmodel.Compile(clientmodel.CompileOptions{
		CompiledModelPath: opts.ModelPath,
		ModelKey:          opts.ModelKey,
		InputBytes:        []int{int(opts.InputBytes)},
		OutputBytes:       []int{int(opts.OutputBytes)},
	})
	if err != nil {
		return nil, fmt.Errorf("pipeline open: compile client kernel: %w", err)
	}
	waitEvent, err := clientmodel.NewSharedEvent()
	if err != nil {
		k.Close()
		return nil, fmt.Errorf("pipeline open: create wait event: %w", err)
	}
	signalEvent, err := clientmodel.NewSharedEvent()
	if err != nil {
		_ = waitEvent.Close()
		k.Close()
		return nil, fmt.Errorf("pipeline open: create signal event: %w", err)
	}

	return &Runner{
		k:           k,
		waitEvent:   waitEvent,
		signalEvent: signalEvent,
		timeoutMS:   uint64(opts.WaitTimeoutMS),
		inputBytes:  make([]byte, opts.InputBytes),
		outputBytes: make([]byte, opts.OutputBytes),
	}, nil
}

// Close releases client and event resources.
func (r *Runner) Close() error {
	if r == nil {
		return nil
	}
	if r.signalEvent != nil {
		_ = r.signalEvent.Close()
		r.signalEvent = nil
	}
	if r.waitEvent != nil {
		_ = r.waitEvent.Close()
		r.waitEvent = nil
	}
	if r.k != nil {
		r.k.Close()
		r.k = nil
	}
	r.inputBytes = nil
	r.outputBytes = nil
	return nil
}

// WaitPort returns the Metal->ANE wait event port.
func (r *Runner) WaitPort() uint32 {
	if r == nil || r.waitEvent == nil {
		return 0
	}
	return r.waitEvent.Port()
}

// SignalPort returns the ANE completion signal event port.
func (r *Runner) SignalPort() uint32 {
	if r == nil || r.signalEvent == nil {
		return 0
	}
	return r.signalEvent.Port()
}

// SignalWaitFromCPU increments the wait event from CPU.
func (r *Runner) SignalWaitFromCPU(value uint64) error {
	if r == nil || r.waitEvent == nil {
		return fmt.Errorf("signal wait event: runner is closed")
	}
	return r.waitEvent.Signal(value)
}

// WaitForSignal waits for the ANE completion signal event to reach value.
func (r *Runner) WaitForSignal(value uint64, timeout time.Duration) (bool, error) {
	if r == nil || r.signalEvent == nil {
		return false, fmt.Errorf("wait for signal: runner is closed")
	}
	if timeout == 0 {
		timeout = time.Duration(r.timeoutMS) * time.Millisecond
	}
	return r.signalEvent.Wait(value, timeout)
}

// Eval dispatches one wait+signal evaluation.
//
// This uses the direct clientmodel bidirectional shared-event path.
func (r *Runner) Eval(waitValue, signalValue uint64, input, output []float32) error {
	if r == nil || r.k == nil || r.waitEvent == nil || r.signalEvent == nil {
		return fmt.Errorf("pipeline eval: runner is closed")
	}
	if len(input)*4 > len(r.inputBytes) {
		return fmt.Errorf("pipeline eval: input is %d bytes, want <= %d", len(input)*4, len(r.inputBytes))
	}
	if len(output)*4 > len(r.outputBytes) {
		return fmt.Errorf("pipeline eval: output is %d bytes, want <= %d", len(output)*4, len(r.outputBytes))
	}
	for i, v := range input {
		binary.LittleEndian.PutUint32(r.inputBytes[i*4:], math.Float32bits(v))
	}
	if err := r.k.WriteInput(0, r.inputBytes); err != nil {
		return fmt.Errorf("pipeline eval: write input: %w", err)
	}
	if err := r.k.EvalBidirectional(
		r.waitEvent.Port(),
		waitValue,
		r.signalEvent.Port(),
		signalValue,
		clientmodel.SharedEventEvalOptions{
			DisableIOFencesUseSharedEvents: true,
			EnableFWToFWSignal:             false,
		},
	); err != nil {
		return fmt.Errorf("pipeline eval: eval bidirectional: %w", err)
	}
	if err := r.k.ReadOutput(0, r.outputBytes); err != nil {
		return fmt.Errorf("pipeline eval: read output: %w", err)
	}
	for i := range output {
		output[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.outputBytes[i*4:]))
	}
	return nil
}

// StepCPUOnly is a convenience step for CPU-driven event signaling.
func (r *Runner) StepCPUOnly(waitValue, signalValue uint64, input, output []float32) error {
	if err := r.SignalWaitFromCPU(waitValue); err != nil {
		return fmt.Errorf("pipeline step: %w", err)
	}
	if err := r.Eval(waitValue, signalValue, input, output); err != nil {
		return fmt.Errorf("pipeline step: eval: %w", err)
	}
	ok, err := r.WaitForSignal(signalValue, 0)
	if err != nil {
		return fmt.Errorf("pipeline step: wait signal: %w", err)
	}
	if !ok {
		return fmt.Errorf("pipeline step: wait signal timed out")
	}
	return nil
}
