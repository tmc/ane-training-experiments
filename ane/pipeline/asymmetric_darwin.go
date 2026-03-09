//go:build darwin

// Package pipeline provides supported Metal->ANE asymmetric synchronization helpers.
package pipeline

import (
	"fmt"
	"time"

	"github.com/maderix/ANE/ane/model"
	"github.com/maderix/ANE/internal/evalbuffer"
	"github.com/maderix/ANE/internal/kernelio"
	xane "github.com/tmc/apple/x/ane"
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
	k           *model.Kernel
	waitEvent   *xane.SharedEvent
	signalEvent *xane.SharedEvent
	timeoutMS   uint64
	buf         *evalbuffer.Buffers
}

// Open creates a new asymmetric pipeline runner.
func Open(opts Options) (*Runner, error) {
	if opts.ModelPath == "" {
		return nil, fmt.Errorf("pipeline open: model path is empty")
	}
	if opts.InputBytes == 0 || opts.OutputBytes == 0 {
		return nil, fmt.Errorf("pipeline open: input and output bytes must be > 0")
	}
	modelKey := opts.ModelKey
	if modelKey == "" {
		modelKey = "s"
	}
	k, err := model.Compile(model.CompileOptions{
		PackagePath: opts.ModelPath,
		ModelKey:    modelKey,
	})
	if err != nil {
		return nil, fmt.Errorf("pipeline open: compile kernel: %w", err)
	}
	if opts.WaitTimeoutMS == 0 {
		opts.WaitTimeoutMS = 5000
	}
	waitEvent, err := xane.NewSharedEvent()
	if err != nil {
		k.Close()
		return nil, fmt.Errorf("pipeline open: create wait event: %w", err)
	}
	signalEvent, err := xane.NewSharedEvent()
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
		buf:         evalbuffer.New(int(opts.InputBytes), int(opts.OutputBytes)),
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
	r.buf = nil
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
	r.waitEvent.Signal(value)
	return nil
}

// WaitForSignal waits for the ANE completion signal event to reach value.
func (r *Runner) WaitForSignal(value uint64, timeout time.Duration) (bool, error) {
	if r == nil || r.signalEvent == nil {
		return false, fmt.Errorf("wait for signal: runner is closed")
	}
	if timeout == 0 {
		timeout = time.Duration(r.timeoutMS) * time.Millisecond
	}
	return r.signalEvent.Wait(value, timeout), nil
}

// Eval dispatches one wait+signal evaluation.
func (r *Runner) Eval(waitValue, signalValue uint64, input, output []float32) error {
	if r == nil || r.k == nil || r.waitEvent == nil || r.signalEvent == nil {
		return fmt.Errorf("pipeline eval: runner is closed")
	}
	if err := r.buf.StageF32(input); err != nil {
		return fmt.Errorf("pipeline eval: %w", err)
	}
	if err := kernelio.WriteF32(r.k, r.buf, "pipeline eval"); err != nil {
		return err
	}
	if err := r.k.EvalBidirectional(
		r.waitEvent.Port(),
		waitValue,
		r.signalEvent.Port(),
		signalValue,
		xane.SharedEventEvalOptions{
			DisableIOFencesUseSharedEvents: true,
			EnableFWToFWSignal:             false,
		},
	); err != nil {
		return fmt.Errorf("pipeline eval: eval bidirectional: %w", err)
	}
	return kernelio.ReadF32(r.k, r.buf, output, "pipeline eval")
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
