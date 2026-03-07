//go:build darwin

package ane

import (
	"fmt"

	"github.com/maderix/ANE/ane/pipeline"
)

// Evaluator provides a high-level model eval entrypoint.
type Evaluator struct {
	r *pipeline.EvalRunner
}

// OpenEvaluator opens a model evaluator with optional Espresso-backed I/O.
func OpenEvaluator(opts EvalOptions) (*Evaluator, error) {
	r, err := pipeline.OpenEval(pipeline.EvalOptions{
		ModelPath:        opts.ModelPath,
		ModelPackagePath: opts.ModelPackagePath,
		ModelKey:         opts.ModelKey,
		ModelType:        opts.ModelType,
		NetPlistFilename: opts.NetPlistFilename,
		QoS:              opts.QoS,
		InputBytes:       opts.InputBytes,
		OutputBytes:      opts.OutputBytes,
		UseEspressoIO:    opts.UseEspressoIO,
		EspressoFrames:   opts.EspressoFrames,
	})
	if err != nil {
		return nil, err
	}
	return &Evaluator{r: r}, nil
}

// Close releases evaluator resources.
func (e *Evaluator) Close() error {
	if e == nil || e.r == nil {
		return nil
	}
	err := e.r.Close()
	e.r = nil
	return err
}

// EspressoEnabled reports whether Espresso I/O mode is active.
func (e *Evaluator) EspressoEnabled() bool {
	return e != nil && e.r != nil && e.r.EspressoEnabled()
}

// EvalBytes runs one evaluation with byte input/output.
func (e *Evaluator) EvalBytes(input, output []byte) error {
	if e == nil || e.r == nil {
		return fmt.Errorf("ane evaluator is closed")
	}
	return e.r.EvalBytes(input, output)
}

// EvalF32 runs one evaluation with float32 input/output.
func (e *Evaluator) EvalF32(input, output []float32) error {
	if e == nil || e.r == nil {
		return fmt.Errorf("ane evaluator is closed")
	}
	return e.r.EvalF32(input, output)
}
