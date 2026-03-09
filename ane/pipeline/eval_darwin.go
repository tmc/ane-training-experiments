//go:build darwin

package pipeline

import (
	"fmt"

	"github.com/maderix/ANE/ane/model"
	"github.com/maderix/ANE/internal/evalbuffer"
	"github.com/maderix/ANE/internal/kernelio"
)

// EvalOptions configures a synchronous eval runner.
type EvalOptions struct {
	ModelPath        string
	ModelPackagePath string
	ModelKey         string
	QoS              uint32
	InputBytes       uint32
	OutputBytes      uint32
}

// EvalRunner is a high-level entrypoint for model eval.
type EvalRunner struct {
	k   evalKernel
	buf *evalbuffer.Buffers
}

type evalKernel interface {
	kernelio.Kernel
	Eval() error
	Close()
}

// OpenEval creates an eval runner.
func OpenEval(opts EvalOptions) (*EvalRunner, error) {
	k, buf, err := openModelEval(opts)
	if err != nil {
		return nil, err
	}
	return &EvalRunner{k: k, buf: buf}, nil
}

// Close releases resources.
func (r *EvalRunner) Close() error {
	if r == nil {
		return nil
	}
	if r.k != nil {
		r.k.Close()
		r.k = nil
	}
	r.buf = nil
	return nil
}

// EvalBytes runs one evaluation with byte buffers.
func (r *EvalRunner) EvalBytes(input, output []byte) error {
	if r == nil || r.k == nil {
		return fmt.Errorf("eval bytes: runner is closed")
	}
	if err := r.buf.StageBytes(input); err != nil {
		return fmt.Errorf("eval bytes: %w", err)
	}
	if err := r.writeBytesInput(); err != nil {
		return err
	}
	if err := r.eval("eval bytes"); err != nil {
		return err
	}
	if err := r.readBytesOutput(); err != nil {
		return err
	}
	return r.buf.CopyBytes(output)
}

// EvalF32 runs one evaluation with float32 input/output.
func (r *EvalRunner) EvalF32(input, output []float32) error {
	if r == nil || r.k == nil {
		return fmt.Errorf("eval f32: runner is closed")
	}
	if err := r.buf.StageF32(input); err != nil {
		return fmt.Errorf("eval f32: %w", err)
	}
	if err := kernelio.WriteF32(r.k, r.buf, "eval f32"); err != nil {
		return err
	}
	if err := r.eval("eval f32"); err != nil {
		return err
	}
	return kernelio.ReadF32(r.k, r.buf, output, "eval f32")
}

func openModelEval(opts EvalOptions) (*model.Kernel, *evalbuffer.Buffers, error) {
	modelPath := opts.ModelPackagePath
	if modelPath == "" {
		modelPath = opts.ModelPath
	}
	if modelPath == "" {
		return nil, nil, fmt.Errorf("open eval: model path is empty")
	}
	modelKey := opts.ModelKey
	if modelKey == "" {
		modelKey = "s"
	}
	k, err := model.Compile(model.CompileOptions{
		PackagePath: modelPath,
		ModelKey:    modelKey,
		QoS:         opts.QoS,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("open eval: compile kernel: %w", err)
	}
	if k.NumInputs() != 1 || k.NumOutputs() != 1 {
		k.Close()
		return nil, nil, fmt.Errorf("open eval: compiled model reported %d inputs and %d outputs; want 1 input and 1 output", k.NumInputs(), k.NumOutputs())
	}
	if err := validateBufferSize("input", int(opts.InputBytes), k.InputBytes(0)); err != nil {
		k.Close()
		return nil, nil, fmt.Errorf("open eval: %w", err)
	}
	if err := validateBufferSize("output", int(opts.OutputBytes), k.OutputBytes(0)); err != nil {
		k.Close()
		return nil, nil, fmt.Errorf("open eval: %w", err)
	}
	return k, evalbuffer.New(k.InputBytes(0), k.OutputBytes(0)), nil
}

func validateBufferSize(name string, got, want int) error {
	if got == 0 || got == want {
		return nil
	}
	return fmt.Errorf("%s bytes = %d, want %d", name, got, want)
}

func (r *EvalRunner) writeBytesInput() error {
	return kernelio.WriteBytes(r.k, r.buf, "eval bytes")
}

func (r *EvalRunner) readBytesOutput() error {
	return kernelio.ReadBytes(r.k, r.buf, "eval bytes")
}

func (r *EvalRunner) eval(op string) error {
	if err := r.k.Eval(); err != nil {
		return fmt.Errorf("%s: eval: %w", op, err)
	}
	return nil
}
