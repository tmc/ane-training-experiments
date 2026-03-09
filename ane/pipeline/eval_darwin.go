//go:build darwin

package pipeline

import (
	"fmt"

	"github.com/maderix/ANE/ane/clientmodel"
	"github.com/maderix/ANE/ane/model"
	"github.com/maderix/ANE/internal/clientkernel"
	"github.com/maderix/ANE/internal/espressosurface"
	"github.com/maderix/ANE/internal/evalbuffer"
	"github.com/maderix/ANE/internal/kernelio"
	"github.com/tmc/apple/coregraphics"
	xespresso "github.com/tmc/apple/x/espresso"
)

// EvalOptions configures a synchronous eval runner.
type EvalOptions struct {
	ModelPath        string
	ModelPackagePath string
	ModelKey         string
	ModelType        string
	NetPlistFilename string
	QoS              uint32
	InputBytes       uint32
	OutputBytes      uint32
	UseEspressoIO    bool
	EspressoFrames   uint64
}

// EvalRunner is a high-level entrypoint for model eval with optional Espresso-backed I/O.
type EvalRunner struct {
	k          evalKernel
	buf        *evalbuffer.Buffers
	espresso   bool
	inPool     *xespresso.ANESurface
	outPool    *xespresso.ANESurface
	frameCount uint64
	frameIndex uint64
}

type evalKernel interface {
	kernelio.Kernel
	Eval() error
	Close()
}

// OpenEval creates an eval runner.
func OpenEval(opts EvalOptions) (*EvalRunner, error) {
	if opts.EspressoFrames == 0 {
		opts.EspressoFrames = 1
	}
	if useClientEvalPath(opts) {
		k, buf, err := openClientEval(opts)
		if err != nil {
			return nil, err
		}
		r := &EvalRunner{
			k:          k,
			buf:        buf,
			espresso:   opts.UseEspressoIO,
			frameCount: opts.EspressoFrames,
		}
		if !opts.UseEspressoIO {
			return r, nil
		}
		if err := r.openEspresso(k, buf); err != nil {
			r.Close()
			return nil, err
		}
		return r, nil
	}
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
	if r.inPool != nil {
		r.inPool.Cleanup()
		r.inPool = nil
	}
	if r.outPool != nil {
		r.outPool.Cleanup()
		r.outPool = nil
	}
	if r.k != nil {
		r.k.Close()
		r.k = nil
	}
	r.buf = nil
	return nil
}

// EspressoEnabled reports whether Espresso I/O mode is active.
func (r *EvalRunner) EspressoEnabled() bool {
	return r != nil && r.espresso
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
	if !r.espresso {
		if err := kernelio.WriteF32(r.k, r.buf, "eval f32"); err != nil {
			return err
		}
		if err := r.eval("eval f32"); err != nil {
			return err
		}
		return kernelio.ReadF32(r.k, r.buf, output, "eval f32")
	}
	if r.buf.TypedF32() {
		if err := r.writeF32Input(); err != nil {
			return err
		}
		if err := r.eval("eval f32"); err != nil {
			return err
		}
		if err := r.readF32Output(); err != nil {
			return err
		}
		return r.buf.DecodeF32(output)
	}
	if err := r.writeBytesInput(); err != nil {
		return err
	}
	if err := r.eval("eval f32"); err != nil {
		return err
	}
	if err := r.readBytesOutput(); err != nil {
		return err
	}
	if err := r.buf.DecodeF32(output); err != nil {
		return fmt.Errorf("eval f32: %w", err)
	}
	return nil
}

func useClientEvalPath(opts EvalOptions) bool {
	return opts.UseEspressoIO || opts.ModelType != "" || opts.NetPlistFilename != ""
}

func openClientEval(opts EvalOptions) (*clientmodel.Kernel, *evalbuffer.Buffers, error) {
	cfg := clientkernel.EvalOptions{
		ModelPath:        opts.ModelPath,
		ModelPackagePath: opts.ModelPackagePath,
		ModelKey:         opts.ModelKey,
		ModelType:        opts.ModelType,
		NetPlistFilename: opts.NetPlistFilename,
		QoS:              opts.QoS,
		InputBytes:       opts.InputBytes,
		OutputBytes:      opts.OutputBytes,
	}
	cfg = clientkernel.WithDefaults(cfg)
	if err := clientkernel.Validate(cfg); err != nil {
		return nil, nil, fmt.Errorf("open eval: %w", err)
	}
	k, err := clientkernel.Compile(cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("open eval: compile client kernel: %w", err)
	}
	return k, evalbuffer.New(int(cfg.InputBytes), int(cfg.OutputBytes)), nil
}

func openModelEval(opts EvalOptions) (*model.Kernel, *evalbuffer.Buffers, error) {
	modelPath := opts.ModelPackagePath
	if modelPath == "" {
		modelPath = opts.ModelPath
	}
	if modelPath == "" {
		return nil, nil, fmt.Errorf("open eval: model path is empty")
	}
	cfg := clientkernel.WithDefaults(clientkernel.EvalOptions{
		ModelPath:        opts.ModelPath,
		ModelPackagePath: opts.ModelPackagePath,
		ModelKey:         opts.ModelKey,
		QoS:              opts.QoS,
	})
	k, err := model.Compile(model.CompileOptions{
		PackagePath: modelPath,
		ModelKey:    cfg.ModelKey,
		QoS:         cfg.QoS,
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

func (r *EvalRunner) openEspresso(k *clientmodel.Kernel, buf *evalbuffer.Buffers) error {
	inRef, err := k.InputSurfaceRef(0)
	if err != nil {
		return fmt.Errorf("open eval: input surface ref: %w", err)
	}
	outRef, err := k.OutputSurfaceRef(0)
	if err != nil {
		return fmt.Errorf("open eval: output surface ref: %w", err)
	}

	r.inPool, err = espressosurface.Open(len(buf.InputBytesScratch()), r.frameCount)
	if err != nil {
		return fmt.Errorf("open eval: create input pool: %w", err)
	}
	r.outPool, err = espressosurface.Open(len(buf.OutputBytesScratch()), r.frameCount)
	if err != nil {
		return fmt.Errorf("open eval: create output pool: %w", err)
	}
	for i := range r.frameCount {
		r.inPool.SetExternalStorage(i, coregraphics.IOSurfaceRef(inRef))
		if got, err := r.inPool.IOSurfaceForFrame(i); err != nil || uintptr(got) != inRef {
			if err != nil {
				return fmt.Errorf("open eval: bind input frame %d: %w", i, err)
			}
			return fmt.Errorf("open eval: bind input frame %d: surface mismatch", i)
		}
		r.outPool.SetExternalStorage(i, coregraphics.IOSurfaceRef(outRef))
		if got, err := r.outPool.IOSurfaceForFrame(i); err != nil || uintptr(got) != outRef {
			if err != nil {
				return fmt.Errorf("open eval: bind output frame %d: %w", i, err)
			}
			return fmt.Errorf("open eval: bind output frame %d: surface mismatch", i)
		}
	}
	return nil
}

func (r *EvalRunner) writeBytesInput() error {
	if r.espresso {
		frame := r.nextFrame()
		if err := r.inPool.WriteFrame(frame, r.buf.InputBytesScratch()); err != nil {
			return fmt.Errorf("eval bytes: espresso write input frame=%d: %w", frame, err)
		}
		return nil
	}
	return kernelio.WriteBytes(r.k, r.buf, "eval bytes")
}

func (r *EvalRunner) readBytesOutput() error {
	if r.espresso {
		frame := r.currentFrame()
		if err := r.outPool.ReadFrame(frame, r.buf.OutputBytesScratch()); err != nil {
			return fmt.Errorf("eval bytes: espresso read output frame=%d: %w", frame, err)
		}
		return nil
	}
	return kernelio.ReadBytes(r.k, r.buf, "eval bytes")
}

func (r *EvalRunner) writeF32Input() error {
	if r.espresso {
		frame := r.nextFrame()
		if err := r.inPool.WriteFrameF32(frame, r.buf.InputF32Scratch()); err != nil {
			return fmt.Errorf("eval f32: espresso write input frame=%d: %w", frame, err)
		}
		return nil
	}
	return kernelio.WriteF32(r.k, r.buf, "eval f32")
}

func (r *EvalRunner) readF32Output() error {
	if r.espresso {
		frame := r.currentFrame()
		if err := r.outPool.ReadFrameF32(frame, r.buf.OutputF32Scratch()); err != nil {
			return fmt.Errorf("eval f32: espresso read output frame=%d: %w", frame, err)
		}
		return nil
	}
	return nil
}

func (r *EvalRunner) eval(op string) error {
	if err := r.k.Eval(); err != nil {
		return fmt.Errorf("%s: eval: %w", op, err)
	}
	return nil
}

func (r *EvalRunner) nextFrame() uint64 {
	if r.frameCount == 0 {
		return 0
	}
	f := r.frameIndex % r.frameCount
	r.frameIndex++
	return f
}

func (r *EvalRunner) currentFrame() uint64 {
	if r.frameCount == 0 {
		return 0
	}
	return (r.frameIndex - 1) % r.frameCount
}
