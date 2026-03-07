//go:build darwin

package pipeline

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/maderix/ANE/ane/clientmodel"
	"github.com/maderix/ANE/ane/espressoio"
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
	k          *clientmodel.Kernel
	inputBuf   []byte
	outputBuf  []byte
	espresso   bool
	inPool     *espressoio.Pool
	outPool    *espressoio.Pool
	frameCount uint64
	frameIndex uint64
}

// OpenEval creates an eval runner.
func OpenEval(opts EvalOptions) (*EvalRunner, error) {
	if opts.ModelPath == "" && opts.ModelPackagePath == "" {
		return nil, fmt.Errorf("open eval: model path is empty")
	}
	if opts.InputBytes == 0 || opts.OutputBytes == 0 {
		return nil, fmt.Errorf("open eval: input and output bytes must be > 0")
	}
	if opts.ModelKey == "" {
		opts.ModelKey = "s"
	}
	if opts.EspressoFrames == 0 {
		opts.EspressoFrames = 1
	}

	k, err := clientmodel.Compile(clientmodel.CompileOptions{
		CompiledModelPath: opts.ModelPath,
		ModelPackagePath:  opts.ModelPackagePath,
		ModelKey:          opts.ModelKey,
		ModelType:         opts.ModelType,
		NetPlistFilename:  opts.NetPlistFilename,
		QoS:               opts.QoS,
		InputBytes:        []int{int(opts.InputBytes)},
		OutputBytes:       []int{int(opts.OutputBytes)},
	})
	if err != nil {
		return nil, fmt.Errorf("open eval: compile client kernel: %w", err)
	}

	r := &EvalRunner{
		k:          k,
		inputBuf:   make([]byte, opts.InputBytes),
		outputBuf:  make([]byte, opts.OutputBytes),
		espresso:   opts.UseEspressoIO,
		frameCount: opts.EspressoFrames,
	}
	if !opts.UseEspressoIO {
		return r, nil
	}

	inRef, err := k.InputSurfaceRef(0)
	if err != nil {
		r.Close()
		return nil, fmt.Errorf("open eval: input surface ref: %w", err)
	}
	outRef, err := k.OutputSurfaceRef(0)
	if err != nil {
		r.Close()
		return nil, fmt.Errorf("open eval: output surface ref: %w", err)
	}

	r.inPool, err = espressoio.Open(int(opts.InputBytes), opts.EspressoFrames)
	if err != nil {
		r.Close()
		return nil, fmt.Errorf("open eval: create input pool: %w", err)
	}
	r.outPool, err = espressoio.Open(int(opts.OutputBytes), opts.EspressoFrames)
	if err != nil {
		r.Close()
		return nil, fmt.Errorf("open eval: create output pool: %w", err)
	}
	for i := range opts.EspressoFrames {
		if err := r.inPool.SetExternalFrameStorage(i, inRef); err != nil {
			r.Close()
			return nil, fmt.Errorf("open eval: bind input frame %d: %w", i, err)
		}
		if err := r.outPool.SetExternalFrameStorage(i, outRef); err != nil {
			r.Close()
			return nil, fmt.Errorf("open eval: bind output frame %d: %w", i, err)
		}
	}
	return r, nil
}

// Close releases resources.
func (r *EvalRunner) Close() error {
	if r == nil {
		return nil
	}
	if r.inPool != nil {
		r.inPool.Close()
		r.inPool = nil
	}
	if r.outPool != nil {
		r.outPool.Close()
		r.outPool = nil
	}
	if r.k != nil {
		r.k.Close()
		r.k = nil
	}
	r.inputBuf = nil
	r.outputBuf = nil
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
	if len(input) > len(r.inputBuf) {
		return fmt.Errorf("eval bytes: input is %d bytes, want <= %d", len(input), len(r.inputBuf))
	}
	if len(output) > len(r.outputBuf) {
		return fmt.Errorf("eval bytes: output is %d bytes, want <= %d", len(output), len(r.outputBuf))
	}

	copy(r.inputBuf, input)
	if len(input) < len(r.inputBuf) {
		zero(r.inputBuf[len(input):])
	}

	if r.espresso {
		frame := r.nextFrame()
		if err := r.inPool.WriteFrame(frame, r.inputBuf); err != nil {
			return fmt.Errorf("eval bytes: espresso write input frame=%d: %w", frame, err)
		}
	} else {
		if err := r.k.WriteInput(0, r.inputBuf); err != nil {
			return fmt.Errorf("eval bytes: write input: %w", err)
		}
	}

	if err := r.k.Eval(); err != nil {
		return fmt.Errorf("eval bytes: eval: %w", err)
	}

	if r.espresso {
		frame := r.currentFrame()
		if err := r.outPool.ReadFrame(frame, r.outputBuf); err != nil {
			return fmt.Errorf("eval bytes: espresso read output frame=%d: %w", frame, err)
		}
	} else {
		if err := r.k.ReadOutput(0, r.outputBuf); err != nil {
			return fmt.Errorf("eval bytes: read output: %w", err)
		}
	}

	copy(output, r.outputBuf[:len(output)])
	return nil
}

// EvalF32 runs one evaluation with float32 input/output.
func (r *EvalRunner) EvalF32(input, output []float32) error {
	if len(input)*4 > len(r.inputBuf) {
		return fmt.Errorf("eval f32: input is %d bytes, want <= %d", len(input)*4, len(r.inputBuf))
	}
	if len(output)*4 > len(r.outputBuf) {
		return fmt.Errorf("eval f32: output is %d bytes, want <= %d", len(output)*4, len(r.outputBuf))
	}
	for i, v := range input {
		binary.LittleEndian.PutUint32(r.inputBuf[i*4:], math.Float32bits(v))
	}
	if err := r.EvalBytes(r.inputBuf[:len(input)*4], r.outputBuf[:len(output)*4]); err != nil {
		return err
	}
	for i := range output {
		output[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.outputBuf[i*4:]))
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

func zero(b []byte) {
	for i := range b {
		b[i] = 0
	}
}
