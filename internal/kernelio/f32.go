package kernelio

import (
	"fmt"

	"github.com/maderix/ANE/internal/evalbuffer"
)

// Kernel is the shared ANE kernel contract used by the direct model and
// pipeline execution paths.
type Kernel interface {
	WriteInput(int, []byte) error
	ReadOutput(int, []byte) error
	WriteInputF32(int, []float32) error
	ReadOutputF32(int, []float32) error
}

// WriteBytes writes the staged byte input.
func WriteBytes(k Kernel, buf *evalbuffer.Buffers, op string) error {
	if err := k.WriteInput(0, buf.InputBytesScratch()); err != nil {
		return fmt.Errorf("%s: write input: %w", op, err)
	}
	return nil
}

// ReadBytes reads output into the staged byte buffer.
func ReadBytes(k Kernel, buf *evalbuffer.Buffers, op string) error {
	if err := k.ReadOutput(0, buf.OutputBytesScratch()); err != nil {
		return fmt.Errorf("%s: read output: %w", op, err)
	}
	return nil
}

// WriteF32 writes the staged float32 input using typed I/O when available and
// falls back to byte I/O otherwise.
func WriteF32(k Kernel, buf *evalbuffer.Buffers, op string) error {
	if buf.TypedF32() {
		if err := k.WriteInputF32(0, buf.InputF32Scratch()); err != nil {
			return fmt.Errorf("%s: write input: %w", op, err)
		}
		return nil
	}
	if err := k.WriteInput(0, buf.InputBytesScratch()); err != nil {
		return fmt.Errorf("%s: write input: %w", op, err)
	}
	return nil
}

// ReadF32 reads and decodes float32 output using typed I/O when available and
// falls back to byte I/O otherwise.
func ReadF32(k Kernel, buf *evalbuffer.Buffers, output []float32, op string) error {
	if buf.TypedF32() {
		if err := k.ReadOutputF32(0, buf.OutputF32Scratch()); err != nil {
			return fmt.Errorf("%s: read output: %w", op, err)
		}
	} else {
		if err := k.ReadOutput(0, buf.OutputBytesScratch()); err != nil {
			return fmt.Errorf("%s: read output: %w", op, err)
		}
	}
	if err := buf.DecodeF32(output); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return nil
}
