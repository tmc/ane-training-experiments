package evalbuffer

import (
	"fmt"
	"math"
)

// Buffers manages reusable byte and float32 scratch buffers for ANE eval paths.
type Buffers struct {
	inputBytes  []byte
	outputBytes []byte
	inputF32    []float32
	outputF32   []float32
	typedF32    bool
}

// New allocates reusable buffers for the given input and output sizes.
func New(inputBytes, outputBytes int) *Buffers {
	return &Buffers{
		inputBytes:  make([]byte, inputBytes),
		outputBytes: make([]byte, outputBytes),
		inputF32:    make([]float32, inputBytes/4),
		outputF32:   make([]float32, outputBytes/4),
		typedF32:    inputBytes%4 == 0 && outputBytes%4 == 0,
	}
}

// TypedF32 reports whether the buffers support direct float32 I/O.
func (b *Buffers) TypedF32() bool {
	return b != nil && b.typedF32
}

// InputBytesScratch returns the full input byte scratch buffer.
func (b *Buffers) InputBytesScratch() []byte {
	if b == nil {
		return nil
	}
	return b.inputBytes
}

// OutputBytesScratch returns the full output byte scratch buffer.
func (b *Buffers) OutputBytesScratch() []byte {
	if b == nil {
		return nil
	}
	return b.outputBytes
}

// InputF32Scratch returns the full input float32 scratch buffer.
func (b *Buffers) InputF32Scratch() []float32 {
	if b == nil {
		return nil
	}
	return b.inputF32
}

// OutputF32Scratch returns the full output float32 scratch buffer.
func (b *Buffers) OutputF32Scratch() []float32 {
	if b == nil {
		return nil
	}
	return b.outputF32
}

// StageBytes copies input into the full input byte buffer and zero-fills the tail.
func (b *Buffers) StageBytes(input []byte) error {
	if len(input) > len(b.inputBytes) {
		return fmt.Errorf("input is %d bytes, want <= %d", len(input), len(b.inputBytes))
	}
	copy(b.inputBytes, input)
	zeroBytes(b.inputBytes[len(input):])
	return nil
}

// StageF32 prepares input for float32 eval. On typed paths it fills the float32
// buffer directly; otherwise it packs input into the byte buffer.
func (b *Buffers) StageF32(input []float32) error {
	if len(input)*4 > len(b.inputBytes) {
		return fmt.Errorf("input is %d bytes, want <= %d", len(input)*4, len(b.inputBytes))
	}
	if b.typedF32 {
		copy(b.inputF32, input)
		zeroF32(b.inputF32[len(input):])
		return nil
	}
	for i, v := range input {
		u := math.Float32bits(v)
		b.inputBytes[i*4] = byte(u)
		b.inputBytes[i*4+1] = byte(u >> 8)
		b.inputBytes[i*4+2] = byte(u >> 16)
		b.inputBytes[i*4+3] = byte(u >> 24)
	}
	zeroBytes(b.inputBytes[len(input)*4:])
	return nil
}

// CopyBytes copies the current output byte buffer into dst.
func (b *Buffers) CopyBytes(dst []byte) error {
	if len(dst) > len(b.outputBytes) {
		return fmt.Errorf("output is %d bytes, want <= %d", len(dst), len(b.outputBytes))
	}
	copy(dst, b.outputBytes[:len(dst)])
	return nil
}

// DecodeF32 copies or decodes the current output buffers into dst.
func (b *Buffers) DecodeF32(dst []float32) error {
	if len(dst)*4 > len(b.outputBytes) {
		return fmt.Errorf("output is %d bytes, want <= %d", len(dst)*4, len(b.outputBytes))
	}
	if b.typedF32 {
		copy(dst, b.outputF32[:len(dst)])
		return nil
	}
	for i := range dst {
		u := uint32(b.outputBytes[i*4]) |
			uint32(b.outputBytes[i*4+1])<<8 |
			uint32(b.outputBytes[i*4+2])<<16 |
			uint32(b.outputBytes[i*4+3])<<24
		dst[i] = math.Float32frombits(u)
	}
	return nil
}

func zeroBytes(b []byte) {
	for i := range b {
		b[i] = 0
	}
}

func zeroF32(v []float32) {
	for i := range v {
		v[i] = 0
	}
}
