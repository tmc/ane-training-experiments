//go:build !darwin

package model

import (
	"fmt"

	"github.com/tmc/apple/coregraphics"
	xane "github.com/tmc/apple/x/ane"
)

type CompileOptions struct {
	MILText       string
	WeightBlob    []byte
	WeightPath    string
	WeightFiles   []WeightFile
	PackagePath   string
	ModelKey      string
	QoS           uint32
	PerfStatsMask uint32
}

type WeightFile struct {
	Path string
	Blob []byte
}

type EvalStats struct {
	HWExecutionNS uint64
}

type Kernel struct{}

func Compile(CompileOptions) (*Kernel, error) { return nil, fmt.Errorf("ane model requires darwin") }
func (k *Kernel) InputBytes(int) int          { return 0 }
func (k *Kernel) NumInputs() int              { return 0 }
func (k *Kernel) OutputBytes(int) int         { return 0 }
func (k *Kernel) NumOutputs() int             { return 0 }
func (k *Kernel) InputSurface(int) coregraphics.IOSurfaceRef {
	return 0
}
func (k *Kernel) InputLayout(int) xane.TensorLayout { return xane.TensorLayout{} }
func (k *Kernel) OutputSurface(int) coregraphics.IOSurfaceRef {
	return 0
}
func (k *Kernel) OutputLayout(int) xane.TensorLayout          { return xane.TensorLayout{} }
func (k *Kernel) InputSurfaces() []coregraphics.IOSurfaceRef  { return nil }
func (k *Kernel) OutputSurfaces() []coregraphics.IOSurfaceRef { return nil }
func (k *Kernel) WriteInput(int, []byte) error                { return fmt.Errorf("ane model requires darwin") }
func (k *Kernel) WriteInputF32(int, []float32) error {
	return fmt.Errorf("ane model requires darwin")
}
func (k *Kernel) WriteInputFP16(int, []float32) error {
	return fmt.Errorf("ane model requires darwin")
}
func (k *Kernel) ReadOutput(int, []byte) error { return fmt.Errorf("ane model requires darwin") }
func (k *Kernel) ReadOutputF32(int, []float32) error {
	return fmt.Errorf("ane model requires darwin")
}
func (k *Kernel) ReadOutputFP16(int, []float32) error {
	return fmt.Errorf("ane model requires darwin")
}
func (k *Kernel) Eval() error { return fmt.Errorf("ane model requires darwin") }
func (k *Kernel) EvalWithStats() (EvalStats, error) {
	return EvalStats{}, fmt.Errorf("ane model requires darwin")
}
func (k *Kernel) EvalAsync() <-chan error {
	ch := make(chan error, 1)
	ch <- fmt.Errorf("ane model requires darwin")
	return ch
}
func (k *Kernel) EvalAsyncWithCallback(fn func(error)) { fn(fmt.Errorf("ane model requires darwin")) }
func (k *Kernel) Diagnostics() xane.Diagnostics        { return xane.Diagnostics{} }
func (k *Kernel) EvalWithSignalEvent(uint32, uint64, xane.SharedEventEvalOptions) error {
	return fmt.Errorf("ane model requires darwin")
}
func (k *Kernel) EvalBidirectional(uint32, uint64, uint32, uint64, xane.SharedEventEvalOptions) error {
	return fmt.Errorf("ane model requires darwin")
}
func (k *Kernel) Close() {}

func CopyOutputChannelsToInput(*Kernel, int, int, *Kernel, int, int, int) error {
	return fmt.Errorf("ane model requires darwin")
}

func Float32ToFP16(float32) uint16 { return 0 }
func FP16ToFloat32(uint16) float32 { return 0 }
