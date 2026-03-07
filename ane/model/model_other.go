//go:build !darwin

package model

import "fmt"

type CompileOptions struct {
	MILText     string
	WeightBlob  []byte
	InputBytes  []int
	OutputBytes []int
	QoS         uint32
	PerfStats   bool
	PerfMask    uint32
}

type EvalStats struct {
	HWExecutionNS uint64
}

type Kernel struct{}

func Compile(CompileOptions) (*Kernel, error)  { return nil, fmt.Errorf("ane model requires darwin") }
func (k *Kernel) InputBytes(int) int           { return 0 }
func (k *Kernel) OutputBytes(int) int          { return 0 }
func (k *Kernel) WriteInput(int, []byte) error { return fmt.Errorf("ane model requires darwin") }
func (k *Kernel) ReadOutput(int, []byte) error { return fmt.Errorf("ane model requires darwin") }
func (k *Kernel) Eval() error                  { return fmt.Errorf("ane model requires darwin") }
func (k *Kernel) EvalWithStats() (EvalStats, error) {
	return EvalStats{}, fmt.Errorf("ane model requires darwin")
}
func (k *Kernel) Close() {}
