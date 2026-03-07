//go:build !darwin

package clientmodel

import "fmt"

// CompileOptions configures the _ANEClient + _ANEModel execution path.
type CompileOptions struct {
	ModelPackagePath  string
	CompiledModelPath string
	ModelKey          string
	ModelType         string
	NetPlistFilename  string
	QoS               uint32
	InputBytes        []int
	OutputBytes       []int
}

// Kernel wraps a loaded _ANEModel and request I/O surfaces.
type Kernel struct{}

func Compile(CompileOptions) (*Kernel, error) {
	return nil, fmt.Errorf("ane client model requires darwin")
}
func (k *Kernel) InputBytes(int) int   { return 0 }
func (k *Kernel) OutputBytes(int) int  { return 0 }
func (k *Kernel) CompiledPath() string { return "" }
func (k *Kernel) VirtualClientConnect() (uint32, bool) {
	return 0, false
}
func (k *Kernel) SupportsCompletionEventEval() bool { return false }
func (k *Kernel) WriteInput(int, []byte) error      { return fmt.Errorf("ane client model requires darwin") }
func (k *Kernel) ReadOutput(int, []byte) error      { return fmt.Errorf("ane client model requires darwin") }
func (k *Kernel) InputSurfaceRef(int) (uintptr, error) {
	return 0, fmt.Errorf("ane client model requires darwin")
}
func (k *Kernel) OutputSurfaceRef(int) (uintptr, error) {
	return 0, fmt.Errorf("ane client model requires darwin")
}
func (k *Kernel) Eval() error { return fmt.Errorf("ane client model requires darwin") }
func (k *Kernel) Close()      {}

func CompileCount() uint32 { return 0 }
func ResetCompileCount()   {}
