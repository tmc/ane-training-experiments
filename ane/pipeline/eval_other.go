//go:build !darwin

package pipeline

import "fmt"

// EvalOptions configures a synchronous eval runner.
type EvalOptions struct {
	ModelPath        string
	ModelPackagePath string
	ModelKey         string
	QoS              uint32
	InputBytes       uint32
	OutputBytes      uint32
}

// EvalRunner is unavailable on non-darwin platforms.
type EvalRunner struct{}

func OpenEval(EvalOptions) (*EvalRunner, error) {
	return nil, fmt.Errorf("eval runner is only supported on darwin")
}

func (r *EvalRunner) Close() error { return nil }
func (r *EvalRunner) EvalBytes([]byte, []byte) error {
	return fmt.Errorf("eval runner is only supported on darwin")
}
func (r *EvalRunner) EvalF32([]float32, []float32) error {
	return fmt.Errorf("eval runner is only supported on darwin")
}
