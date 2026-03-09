package clientkernel

import (
	"fmt"

	"github.com/maderix/ANE/ane/clientmodel"
)

const defaultModelKey = "s"

// EvalOptions describes the one-input, one-output clientmodel compile path
// used by the high-level evaluator and pipeline helpers.
type EvalOptions struct {
	ModelPath         string
	ModelPackagePath  string
	ModelKey          string
	ModelType         string
	NetPlistFilename  string
	ForceNewClient    bool
	PreferPrivateConn bool
	QoS               uint32
	InputBytes        uint32
	OutputBytes       uint32
}

// Validate checks the shared high-level compile requirements.
func Validate(opts EvalOptions) error {
	if opts.ModelPath == "" && opts.ModelPackagePath == "" {
		return fmt.Errorf("model path is empty")
	}
	if opts.InputBytes == 0 || opts.OutputBytes == 0 {
		return fmt.Errorf("input and output bytes must be > 0")
	}
	return nil
}

// WithDefaults applies shared defaults.
func WithDefaults(opts EvalOptions) EvalOptions {
	if opts.ModelKey == "" {
		opts.ModelKey = defaultModelKey
	}
	return opts
}

// Compile compiles a one-input, one-output clientmodel kernel.
func Compile(opts EvalOptions) (*clientmodel.Kernel, error) {
	opts = WithDefaults(opts)
	if err := Validate(opts); err != nil {
		return nil, err
	}
	return clientmodel.Compile(clientmodel.CompileOptions{
		CompiledModelPath: opts.ModelPath,
		ModelPackagePath:  opts.ModelPackagePath,
		ModelKey:          opts.ModelKey,
		ModelType:         opts.ModelType,
		NetPlistFilename:  opts.NetPlistFilename,
		ForceNewClient:    opts.ForceNewClient,
		PreferPrivateConn: opts.PreferPrivateConn,
		QoS:               opts.QoS,
		InputBytes:        []int{int(opts.InputBytes)},
		OutputBytes:       []int{int(opts.OutputBytes)},
	})
}
