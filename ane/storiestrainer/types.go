// Package storiestrainer exposes a Go API for ANE Stories training.
package storiestrainer

import "time"

const (
	// DefaultQoS is the ANE QoS used by stories training flows.
	DefaultQoS uint32 = 21

	// DefaultCompileBudget matches the default compile-budget restart threshold.
	DefaultCompileBudget uint32 = 100

	// BackendBridge is reserved for compatibility, but currently unsupported in pure-Go mode.
	BackendBridge = "bridge"

	// BackendDirect indicates the direct Go _ANEClient implementation.
	BackendDirect = "direct"

	// BackendAuto selects the direct-Go implementation.
	BackendAuto = "auto"
)

// Options configures an ANE Stories trainer.
type Options struct {
	// ModelPath is the .mlmodelc path used by the daemon-backed _ANEClient path.
	ModelPath string

	// ModelKey is the _ANEModel key (usually "s").
	ModelKey string

	// DataPath is the TinyStories uint16 token data path.
	DataPath string

	// InputBytes and OutputBytes define mapped tensor byte sizes.
	InputBytes  uint32
	OutputBytes uint32

	// SequenceLength sets the token window for direct .bin training.
	// Zero uses the runtime default.
	SequenceLength uint32

	// AccumSteps controls optimizer update batching for direct .bin training.
	// Zero uses the runtime default.
	AccumSteps uint32

	// Steps limits the number of training steps. Zero means unbounded.
	Steps uint32

	// LearningRate controls optimizer step size.
	LearningRate float32

	// DisableANEExtras disables optional ANE extras path.
	DisableANEExtras bool

	// CompileBudget enables restart signaling after N compiles.
	// Zero means "use default" unless DisableCompileBudget is true.
	CompileBudget uint32

	// DisableCompileBudget disables restart signaling based on compile count.
	DisableCompileBudget bool

	// RecompileEachStep forces recompilation of the underlying ANE kernel each step.
	//
	// This is useful for parity experiments with compile-heavy training loops.
	RecompileEachStep bool

	// QoS is currently reserved and defaults to DefaultQoS.
	QoS uint32

	// Backend selects trainer implementation: auto, bridge, or direct.
	//
	// auto maps to direct-Go; bridge returns an explicit unsupported error.
	Backend string
}

// StepStats reports one trainer step.
type StepStats struct {
	Step            uint32
	Loss            float32
	StepDuration    time.Duration
	CompileDuration time.Duration
	WriteDuration   time.Duration
	EvalDuration    time.Duration
	ReadDuration    time.Duration
	Compiles        uint32
	RestartRequired bool
}

// Diagnostics reports best-effort runtime signals for the trainer backend.
type Diagnostics struct {
	Backend                      string
	HasVirtualClient             bool
	VirtualClientClass           string
	AllowRestrictedAccess        bool
	AllowRestrictedAccessKnown   bool
	IsVirtualClient              bool
	IsVirtualClientKnown         bool
	ModelQueueDepth              int
	ModelQueueDepthKnown         bool
	ProgramClass                 string
	ProgramQueueDepth            int
	ProgramQueueDepthKnown       bool
	CurrentAsyncRequestsInFlight int64
	CurrentAsyncRequestsKnown    bool
	RequestsInFlightCount        int
	RequestsInFlightCountKnown   bool
}
