package ane

// EvalOptions configures a high-level ANE evaluator.
//
// On darwin, both ModelPath (.mlmodelc) and ModelPackagePath (.mlpackage)
// are supported. On non-darwin platforms, OpenEvaluator returns an error.
// InputBytes and OutputBytes are optional on the default x/ane-backed path and
// are inferred from compiled model layouts when omitted. They are still
// required when forcing the older private-client compile path through
// ModelType or NetPlistFilename.
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
