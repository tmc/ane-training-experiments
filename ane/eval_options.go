package ane

// EvalOptions configures a high-level ANE evaluator.
//
// On darwin, both ModelPath (.mlmodelc) and ModelPackagePath (.mlpackage)
// are supported. On non-darwin platforms, OpenEvaluator returns an error.
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
