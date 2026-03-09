package ane

// EvalOptions configures a high-level ANE evaluator.
//
// On darwin, both ModelPath (.mlmodelc) and ModelPackagePath (.mlpackage)
// are supported. On non-darwin platforms, OpenEvaluator returns an error.
// InputBytes and OutputBytes are optional and are inferred from compiled model
// layouts when omitted.
type EvalOptions struct {
	ModelPath        string
	ModelPackagePath string
	ModelKey         string
	QoS              uint32
	InputBytes       uint32
	OutputBytes      uint32
}
