//go:build darwin

package model

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/maderix/ANE/ane/iosurface"
	aneruntime "github.com/maderix/ANE/ane/runtime"
	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

const defaultQoS = uint32(21)

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

type Kernel struct {
	model     objc.ID
	request   objc.ID
	perfStats objc.ID
	qos       uint32
	tmpDir    string
	inputs    []*iosurface.Surface
	outputs   []*iosurface.Surface
}

func idObject(id objc.ID) objectivec.Object {
	return objectivec.Object{ID: id}
}

func Compile(opts CompileOptions) (*Kernel, error) {
	if err := aneruntime.EnsureLoaded(); err != nil {
		return nil, err
	}
	if len(opts.InputBytes) == 0 || len(opts.OutputBytes) == 0 {
		return nil, fmt.Errorf("compile: at least one input and one output are required")
	}

	if opts.QoS == 0 {
		opts.QoS = defaultQoS
	}

	descClassName, descClass := aneruntime.FirstClass("_ANEInMemoryModelDescriptor", "ANEInMemoryModelDescriptor")
	if descClass == 0 {
		return nil, fmt.Errorf("compile: ANE descriptor class not found")
	}
	_ = descClassName

	milData := nsDataFromBytes([]byte(opts.MILText))
	weightsObj := objc.Send[objc.ID](objc.ID(objc.GetClass("NSDictionary")), objc.Sel("dictionary"))
	if len(opts.WeightBlob) > 0 {
		weightsObj = weightsDictionary(opts.WeightBlob)
	}
	desc, usedMILInit, err := newMILDescriptor(descClass, milData, weightsObj)
	if err != nil {
		return nil, fmt.Errorf("compile: %w", err)
	}
	_ = usedMILInit
	if desc == 0 {
		return nil, fmt.Errorf("compile: model descriptor creation returned nil")
	}

	mdl := appleneuralengine.GetANEInMemoryModelClass().InMemoryModelWithDescriptor(idObject(desc)).GetID()
	if mdl == 0 {
		return nil, fmt.Errorf("compile: inMemoryModelWithDescriptor returned nil")
	}
	inMemModel := appleneuralengine.ANEInMemoryModelFromID(mdl)
	if opts.PerfStats && objc.RespondsToSelector(mdl, objc.Sel("setPerfStatsMask:")) {
		mask := opts.PerfMask
		if mask == 0 {
			mask = ^uint32(0)
		}
		inMemModel.SetPerfStatsMask(mask)
	}

	hex := inMemModel.HexStringIdentifier()
	tmpDir := filepath.Join(os.TempDir(), hex)
	if err := os.MkdirAll(filepath.Join(tmpDir, "weights"), 0o755); err != nil {
		return nil, fmt.Errorf("compile: create temp dir: %w", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "model.mil"), []byte(opts.MILText), 0o644); err != nil {
		return nil, fmt.Errorf("compile: write model.mil: %w", err)
	}
	if len(opts.WeightBlob) > 0 {
		if err := os.WriteFile(filepath.Join(tmpDir, "weights", "weight.bin"), opts.WeightBlob, 0o644); err != nil {
			return nil, fmt.Errorf("compile: write weight.bin: %w", err)
		}
	}

	emptyDict := objc.Send[objc.ID](objc.ID(objc.GetClass("NSDictionary")), objc.Sel("dictionary"))
	if ok, err := inMemModel.CompileWithQoSOptionsError(opts.QoS, idObject(emptyDict)); !ok {
		return nil, fmt.Errorf("compile: compileWithQoS failed: %v", err)
	}
	if ok, err := inMemModel.LoadWithQoSOptionsError(opts.QoS, idObject(emptyDict)); !ok {
		return nil, fmt.Errorf("compile: loadWithQoS failed: %v", err)
	}

	ins, err := newSurfaceList(opts.InputBytes)
	if err != nil {
		return nil, fmt.Errorf("compile: inputs: %w", err)
	}
	outs, err := newSurfaceList(opts.OutputBytes)
	if err != nil {
		closeAll(ins)
		return nil, fmt.Errorf("compile: outputs: %w", err)
	}

	request, perfStats, err := buildRequest(ins, outs, opts.PerfStats)
	if err != nil {
		closeAll(ins)
		closeAll(outs)
		return nil, fmt.Errorf("compile: build request: %w", err)
	}
	if err := mapRequestWithRetry(mdl, request, true); err != nil {
		closeAll(ins)
		closeAll(outs)
		_, _ = inMemModel.UnloadWithQoSError(opts.QoS)
		return nil, fmt.Errorf("compile: map request: %w", err)
	}

	k := &Kernel{model: mdl, request: request, perfStats: perfStats, qos: opts.QoS, tmpDir: tmpDir, inputs: ins, outputs: outs}
	runtime.SetFinalizer(k, (*Kernel).Close)
	return k, nil
}

func (k *Kernel) InputBytes(i int) int  { return k.inputs[i].Bytes() }
func (k *Kernel) OutputBytes(i int) int { return k.outputs[i].Bytes() }

func (k *Kernel) WriteInput(i int, b []byte) error {
	if i < 0 || i >= len(k.inputs) {
		return fmt.Errorf("write input: index %d out of range", i)
	}
	return k.inputs[i].Write(b)
}

func (k *Kernel) ReadOutput(i int, b []byte) error {
	if i < 0 || i >= len(k.outputs) {
		return fmt.Errorf("read output: index %d out of range", i)
	}
	return k.outputs[i].Read(b)
}

func (k *Kernel) Eval() error {
	_, err := k.EvalWithStats()
	return err
}

func (k *Kernel) EvalWithStats() (EvalStats, error) {
	var st EvalStats
	if !appleneuralengine.ANERequestFromID(k.request).Validate() {
		return st, fmt.Errorf("evaluateWithQoS failed: invalid request")
	}
	emptyDict := objc.Send[objc.ID](objc.ID(objc.GetClass("NSDictionary")), objc.Sel("dictionary"))
	if ok, err := appleneuralengine.ANEInMemoryModelFromID(k.model).
		EvaluateWithQoSOptionsRequestError(k.qos, idObject(emptyDict), idObject(k.request)); !ok {
		return st, fmt.Errorf("evaluateWithQoS failed: %v", err)
	}
	if k.perfStats != 0 {
		st.HWExecutionNS = appleneuralengine.ANEPerformanceStatsFromID(k.perfStats).HwExecutionTime()
	}
	return st, nil
}

func (k *Kernel) Close() {
	if k == nil {
		return
	}
	if k.model != 0 {
		if k.request != 0 && objc.RespondsToSelector(k.model, objc.Sel("unmapIOSurfacesWithRequest:")) {
			appleneuralengine.ANEInMemoryModelFromID(k.model).UnmapIOSurfacesWithRequest(idObject(k.request))
		}
		_, _ = appleneuralengine.ANEInMemoryModelFromID(k.model).UnloadWithQoSError(k.qos)
		k.model = 0
	}
	k.request = 0
	k.perfStats = 0
	closeAll(k.inputs)
	closeAll(k.outputs)
	k.inputs = nil
	k.outputs = nil
	if k.tmpDir != "" {
		_ = os.RemoveAll(k.tmpDir)
		k.tmpDir = ""
	}
}

func newSurfaceList(sizes []int) ([]*iosurface.Surface, error) {
	out := make([]*iosurface.Surface, 0, len(sizes))
	for _, sz := range sizes {
		s, err := iosurface.Create(sz)
		if err != nil {
			closeAll(out)
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

func closeAll(ss []*iosurface.Surface) {
	for _, s := range ss {
		if s != nil {
			s.Close()
		}
	}
}

func buildRequest(ins, outs []*iosurface.Surface, withPerfStats bool) (objc.ID, objc.ID, error) {
	reqClass := appleneuralengine.GetANERequestClass()

	inputObjs, inputIdx := mapSurfaces(ins)
	outputObjs, outputIdx := mapSurfaces(outs)
	procIndex := numberInt(0)

	perfObj := objc.ID(0)
	if withPerfStats {
		perfObj = appleneuralengine.NewANEPerformanceStats().GetID()
	}
	req := reqClass.RequestWithInputsInputIndicesOutputsOutputIndicesWeightsBufferPerfStatsProcedureIndex(
		idObject(objectivec.IDSliceToNSArray(inputObjs)),
		idObject(objectivec.IDSliceToNSArray(inputIdx)),
		idObject(objectivec.IDSliceToNSArray(outputObjs)),
		idObject(objectivec.IDSliceToNSArray(outputIdx)),
		idObject(0),
		idObject(perfObj),
		idObject(procIndex),
	).GetID()
	if req == 0 {
		return 0, 0, fmt.Errorf("requestWithInputs returned nil")
	}
	return req, perfObj, nil
}

func mapSurfaces(ss []*iosurface.Surface) (objs []objc.ID, idx []objc.ID) {
	iosClass := appleneuralengine.GetANEIOSurfaceObjectClass()
	objs = make([]objc.ID, 0, len(ss))
	idx = make([]objc.ID, 0, len(ss))
	for i, s := range ss {
		obj := iosClass.ObjectWithIOSurface(coregraphics.IOSurfaceRef(s.Ref())).GetID()
		objs = append(objs, obj)
		idx = append(idx, numberInt(i))
	}
	return objs, idx
}

func mapRequestWithRetry(model, request objc.ID, cacheInference bool) error {
	sel := objc.Sel("mapIOSurfacesWithRequest:cacheInference:error:")
	if !objc.RespondsToSelector(model, sel) {
		return nil
	}
	cacheModes := []bool{cacheInference}
	if cacheInference {
		cacheModes = append(cacheModes, false)
	}
	inMemModel := appleneuralengine.ANEInMemoryModelFromID(model)
	var mapErr error
	for _, cache := range cacheModes {
		mapErr = nil
		if ok, err := inMemModel.MapIOSurfacesWithRequestCacheInferenceError(idObject(request), cache); ok {
			return nil
		} else if err != nil {
			mapErr = err
		}
	}
	if mapErr == nil {
		return fmt.Errorf("mapIOSurfacesWithRequest failed")
	}
	return fmt.Errorf("mapIOSurfacesWithRequest failed: %v", mapErr)
}

func numberInt(v int) objc.ID {
	return objc.Send[objc.ID](objc.ID(objc.GetClass("NSNumber")), objc.Sel("numberWithInt:"), int32(v))
}

func nsDataFromBytes(b []byte) objc.ID {
	if len(b) == 0 {
		return objc.ID(0)
	}
	d := objc.Send[objc.ID](objc.ID(objc.GetClass("NSData")), objc.Sel("dataWithBytes:length:"), unsafe.Pointer(&b[0]), uint64(len(b)))
	runtime.KeepAlive(b)
	return d
}

func weightsDictionary(weightBlob []byte) objc.ID {
	data := nsDataFromBytes(weightBlob)
	offset := numberInt(0)
	innerVals := []objc.ID{offset, data}
	innerKeys := []objc.ID{objc.String("offset"), objc.String("data")}

	inner := objc.Send[objc.ID](
		objc.ID(objc.GetClass("NSDictionary")),
		objc.Sel("dictionaryWithObjects:forKeys:count:"),
		innerVals,
		innerKeys,
		uint64(2),
	)
	key := objc.String("@model_path/weights/weight.bin")
	outerVals := []objc.ID{inner}
	outerKeys := []objc.ID{key}
	outer := objc.Send[objc.ID](
		objc.ID(objc.GetClass("NSDictionary")),
		objc.Sel("dictionaryWithObjects:forKeys:count:"),
		outerVals,
		outerKeys,
		uint64(1),
	)
	return outer
}

func objcErrorString(errID objc.ID) string {
	if errID == 0 {
		return "unknown error"
	}
	desc := objc.Send[objc.ID](errID, objc.Sel("description"))
	if desc == 0 {
		return "error without description"
	}
	return objc.IDToString(desc)
}

func newMILDescriptor(descClass objc.Class, milData, weights objc.ID) (desc objc.ID, usedMILInit bool, err error) {
	if descClass == 0 {
		return 0, false, fmt.Errorf("ANE descriptor class not found")
	}
	descClassID := objc.ID(descClass)
	plist := objc.ID(0)

	// Preferred path: explicitly mark descriptor as MIL-backed.
	if objc.RespondsToSelector(descClassID, objc.Sel("modelWithMILText:weights:optionsPlist:isMILModel:")) {
		desc = appleneuralengine.GetANEInMemoryModelDescriptorClass().
			ModelWithMILTextWeightsOptionsPlistIsMILModel(
				idObject(milData),
				idObject(weights),
				idObject(plist),
				true,
			).GetID()
		if desc != 0 {
			return desc, true, nil
		}
	}

	alloc := objc.Send[objc.ID](descClassID, objc.Sel("alloc"))
	if alloc != 0 {
		for _, selector := range []string{
			"initWithNetworkDescription:weights:optionsPlist:isMILModel:",
			"initWithNetworkText:weights:optionsPlist:isMILModel:",
			"initWithMILText:weights:optionsPlist:isMILModel:",
		} {
			sel := objc.Sel(selector)
			if !objc.RespondsToSelector(alloc, sel) {
				continue
			}
			desc = objc.Send[objc.ID](alloc, sel, milData, weights, plist, true)
			if desc != 0 {
				return desc, true, nil
			}
		}
	}

	// Fallback path used by older runtime versions.
	desc = objc.Send[objc.ID](
		descClassID,
		objc.Sel("modelWithMILText:weights:optionsPlist:"),
		milData,
		weights,
		plist,
	)
	if desc == 0 {
		return 0, false, fmt.Errorf("modelWithMILText returned nil")
	}
	return desc, false, nil
}
