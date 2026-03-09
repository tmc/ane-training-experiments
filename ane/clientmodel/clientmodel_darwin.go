//go:build darwin

package clientmodel

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	aneiosurface "github.com/maderix/ANE/ane/iosurface"
	aneruntime "github.com/maderix/ANE/ane/runtime"
	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/coreml"
	"github.com/tmc/apple/foundation"
	appiosurface "github.com/tmc/apple/iosurface"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

const (
	defaultQoS      = uint32(21)
	defaultModelKey = "s"
)

const (
	eventTypeSignal = int64(5)
)

// CompileOptions configures the _ANEClient + _ANEModel execution path.
type CompileOptions struct {
	ModelPackagePath  string
	CompiledModelPath string
	ModelKey          string
	ModelType         string
	NetPlistFilename  string
	ForceNewClient    bool
	PreferPrivateConn bool
	QoS               uint32
	InputBytes        []int
	OutputBytes       []int
}

type compileProfile struct {
	modelType string
	netPlist  string
}

// Kernel wraps a loaded _ANEModel and request I/O surfaces.
type Kernel struct {
	client       objc.ID
	ownsClient   bool
	model        objc.ID
	request      objc.ID
	compiledPath string
	qos          uint32
	mapped       bool
	directEval   bool
	inputs       []*aneiosurface.Surface
	outputs      []*aneiosurface.Surface
}

var compileCounter atomic.Uint32

// SharedEventEvalOptions configures shared-event execution behavior.
type SharedEventEvalOptions struct {
	// DisableIOFencesUseSharedEvents sets kANEFDisableIOFencesUseSharedEventsKey.
	DisableIOFencesUseSharedEvents bool
	// EnableFWToFWSignal sets kANEFEnableFWToFWSignal.
	//
	// NOTE: keep this false for ANE->Metal signal direction on physical hosts.
	EnableFWToFWSignal bool
}

// Diagnostics reports runtime availability and queue-state probes.
type Diagnostics struct {
	HasVirtualClient               bool
	VirtualClientClass             string
	VirtualClientConnectCode       uint32
	VirtualClientConnectKnown      bool
	SupportsCompletionEventEval    bool
	AllowRestrictedAccess          bool
	AllowRestrictedAccessKnown     bool
	IsVirtualClient                bool
	IsVirtualClientKnown           bool
	ModelQueueDepth                int
	ModelQueueDepthKnown           bool
	ProgramClass                   string
	ProgramQueueDepth              int
	ProgramQueueDepthKnown         bool
	CurrentAsyncRequestsInFlight   int64
	CurrentAsyncRequestsInFlightOK bool
	RequestsInFlightCount          int
	RequestsInFlightCountKnown     bool
}

// Compile compiles/loads a model through _ANEClient and prepares request surfaces.
func Compile(opts CompileOptions) (*Kernel, error) {
	if err := aneruntime.EnsureLoaded(); err != nil {
		return nil, err
	}
	opts = withDefaults(opts)
	if err := validateCompileOptions(opts); err != nil {
		return nil, err
	}
	if opts.CompiledModelPath == "" {
		if err := aneruntime.EnsureCoreMLLoaded(); err != nil {
			return nil, err
		}
	}

	compiledURL, compiledPath, err := resolveCompiledModelURL(opts)
	if err != nil {
		return nil, err
	}

	client, ownsClient := sharedClient(opts.ForceNewClient, opts.PreferPrivateConn)
	if client.GetID() == 0 {
		return nil, fmt.Errorf("compile: _ANEClient shared connection unavailable")
	}
	modelObj := appleneuralengine.GetANEModelClass().ModelAtURLKey(compiledURL, idObject(objc.String(opts.ModelKey)))
	model := modelObj.GetID()
	if model == 0 {
		return nil, fmt.Errorf("compile: modelAtURL returned nil")
	}

	compileOptions := compileOptionsDictionary(opts.ModelType, opts.NetPlistFilename)
	emptyOptions := emptyDictionary()

	if err := compileModelWithFallback(client, model, compileOptions, opts); err != nil {
		return nil, err
	}

	if err := loadModelWithRetry(client, model, emptyOptions, opts.QoS); err != nil {
		return nil, err
	}

	ins, err := newSurfaceList(opts.InputBytes)
	if err != nil {
		_ = unloadModel(client, model, opts.QoS)
		return nil, fmt.Errorf("compile: inputs: %w", err)
	}
	outs, err := newSurfaceList(opts.OutputBytes)
	if err != nil {
		closeAll(ins)
		_ = unloadModel(client, model, opts.QoS)
		return nil, fmt.Errorf("compile: outputs: %w", err)
	}

	request, err := buildRequest(ins, outs)
	if err != nil {
		closeAll(ins)
		closeAll(outs)
		_ = unloadModel(client, model, opts.QoS)
		return nil, fmt.Errorf("compile: build request: %w", err)
	}
	if err := mapModelRequest(client, model, request, true); err != nil {
		closeAll(ins)
		closeAll(outs)
		_ = unloadModel(client, model, opts.QoS)
		return nil, fmt.Errorf("compile: map request: %w", err)
	}

	directEval := objc.RespondsToSelector(client.GetID(), objc.Sel("doEvaluateDirectWithModel:options:request:qos:error:"))

	k := &Kernel{
		client:       client.GetID(),
		ownsClient:   ownsClient,
		model:        model,
		request:      request,
		compiledPath: compiledPath,
		qos:          opts.QoS,
		mapped:       true,
		directEval:   directEval,
		inputs:       ins,
		outputs:      outs,
	}
	compileCounter.Add(1)
	runtime.SetFinalizer(k, (*Kernel).Close)
	return k, nil
}

// CompileCount reports successful Compile calls in this process.
func CompileCount() uint32 {
	return compileCounter.Load()
}

// ResetCompileCount resets process-local CompileCount.
//
// This is primarily useful for deterministic tests/benchmarks.
func ResetCompileCount() {
	compileCounter.Store(0)
}

func withDefaults(opts CompileOptions) CompileOptions {
	if opts.ModelKey == "" {
		opts.ModelKey = defaultModelKey
	}
	if opts.QoS == 0 {
		opts.QoS = defaultQoS
	}
	return opts
}

func validateCompileOptions(opts CompileOptions) error {
	if len(opts.InputBytes) == 0 || len(opts.OutputBytes) == 0 {
		return fmt.Errorf("compile: at least one input and one output are required")
	}
	for i, n := range opts.InputBytes {
		if n <= 0 {
			return fmt.Errorf("compile: input[%d] bytes must be > 0", i)
		}
	}
	for i, n := range opts.OutputBytes {
		if n <= 0 {
			return fmt.Errorf("compile: output[%d] bytes must be > 0", i)
		}
	}
	if opts.CompiledModelPath == "" && opts.ModelPackagePath == "" {
		return fmt.Errorf("compile: either ModelPackagePath or CompiledModelPath is required")
	}
	return nil
}

func resolveCompiledModelURL(opts CompileOptions) (foundation.NSURL, string, error) {
	if opts.CompiledModelPath != "" {
		path, err := filepath.Abs(opts.CompiledModelPath)
		if err != nil {
			return foundation.NSURL{}, "", fmt.Errorf("compile: resolve compiled model path: %w", err)
		}
		if _, err := os.Stat(path); err != nil {
			return foundation.NSURL{}, "", fmt.Errorf("compile: stat compiled model path: %w", err)
		}
		url := fileURL(path)
		if url.GetID() == 0 {
			return foundation.NSURL{}, "", fmt.Errorf("compile: create file URL for compiled model path")
		}
		return url, path, nil
	}

	modelClass := objc.GetClass("MLModel")
	if modelClass == 0 {
		return foundation.NSURL{}, "", fmt.Errorf("compile: MLModel class not found")
	}
	pkgPath, err := filepath.Abs(opts.ModelPackagePath)
	if err != nil {
		return foundation.NSURL{}, "", fmt.Errorf("compile: resolve model package path: %w", err)
	}
	if _, err := os.Stat(pkgPath); err != nil {
		return foundation.NSURL{}, "", fmt.Errorf("compile: stat model package path: %w", err)
	}
	pkgURL := fileURL(pkgPath)
	if pkgURL.GetID() == 0 {
		return foundation.NSURL{}, "", fmt.Errorf("compile: create file URL for model package path")
	}

	compiledURL, err := coreml.GetMLModelClass().CompileModelAtURLError(pkgURL)
	if err != nil || compiledURL.GetID() == 0 {
		msg := "unknown error"
		if err != nil {
			msg = err.Error()
		}
		return foundation.NSURL{}, "", fmt.Errorf(
			"compile: compileModelAtURL failed: %s (hint: pass CompiledModelPath to skip runtime CoreML compilation; on newer macOS releases, compileModelAtURL may fail for some models)",
			msg,
		)
	}
	compiledPath := compiledURL.Path()
	return compiledURL, compiledPath, nil
}

func fileURL(path string) foundation.NSURL {
	return foundation.GetNSURLClass().FileURLWithPath(path)
}

func sharedClient(forceNew, preferPrivate bool) (appleneuralengine.ANEClient, bool) {
	cls := appleneuralengine.GetANEClientClass()
	if !forceNew {
		if preferPrivate {
			if c := cls.SharedPrivateConnection(); c.GetID() != 0 {
				return c, false
			}
		}
		if c := cls.SharedConnection(); c.GetID() != 0 {
			return c, false
		}
		if !preferPrivate {
			if c := cls.SharedPrivateConnection(); c.GetID() != 0 {
				return c, false
			}
		}
	}

	client := cls.Alloc()
	if client.GetID() != 0 {
		client = client.InitWithRestrictedAccessAllowed(true)
		if client.GetID() != 0 {
			return client, true
		}
	}

	client = appleneuralengine.NewANEClientWithRestrictedAccessAllowed(true)
	if client.GetID() != 0 {
		return client, true
	}
	client = appleneuralengine.NewANEClient()
	if client.GetID() != 0 {
		return client, true
	}
	return appleneuralengine.ANEClient{}, false
}

func compileOptionsDictionary(modelType, plistFilename string) objc.ID {
	keys := make([]objc.ID, 0, 2)
	vals := make([]objc.ID, 0, 2)
	if modelType != "" {
		keys = append(keys, optionKeyID(appleneuralengine.KANEFModelTypeKey, "kANEFModelType"))
		vals = append(vals, objc.String(modelType))
	}
	if plistFilename != "" {
		keys = append(keys, optionKeyID(appleneuralengine.KANEFNetPlistFilenameKey, "kANEFNetPlistFilenameKey"))
		vals = append(vals, objc.String(plistFilename))
	}
	if len(vals) == 0 {
		return emptyDictionary()
	}
	return objc.Send[objc.ID](
		objc.ID(objc.GetClass("NSDictionary")),
		objc.Sel("dictionaryWithObjects:forKeys:count:"),
		vals,
		keys,
		uint64(len(vals)),
	)
}

func compileModelWithFallback(client appleneuralengine.ANEClient, model, compileOptions objc.ID, opts CompileOptions) error {
	ok, err := client.CompileModelOptionsQosError(
		idObject(model),
		idObject(compileOptions),
		opts.QoS,
	)
	if ok {
		return nil
	}
	if !isInvalidMILCompileErr(err) {
		return fmt.Errorf("compile: compileModel failed: %v", err)
	}

	profiles := buildCompileFallbackProfiles(opts)
	if len(profiles) == 0 {
		return fmt.Errorf("compile: compileModel failed: %v", err)
	}
	for _, p := range profiles {
		fallbackOpts := compileOptionsDictionary(p.modelType, p.netPlist)
		if ok, _ := client.CompileModelOptionsQosError(idObject(model), idObject(fallbackOpts), opts.QoS); ok {
			return nil
		}
	}
	return fmt.Errorf("compile: compileModel failed: %v", err)
}

func buildCompileFallbackProfiles(opts CompileOptions) []compileProfile {
	primary := compileProfile{modelType: opts.ModelType, netPlist: opts.NetPlistFilename}
	// Always try empty profile first when primary is non-empty.
	base := []compileProfile{}
	if primary.modelType != "" || primary.netPlist != "" {
		base = append(base, compileProfile{})
	}
	base = append(base, parseCompileFallbackProfiles(os.Getenv("ANE_COMPILE_FALLBACK_PROFILES"))...)
	if len(base) == 0 {
		return nil
	}
	seen := map[compileProfile]bool{primary: true}
	out := make([]compileProfile, 0, len(base))
	for _, p := range base {
		if seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}

func isInvalidMILCompileErr(err any) bool {
	if err == nil {
		return false
	}
	return strings.Contains(strings.ToLower(fmt.Sprint(err)), "invalidmilprogram")
}

func parseCompileFallbackProfiles(raw string) []compileProfile {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	seen := make(map[compileProfile]bool, len(parts))
	out := make([]compileProfile, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		p := compileProfile{}
		if i := strings.Index(part, ":"); i >= 0 {
			p.modelType = normalizeCompileProfileField(part[:i])
			p.netPlist = normalizeCompileProfileField(part[i+1:])
		} else {
			p.modelType = normalizeCompileProfileField(part)
		}
		if seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}

func normalizeCompileProfileField(s string) string {
	s = strings.TrimSpace(s)
	switch strings.ToLower(s) {
	case "", "-", "<empty>", "<none>", "none", "null":
		return ""
	default:
		return s
	}
}

func emptyDictionary() objc.ID {
	return objc.Send[objc.ID](objc.ID(objc.GetClass("NSDictionary")), objc.Sel("dictionary"))
}

func optionKeyID(key objectivec.Object, fallback string) objc.ID {
	if key.ID != 0 {
		return key.ID
	}
	return objc.String(fallback)
}

func idObject(id objc.ID) objectivec.Object {
	return objectivec.Object{ID: id}
}

func surfaceRef(s *aneiosurface.Surface) coregraphics.IOSurfaceRef {
	if s == nil {
		return 0
	}
	return coregraphics.IOSurfaceRef(s.Ref())
}

func newSurfaceList(sizes []int) ([]*aneiosurface.Surface, error) {
	out := make([]*aneiosurface.Surface, 0, len(sizes))
	for _, sz := range sizes {
		s, err := aneiosurface.Create(sz)
		if err != nil {
			closeAll(out)
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

func closeAll(ss []*aneiosurface.Surface) {
	for _, s := range ss {
		if s != nil {
			s.Close()
		}
	}
}

func buildRequest(ins, outs []*aneiosurface.Surface) (objc.ID, error) {
	inputObjs, inputIdx := mapSurfaces(ins)
	outputObjs, outputIdx := mapSurfaces(outs)
	procIndex := numberInt(0)

	req := appleneuralengine.GetANERequestClass().
		RequestWithInputsInputIndicesOutputsOutputIndicesWeightsBufferPerfStatsProcedureIndex(
			idObject(objectivec.IDSliceToNSArray(inputObjs)),
			idObject(objectivec.IDSliceToNSArray(inputIdx)),
			idObject(objectivec.IDSliceToNSArray(outputObjs)),
			idObject(objectivec.IDSliceToNSArray(outputIdx)),
			idObject(objc.ID(0)),
			idObject(objc.ID(0)),
			idObject(procIndex),
		).GetID()
	if req == 0 {
		return 0, fmt.Errorf("requestWithInputs returned nil")
	}
	return req, nil
}

func mapSurfaces(ss []*aneiosurface.Surface) (objs []objc.ID, idx []objc.ID) {
	iosClass := appleneuralengine.GetANEIOSurfaceObjectClass()
	objs = make([]objc.ID, 0, len(ss))
	idx = make([]objc.ID, 0, len(ss))
	for i, s := range ss {
		obj := iosClass.ObjectWithIOSurface(surfaceRef(s))
		objs = append(objs, obj.GetID())
		idx = append(idx, numberInt(i))
	}
	return objs, idx
}

func numberInt(v int) objc.ID {
	return objc.Send[objc.ID](objc.ID(objc.GetClass("NSNumber")), objc.Sel("numberWithInt:"), int32(v))
}

func unloadModel(client appleneuralengine.ANEClient, model objc.ID, qos uint32) error {
	if client.GetID() == 0 || model == 0 {
		return nil
	}
	if ok, err := client.UnloadModelOptionsQosError(idObject(model), idObject(emptyDictionary()), qos); !ok {
		return fmt.Errorf("unloadModel failed: %v", err)
	}
	return nil
}

func loadModelWithRetry(client appleneuralengine.ANEClient, model, options objc.ID, qos uint32) error {
	if ok, _ := client.LoadModelOptionsQosError(idObject(model), idObject(options), qos); ok {
		return nil
	}
	time.Sleep(100 * time.Millisecond)
	if ok, _ := client.LoadModelOptionsQosError(idObject(model), idObject(options), qos); ok {
		return nil
	}
	_, err := client.LoadModelOptionsQosError(idObject(model), idObject(options), qos)
	return fmt.Errorf("compile: loadModel failed: %v", err)
}

func mapModelRequest(client appleneuralengine.ANEClient, model, request objc.ID, cacheInference bool) error {
	cacheModes := []bool{cacheInference}
	if cacheInference {
		cacheModes = append(cacheModes, false)
	}
	var mapErr error
	var virtualErr error
	for _, cache := range cacheModes {
		if ok, _ := client.MapIOSurfacesWithModelRequestCacheInferenceError(idObject(model), idObject(request), cache); ok {
			return nil
		}
		if mapErr == nil {
			_, mapErr = client.MapIOSurfacesWithModelRequestCacheInferenceError(idObject(model), idObject(request), cache)
		}
		if vc := attachedVirtualClient(client); vc != nil {
			if ok, _ := vc.DoMapIOSurfacesWithModelRequestCacheInferenceError(idObject(model), idObject(request), cache); ok {
				return nil
			}
			if virtualErr == nil {
				_, virtualErr = vc.DoMapIOSurfacesWithModelRequestCacheInferenceError(idObject(model), idObject(request), cache)
			}
		}
	}
	if virtualErr != nil {
		return fmt.Errorf("map request: mapIOSurfacesWithModel failed: %v; virtual client map failed: %v", mapErr, virtualErr)
	}
	return fmt.Errorf("map request: mapIOSurfacesWithModel failed: %v", mapErr)
}

func unmapModelRequest(client appleneuralengine.ANEClient, model, request objc.ID) {
	if client.GetID() == 0 || model == 0 || request == 0 {
		return
	}
	client.UnmapIOSurfacesWithModelRequest(idObject(model), idObject(request))
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

func (k *Kernel) InputBytes(i int) int  { return k.inputs[i].Bytes() }
func (k *Kernel) OutputBytes(i int) int { return k.outputs[i].Bytes() }

func (k *Kernel) CompiledPath() string {
	if k == nil {
		return ""
	}
	return k.compiledPath
}

// HasVirtualClient reports whether the backing _ANEClient exposes a non-nil
// virtual client object.
func (k *Kernel) HasVirtualClient() bool {
	if k == nil || k.client == 0 {
		return false
	}
	client := appleneuralengine.ANEClientFromID(k.client)
	return client.VirtualClient() != nil
}

// IsVirtualClient reports _ANEClient.isVirtualClient when available.
//
// The second return value is false when the selector is unavailable.
func (k *Kernel) IsVirtualClient() (bool, bool) {
	if k == nil || k.client == 0 {
		return false, false
	}
	return appleneuralengine.ANEClientFromID(k.client).IsVirtualClient(), true
}

// QueueDepth reports model queueDepth when available.
//
// The second return value is false when the selector is unavailable.
func (k *Kernel) QueueDepth() (int, bool) {
	if k == nil || k.model == 0 {
		return 0, false
	}
	return int(appleneuralengine.ANEModelFromID(k.model).QueueDepth()), true
}

// VirtualClientConnect reports the attached _ANEVirtualClient connect status.
//
// The second return value is false when no attached virtual client is present.
func (k *Kernel) VirtualClientConnect() (uint32, bool) {
	if k == nil || k.client == 0 {
		return 0, false
	}
	vc := attachedVirtualClient(appleneuralengine.ANEClientFromID(k.client))
	if vc == nil {
		return 0, false
	}
	return vc.Connect(), true
}

// SupportsCompletionEventEval reports whether an attached virtual client is
// available for completionEvent-based evaluation helpers.
func (k *Kernel) SupportsCompletionEventEval() bool {
	if k == nil || k.client == 0 {
		return false
	}
	return attachedVirtualClient(appleneuralengine.ANEClientFromID(k.client)) != nil
}

// Diagnostics returns best-effort model/client/program diagnostics.
func (k *Kernel) Diagnostics() Diagnostics {
	var d Diagnostics
	if k == nil {
		return d
	}

	d.HasVirtualClient = k.HasVirtualClient()
	client := appleneuralengine.ANEClientFromID(k.client)
	d.AllowRestrictedAccess = client.AllowRestrictedAccess()
	d.AllowRestrictedAccessKnown = true
	if d.HasVirtualClient {
		if vc := client.VirtualClient(); vc != nil {
			d.VirtualClientClass = className(vc.GetID())
		}
	}
	d.VirtualClientConnectCode, d.VirtualClientConnectKnown = k.VirtualClientConnect()
	d.SupportsCompletionEventEval = k.SupportsCompletionEventEval()
	d.IsVirtualClient, d.IsVirtualClientKnown = k.IsVirtualClient()
	d.ModelQueueDepth, d.ModelQueueDepthKnown = k.QueueDepth()

	program := programForModel(k.model)
	if program == nil {
		return d
	}
	d.ProgramClass = className(program.GetID())
	if program.GetID() != 0 {
		d.ProgramQueueDepth = int(program.QueueDepth())
		d.ProgramQueueDepthKnown = true
		d.CurrentAsyncRequestsInFlight = program.CurrentAsyncRequestsInFlight()
		d.CurrentAsyncRequestsInFlightOK = true
		reqs := program.RequestsInFlight()
		if reqs.GetID() != 0 {
			d.RequestsInFlightCount = int(foundation.NSArrayFromID(reqs.GetID()).Count())
			d.RequestsInFlightCountKnown = true
		}
	}
	return d
}

func className(id objc.ID) string {
	if id == 0 {
		return ""
	}
	cls := objc.Send[objc.ID](id, objc.Sel("class"))
	if cls == 0 {
		return ""
	}
	name := objc.Send[objc.ID](cls, objc.Sel("description"))
	if name == 0 {
		return ""
	}
	return objc.IDToString(name)
}

func programForModel(model objc.ID) appleneuralengine.IANEProgramForEvaluation {
	if model == 0 {
		return nil
	}
	m := appleneuralengine.ANEModelFromID(model)
	if p := m.Program(); p != nil {
		return p
	}
	for _, selName := range []string{"program", "programForEvaluation"} {
		sel := objc.Sel(selName)
		if objc.RespondsToSelector(model, sel) {
			if program := objc.Send[objc.ID](model, sel); program != 0 {
				p := appleneuralengine.ANEProgramForEvaluationFromID(program)
				return &p
			}
		}
	}

	if objc.RespondsToSelector(model, objc.Sel("valueForKey:")) {
		for _, key := range []string{"program", "_program"} {
			program := objc.Send[objc.ID](model, objc.Sel("valueForKey:"), objc.String(key))
			if program != 0 {
				p := appleneuralengine.ANEProgramForEvaluationFromID(program)
				return &p
			}
		}
	}
	return nil
}

func (k *Kernel) WriteInput(i int, b []byte) error {
	if k == nil {
		return fmt.Errorf("write input: kernel is nil")
	}
	if i < 0 || i >= len(k.inputs) {
		return fmt.Errorf("write input: index %d out of range", i)
	}
	return k.inputs[i].Write(b)
}

func (k *Kernel) ReadOutput(i int, b []byte) error {
	if k == nil {
		return fmt.Errorf("read output: kernel is nil")
	}
	if i < 0 || i >= len(k.outputs) {
		return fmt.Errorf("read output: index %d out of range", i)
	}
	return k.outputs[i].Read(b)
}

func (k *Kernel) WriteInputF32(i int, data []float32) error {
	if k == nil {
		return fmt.Errorf("write input f32: kernel is nil")
	}
	if i < 0 || i >= len(k.inputs) {
		return fmt.Errorf("write input f32: index %d out of range", i)
	}
	return k.inputs[i].WriteF32(data)
}

func (k *Kernel) ReadOutputF32(i int, data []float32) error {
	if k == nil {
		return fmt.Errorf("read output f32: kernel is nil")
	}
	if i < 0 || i >= len(k.outputs) {
		return fmt.Errorf("read output f32: index %d out of range", i)
	}
	return k.outputs[i].ReadF32(data)
}

// InputSurfaceRef returns the underlying IOSurfaceRef handle for input i.
func (k *Kernel) InputSurfaceRef(i int) (uintptr, error) {
	if k == nil {
		return 0, fmt.Errorf("input surface ref: kernel is nil")
	}
	if i < 0 || i >= len(k.inputs) {
		return 0, fmt.Errorf("input surface ref: index %d out of range", i)
	}
	if k.inputs[i] == nil {
		return 0, fmt.Errorf("input surface ref: surface %d is nil", i)
	}
	return k.inputs[i].Ref(), nil
}

// OutputSurfaceRef returns the underlying IOSurfaceRef handle for output i.
func (k *Kernel) OutputSurfaceRef(i int) (uintptr, error) {
	if k == nil {
		return 0, fmt.Errorf("output surface ref: kernel is nil")
	}
	if i < 0 || i >= len(k.outputs) {
		return 0, fmt.Errorf("output surface ref: index %d out of range", i)
	}
	if k.outputs[i] == nil {
		return 0, fmt.Errorf("output surface ref: surface %d is nil", i)
	}
	return k.outputs[i].Ref(), nil
}

func (k *Kernel) Eval() error {
	if k == nil || k.client == 0 || k.model == 0 {
		return fmt.Errorf("eval: kernel is closed")
	}
	client := appleneuralengine.ANEClientFromID(k.client)
	if !appleneuralengine.ANERequestFromID(k.request).Validate() {
		return fmt.Errorf("eval: invalid request")
	}
	opts := idObject(emptyDictionary())
	if k.directEval {
		if ok, _ := client.DoEvaluateDirectWithModelOptionsRequestQosError(
			idObject(k.model),
			opts,
			idObject(k.request),
			k.qos,
		); ok {
			return nil
		}
		if ok, _ := client.EvaluateWithModelOptionsRequestQosError(
			idObject(k.model),
			opts,
			idObject(k.request),
			k.qos,
		); ok {
			// Direct dispatch is accepted on some builds but still fails at runtime.
			// After one successful fallback, avoid paying the failed direct call cost.
			k.directEval = false
			return nil
		}
		_, directErr := client.DoEvaluateDirectWithModelOptionsRequestQosError(
			idObject(k.model),
			opts,
			idObject(k.request),
			k.qos,
		)
		_, evalErr := client.EvaluateWithModelOptionsRequestQosError(
			idObject(k.model),
			opts,
			idObject(k.request),
			k.qos,
		)
		return fmt.Errorf(
			"doEvaluateDirectWithModel failed: %v; evaluateWithModel failed: %v",
			directErr,
			evalErr,
		)
	}

	if ok, err := client.EvaluateWithModelOptionsRequestQosError(
		idObject(k.model),
		opts,
		idObject(k.request),
		k.qos,
	); !ok {
		if vc := attachedVirtualClient(client); vc != nil {
			if ok, _ := vc.EvaluateWithModelOptionsRequestQosError(
				idObject(k.model),
				opts,
				idObject(k.request),
				k.qos,
			); ok {
				return nil
			}
			_, vcErr := vc.EvaluateWithModelOptionsRequestQosError(
				idObject(k.model),
				opts,
				idObject(k.request),
				k.qos,
			)
			return fmt.Errorf("evaluateWithModel failed: %v; virtual client evaluate failed: %v", err, vcErr)
		}
		return fmt.Errorf("evaluateWithModel failed: %v", err)
	}
	return nil
}

// EvalWithSignalEvent executes one request with an ANE signal event only.
//
// This is the ANE->(event ring) direction and should generally use
// EnableFWToFWSignal=false for stability.
func (k *Kernel) EvalWithSignalEvent(signalPort uint32, signalValue uint64, cfg SharedEventEvalOptions) error {
	if signalPort == 0 {
		return fmt.Errorf("eval with signal event: signal port is zero")
	}
	return k.evalWithSharedEvents(sharedEventSpec{
		signalPort:  signalPort,
		signalValue: signalValue,
	}, cfg)
}

// EvalBidirectional executes one request with wait+signal shared events.
//
// Typical usage:
//  1. producer (CPU/GPU) signals waitPort:waitValue
//  2. this call runs
//  3. request completion path signals signalPort:signalValue
func (k *Kernel) EvalBidirectional(waitPort uint32, waitValue uint64, signalPort uint32, signalValue uint64, cfg SharedEventEvalOptions) error {
	if waitPort == 0 || signalPort == 0 {
		return fmt.Errorf("eval bidirectional: wait and signal ports must be non-zero")
	}
	return k.evalWithSharedEvents(sharedEventSpec{
		waitPort:    waitPort,
		waitValue:   waitValue,
		signalPort:  signalPort,
		signalValue: signalValue,
	}, cfg)
}

type sharedEventSpec struct {
	waitPort    uint32
	waitValue   uint64
	signalPort  uint32
	signalValue uint64
}

func (k *Kernel) evalWithSharedEvents(spec sharedEventSpec, cfg SharedEventEvalOptions) error {
	if k == nil || k.client == 0 || k.model == 0 || k.request == 0 {
		return fmt.Errorf("shared-events eval: kernel is closed")
	}
	req := appleneuralengine.ANERequestFromID(k.request)
	if !req.Validate() {
		return fmt.Errorf("shared-events eval: invalid request")
	}

	sharedEventsID, releaseEvents, err := buildSharedEvents(spec)
	if err != nil {
		return err
	}
	defer releaseEvents()

	sharedEvents := appleneuralengine.ANESharedEventsFromID(sharedEventsID)
	req.SetSharedEvents(&sharedEvents)
	defer objc.Send[struct{}](k.request, objc.Sel("setSharedEvents:"), objc.ID(0))

	waitDone := make(chan completionResult, 1)
	releaseHandler, err := setRequestCompletionHandler(k.request, waitDone)
	if err != nil {
		return fmt.Errorf("shared-events eval: set completion handler: %w", err)
	}
	defer releaseHandler()

	client := appleneuralengine.ANEClientFromID(k.client)
	opts := idObject(sharedEventOptions(cfg))
	if ok, evalErr := client.DoEvaluateDirectWithModelOptionsRequestQosError(
		idObject(k.model),
		opts,
		idObject(k.request),
		k.qos,
	); !ok {
		if vc := attachedVirtualClient(client); vc != nil {
			releaseVC, vcErr := startVirtualCompletionEval(vc, k.model, k.request, k.qos, opts, waitDone)
			if vcErr == nil {
				defer releaseVC()
			} else {
				return fmt.Errorf("shared-events eval: doEvaluateDirectWithModel failed: %v; virtual client completion eval failed: %v", evalErr, vcErr)
			}
		} else {
			return fmt.Errorf("shared-events eval: doEvaluateDirectWithModel failed: %v", evalErr)
		}
	}

	select {
	case r := <-waitDone:
		if r.err != nil {
			return fmt.Errorf("shared-events eval: completion error: %w", r.err)
		}
		if !r.ok {
			return fmt.Errorf("shared-events eval: completion reported failure")
		}
		return nil
	case <-time.After(5 * time.Second):
		return fmt.Errorf("shared-events eval: timeout waiting for completion")
	}
}

func sharedEventOptions(cfg SharedEventEvalOptions) objc.ID {
	keys := make([]objc.ID, 0, 2)
	vals := make([]objc.ID, 0, 2)
	num := foundation.GetNSNumberClass()
	keys = append(keys, optionKeyID(appleneuralengine.KANEFDisableIOFencesUseSharedEventsKey, "kANEFDisableIOFencesUseSharedEventsKey"))
	vals = append(vals, num.NumberWithBool(cfg.DisableIOFencesUseSharedEvents).GetID())
	keys = append(keys, optionKeyID(appleneuralengine.KANEFEnableFWToFWSignal, "kANEFEnableFWToFWSignal"))
	vals = append(vals, num.NumberWithBool(cfg.EnableFWToFWSignal).GetID())
	return objc.Send[objc.ID](
		objc.ID(objc.GetClass("NSDictionary")),
		objc.Sel("dictionaryWithObjects:forKeys:count:"),
		vals,
		keys,
		uint64(len(vals)),
	)
}

func buildSharedEvents(spec sharedEventSpec) (objc.ID, func(), error) {
	waitIDs := make([]objc.ID, 0, 1)
	signalIDs := make([]objc.ID, 0, 1)
	release := make([]func(), 0, 2)
	cleanup := func() {
		for i := len(release) - 1; i >= 0; i-- {
			release[i]()
		}
	}

	if spec.waitPort != 0 {
		waitEventObj := appiosurface.NewIOSurfaceSharedEventWithMachPort(spec.waitPort)
		if waitEventObj.GetID() == 0 {
			return 0, nil, fmt.Errorf("shared-events eval: bind wait event port %d failed", spec.waitPort)
		}
		release = append(release, waitEventObj.Release)
		waitID := appleneuralengine.GetANESharedWaitEventClass().
			WaitEventWithValueSharedEvent(spec.waitValue, idObject(waitEventObj.GetID())).GetID()
		if waitID == 0 {
			cleanup()
			return 0, nil, fmt.Errorf("shared-events eval: wait event creation failed")
		}
		waitIDs = append(waitIDs, waitID)
	}
	if spec.signalPort != 0 {
		signalEventObj := appiosurface.NewIOSurfaceSharedEventWithMachPort(spec.signalPort)
		if signalEventObj.GetID() == 0 {
			cleanup()
			return 0, nil, fmt.Errorf("shared-events eval: bind signal event port %d failed", spec.signalPort)
		}
		release = append(release, signalEventObj.Release)
		sigID := appleneuralengine.GetANESharedSignalEventClass().
			SignalEventWithValueSymbolIndexEventTypeSharedEvent(
				spec.signalValue,
				0,
				eventTypeSignal,
				idObject(signalEventObj.GetID()),
			).GetID()
		if sigID == 0 {
			cleanup()
			return 0, nil, fmt.Errorf("shared-events eval: signal event creation failed")
		}
		signalIDs = append(signalIDs, sigID)
	}

	waitArray := objectivec.IDSliceToNSArray(waitIDs)
	signalArray := objectivec.IDSliceToNSArray(signalIDs)
	sharedID := appleneuralengine.GetANESharedEventsClass().
		SharedEventsWithSignalEventsWaitEvents(
			idObject(signalArray),
			idObject(waitArray),
		).GetID()
	if sharedID == 0 {
		cleanup()
		return 0, nil, fmt.Errorf("shared-events eval: sharedEventsWithSignalEvents failed")
	}
	return sharedID, cleanup, nil
}

type completionResult struct {
	ok  bool
	err error
}

func setRequestCompletionHandler(requestID objc.ID, done chan<- completionResult) (func(), error) {
	if requestID == 0 {
		return nil, fmt.Errorf("request is nil")
	}
	var once sync.Once
	appleneuralengine.ANERequestFromID(requestID).SetCompletionHandler(func(ok bool, err error) {
		once.Do(func() {
			done <- completionResult{ok: ok, err: err}
		})
	})
	return func() {
		objc.Send[struct{}](requestID, objc.Sel("setCompletionHandler:"), objc.ID(0))
	}, nil
}

func attachedVirtualClient(client appleneuralengine.ANEClient) appleneuralengine.IANEVirtualClient {
	if client.GetID() == 0 {
		return nil
	}
	vc := client.VirtualClient()
	if vc == nil || vc.GetID() == 0 {
		return nil
	}
	return vc
}

func startVirtualCompletionEval(vc appleneuralengine.IANEVirtualClient, model, request objc.ID, qos uint32, opts objectivec.Object, done chan<- completionResult) (func(), error) {
	if vc == nil || vc.GetID() == 0 {
		return nil, fmt.Errorf("virtual client unavailable")
	}
	handler := func(ok bool, err error) {
		select {
		case done <- completionResult{ok: ok, err: err}:
		default:
		}
	}
	block, cleanup := appleneuralengine.NewBoolErrorBlock(handler)
	ok, err := vc.DoEvaluateWithModelOptionsRequestQosCompletionEventError(
		idObject(model),
		opts,
		idObject(request),
		qos,
		idObject(block),
	)
	if ok {
		return cleanup, nil
	}
	ok, legacyErr := vc.DoEvaluateWithModelLegacyOptionsRequestQosCompletionEventError(
		idObject(model),
		opts,
		idObject(request),
		qos,
		idObject(block),
	)
	if ok {
		return cleanup, nil
	}
	cleanup()
	if legacyErr != nil {
		return nil, legacyErr
	}
	return nil, err
}

func (k *Kernel) Close() {
	if k == nil {
		return
	}
	client := appleneuralengine.ANEClientFromID(k.client)
	if k.mapped {
		unmapModelRequest(client, k.model, k.request)
		k.mapped = false
	}
	_ = unloadModel(client, k.model, k.qos)
	if k.ownsClient && k.client != 0 {
		client.Release()
	}
	k.client = 0
	k.ownsClient = false
	k.model = 0
	k.request = 0
	closeAll(k.inputs)
	closeAll(k.outputs)
	k.inputs = nil
	k.outputs = nil
}
