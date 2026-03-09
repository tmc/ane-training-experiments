//go:build darwin

package model

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/apple/coregraphics"
	appleiosurface "github.com/tmc/apple/iosurface"
	xane "github.com/tmc/apple/x/ane"
)

const defaultQoS = uint32(21)

type CompileOptions struct {
	MILText       string
	WeightBlob    []byte
	WeightPath    string
	WeightFiles   []WeightFile
	PackagePath   string
	ModelKey      string
	QoS           uint32
	PerfStatsMask uint32
}

type WeightFile struct {
	Path string
	Blob []byte
}

type EvalStats struct {
	HWExecutionNS uint64
}

// Kernel adapts github.com/tmc/apple/x/ane for the local model API.
type Kernel struct {
	rt *xane.Runtime
	k  *xane.Kernel

	inputBytes   []int
	outputBytes  []int
	inputAlloc   [][]byte
	outputAlloc  [][]byte
	inputLayout  []xane.TensorLayout
	outputLayout []xane.TensorLayout
}

// CopyOutputChannelsToInput copies a contiguous block of channels from src output
// into dst input without converting through float32.
//
// Both tensors must have matching element size and spatial shape. This mirrors
// the native io_copy helper used to chain ANE kernels through IOSurfaces.
func CopyOutputChannelsToInput(dst *Kernel, dstInput, dstChannel int, src *Kernel, srcOutput, srcChannel, channels int) error {
	if dst == nil || dst.k == nil {
		return fmt.Errorf("copy output to input: destination kernel is closed")
	}
	if src == nil || src.k == nil {
		return fmt.Errorf("copy output to input: source kernel is closed")
	}
	if channels < 0 {
		return fmt.Errorf("copy output to input: channels=%d must be >= 0", channels)
	}
	if channels == 0 {
		return nil
	}
	if dstInput < 0 || dstInput >= len(dst.inputLayout) {
		return fmt.Errorf("copy output to input: destination input index %d out of range", dstInput)
	}
	if srcOutput < 0 || srcOutput >= len(src.outputLayout) {
		return fmt.Errorf("copy output to input: source output index %d out of range", srcOutput)
	}
	dstLayout := dst.inputLayout[dstInput]
	srcLayout := src.outputLayout[srcOutput]
	if dstLayout.Height != 1 || srcLayout.Height != 1 {
		return fmt.Errorf("copy output to input: height > 1 is not supported")
	}
	if dstLayout.Width != srcLayout.Width {
		return fmt.Errorf("copy output to input: width mismatch dst=%d src=%d", dstLayout.Width, srcLayout.Width)
	}
	if dstLayout.ElemSize != srcLayout.ElemSize {
		return fmt.Errorf("copy output to input: elem size mismatch dst=%d src=%d", dstLayout.ElemSize, srcLayout.ElemSize)
	}
	if srcChannel < 0 || srcChannel+channels > srcLayout.Channels {
		return fmt.Errorf("copy output to input: source channels [%d,%d) out of range [0,%d)", srcChannel, srcChannel+channels, srcLayout.Channels)
	}
	if dstChannel < 0 || dstChannel+channels > dstLayout.Channels {
		return fmt.Errorf("copy output to input: destination channels [%d,%d) out of range [0,%d)", dstChannel, dstChannel+channels, dstLayout.Channels)
	}
	return copySurfaceChannels(
		dst.k.InputSurface(dstInput), dstLayout, dstChannel,
		src.k.OutputSurface(srcOutput), srcLayout, srcChannel,
		channels,
	)
}

func Compile(opts CompileOptions) (*Kernel, error) {
	if opts.QoS == 0 {
		opts.QoS = defaultQoS
	}
	if opts.PackagePath == "" && opts.MILText == "" {
		return nil, fmt.Errorf("compile: MILText or PackagePath is required")
	}

	rt, err := xane.Open()
	if err != nil {
		return nil, fmt.Errorf("compile: open runtime: %w", err)
	}
	k, err := rt.Compile(xaneCompileOptions(opts))
	if err != nil {
		_ = rt.Close()
		return nil, fmt.Errorf("compile: %w", err)
	}

	out := &Kernel{
		rt: rt,
		k:  k,
	}
	if err := out.initIO(); err != nil {
		out.Close()
		return nil, err
	}
	runtime.SetFinalizer(out, (*Kernel).Close)
	return out, nil
}

func (k *Kernel) InputBytes(i int) int {
	if k == nil || i < 0 || i >= len(k.inputBytes) {
		return 0
	}
	return k.inputBytes[i]
}

func (k *Kernel) NumInputs() int {
	if k == nil || k.k == nil {
		return 0
	}
	return k.k.NumInputs()
}

func (k *Kernel) OutputBytes(i int) int {
	if k == nil || i < 0 || i >= len(k.outputBytes) {
		return 0
	}
	return k.outputBytes[i]
}

func (k *Kernel) NumOutputs() int {
	if k == nil || k.k == nil {
		return 0
	}
	return k.k.NumOutputs()
}

func (k *Kernel) InputSurface(i int) coregraphics.IOSurfaceRef {
	if k == nil || k.k == nil || i < 0 || i >= k.k.NumInputs() {
		return 0
	}
	return k.k.InputSurface(i)
}

func (k *Kernel) InputLayout(i int) xane.TensorLayout {
	if k == nil || i < 0 || i >= len(k.inputLayout) {
		return xane.TensorLayout{}
	}
	return k.inputLayout[i]
}

func (k *Kernel) OutputSurface(i int) coregraphics.IOSurfaceRef {
	if k == nil || k.k == nil || i < 0 || i >= k.k.NumOutputs() {
		return 0
	}
	return k.k.OutputSurface(i)
}

func (k *Kernel) OutputLayout(i int) xane.TensorLayout {
	if k == nil || i < 0 || i >= len(k.outputLayout) {
		return xane.TensorLayout{}
	}
	return k.outputLayout[i]
}

func (k *Kernel) InputSurfaces() []coregraphics.IOSurfaceRef {
	if k == nil || k.k == nil {
		return nil
	}
	return k.k.InputSurfaces()
}

func (k *Kernel) OutputSurfaces() []coregraphics.IOSurfaceRef {
	if k == nil || k.k == nil {
		return nil
	}
	return k.k.OutputSurfaces()
}

func (k *Kernel) WriteInput(i int, b []byte) error {
	if k == nil || k.k == nil {
		return fmt.Errorf("write input: kernel is closed")
	}
	if i < 0 || i >= k.k.NumInputs() {
		return fmt.Errorf("write input: index %d out of range", i)
	}
	if len(b) != k.InputBytes(i) {
		return fmt.Errorf("write input: got %d bytes, want %d", len(b), k.InputBytes(i))
	}
	if buf := k.inputAlloc[i]; buf != nil {
		copy(buf, b)
		return k.k.WriteInput(i, buf)
	}
	return k.k.WriteInput(i, b)
}

func (k *Kernel) ReadOutput(i int, b []byte) error {
	if k == nil || k.k == nil {
		return fmt.Errorf("read output: kernel is closed")
	}
	if i < 0 || i >= k.k.NumOutputs() {
		return fmt.Errorf("read output: index %d out of range", i)
	}
	if len(b) != k.OutputBytes(i) {
		return fmt.Errorf("read output: got %d bytes, want %d", len(b), k.OutputBytes(i))
	}
	if buf := k.outputAlloc[i]; buf != nil {
		if err := k.k.ReadOutput(i, buf); err != nil {
			return err
		}
		copy(b, buf[:len(b)])
		return nil
	}
	return k.k.ReadOutput(i, b)
}

func (k *Kernel) WriteInputF32(i int, data []float32) error {
	if k == nil || k.k == nil {
		return fmt.Errorf("write input f32: kernel is closed")
	}
	return k.k.WriteInputF32(i, data)
}

func (k *Kernel) ReadOutputF32(i int, data []float32) error {
	if k == nil || k.k == nil {
		return fmt.Errorf("read output f32: kernel is closed")
	}
	return k.k.ReadOutputF32(i, data)
}

func (k *Kernel) WriteInputFP16(i int, data []float32) error {
	if k == nil || k.k == nil {
		return fmt.Errorf("write input fp16: kernel is closed")
	}
	return k.k.WriteInputFP16(i, data)
}

func (k *Kernel) ReadOutputFP16(i int, data []float32) error {
	if k == nil || k.k == nil {
		return fmt.Errorf("read output fp16: kernel is closed")
	}
	return k.k.ReadOutputFP16(i, data)
}

func (k *Kernel) Eval() error {
	if k == nil || k.k == nil {
		return fmt.Errorf("eval: kernel is closed")
	}
	return k.k.Eval()
}

func (k *Kernel) EvalWithStats() (EvalStats, error) {
	if k == nil || k.k == nil {
		return EvalStats{}, fmt.Errorf("eval with stats: kernel is closed")
	}
	st, err := k.k.EvalWithStats()
	if err != nil {
		return EvalStats{}, err
	}
	return EvalStats{HWExecutionNS: st.HWExecutionNS}, nil
}

func (k *Kernel) EvalAsync() <-chan error {
	if k == nil || k.k == nil {
		ch := make(chan error, 1)
		ch <- fmt.Errorf("eval async: kernel is closed")
		return ch
	}
	return k.k.EvalAsync()
}

func (k *Kernel) EvalAsyncWithCallback(fn func(error)) {
	if k == nil || k.k == nil {
		fn(fmt.Errorf("eval async: kernel is closed"))
		return
	}
	k.k.EvalAsyncWithCallback(fn)
}

func (k *Kernel) Diagnostics() xane.Diagnostics {
	if k == nil || k.k == nil {
		return xane.Diagnostics{}
	}
	return k.k.Diagnostics()
}

func (k *Kernel) EvalWithSignalEvent(signalPort uint32, signalValue uint64, cfg xane.SharedEventEvalOptions) error {
	if k == nil || k.k == nil {
		return fmt.Errorf("eval with signal event: kernel is closed")
	}
	return k.k.EvalWithSignalEvent(signalPort, signalValue, cfg)
}

func (k *Kernel) EvalBidirectional(waitPort uint32, waitValue uint64, signalPort uint32, signalValue uint64, cfg xane.SharedEventEvalOptions) error {
	if k == nil || k.k == nil {
		return fmt.Errorf("eval bidirectional: kernel is closed")
	}
	return k.k.EvalBidirectional(waitPort, waitValue, signalPort, signalValue, cfg)
}

func (k *Kernel) Close() {
	if k == nil {
		return
	}
	runtime.SetFinalizer(k, nil)
	if k.k != nil {
		_ = k.k.Close()
		k.k = nil
	}
	if k.rt != nil {
		_ = k.rt.Close()
		k.rt = nil
	}
}

func xaneCompileOptions(opts CompileOptions) xane.CompileOptions {
	xc := xane.CompileOptions{
		QoS:           opts.QoS,
		PerfStatsMask: opts.PerfStatsMask,
	}
	if opts.PackagePath != "" {
		xc.ModelType = xane.ModelTypePackage
		xc.PackagePath = opts.PackagePath
		xc.ModelKey = opts.ModelKey
		return xc
	}
	xc.ModelType = xane.ModelTypeMIL
	xc.MILText = []byte(opts.MILText)
	xc.WeightBlob = opts.WeightBlob
	xc.WeightPath = opts.WeightPath
	if len(opts.WeightFiles) > 0 {
		xc.WeightFiles = make([]xane.WeightFile, len(opts.WeightFiles))
		for i, wf := range opts.WeightFiles {
			xc.WeightFiles[i] = xane.WeightFile{Path: wf.Path, Blob: wf.Blob}
		}
	}
	return xc
}

func (k *Kernel) initIO() error {
	nIn := k.k.NumInputs()
	nOut := k.k.NumOutputs()
	if nIn == 0 || nOut == 0 {
		return fmt.Errorf("compile: compiled model reported %d inputs and %d outputs", nIn, nOut)
	}

	k.inputBytes = make([]int, nIn)
	k.outputBytes = make([]int, nOut)
	k.inputAlloc = make([][]byte, nIn)
	k.outputAlloc = make([][]byte, nOut)
	k.inputLayout = make([]xane.TensorLayout, nIn)
	k.outputLayout = make([]xane.TensorLayout, nOut)

	for i := range nIn {
		layout := k.k.InputLayout(i)
		if layout.LogicalBytes() <= 0 {
			return fmt.Errorf("compile: input[%d] has invalid logical size %d", i, layout.LogicalBytes())
		}
		k.inputLayout[i] = layout
		k.inputBytes[i] = layout.LogicalBytes()
		if alloc := k.k.InputAllocSize(i); alloc > layout.LogicalBytes() {
			k.inputAlloc[i] = make([]byte, alloc)
		}
	}
	for i := range nOut {
		layout := k.k.OutputLayout(i)
		if layout.LogicalBytes() <= 0 {
			return fmt.Errorf("compile: output[%d] has invalid logical size %d", i, layout.LogicalBytes())
		}
		k.outputLayout[i] = layout
		k.outputBytes[i] = layout.LogicalBytes()
		if alloc := k.k.OutputAllocSize(i); alloc > layout.LogicalBytes() {
			k.outputAlloc[i] = make([]byte, alloc)
		}
	}
	return nil
}

func Float32ToFP16(f float32) uint16 {
	return xane.Float32ToFP16(f)
}

func FP16ToFloat32(h uint16) float32 {
	return xane.FP16ToFloat32(h)
}

func copySurfaceChannels(
	dstRef coregraphics.IOSurfaceRef,
	dstLayout xane.TensorLayout,
	dstChannel int,
	srcRef coregraphics.IOSurfaceRef,
	srcLayout xane.TensorLayout,
	srcChannel int,
	channels int,
) error {
	if channels == 0 {
		return nil
	}
	rowBytes := srcLayout.Width * srcLayout.ElemSize
	if rowBytes <= 0 {
		return fmt.Errorf("copy output to input: invalid row bytes %d", rowBytes)
	}
	dstSurf := appleiosurface.IOSurfaceRef(dstRef)
	srcSurf := appleiosurface.IOSurfaceRef(srcRef)
	appleiosurface.IOSurfaceLock(dstSurf, 0, nil)
	appleiosurface.IOSurfaceLock(srcSurf, appleiosurface.KIOSurfaceLockReadOnly, nil)
	defer appleiosurface.IOSurfaceUnlock(srcSurf, appleiosurface.KIOSurfaceLockReadOnly, nil)
	defer appleiosurface.IOSurfaceUnlock(dstSurf, 0, nil)

	dstBase := appleiosurface.IOSurfaceGetBaseAddress(dstSurf)
	srcBase := appleiosurface.IOSurfaceGetBaseAddress(srcSurf)
	if dstBase == nil || srcBase == nil {
		return fmt.Errorf("copy output to input: nil IOSurface base address")
	}
	dstAlloc := dstLayout.AllocSize()
	srcAlloc := srcLayout.AllocSize()
	dstBytes := unsafe.Slice((*byte)(dstBase), dstAlloc)
	srcBytes := unsafe.Slice((*byte)(srcBase), srcAlloc)
	for c := 0; c < channels; c++ {
		dstOff := (dstChannel + c) * dstLayout.PlaneStride
		srcOff := (srcChannel + c) * srcLayout.PlaneStride
		if dstOff < 0 || dstOff+rowBytes > len(dstBytes) {
			return fmt.Errorf("copy output to input: destination offset %d out of range", dstOff)
		}
		if srcOff < 0 || srcOff+rowBytes > len(srcBytes) {
			return fmt.Errorf("copy output to input: source offset %d out of range", srcOff)
		}
		copy(dstBytes[dstOff:dstOff+rowBytes], srcBytes[srcOff:srcOff+rowBytes])
	}
	return nil
}
