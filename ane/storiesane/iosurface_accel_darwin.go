//go:build darwin && cgo

package storiesane

/*
#cgo darwin LDFLAGS: -framework IOSurface
#cgo CFLAGS: -O3
#include <IOSurface/IOSurface.h>
#include <string.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

static void *iosurface_lock_and_get_base(IOSurfaceRef surf, uint32_t options) {
	IOSurfaceLock(surf, options, NULL);
	return IOSurfaceGetBaseAddress(surf);
}

static void iosurface_unlock(IOSurfaceRef surf, uint32_t options) {
	IOSurfaceUnlock(surf, options, NULL);
}

// iosurface_copy_range copies a contiguous width range for a block of channels
// from src output surface to dst input surface without going through purego.
// This replaces 6 purego calls (2 locks + 2 getbase + 2 unlocks) with 2 CGo calls.
static int iosurface_copy_range(
	IOSurfaceRef dstSurf, int dstPlaneStride, int dstChannel, int dstOffset, int dstAllocSize,
	IOSurfaceRef srcSurf, int srcPlaneStride, int srcChannel, int srcOffset, int srcAllocSize,
	int channels, int rowBytes, int elemSize
) {
	void *dstBase = iosurface_lock_and_get_base(dstSurf, 0);
	void *srcBase = iosurface_lock_and_get_base(srcSurf, 1); // kIOSurfaceLockReadOnly
	if (!dstBase || !srcBase) {
		if (srcBase) iosurface_unlock(srcSurf, 1);
		if (dstBase) iosurface_unlock(dstSurf, 0);
		return -1;
	}
	for (int c = 0; c < channels; c++) {
		int dstOff = (dstChannel + c) * dstPlaneStride + dstOffset * elemSize;
		int srcOff = (srcChannel + c) * srcPlaneStride + srcOffset * elemSize;
		if (dstOff < 0 || dstOff + rowBytes > dstAllocSize) { break; }
		if (srcOff < 0 || srcOff + rowBytes > srcAllocSize) { break; }
		memcpy((char*)dstBase + dstOff, (char*)srcBase + srcOff, rowBytes);
	}
	iosurface_unlock(srcSurf, 1);
	iosurface_unlock(dstSurf, 0);
	return 0;
}
// iosurface_copy_range2 copies two contiguous width ranges from the same source
// surface to the same destination surface in a single lock/unlock pair.
static int iosurface_copy_range2(
	IOSurfaceRef dstSurf, int dstPlaneStride, int dstAllocSize,
	int dst1Channel, int dst1Offset,
	int dst2Channel, int dst2Offset,
	IOSurfaceRef srcSurf, int srcPlaneStride, int srcAllocSize,
	int src1Channel, int src1Offset,
	int src2Channel, int src2Offset,
	int channels, int rowBytes, int elemSize
) {
	void *dstBase = iosurface_lock_and_get_base(dstSurf, 0);
	void *srcBase = iosurface_lock_and_get_base(srcSurf, 1);
	if (!dstBase || !srcBase) {
		if (srcBase) iosurface_unlock(srcSurf, 1);
		if (dstBase) iosurface_unlock(dstSurf, 0);
		return -1;
	}
	for (int c = 0; c < channels; c++) {
		int dOff = (dst1Channel + c) * dstPlaneStride + dst1Offset * elemSize;
		int sOff = (src1Channel + c) * srcPlaneStride + src1Offset * elemSize;
		if (dOff >= 0 && dOff + rowBytes <= dstAllocSize && sOff >= 0 && sOff + rowBytes <= srcAllocSize)
			memcpy((char*)dstBase + dOff, (char*)srcBase + sOff, rowBytes);
	}
	for (int c = 0; c < channels; c++) {
		int dOff = (dst2Channel + c) * dstPlaneStride + dst2Offset * elemSize;
		int sOff = (src2Channel + c) * srcPlaneStride + src2Offset * elemSize;
		if (dOff >= 0 && dOff + rowBytes <= dstAllocSize && sOff >= 0 && sOff + rowBytes <= srcAllocSize)
			memcpy((char*)dstBase + dOff, (char*)srcBase + sOff, rowBytes);
	}
	iosurface_unlock(srcSurf, 1);
	iosurface_unlock(dstSurf, 0);
	return 0;
}
// iosurface_copy_and_write_fp16 copies FP16 data from a source output surface
// AND writes FP32→FP16 converted data to the same destination input surface in
// a single lock/unlock cycle. This fuses two IOSurface operations that target
// the same destination, saving one lock+unlock pair (~33µs per call).
//
// The copy reads from srcSurf output channels [srcChannel, srcChannel+channels)
// at offset srcOffset, writing to dstSurf input at (channel 0, offset 0).
// The two FP32 writes go to the same destination at specified offsets.
static int iosurface_copy_and_write_fp16(
	IOSurfaceRef dstSurf, int dstPlaneStride, int dstAllocSize,
	IOSurfaceRef srcSurf, int srcPlaneStride, int srcAllocSize,
	int srcChannel, int srcOffset,
	int channels, int copyRowBytes, int elemSize,
	const float *data1, int d1Offset, int d1Len,
	const float *data2, int d2Offset, int d2Len,
	int writeWidth
) {
	void *dstBase = iosurface_lock_and_get_base(dstSurf, 0);
	void *srcBase = iosurface_lock_and_get_base(srcSurf, 1);
	if (!dstBase || !srcBase) {
		if (srcBase) iosurface_unlock(srcSurf, 1);
		if (dstBase) iosurface_unlock(dstSurf, 0);
		return -1;
	}
	// Copy FP16 from src output to dst input at channel 0, offset 0.
	int dstElemStride = dstPlaneStride / elemSize;
	for (int c = 0; c < channels; c++) {
		int dstOff = c * dstPlaneStride;
		int srcOff = (srcChannel + c) * srcPlaneStride + srcOffset * elemSize;
		if (dstOff + copyRowBytes <= dstAllocSize && srcOff + copyRowBytes <= srcAllocSize)
			memcpy((char*)dstBase + dstOff, (char*)srcBase + srcOff, copyRowBytes);
	}
	iosurface_unlock(srcSurf, 1);
	// Write FP32→FP16 data1 and data2 into the still-locked dst surface.
	_Float16 *dst16 = (_Float16 *)dstBase;
#ifdef __aarch64__
	for (int c = 0; c < channels && c * writeWidth < d1Len; c++) {
		int srcOff = c * writeWidth;
		int n = writeWidth;
		if (srcOff + n > d1Len) n = d1Len - srcOff;
		int dOff = c * dstElemStride + d1Offset;
		if (dOff + n > dstAllocSize / 2) break;
		const float *s = data1 + srcOff;
		_Float16 *d = dst16 + dOff;
		int i = 0;
		for (; i + 7 < n; i += 8) {
			float16x8_t h = vcombine_f16(
				vcvt_f16_f32(vld1q_f32(s + i)),
				vcvt_f16_f32(vld1q_f32(s + i + 4)));
			vst1q_f16((__fp16 *)(d + i), h);
		}
		for (; i < n; i++) d[i] = (_Float16)s[i];
	}
	for (int c = 0; c < channels && c * writeWidth < d2Len; c++) {
		int srcOff = c * writeWidth;
		int n = writeWidth;
		if (srcOff + n > d2Len) n = d2Len - srcOff;
		int dOff = c * dstElemStride + d2Offset;
		if (dOff + n > dstAllocSize / 2) break;
		const float *s = data2 + srcOff;
		_Float16 *d = dst16 + dOff;
		int i = 0;
		for (; i + 7 < n; i += 8) {
			float16x8_t h = vcombine_f16(
				vcvt_f16_f32(vld1q_f32(s + i)),
				vcvt_f16_f32(vld1q_f32(s + i + 4)));
			vst1q_f16((__fp16 *)(d + i), h);
		}
		for (; i < n; i++) d[i] = (_Float16)s[i];
	}
#else
	for (int c = 0; c < channels && c * writeWidth < d1Len; c++) {
		int srcOff = c * writeWidth;
		int n = writeWidth;
		if (srcOff + n > d1Len) n = d1Len - srcOff;
		int dOff = c * dstElemStride + d1Offset;
		if (dOff + n > dstAllocSize / 2) break;
		for (int i = 0; i < n; i++) dst16[dOff + i] = (_Float16)data1[srcOff + i];
	}
	for (int c = 0; c < channels && c * writeWidth < d2Len; c++) {
		int srcOff = c * writeWidth;
		int n = writeWidth;
		if (srcOff + n > d2Len) n = d2Len - srcOff;
		int dOff = c * dstElemStride + d2Offset;
		if (dOff + n > dstAllocSize / 2) break;
		for (int i = 0; i < n; i++) dst16[dOff + i] = (_Float16)data2[srcOff + i];
	}
#endif
	iosurface_unlock(dstSurf, 0);
	return 0;
}
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/maderix/ANE/ane/model"
	coregraphics "github.com/tmc/apple/coregraphics"
	xane "github.com/tmc/apple/x/ane"
)

var errNilIOSurfaceBase = errors.New("nil IOSurface base address")

// withLockedFP16InputCGo is a CGo-based replacement for withLockedFP16Input,
// replacing 3 purego calls (Lock, GetBaseAddress, Unlock) with 2 CGo calls.
func withLockedFP16InputCGo(surfRef coregraphics.IOSurfaceRef, layout xane.TensorLayout, fn func(layout xane.TensorLayout, data []uint16) error) error {
	base := C.iosurface_lock_and_get_base(C.IOSurfaceRef(unsafe.Pointer(surfRef)), 0)
	if base == nil {
		C.iosurface_unlock(C.IOSurfaceRef(unsafe.Pointer(surfRef)), 0)
		return errNilIOSurfaceBase
	}
	data := unsafe.Slice((*uint16)(base), layout.AllocSize()/2)
	err := fn(layout, data)
	C.iosurface_unlock(C.IOSurfaceRef(unsafe.Pointer(surfRef)), 0)
	return err
}

// withLockedFP16OutputCGo is a CGo-based replacement for withLockedFP16Output.
func withLockedFP16OutputCGo(surfRef coregraphics.IOSurfaceRef, layout xane.TensorLayout, fn func(layout xane.TensorLayout, data []uint16) error) error {
	const readOnly = 1 // kIOSurfaceLockReadOnly
	base := C.iosurface_lock_and_get_base(C.IOSurfaceRef(unsafe.Pointer(surfRef)), readOnly)
	if base == nil {
		C.iosurface_unlock(C.IOSurfaceRef(unsafe.Pointer(surfRef)), readOnly)
		return errNilIOSurfaceBase
	}
	data := unsafe.Slice((*uint16)(base), layout.AllocSize()/2)
	err := fn(layout, data)
	C.iosurface_unlock(C.IOSurfaceRef(unsafe.Pointer(surfRef)), readOnly)
	return err
}

// copyOutputRangeToInputCGo is a CGo-based replacement for model.CopyOutputRangeToInput,
// replacing 6 purego calls (2 locks + 2 getbase + 2 unlocks) with 2 CGo calls
// for the IOSurface operations.
func copyOutputRangeToInputCGo(dst *model.Kernel, dstInput, dstChannel, dstOffset int, src *model.Kernel, srcOutput, srcChannel, srcOffset, channels, width int) error {
	if dst == nil || src == nil {
		return fmt.Errorf("copy output to input cgo: nil kernel")
	}
	if channels <= 0 {
		return nil
	}
	dstLayout := dst.InputLayout(dstInput)
	srcLayout := src.OutputLayout(srcOutput)
	if dstLayout.Height != 1 || srcLayout.Height != 1 {
		return fmt.Errorf("copy output to input cgo: height > 1 not supported")
	}
	if width < 0 {
		if dstLayout.Width != srcLayout.Width {
			return fmt.Errorf("copy output to input cgo: width mismatch dst=%d src=%d", dstLayout.Width, srcLayout.Width)
		}
		width = srcLayout.Width
	}
	if dstLayout.ElemSize != srcLayout.ElemSize {
		return fmt.Errorf("copy output to input cgo: elem size mismatch dst=%d src=%d", dstLayout.ElemSize, srcLayout.ElemSize)
	}
	rowBytes := width * srcLayout.ElemSize
	dstRef := dst.InputSurface(dstInput)
	srcRef := src.OutputSurface(srcOutput)
	if dstRef == 0 || srcRef == 0 {
		return fmt.Errorf("copy output to input cgo: nil surface ref")
	}
	rc := C.iosurface_copy_range(
		C.IOSurfaceRef(unsafe.Pointer(dstRef)),
		C.int(dstLayout.PlaneStride), C.int(dstChannel), C.int(dstOffset), C.int(dstLayout.AllocSize()),
		C.IOSurfaceRef(unsafe.Pointer(srcRef)),
		C.int(srcLayout.PlaneStride), C.int(srcChannel), C.int(srcOffset), C.int(srcLayout.AllocSize()),
		C.int(channels), C.int(rowBytes), C.int(srcLayout.ElemSize),
	)
	if rc != 0 {
		return errNilIOSurfaceBase
	}
	return nil
}

// copyOutputChannelsToInputCGo is a convenience wrapper.
func copyOutputChannelsToInputCGo(dst *model.Kernel, dstInput, dstChannel int, src *model.Kernel, srcOutput, srcChannel, channels int) error {
	return copyOutputRangeToInputCGo(dst, dstInput, dstChannel, 0, src, srcOutput, srcChannel, 0, channels, -1)
}

// copyOutputRange2ToInputCGo copies two channel ranges from the same source
// output surface to the same destination input surface in a single lock/unlock.
func copyOutputRange2ToInputCGo(
	dst *model.Kernel, dstInput int,
	dst1Channel, dst1Offset int,
	dst2Channel, dst2Offset int,
	src *model.Kernel, srcOutput int,
	src1Channel, src1Offset int,
	src2Channel, src2Offset int,
	channels, width int,
) error {
	if dst == nil || src == nil {
		return fmt.Errorf("copy output range2 to input cgo: nil kernel")
	}
	if channels <= 0 {
		return nil
	}
	dstLayout := dst.InputLayout(dstInput)
	srcLayout := src.OutputLayout(srcOutput)
	if dstLayout.Height != 1 || srcLayout.Height != 1 {
		return fmt.Errorf("copy output range2 to input cgo: height > 1 not supported")
	}
	if width < 0 {
		width = srcLayout.Width
	}
	if dstLayout.ElemSize != srcLayout.ElemSize {
		return fmt.Errorf("copy output range2 to input cgo: elem size mismatch")
	}
	rowBytes := width * srcLayout.ElemSize
	dstRef := dst.InputSurface(dstInput)
	srcRef := src.OutputSurface(srcOutput)
	if dstRef == 0 || srcRef == 0 {
		return fmt.Errorf("copy output range2 to input cgo: nil surface ref")
	}
	rc := C.iosurface_copy_range2(
		C.IOSurfaceRef(unsafe.Pointer(dstRef)),
		C.int(dstLayout.PlaneStride), C.int(dstLayout.AllocSize()),
		C.int(dst1Channel), C.int(dst1Offset),
		C.int(dst2Channel), C.int(dst2Offset),
		C.IOSurfaceRef(unsafe.Pointer(srcRef)),
		C.int(srcLayout.PlaneStride), C.int(srcLayout.AllocSize()),
		C.int(src1Channel), C.int(src1Offset),
		C.int(src2Channel), C.int(src2Offset),
		C.int(channels), C.int(rowBytes), C.int(srcLayout.ElemSize),
	)
	if rc != 0 {
		return errNilIOSurfaceBase
	}
	return nil
}

// writeInputFP16CGo writes float32 data to a kernel's input IOSurface using CGo
// instead of purego, bypassing the model package's purego IOSurface path.
// Uses NEON-vectorized FP16 conversion from fp16_pack_darwin.go.
func writeInputFP16CGo(k *model.Kernel, input int, data []float32) error {
	if k == nil {
		return fmt.Errorf("write input fp16 cgo: kernel is nil")
	}
	layout := k.InputLayout(input)
	ref := k.InputSurface(input)
	if ref == 0 {
		return fmt.Errorf("write input fp16 cgo: input surface %d is nil", input)
	}
	return withLockedFP16InputCGo(ref, layout, func(layout xane.TensorLayout, surfData []uint16) error {
		writeChannelFirstActsOffsetFP16(surfData, layout, 0, 0, layout.Width, data)
		return nil
	})
}

// writeInputFP16ChannelsCGo writes channel-offset float32 data using CGo.
func writeInputFP16ChannelsCGo(k *model.Kernel, input, channel int, data []float32) error {
	if k == nil {
		return fmt.Errorf("write input fp16 channels cgo: kernel is nil")
	}
	layout := k.InputLayout(input)
	ref := k.InputSurface(input)
	if ref == 0 {
		return fmt.Errorf("write input fp16 channels cgo: input surface %d is nil", input)
	}
	channelElems := layout.Height * layout.Width
	if channelElems <= 0 || len(data)%channelElems != 0 {
		return fmt.Errorf("write input fp16 channels cgo: data len %d, channelElems %d", len(data), channelElems)
	}
	channels := len(data) / channelElems
	return withLockedFP16InputCGo(ref, layout, func(layout xane.TensorLayout, surfData []uint16) error {
		writeChannelFirstActsOffsetFP16(surfData, layout, channel, 0, channelElems, data)
		_ = channels
		return nil
	})
}

// writeInputFP16Channels2CGo writes two channel-offset regions to the same
// IOSurface input in a single lock/unlock, halving the kernel synchronization
// overhead compared to two separate writeInputFP16ChannelsCGo calls.
func writeInputFP16Channels2CGo(k *model.Kernel, input int, ch1 int, data1 []float32, ch2 int, data2 []float32) error {
	if k == nil {
		return fmt.Errorf("write input fp16 channels2 cgo: kernel is nil")
	}
	layout := k.InputLayout(input)
	ref := k.InputSurface(input)
	if ref == 0 {
		return fmt.Errorf("write input fp16 channels2 cgo: input surface %d is nil", input)
	}
	channelElems := layout.Height * layout.Width
	if channelElems <= 0 {
		return fmt.Errorf("write input fp16 channels2 cgo: invalid channel elems %d", channelElems)
	}
	return withLockedFP16InputCGo(ref, layout, func(layout xane.TensorLayout, surfData []uint16) error {
		writeChannelFirstActsOffsetFP16(surfData, layout, ch1, 0, channelElems, data1)
		writeChannelFirstActsOffsetFP16(surfData, layout, ch2, 0, channelElems, data2)
		return nil
	})
}

// writeInputFP16Channels3CGo writes three channel-offset regions to the same
// IOSurface input in a single lock/unlock.
func writeInputFP16Channels3CGo(k *model.Kernel, input int, ch1 int, data1 []float32, ch2 int, data2 []float32, ch3 int, data3 []float32) error {
	if k == nil {
		return fmt.Errorf("write input fp16 channels3 cgo: kernel is nil")
	}
	layout := k.InputLayout(input)
	ref := k.InputSurface(input)
	if ref == 0 {
		return fmt.Errorf("write input fp16 channels3 cgo: input surface %d is nil", input)
	}
	channelElems := layout.Height * layout.Width
	if channelElems <= 0 {
		return fmt.Errorf("write input fp16 channels3 cgo: invalid channel elems %d", channelElems)
	}
	return withLockedFP16InputCGo(ref, layout, func(layout xane.TensorLayout, surfData []uint16) error {
		writeChannelFirstActsOffsetFP16(surfData, layout, ch1, 0, channelElems, data1)
		writeChannelFirstActsOffsetFP16(surfData, layout, ch2, 0, channelElems, data2)
		writeChannelFirstActsOffsetFP16(surfData, layout, ch3, 0, channelElems, data3)
		return nil
	})
}

// readOutputFP16Channels2CGo reads two channel-offset regions from the same
// IOSurface output in a single lock/unlock.
func readOutputFP16Channels2CGo(k *model.Kernel, output int, ch1 int, data1 []float32, ch2 int, data2 []float32) error {
	if k == nil {
		return fmt.Errorf("read output fp16 channels2 cgo: kernel is nil")
	}
	layout := k.OutputLayout(output)
	ref := k.OutputSurface(output)
	if ref == 0 {
		return fmt.Errorf("read output fp16 channels2 cgo: output surface %d is nil", output)
	}
	channelElems := layout.Height * layout.Width
	if channelElems <= 0 || len(data1)%channelElems != 0 || len(data2)%channelElems != 0 {
		return fmt.Errorf("read output fp16 channels2 cgo: invalid data lens %d %d channelElems %d", len(data1), len(data2), channelElems)
	}
	return withLockedFP16OutputCGo(ref, layout, func(layout xane.TensorLayout, surfData []uint16) error {
		readChannelFirstActsOffsetFP16(data1, surfData, layout, ch1, 0, channelElems)
		readChannelFirstActsOffsetFP16(data2, surfData, layout, ch2, 0, channelElems)
		return nil
	})
}

// readOutputFP16ChannelsCGo reads channel-offset float32 data using CGo.
func readOutputFP16ChannelsCGo(k *model.Kernel, output, channel int, data []float32) error {
	if k == nil {
		return fmt.Errorf("read output fp16 channels cgo: kernel is nil")
	}
	layout := k.OutputLayout(output)
	ref := k.OutputSurface(output)
	if ref == 0 {
		return fmt.Errorf("read output fp16 channels cgo: output surface %d is nil", output)
	}
	channelElems := layout.Height * layout.Width
	if channelElems <= 0 || len(data)%channelElems != 0 {
		return fmt.Errorf("read output fp16 channels cgo: data len %d, channelElems %d", len(data), channelElems)
	}
	return withLockedFP16OutputCGo(ref, layout, func(layout xane.TensorLayout, surfData []uint16) error {
		readChannelFirstActsOffsetFP16(data, surfData, layout, channel, 0, channelElems)
		return nil
	})
}

// readOutputFP16CGo reads float32 data from a kernel's output IOSurface using CGo.
// Uses NEON-vectorized FP16 conversion from fp16_pack_darwin.go.
func readOutputFP16CGo(k *model.Kernel, output int, data []float32) error {
	if k == nil {
		return fmt.Errorf("read output fp16 cgo: kernel is nil")
	}
	layout := k.OutputLayout(output)
	ref := k.OutputSurface(output)
	if ref == 0 {
		return fmt.Errorf("read output fp16 cgo: output surface %d is nil", output)
	}
	return withLockedFP16OutputCGo(ref, layout, func(layout xane.TensorLayout, surfData []uint16) error {
		readChannelFirstActsOffsetFP16(data, surfData, layout, 0, 0, layout.Width)
		return nil
	})
}

// copyAndWriteFP16CGo copies FP16 channels from src output to dst input AND
// writes two FP32→FP16 data regions to the same dst input in a single
// lock/unlock cycle. Saves one lock+unlock pair vs separate copy + write calls.
func copyAndWriteFP16CGo(
	dst *model.Kernel, dstInput int,
	src *model.Kernel, srcOutput, srcChannel, channels int,
	data1 []float32, d1WidthOffset int,
	data2 []float32, d2WidthOffset int,
	writeWidth int,
) error {
	if dst == nil || src == nil {
		return fmt.Errorf("copy and write fp16 cgo: nil kernel")
	}
	if channels <= 0 {
		return nil
	}
	dstLayout := dst.InputLayout(dstInput)
	srcLayout := src.OutputLayout(srcOutput)
	if dstLayout.Height != 1 || srcLayout.Height != 1 {
		return fmt.Errorf("copy and write fp16 cgo: height > 1 not supported")
	}
	if dstLayout.ElemSize != srcLayout.ElemSize || dstLayout.ElemSize != 2 {
		return fmt.Errorf("copy and write fp16 cgo: elem size dst=%d src=%d, want 2", dstLayout.ElemSize, srcLayout.ElemSize)
	}
	dstRef := dst.InputSurface(dstInput)
	srcRef := src.OutputSurface(srcOutput)
	if dstRef == 0 || srcRef == 0 {
		return fmt.Errorf("copy and write fp16 cgo: nil surface ref")
	}
	copyRowBytes := writeWidth * srcLayout.ElemSize // FP16 copy width = writeWidth * 2
	rc := C.iosurface_copy_and_write_fp16(
		C.IOSurfaceRef(unsafe.Pointer(dstRef)),
		C.int(dstLayout.PlaneStride), C.int(dstLayout.AllocSize()),
		C.IOSurfaceRef(unsafe.Pointer(srcRef)),
		C.int(srcLayout.PlaneStride), C.int(srcLayout.AllocSize()),
		C.int(srcChannel), C.int(0),
		C.int(channels), C.int(copyRowBytes), C.int(srcLayout.ElemSize),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(data1))), C.int(d1WidthOffset), C.int(len(data1)),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(data2))), C.int(d2WidthOffset), C.int(len(data2)),
		C.int(writeWidth),
	)
	if rc != 0 {
		return errNilIOSurfaceBase
	}
	return nil
}
