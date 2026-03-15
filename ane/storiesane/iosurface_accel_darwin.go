//go:build darwin && cgo

package storiesane

/*
#cgo darwin LDFLAGS: -framework IOSurface
#include <IOSurface/IOSurface.h>
#include <string.h>

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
// iosurface_copy_3range_2src copies 3 ranges: 2 from src1 and 1 from src2, all
// to the same destination surface, in a single destination lock/unlock.
static int iosurface_copy_3range_2src(
	IOSurfaceRef dstSurf, int dstPlaneStride, int dstAllocSize, int elemSize,
	IOSurfaceRef src1Surf, int src1PlaneStride, int src1AllocSize,
	int dst1aCh, int dst1aOff, int src1aCh, int src1aOff, int ch1a, int rowBytes1a,
	int dst1bCh, int dst1bOff, int src1bCh, int src1bOff, int ch1b, int rowBytes1b,
	IOSurfaceRef src2Surf, int src2PlaneStride, int src2AllocSize,
	int dst2Ch, int dst2Off, int src2Ch, int src2Off, int ch2, int rowBytes2
) {
	void *dstBase = iosurface_lock_and_get_base(dstSurf, 0);
	void *src1Base = iosurface_lock_and_get_base(src1Surf, 1);
	void *src2Base = iosurface_lock_and_get_base(src2Surf, 1);
	if (!dstBase || !src1Base || !src2Base) {
		if (src2Base) iosurface_unlock(src2Surf, 1);
		if (src1Base) iosurface_unlock(src1Surf, 1);
		if (dstBase) iosurface_unlock(dstSurf, 0);
		return -1;
	}
	for (int c = 0; c < ch1a; c++) {
		int dOff = (dst1aCh + c) * dstPlaneStride + dst1aOff * elemSize;
		int sOff = (src1aCh + c) * src1PlaneStride + src1aOff * elemSize;
		if (dOff >= 0 && dOff + rowBytes1a <= dstAllocSize && sOff >= 0 && sOff + rowBytes1a <= src1AllocSize)
			memcpy((char*)dstBase + dOff, (char*)src1Base + sOff, rowBytes1a);
	}
	for (int c = 0; c < ch1b; c++) {
		int dOff = (dst1bCh + c) * dstPlaneStride + dst1bOff * elemSize;
		int sOff = (src1bCh + c) * src1PlaneStride + src1bOff * elemSize;
		if (dOff >= 0 && dOff + rowBytes1b <= dstAllocSize && sOff >= 0 && sOff + rowBytes1b <= src1AllocSize)
			memcpy((char*)dstBase + dOff, (char*)src1Base + sOff, rowBytes1b);
	}
	for (int c = 0; c < ch2; c++) {
		int dOff = (dst2Ch + c) * dstPlaneStride + dst2Off * elemSize;
		int sOff = (src2Ch + c) * src2PlaneStride + src2Off * elemSize;
		if (dOff >= 0 && dOff + rowBytes2 <= dstAllocSize && sOff >= 0 && sOff + rowBytes2 <= src2AllocSize)
			memcpy((char*)dstBase + dOff, (char*)src2Base + sOff, rowBytes2);
	}
	iosurface_unlock(src2Surf, 1);
	iosurface_unlock(src1Surf, 1);
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

// copy3Range2SrcToInputCGo copies 3 channel ranges (2 from src1, 1 from src2)
// to a single destination input surface in one lock of the destination.
// This saves 2 lock/unlock pairs vs 2 separate copy calls.
func copy3Range2SrcToInputCGo(
	dst *model.Kernel, dstInput int,
	src1 *model.Kernel, src1Output int,
	dst1aCh, dst1aOff, src1aCh, src1aOff, ch1a, width1a int,
	dst1bCh, dst1bOff, src1bCh, src1bOff, ch1b, width1b int,
	src2 *model.Kernel, src2Output int,
	dst2Ch, dst2Off, src2Ch, src2Off, ch2, width2 int,
) error {
	if dst == nil || src1 == nil || src2 == nil {
		return fmt.Errorf("copy 3range 2src to input cgo: nil kernel")
	}
	dstLayout := dst.InputLayout(dstInput)
	src1Layout := src1.OutputLayout(src1Output)
	src2Layout := src2.OutputLayout(src2Output)
	elemSize := dstLayout.ElemSize
	rowBytes1a := width1a * elemSize
	rowBytes1b := width1b * elemSize
	rowBytes2 := width2 * elemSize
	dstRef := dst.InputSurface(dstInput)
	src1Ref := src1.OutputSurface(src1Output)
	src2Ref := src2.OutputSurface(src2Output)
	if dstRef == 0 || src1Ref == 0 || src2Ref == 0 {
		return fmt.Errorf("copy 3range 2src to input cgo: nil surface ref")
	}
	rc := C.iosurface_copy_3range_2src(
		C.IOSurfaceRef(unsafe.Pointer(dstRef)),
		C.int(dstLayout.PlaneStride), C.int(dstLayout.AllocSize()), C.int(elemSize),
		C.IOSurfaceRef(unsafe.Pointer(src1Ref)),
		C.int(src1Layout.PlaneStride), C.int(src1Layout.AllocSize()),
		C.int(dst1aCh), C.int(dst1aOff), C.int(src1aCh), C.int(src1aOff), C.int(ch1a), C.int(rowBytes1a),
		C.int(dst1bCh), C.int(dst1bOff), C.int(src1bCh), C.int(src1bOff), C.int(ch1b), C.int(rowBytes1b),
		C.IOSurfaceRef(unsafe.Pointer(src2Ref)),
		C.int(src2Layout.PlaneStride), C.int(src2Layout.AllocSize()),
		C.int(dst2Ch), C.int(dst2Off), C.int(src2Ch), C.int(src2Off), C.int(ch2), C.int(rowBytes2),
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
