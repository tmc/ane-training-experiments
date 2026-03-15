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
