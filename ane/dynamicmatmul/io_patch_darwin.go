//go:build darwin && cgo

package dynamicmatmul

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

// iosurface_write_strided_f32 writes float32 data to an IOSurface with
// the given layout. Single CGo crossing replaces 3 purego calls.
static int iosurface_write_strided_f32(
	IOSurfaceRef surf,
	const float* data,
	int channels, int height, int width,
	int planeStride, int rowStride,
	int allocSize, int dataLen
) {
	void *base = iosurface_lock_and_get_base(surf, 0);
	if (!base) {
		iosurface_unlock(surf, 0);
		return -1;
	}
	char *dst = (char*)base;
	int hw = height * width;
	int limit = dataLen;
	int logical = channels * hw;
	if (limit > logical) limit = logical;
	if (limit == 0) {
		iosurface_unlock(surf, 0);
		return 0;
	}
	for (int c = 0; c < channels; c++) {
		for (int h = 0; h < height; h++) {
			int srcIdx = c * hw + h * width;
			if (srcIdx >= limit) goto done;
			int n = width;
			int remain = limit - srcIdx;
			if (remain < n) n = remain;
			int off = c * planeStride + h * rowStride;
			if (off + n * 4 > allocSize) goto done;
			memcpy(dst + off, data + srcIdx, n * 4);
		}
	}
done:
	iosurface_unlock(surf, 0);
	return 0;
}

// iosurface_read_strided_f32 reads float32 data from an IOSurface.
// Single CGo crossing replaces 3 purego calls.
static int iosurface_read_strided_f32(
	IOSurfaceRef surf,
	float* data,
	int channels, int height, int width,
	int planeStride, int rowStride,
	int allocSize, int dataLen
) {
	void *base = iosurface_lock_and_get_base(surf, 1); // kIOSurfaceLockReadOnly
	if (!base) {
		iosurface_unlock(surf, 1);
		return -1;
	}
	char *src = (char*)base;
	int hw = height * width;
	int limit = dataLen;
	int logical = channels * hw;
	if (limit > logical) limit = logical;
	if (limit == 0) {
		iosurface_unlock(surf, 1);
		return 0;
	}
	for (int c = 0; c < channels; c++) {
		for (int h = 0; h < height; h++) {
			int dstIdx = c * hw + h * width;
			if (dstIdx >= limit) goto done;
			int n = width;
			int remain = limit - dstIdx;
			if (remain < n) n = remain;
			int off = c * planeStride + h * rowStride;
			if (off + n * 4 > allocSize) goto done;
			memcpy(data + dstIdx, src + off, n * 4);
		}
	}
done:
	iosurface_unlock(surf, 1);
	return 0;
}

// iosurface_write_rows_f32 writes specific rows (channels) of float32 data
// to an IOSurface. Used for partial weight updates.
static int iosurface_write_rows_f32(
	IOSurfaceRef surf,
	const float* inputPacked,
	const int* rows, int nRows,
	int rowWidth, int planeStride,
	int channels, int allocSize
) {
	void *base = iosurface_lock_and_get_base(surf, 0);
	if (!base) {
		iosurface_unlock(surf, 0);
		return -1;
	}
	char *dst = (char*)base;
	int rowBytes = rowWidth * 4;
	for (int r = 0; r < nRows; r++) {
		int row = rows[r];
		if (row < 0 || row >= channels) {
			iosurface_unlock(surf, 0);
			return -2;
		}
		int off = row * planeStride;
		if (off < 0 || off + rowBytes > allocSize) {
			iosurface_unlock(surf, 0);
			return -3;
		}
		memcpy(dst + off, inputPacked + row * rowWidth, rowBytes);
	}
	iosurface_unlock(surf, 0);
	return 0;
}
// iosurface_write_activation_cols_f32 writes only the first batchCols elements
// of each channel row to an IOSurface, leaving the remaining (weight) columns
// intact. This avoids re-writing weights that were already primed.
static int iosurface_write_activation_cols_f32(
	IOSurfaceRef surf,
	const float* data,
	int channels, int width,
	int planeStride, int batchCols,
	int allocSize
) {
	void *base = iosurface_lock_and_get_base(surf, 0);
	if (!base) {
		iosurface_unlock(surf, 0);
		return -1;
	}
	char *dst = (char*)base;
	int copyBytes = batchCols * 4;
	for (int c = 0; c < channels; c++) {
		int off = c * planeStride;
		if (off + copyBytes > allocSize) break;
		int srcIdx = c * width;
		memcpy(dst + off, data + srcIdx, copyBytes);
	}
	iosurface_unlock(surf, 0);
	return 0;
}

// iosurface_copy_f32_to_f16 copies channel-first FP32 data from srcSurf
// to FP16 data in dstSurf, converting each element. Both surfaces must
// have height=1 and the same width.
static int iosurface_copy_f32_to_f16(
	IOSurfaceRef dstSurf, int dstPlaneStride, int dstChannel, int dstAllocSize,
	IOSurfaceRef srcSurf, int srcPlaneStride, int srcChannel, int srcAllocSize,
	int channels, int width
) {
	void *dstBase = iosurface_lock_and_get_base(dstSurf, 0);
	void *srcBase = iosurface_lock_and_get_base(srcSurf, 1);
	if (!dstBase || !srcBase) {
		if (srcBase) iosurface_unlock(srcSurf, 1);
		if (dstBase) iosurface_unlock(dstSurf, 0);
		return -1;
	}
	for (int c = 0; c < channels; c++) {
		int dstOff = (dstChannel + c) * dstPlaneStride;
		int srcOff = (srcChannel + c) * srcPlaneStride;
		if (dstOff + width * 2 > dstAllocSize) break;
		if (srcOff + width * 4 > srcAllocSize) break;
		const float *src = (const float*)((char*)srcBase + srcOff);
		__fp16 *dst = (__fp16*)((char*)dstBase + dstOff);
		for (int w = 0; w < width; w++) {
			dst[w] = (__fp16)src[w];
		}
	}
	iosurface_unlock(srcSurf, 1);
	iosurface_unlock(dstSurf, 0);
	return 0;
}
*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/maderix/ANE/ane/model"
	coregraphics "github.com/tmc/apple/coregraphics"
	xane "github.com/tmc/apple/x/ane"
)

func writeFullTileInput(tile *tile) error {
	return writeInputF32CGo(tile.k, 0, tile.inputPacked)
}

func writeTileRows(tile *tile, rows []int) error {
	if len(rows) == 0 {
		return nil
	}
	layout := tile.k.InputLayout(0)
	if layout.Height != 1 || layout.ElemSize != 4 {
		return writeFullTileInput(tile)
	}
	ref := tile.k.InputSurface(0)
	if ref == 0 {
		return fmt.Errorf("dynamic matmul: nil IOSurface ref")
	}
	rc := C.iosurface_write_rows_f32(
		C.IOSurfaceRef(unsafe.Pointer(ref)),
		(*C.float)(unsafe.Pointer(&tile.inputPacked[0])),
		(*C.int)(unsafe.Pointer(&rows[0])), C.int(len(rows)),
		C.int(layout.Width), C.int(layout.PlaneStride),
		C.int(layout.Channels), C.int(layout.AllocSize()),
	)
	if rc != 0 {
		switch rc {
		case -2:
			return fmt.Errorf("dynamic matmul: row out of range")
		case -3:
			return fmt.Errorf("dynamic matmul: row offset out of range")
		default:
			return fmt.Errorf("dynamic matmul: nil IOSurface base address")
		}
	}
	return nil
}

func writeInputF32CGo(k *model.Kernel, input int, data []float32) error {
	layout := k.InputLayout(input)
	ref := k.InputSurface(input)
	if ref == 0 {
		return fmt.Errorf("dynamic matmul: nil input surface %d", input)
	}
	return writeF32ToSurface(ref, data, layout)
}

func readOutputF32CGo(k *model.Kernel, output int, data []float32) error {
	layout := k.OutputLayout(output)
	ref := k.OutputSurface(output)
	if ref == 0 {
		return fmt.Errorf("dynamic matmul: nil output surface %d", output)
	}
	return readF32FromSurface(ref, data, layout)
}

func tileWriteInputF32(tile *tile) error {
	return writeFullTileInput(tile)
}

func tileReadOutputF32(tile *tile) error {
	return readOutputF32CGo(tile.k, 0, tile.outputPacked)
}

func writeF32ToSurface(ref coregraphics.IOSurfaceRef, data []float32, layout xane.TensorLayout) error {
	rc := C.iosurface_write_strided_f32(
		C.IOSurfaceRef(unsafe.Pointer(ref)),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(layout.Channels), C.int(layout.Height), C.int(layout.Width),
		C.int(layout.PlaneStride), C.int(layout.RowStride),
		C.int(layout.AllocSize()), C.int(len(data)),
	)
	if rc != 0 {
		return fmt.Errorf("dynamic matmul: nil IOSurface base address")
	}
	return nil
}

// copyOutputF32ToInputFP16 copies FP32 output channels from src to FP16
// input channels of dst, converting each element. Both surfaces must have
// height=1 and matching widths.
func copyOutputF32ToInputFP16(dst *model.Kernel, dstInput, dstChannel int, src *model.Kernel, srcOutput, srcChannel, channels int) error {
	if dst == nil || src == nil {
		return fmt.Errorf("copy f32 to fp16: nil kernel")
	}
	if channels <= 0 {
		return nil
	}
	dstLayout := dst.InputLayout(dstInput)
	srcLayout := src.OutputLayout(srcOutput)
	if dstLayout.Height != 1 || srcLayout.Height != 1 {
		return fmt.Errorf("copy f32 to fp16: height > 1 not supported")
	}
	if dstLayout.Width != srcLayout.Width {
		return fmt.Errorf("copy f32 to fp16: width mismatch dst=%d src=%d", dstLayout.Width, srcLayout.Width)
	}
	if dstLayout.ElemSize != 2 || srcLayout.ElemSize != 4 {
		return fmt.Errorf("copy f32 to fp16: elem size dst=%d src=%d, want dst=2 src=4", dstLayout.ElemSize, srcLayout.ElemSize)
	}
	dstRef := dst.InputSurface(dstInput)
	srcRef := src.OutputSurface(srcOutput)
	if dstRef == 0 || srcRef == 0 {
		return fmt.Errorf("copy f32 to fp16: nil surface ref")
	}
	rc := C.iosurface_copy_f32_to_f16(
		C.IOSurfaceRef(unsafe.Pointer(dstRef)),
		C.int(dstLayout.PlaneStride), C.int(dstChannel), C.int(dstLayout.AllocSize()),
		C.IOSurfaceRef(unsafe.Pointer(srcRef)),
		C.int(srcLayout.PlaneStride), C.int(srcChannel), C.int(srcLayout.AllocSize()),
		C.int(channels), C.int(dstLayout.Width),
	)
	if rc != 0 {
		return fmt.Errorf("copy f32 to fp16: nil IOSurface base address")
	}
	return nil
}

func tileCopyOutputToInputFP16(dst *model.Kernel, dstInput, dstChannel int, src *model.Kernel, channels int) error {
	return copyOutputF32ToInputFP16(dst, dstInput, dstChannel, src, 0, 0, channels)
}

// writeActivationColsF32 writes only the first batchCols elements of each
// channel row to the IOSurface, leaving weight columns intact.
func writeActivationColsF32(k *model.Kernel, input int, data []float32, batchCols int) error {
	layout := k.InputLayout(input)
	ref := k.InputSurface(input)
	if ref == 0 {
		return fmt.Errorf("dynamic matmul: nil input surface %d", input)
	}
	if layout.Height != 1 {
		return writeF32ToSurface(ref, data, layout)
	}
	rc := C.iosurface_write_activation_cols_f32(
		C.IOSurfaceRef(unsafe.Pointer(ref)),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(layout.Channels), C.int(layout.Width),
		C.int(layout.PlaneStride), C.int(batchCols),
		C.int(layout.AllocSize()),
	)
	if rc != 0 {
		return fmt.Errorf("dynamic matmul: nil IOSurface base address")
	}
	return nil
}

func tileWriteActivationCols(tile *tile, batch int) error {
	return writeActivationColsF32(tile.k, 0, tile.inputPacked, batch)
}

func readF32FromSurface(ref coregraphics.IOSurfaceRef, data []float32, layout xane.TensorLayout) error {
	rc := C.iosurface_read_strided_f32(
		C.IOSurfaceRef(unsafe.Pointer(ref)),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(layout.Channels), C.int(layout.Height), C.int(layout.Width),
		C.int(layout.PlaneStride), C.int(layout.RowStride),
		C.int(layout.AllocSize()), C.int(len(data)),
	)
	if rc != 0 {
		return fmt.Errorf("dynamic matmul: nil IOSurface base address")
	}
	return nil
}
