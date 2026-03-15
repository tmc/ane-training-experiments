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
