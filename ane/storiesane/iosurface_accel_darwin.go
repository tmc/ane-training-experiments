//go:build darwin && cgo

package storiesane

/*
#cgo darwin LDFLAGS: -framework IOSurface
#include <IOSurface/IOSurface.h>

static void *iosurface_lock_and_get_base(IOSurfaceRef surf, uint32_t options) {
	IOSurfaceLock(surf, options, NULL);
	return IOSurfaceGetBaseAddress(surf);
}

static void iosurface_unlock(IOSurfaceRef surf, uint32_t options) {
	IOSurfaceUnlock(surf, options, NULL);
}
*/
import "C"
import (
	"errors"
	"unsafe"

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
