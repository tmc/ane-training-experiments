//go:build darwin

package espressoio

import (
	"fmt"
	"unsafe"

	aneruntime "github.com/maderix/ANE/ane/runtime"
	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/private/espresso"
)

/*
#cgo darwin LDFLAGS: -framework IOSurface -framework CoreFoundation
#include <IOSurface/IOSurface.h>
#include <string.h>
#include <stdint.h>

static int espressoio_surface_write(uintptr_t ref, const void* p, size_t n) {
	IOSurfaceRef s = (IOSurfaceRef)ref;
	if (s == NULL || p == NULL) return -1;
	if (IOSurfaceLock(s, 0, NULL) != kIOReturnSuccess) return -2;
	void* base = IOSurfaceGetBaseAddress(s);
	if (base == NULL) {
		IOSurfaceUnlock(s, 0, NULL);
		return -3;
	}
	memcpy(base, p, n);
	IOSurfaceUnlock(s, 0, NULL);
	return 0;
}

static int espressoio_surface_read(uintptr_t ref, void* p, size_t n) {
	IOSurfaceRef s = (IOSurfaceRef)ref;
	if (s == NULL || p == NULL) return -1;
	if (IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return -2;
	void* base = IOSurfaceGetBaseAddress(s);
	if (base == NULL) {
		IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
		return -3;
	}
	memcpy(p, base, n);
	IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
	return 0;
}
*/
import "C"

// Pool manages EspressoANEIOSurface multi-buffer frames.
type Pool struct {
	surf  espresso.EspressoANEIOSurface
	bytes int
}

// Open creates a multi-buffer IOSurface pool owned by EspressoANEIOSurface.
func Open(bytes int, frames uint64) (*Pool, error) {
	if bytes <= 0 {
		return nil, fmt.Errorf("espressoio open: bytes must be > 0")
	}
	if frames == 0 {
		return nil, fmt.Errorf("espressoio open: frames must be > 0")
	}
	if err := aneruntime.EnsureEspressoLoaded(); err != nil {
		return nil, err
	}

	props := newIOSurfaceProps(uint64(bytes))
	formats := newPixelFormatsSet()
	surf := espresso.NewEspressoANEIOSurfaceWithIOSurfacePropertiesAndPixelFormats(props, formats)
	if surf.GetID() == 0 {
		return nil, fmt.Errorf("espressoio open: EspressoANEIOSurface init failed")
	}
	surf.ResizeForMultipleAsyncBuffers(frames)
	if got := surf.NFrames(); got < frames {
		surf.Release()
		return nil, fmt.Errorf("espressoio open: resize requested %d frames, got %d", frames, got)
	}
	return &Pool{surf: surf, bytes: bytes}, nil
}

// Close releases the EspressoANEIOSurface object.
func (p *Pool) Close() {
	if p == nil || p.surf.GetID() == 0 {
		return
	}
	p.surf.Release()
	p.surf = espresso.EspressoANEIOSurface{}
}

// Bytes reports configured frame bytes.
func (p *Pool) Bytes() int {
	if p == nil {
		return 0
	}
	return p.bytes
}

// Frames reports current number of frames.
func (p *Pool) Frames() uint64 {
	if p == nil || p.surf.GetID() == 0 {
		return 0
	}
	return p.surf.NFrames()
}

// IOSurfaceForFrame returns the frame IOSurfaceRef.
func (p *Pool) IOSurfaceForFrame(frame uint64) (uintptr, error) {
	if p == nil || p.surf.GetID() == 0 {
		return 0, fmt.Errorf("espressoio frame: pool is closed")
	}
	s := p.surf.IoSurfaceForMultiBufferFrame(frame)
	if s == 0 {
		return 0, fmt.Errorf("espressoio frame: frame %d returned nil", frame)
	}
	return uintptr(s), nil
}

// SetExternalFrameStorage binds an external IOSurfaceRef to a frame index.
func (p *Pool) SetExternalFrameStorage(frame uint64, surfaceRef uintptr) error {
	if p == nil || p.surf.GetID() == 0 {
		return fmt.Errorf("espressoio set external: pool is closed")
	}
	if surfaceRef == 0 {
		return fmt.Errorf("espressoio set external: surface is nil")
	}
	p.surf.SetExternalStorageIoSurface(frame, coregraphics.IOSurfaceRef(surfaceRef))
	return nil
}

// RestoreInternalFrameStorage reverts an external frame override.
func (p *Pool) RestoreInternalFrameStorage(frame uint64) error {
	if p == nil || p.surf.GetID() == 0 {
		return fmt.Errorf("espressoio restore: pool is closed")
	}
	p.surf.RestoreInternalStorage(frame)
	return nil
}

// RestoreAllInternalStorage reverts all external overrides.
func (p *Pool) RestoreAllInternalStorage() error {
	if p == nil || p.surf.GetID() == 0 {
		return fmt.Errorf("espressoio restore all: pool is closed")
	}
	p.surf.RestoreInternalStorageForAllMultiBufferFrames()
	return nil
}

// MetalBufferForFrame returns the MTLBuffer for a frame on the given device.
func (p *Pool) MetalBufferForFrame(device metal.MTLDevice, frame uint64) (metal.MTLBuffer, error) {
	if p == nil || p.surf.GetID() == 0 {
		return nil, fmt.Errorf("espressoio metal buffer: pool is closed")
	}
	if device == nil {
		return nil, fmt.Errorf("espressoio metal buffer: device is nil")
	}
	buf := p.surf.MetalBufferWithDeviceMultiBufferFrame(device, frame)
	if buf == nil || buf.GetID() == 0 {
		return nil, fmt.Errorf("espressoio metal buffer: frame %d returned nil", frame)
	}
	return buf, nil
}

// WriteFrame writes bytes into the frame IOSurface.
func (p *Pool) WriteFrame(frame uint64, b []byte) error {
	ref, err := p.IOSurfaceForFrame(frame)
	if err != nil {
		return err
	}
	if len(b) != p.bytes {
		return fmt.Errorf("espressoio write frame: got %d bytes, want %d", len(b), p.bytes)
	}
	if len(b) == 0 {
		return nil
	}
	st := C.espressoio_surface_write(C.uintptr_t(ref), unsafe.Pointer(&b[0]), C.size_t(len(b)))
	if st != 0 {
		return fmt.Errorf("espressoio write frame: status=%d", int(st))
	}
	return nil
}

// ReadFrame reads bytes from the frame IOSurface.
func (p *Pool) ReadFrame(frame uint64, b []byte) error {
	ref, err := p.IOSurfaceForFrame(frame)
	if err != nil {
		return err
	}
	if len(b) != p.bytes {
		return fmt.Errorf("espressoio read frame: got %d bytes, want %d", len(b), p.bytes)
	}
	if len(b) == 0 {
		return nil
	}
	st := C.espressoio_surface_read(C.uintptr_t(ref), unsafe.Pointer(&b[0]), C.size_t(len(b)))
	if st != 0 {
		return fmt.Errorf("espressoio read frame: status=%d", int(st))
	}
	return nil
}

func newIOSurfaceProps(bytes uint64) foundation.NSMutableDictionary {
	num := foundation.GetNSNumberClass()
	props := foundation.NewNSMutableDictionary()
	props.SetObjectForKey(num.NumberWithUnsignedLongLong(bytes), foundation.NewStringWithString("IOSurfaceWidth"))
	props.SetObjectForKey(num.NumberWithUnsignedInt(1), foundation.NewStringWithString("IOSurfaceHeight"))
	props.SetObjectForKey(num.NumberWithUnsignedInt(1), foundation.NewStringWithString("IOSurfaceBytesPerElement"))
	props.SetObjectForKey(num.NumberWithUnsignedLongLong(bytes), foundation.NewStringWithString("IOSurfaceBytesPerRow"))
	props.SetObjectForKey(num.NumberWithUnsignedLongLong(bytes), foundation.NewStringWithString("IOSurfaceAllocSize"))
	props.SetObjectForKey(num.NumberWithUnsignedInt(0), foundation.NewStringWithString("IOSurfacePixelFormat"))
	return props
}

func newPixelFormatsSet() foundation.NSSet {
	num := foundation.GetNSNumberClass()
	return foundation.GetNSSetClass().SetWithObjects(num.NumberWithUnsignedInt(0))
}
