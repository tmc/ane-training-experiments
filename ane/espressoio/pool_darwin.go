//go:build darwin

package espressoio

import (
	"fmt"

	"github.com/maderix/ANE/internal/espressosurface"
	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/metal"
	xespresso "github.com/tmc/apple/x/espresso"
)

// Pool manages multi-buffer ANE surfaces through x/espresso.
type Pool struct {
	surf  *xespresso.ANESurface
	metal *xespresso.MetalDevice
	bytes int
}

// Open creates a multi-buffer IOSurface pool backed by x/espresso.
func Open(bytes int, frames uint64) (*Pool, error) {
	if bytes <= 0 {
		return nil, fmt.Errorf("espressoio open: bytes must be > 0")
	}
	if frames == 0 {
		return nil, fmt.Errorf("espressoio open: frames must be > 0")
	}

	surf, err := espressosurface.Open(bytes, frames)
	if err != nil {
		return nil, fmt.Errorf("espressoio open: %w", err)
	}
	return &Pool{surf: surf, bytes: bytes}, nil
}

// Close releases the underlying ANE surface.
func (p *Pool) Close() {
	if p == nil || p.surf == nil {
		return
	}
	if p.metal != nil {
		_ = p.metal.Close()
		p.metal = nil
	}
	p.surf.Cleanup()
	p.surf = nil
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
	if p == nil || p.surf == nil {
		return 0
	}
	return p.surf.NFrames()
}

// IOSurfaceForFrame returns the frame IOSurfaceRef.
func (p *Pool) IOSurfaceForFrame(frame uint64) (uintptr, error) {
	if p == nil || p.surf == nil {
		return 0, fmt.Errorf("espressoio frame: pool is closed")
	}
	s, err := p.surf.IOSurfaceForFrame(frame)
	if err != nil {
		return 0, fmt.Errorf("espressoio frame: %w", err)
	}
	return uintptr(s), nil
}

// SetExternalFrameStorage binds an external IOSurfaceRef to a frame index.
func (p *Pool) SetExternalFrameStorage(frame uint64, surfaceRef uintptr) error {
	if p == nil || p.surf == nil {
		return fmt.Errorf("espressoio set external: pool is closed")
	}
	if surfaceRef == 0 {
		return fmt.Errorf("espressoio set external: surface is nil")
	}
	p.surf.SetExternalStorage(frame, coregraphics.IOSurfaceRef(surfaceRef))
	return nil
}

// RestoreInternalFrameStorage reverts an external frame override.
func (p *Pool) RestoreInternalFrameStorage(frame uint64) error {
	if p == nil || p.surf == nil {
		return fmt.Errorf("espressoio restore: pool is closed")
	}
	p.surf.RestoreInternalStorage(frame)
	return nil
}

// RestoreAllInternalStorage reverts all external overrides.
func (p *Pool) RestoreAllInternalStorage() error {
	if p == nil || p.surf == nil {
		return fmt.Errorf("espressoio restore all: pool is closed")
	}
	p.surf.RestoreAllInternalStorage()
	return nil
}

// MetalBufferForFrame returns the MTLBuffer for a frame on the given device.
func (p *Pool) MetalBufferForFrame(device metal.MTLDevice, frame uint64) (metal.MTLBuffer, error) {
	if p == nil || p.surf == nil {
		return nil, fmt.Errorf("espressoio metal buffer: pool is closed")
	}
	if device == nil {
		return nil, fmt.Errorf("espressoio metal buffer: device is nil")
	}
	if p.metal == nil {
		metalDev, err := xespresso.OpenMetal()
		if err != nil {
			return nil, fmt.Errorf("espressoio metal buffer: open metal: %w", err)
		}
		p.metal = metalDev
	}
	buf, err := p.surf.MetalBuffer(p.metal, frame)
	if err != nil {
		return nil, fmt.Errorf("espressoio metal buffer: %w", err)
	}
	return buf, nil
}

// WriteFrame writes bytes into the frame IOSurface.
func (p *Pool) WriteFrame(frame uint64, b []byte) error {
	if p == nil || p.surf == nil {
		return fmt.Errorf("espressoio write frame: pool is closed")
	}
	if len(b) != p.bytes {
		return fmt.Errorf("espressoio write frame: got %d bytes, want %d", len(b), p.bytes)
	}
	if len(b) == 0 {
		return nil
	}
	if err := p.surf.WriteFrame(frame, b); err != nil {
		return fmt.Errorf("espressoio write frame: %w", err)
	}
	return nil
}

// WriteFrameF32 writes float32 values into the frame IOSurface.
func (p *Pool) WriteFrameF32(frame uint64, v []float32) error {
	if p == nil || p.surf == nil {
		return fmt.Errorf("espressoio write frame f32: pool is closed")
	}
	if len(v)*4 != p.bytes {
		return fmt.Errorf("espressoio write frame f32: got %d bytes, want %d", len(v)*4, p.bytes)
	}
	if len(v) == 0 {
		return nil
	}
	if err := p.surf.WriteFrameF32(frame, v); err != nil {
		return fmt.Errorf("espressoio write frame f32: %w", err)
	}
	return nil
}

// ReadFrame reads bytes from the frame IOSurface.
func (p *Pool) ReadFrame(frame uint64, b []byte) error {
	if p == nil || p.surf == nil {
		return fmt.Errorf("espressoio read frame: pool is closed")
	}
	if len(b) != p.bytes {
		return fmt.Errorf("espressoio read frame: got %d bytes, want %d", len(b), p.bytes)
	}
	if len(b) == 0 {
		return nil
	}
	if err := p.surf.ReadFrame(frame, b); err != nil {
		return fmt.Errorf("espressoio read frame: %w", err)
	}
	return nil
}

// ReadFrameF32 reads float32 values from the frame IOSurface.
func (p *Pool) ReadFrameF32(frame uint64, v []float32) error {
	if p == nil || p.surf == nil {
		return fmt.Errorf("espressoio read frame f32: pool is closed")
	}
	if len(v)*4 != p.bytes {
		return fmt.Errorf("espressoio read frame f32: got %d bytes, want %d", len(v)*4, p.bytes)
	}
	if len(v) == 0 {
		return nil
	}
	if err := p.surf.ReadFrameF32(frame, v); err != nil {
		return fmt.Errorf("espressoio read frame f32: %w", err)
	}
	return nil
}
