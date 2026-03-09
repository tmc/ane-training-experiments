//go:build darwin

// Package espressosurface provides shared x/espresso ANE-surface setup helpers.
package espressosurface

import (
	"fmt"

	"github.com/tmc/apple/foundation"
	xespresso "github.com/tmc/apple/x/espresso"
)

// Open creates an ANE surface with the standard single-plane byte layout used
// by the local Espresso interoperability paths.
func Open(bytes int, frames uint64) (*xespresso.ANESurface, error) {
	if bytes <= 0 {
		return nil, fmt.Errorf("espresso surface open: bytes must be > 0")
	}
	if frames == 0 {
		return nil, fmt.Errorf("espresso surface open: frames must be > 0")
	}

	props := newIOSurfaceProps(uint64(bytes))
	formats := newPixelFormatsSet()
	surf := xespresso.NewANESurfaceWithProperties(props, formats)
	surf.ResizeForAsync(frames)
	if got := surf.NFrames(); got < frames {
		surf.Cleanup()
		return nil, fmt.Errorf("espresso surface open: resize requested %d frames, got %d", frames, got)
	}
	return surf, nil
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
