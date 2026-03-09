//go:build darwin

package main

/*
#cgo darwin LDFLAGS: -framework IOSurface -framework CoreFoundation
#include <IOSurface/IOSurface.h>
#include <string.h>
#include <stdint.h>

static int bench_surface_write(uintptr_t ref, const void* p, size_t n) {
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

static int bench_surface_read(uintptr_t ref, void* p, size_t n) {
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

import (
	"flag"
	"fmt"
	"time"
	"unsafe"

	aneiosurface "github.com/maderix/ANE/ane/iosurface"
	"github.com/maderix/ANE/internal/espressosurface"
	"github.com/tmc/apple/coregraphics"
	xespresso "github.com/tmc/apple/x/espresso"
)

type rawPool struct {
	frames []*aneiosurface.Surface
}

func newRawPool(bytes int, frames uint64) (*rawPool, error) {
	out := &rawPool{frames: make([]*aneiosurface.Surface, 0, frames)}
	for range frames {
		s, err := aneiosurface.Create(bytes)
		if err != nil {
			out.Close()
			return nil, err
		}
		out.frames = append(out.frames, s)
	}
	return out, nil
}

func (p *rawPool) Close() {
	if p == nil {
		return
	}
	for _, s := range p.frames {
		if s != nil {
			s.Close()
		}
	}
	p.frames = nil
}

func main() {
	var (
		frames   = flag.Uint64("frames", 3, "frame count")
		bytes    = flag.Int("bytes", 4096, "bytes per frame")
		iters    = flag.Int("iters", 20000, "benchmark iterations")
		warmup   = flag.Int("warmup", 2000, "warmup iterations")
		mode     = flag.String("mode", "both", "raw|espresso|both")
		checksum = flag.Bool("checksum", true, "read one byte per iter for anti-DCE")
		metalBuf = flag.Bool("metal", true, "probe metalBufferWithDevice for espresso frames")
		external = flag.Bool("external", true, "run external storage injection check")
	)
	flag.Parse()

	if *frames == 0 || *bytes <= 0 || *iters <= 0 || *warmup < 0 {
		panic("invalid flags")
	}

	payload := make([]byte, *bytes)
	sink := make([]byte, *bytes)
	for i := range payload {
		payload[i] = byte((i % 251) + 1)
	}

	if *mode == "raw" || *mode == "both" {
		if err := runRaw(*bytes, *frames, *warmup, *iters, payload, sink, *checksum); err != nil {
			panic(err)
		}
	}
	if *mode == "espresso" || *mode == "both" {
		if err := runEspresso(*bytes, *frames, *warmup, *iters, payload, sink, *checksum, *metalBuf, *external); err != nil {
			panic(err)
		}
	}
}

func runRaw(bytes int, frames uint64, warmup, iters int, payload, sink []byte, checksum bool) error {
	p, err := newRawPool(bytes, frames)
	if err != nil {
		return fmt.Errorf("raw open: %w", err)
	}
	defer p.Close()

	var sum uint64
	for i := 0; i < warmup; i++ {
		idx := i % len(p.frames)
		if err := p.frames[idx].Write(payload); err != nil {
			return fmt.Errorf("raw warmup write: %w", err)
		}
		if checksum {
			if err := p.frames[idx].Read(sink); err != nil {
				return fmt.Errorf("raw warmup read: %w", err)
			}
			sum += uint64(sink[0])
		}
	}

	t0 := time.Now()
	for i := 0; i < iters; i++ {
		idx := i % len(p.frames)
		if err := p.frames[idx].Write(payload); err != nil {
			return fmt.Errorf("raw write: %w", err)
		}
		if checksum {
			if err := p.frames[idx].Read(sink); err != nil {
				return fmt.Errorf("raw read: %w", err)
			}
			sum += uint64(sink[0])
		}
	}
	d := time.Since(t0)
	fmt.Printf("raw:      frames=%d bytes=%d iters=%d total_ms=%.3f us_per_iter=%.3f checksum=%d\n",
		frames, bytes, iters, float64(d.Microseconds())/1000.0, float64(d.Microseconds())/float64(iters), sum)
	return nil
}

func runEspresso(bytes int, frames uint64, warmup, iters int, payload, sink []byte, checksum, metalBuf, external bool) error {
	p, err := espressosurface.Open(bytes, frames)
	if err != nil {
		return fmt.Errorf("espresso open: %w", err)
	}
	defer p.Cleanup()

	refs := make([]uintptr, frames)
	seen := map[uintptr]struct{}{}
	for i := range frames {
		ref, err := p.IOSurfaceForFrame(i)
		if err != nil {
			return err
		}
		refs[i] = uintptr(ref)
		seen[uintptr(ref)] = struct{}{}
	}
	fmt.Printf("espresso: frames=%d distinct_refs=%d\n", p.NFrames(), len(seen))

	if external {
		raw, err := newRawPool(bytes, frames)
		if err != nil {
			return fmt.Errorf("espresso external raw open: %w", err)
		}
		defer raw.Close()
		for i := range frames {
			p.SetExternalStorage(i, coregraphics.IOSurfaceRef(raw.frames[i].Ref()))
			got, err := p.IOSurfaceForFrame(i)
			if err != nil {
				return err
			}
			if uintptr(got) != raw.frames[i].Ref() {
				return fmt.Errorf("espresso external frame=%d mismatch got=%#x want=%#x", i, uintptr(got), raw.frames[i].Ref())
			}
		}
		p.RestoreAllInternalStorage()
		fmt.Printf("espresso: external storage injection PASS\n")
	}

	if metalBuf {
		dev, err := xespresso.OpenMetal()
		if err == nil {
			defer dev.Close()
			for i := range frames {
				buf, err := p.MetalBuffer(dev, i)
				if err != nil {
					return err
				}
				if n := buf.Length(); n == 0 {
					return fmt.Errorf("espresso metal buffer frame=%d length=0", i)
				}
			}
			fmt.Printf("espresso: metal buffer probe PASS\n")
		} else {
			fmt.Printf("espresso: metal buffer probe SKIP (no default device)\n")
		}
	}

	var sum uint64
	for i := 0; i < warmup; i++ {
		ref := refs[uint64(i)%frames]
		if st := C.bench_surface_write(C.uintptr_t(ref), unsafe.Pointer(&payload[0]), C.size_t(len(payload))); st != 0 {
			return fmt.Errorf("espresso warmup write status=%d", int(st))
		}
		if checksum {
			if st := C.bench_surface_read(C.uintptr_t(ref), unsafe.Pointer(&sink[0]), C.size_t(len(sink))); st != 0 {
				return fmt.Errorf("espresso warmup read status=%d", int(st))
			}
			sum += uint64(sink[0])
		}
	}

	t0 := time.Now()
	for i := 0; i < iters; i++ {
		ref := refs[uint64(i)%frames]
		if st := C.bench_surface_write(C.uintptr_t(ref), unsafe.Pointer(&payload[0]), C.size_t(len(payload))); st != 0 {
			return fmt.Errorf("espresso write status=%d", int(st))
		}
		if checksum {
			if st := C.bench_surface_read(C.uintptr_t(ref), unsafe.Pointer(&sink[0]), C.size_t(len(sink))); st != 0 {
				return fmt.Errorf("espresso read status=%d", int(st))
			}
			sum += uint64(sink[0])
		}
	}
	d := time.Since(t0)
	fmt.Printf("espresso: frames=%d bytes=%d iters=%d total_ms=%.3f us_per_iter=%.3f checksum=%d\n",
		frames, bytes, iters, float64(d.Microseconds())/1000.0, float64(d.Microseconds())/float64(iters), sum)
	return nil
}
