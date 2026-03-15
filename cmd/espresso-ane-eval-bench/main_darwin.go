//go:build darwin

package main

/*
#cgo darwin LDFLAGS: -framework IOSurface -framework CoreFoundation
#include <IOSurface/IOSurface.h>
#include <string.h>
#include <stdint.h>

static int probe_surface_write(uintptr_t ref, const void* p, size_t n) {
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

static int probe_surface_read(uintptr_t ref, void* p, size_t n) {
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

	"github.com/maderix/ANE/internal/espressosurface"
	"github.com/tmc/apple/coregraphics"
	xane "github.com/tmc/apple/x/ane"
	xespresso "github.com/tmc/apple/x/espresso"
)

func main() {
	var (
		model  = flag.String("model", "", "compiled .mlmodelc path")
		key    = flag.String("model-key", "s", "model key")
		bytes  = flag.Int("bytes", 4096, "input/output bytes")
		iters  = flag.Int("iters", 200, "timed iterations")
		warmup = flag.Int("warmup", 20, "warmup iterations")
		mode   = flag.String("mode", "both", "raw|espresso|both")
		metalp = flag.Bool("metal-probe", true, "probe Espresso metal buffer on output surface")
	)
	flag.Parse()

	if *model == "" || *bytes <= 0 || *iters <= 0 || *warmup < 0 {
		panic("set -model and valid bytes/iters/warmup")
	}

	rt, err := xane.Open()
	if err != nil {
		panic(err)
	}
	defer rt.Close()

	k, err := rt.Compile(xane.CompileOptions{
		ModelType:   xane.ModelTypePackage,
		PackagePath: *model,
		ModelKey:    *key,
	})
	if err != nil {
		panic(err)
	}
	defer k.Close()
	if k.NumInputs() != 1 || k.NumOutputs() != 1 {
		panic(fmt.Sprintf("compiled model reported %d inputs and %d outputs; want 1 input and 1 output", k.NumInputs(), k.NumOutputs()))
	}
	if got := k.InputLayout(0).LogicalBytes(); got != *bytes {
		panic(fmt.Sprintf("input logical bytes = %d, want %d", got, *bytes))
	}
	if got := k.OutputLayout(0).LogicalBytes(); got != *bytes {
		panic(fmt.Sprintf("output logical bytes = %d, want %d", got, *bytes))
	}

	in := make([]byte, k.InputAllocSize(0))
	out := make([]byte, k.OutputAllocSize(0))
	for i := range in {
		in[i] = byte((i % 251) + 1)
	}

	if *mode == "raw" || *mode == "both" {
		if err := runRaw(k, in, out, *warmup, *iters); err != nil {
			panic(err)
		}
	}
	if *mode == "espresso" || *mode == "both" {
		if err := runEspressoExternal(k, in, out, *warmup, *iters, *metalp); err != nil {
			panic(err)
		}
	}
}

func runRaw(k *xane.Model, in, out []byte, warmup, iters int) error {
	for i := 0; i < warmup; i++ {
		if err := k.WriteInput(0, in); err != nil {
			return fmt.Errorf("raw warmup write: %w", err)
		}
		if err := k.Eval(); err != nil {
			return fmt.Errorf("raw warmup eval: %w", err)
		}
		if err := k.ReadOutput(0, out); err != nil {
			return fmt.Errorf("raw warmup read: %w", err)
		}
	}

	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if err := k.WriteInput(0, in); err != nil {
			return fmt.Errorf("raw write: %w", err)
		}
		if err := k.Eval(); err != nil {
			return fmt.Errorf("raw eval: %w", err)
		}
		if err := k.ReadOutput(0, out); err != nil {
			return fmt.Errorf("raw read: %w", err)
		}
	}
	d := time.Since(t0)
	fmt.Printf("raw_eval:      iters=%d total_ms=%.3f ms_per_iter=%.3f\n",
		iters, float64(d.Microseconds())/1000.0, float64(d.Microseconds())/1000.0/float64(iters))
	return nil
}

func runEspressoExternal(k *xane.Model, in, out []byte, warmup, iters int, metalProbe bool) error {
	inRef := uintptr(k.InputSurface(0))
	outRef := uintptr(k.OutputSurface(0))

	inPool, err := espressosurface.Open(len(in), 1)
	if err != nil {
		return err
	}
	defer inPool.Cleanup()
	outPool, err := espressosurface.Open(len(out), 1)
	if err != nil {
		return err
	}
	defer outPool.Cleanup()

	inPool.SetExternalStorage(0, coregraphics.IOSurfaceRef(inRef))
	outPool.SetExternalStorage(0, coregraphics.IOSurfaceRef(outRef))

	inFrame, err := inPool.IOSurfaceForFrame(0)
	if err != nil {
		return err
	}
	outFrame, err := outPool.IOSurfaceForFrame(0)
	if err != nil {
		return err
	}
	if uintptr(inFrame) != inRef || uintptr(outFrame) != outRef {
		return fmt.Errorf("espresso external storage mismatch")
	}

	if metalProbe {
		dev, err := xespresso.OpenMetal()
		if err == nil {
			defer dev.Close()
			buf, err := outPool.MetalBuffer(dev, 0)
			if err != nil {
				return err
			}
			fmt.Printf("espresso_eval: metal_output_buffer_length=%d\n", buf.Length())
		}
	}

	for i := 0; i < warmup; i++ {
		if st := C.probe_surface_write(C.uintptr_t(inFrame), unsafe.Pointer(&in[0]), C.size_t(len(in))); st != 0 {
			return fmt.Errorf("espresso warmup write status=%d", int(st))
		}
		if err := k.Eval(); err != nil {
			return fmt.Errorf("espresso warmup eval: %w", err)
		}
		if st := C.probe_surface_read(C.uintptr_t(outFrame), unsafe.Pointer(&out[0]), C.size_t(len(out))); st != 0 {
			return fmt.Errorf("espresso warmup read status=%d", int(st))
		}
	}

	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if st := C.probe_surface_write(C.uintptr_t(inFrame), unsafe.Pointer(&in[0]), C.size_t(len(in))); st != 0 {
			return fmt.Errorf("espresso write status=%d", int(st))
		}
		if err := k.Eval(); err != nil {
			return fmt.Errorf("espresso eval: %w", err)
		}
		if st := C.probe_surface_read(C.uintptr_t(outFrame), unsafe.Pointer(&out[0]), C.size_t(len(out))); st != 0 {
			return fmt.Errorf("espresso read status=%d", int(st))
		}
	}
	d := time.Since(t0)
	fmt.Printf("espresso_eval: iters=%d total_ms=%.3f ms_per_iter=%.3f\n",
		iters, float64(d.Microseconds())/1000.0, float64(d.Microseconds())/1000.0/float64(iters))
	return nil
}
