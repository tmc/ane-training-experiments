//go:build darwin && cgo

package storiesane

/*
#include <dispatch/dispatch.h>

// grad_callback is called by GCD with a context pointer that is really
// an integer index into a Go-side slot table. Defined in Go via //export.
extern void gradCallback(void *ctx);

static void dispatch_grad_job(dispatch_group_t group, dispatch_queue_t queue, void *ctx) {
	dispatch_group_async_f(group, queue, ctx, gradCallback);
}

// dispatch_release wrappers to avoid CGo type-mismatch between
// dispatch_group_t / dispatch_queue_t and dispatch_object_t.
static void release_group(dispatch_group_t g) {
	dispatch_release(g);
}

static void release_queue(dispatch_queue_t q) {
	dispatch_release(q);
}

*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

type gradTasks struct {
	group C.dispatch_group_t
	queue C.dispatch_queue_t
}

func newGradTasks() *gradTasks {
	g := &gradTasks{
		group: C.dispatch_group_create(),
		queue: C.dispatch_queue_create(C.CString("ane.storiesane.dw"), nil),
	}
	runtime.SetFinalizer(g, (*gradTasks).release)
	return g
}

// gradJobSlots avoids passing Go pointers to C. Instead we pass an integer
// slot index as the context pointer and look up the Go function on callback.
var gradJobSlots struct {
	mu    sync.Mutex
	funcs []func()
	free  []int
}

func gradJobAlloc(fn func()) uintptr {
	gradJobSlots.mu.Lock()
	var idx int
	if n := len(gradJobSlots.free); n > 0 {
		idx = gradJobSlots.free[n-1]
		gradJobSlots.free = gradJobSlots.free[:n-1]
		gradJobSlots.funcs[idx] = fn
	} else {
		idx = len(gradJobSlots.funcs)
		gradJobSlots.funcs = append(gradJobSlots.funcs, fn)
	}
	gradJobSlots.mu.Unlock()
	return uintptr(idx + 1) // +1 so zero is never used
}

func gradJobTake(ctx uintptr) func() {
	idx := int(ctx) - 1
	gradJobSlots.mu.Lock()
	fn := gradJobSlots.funcs[idx]
	gradJobSlots.funcs[idx] = nil
	gradJobSlots.free = append(gradJobSlots.free, idx)
	gradJobSlots.mu.Unlock()
	return fn
}

//export gradCallback
func gradCallback(ctx unsafe.Pointer) {
	fn := gradJobTake(uintptr(ctx))
	fn()
}

func (g *gradTasks) Go(fn func()) {
	if g == nil {
		fn()
		return
	}
	slot := gradJobAlloc(fn)
	C.dispatch_grad_job(g.group, g.queue, unsafe.Pointer(slot))
}

func (g *gradTasks) Wait() {
	if g == nil {
		return
	}
	C.dispatch_group_wait(g.group, C.DISPATCH_TIME_FOREVER)
}

func (g *gradTasks) release() {
	if g.group != nil {
		C.release_group(g.group)
		g.group = nil
	}
	if g.queue != nil {
		C.release_queue(g.queue)
		g.queue = nil
	}
}

func (g *gradTasks) Close() {
	if g == nil {
		return
	}
	g.Wait()
	g.release()
	runtime.SetFinalizer(g, nil)
}
