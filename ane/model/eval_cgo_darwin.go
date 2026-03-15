//go:build darwin && cgo

package model

/*
#cgo darwin LDFLAGS: -lobjc
#include <objc/message.h>
#include <objc/runtime.h>
#include <stdint.h>

// ane_eval calls [_ANEInMemoryModel evaluateWithQoS:options:request:error:]
// using a single CGo crossing instead of purego's ObjC messaging.
// Returns 1 on success, 0 on failure (sets *outErr to the NSError pointer).
static int ane_eval(void* model, uint32_t qos, void* request, void** outErr) {
	static SEL sel = 0;
	if (sel == 0) {
		sel = sel_registerName("evaluateWithQoS:options:request:error:");
	}
	// evaluateWithQoS:(uint32)qos options:(id)nil request:(id)req error:(id*)err
	// returns BOOL
	typedef signed char (*EvalFn)(id, SEL, uint32_t, id, id, id*);
	EvalFn fn = (EvalFn)objc_msgSend;
	id errObj = 0;
	signed char ok = fn((id)model, sel, qos, (id)0, (id)request, &errObj);
	if (errObj != 0) {
		// Retain the error so it survives autorelease pool drain.
		((id (*)(id, SEL))objc_msgSend)(errObj, sel_registerName("retain"));
		*outErr = (void*)errObj;
		return 0;
	}
	if (!ok) {
		*outErr = 0;
		return 0;
	}
	return 1;
}
*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objc"
)

func init() {
	evalCGoFunc = evalCGo
}

func evalCGo(h *sharedMILHandle) error {
	if h == nil || h.program == nil {
		return fmt.Errorf("eval: kernel is closed")
	}
	modelID := uintptr(h.program.inMemModel.ID)
	requestID := uintptr(h.request.ID)
	var errPtr unsafe.Pointer
	rc := C.ane_eval(
		unsafe.Pointer(modelID),
		C.uint(h.program.qos),
		unsafe.Pointer(requestID),
		&errPtr,
	)
	if rc == 1 {
		return nil
	}
	if errPtr != nil {
		return fmt.Errorf("eval: %w", foundation.NSErrorFrom(objc.ID(uintptr(errPtr))))
	}
	return fmt.Errorf("eval: returned NO with nil NSError")
}
