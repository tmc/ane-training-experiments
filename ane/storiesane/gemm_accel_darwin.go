//go:build darwin && cgo

package storiesane

/*
#cgo darwin CFLAGS: -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

func accumLinearGradCFAccelerate(dW, dy, x []float32, outCh, inCh, seq int) bool {
	if outCh <= 0 || inCh <= 0 || seq <= 0 {
		return false
	}
	if len(dW) < outCh*inCh || len(dy) < outCh*seq || len(x) < inCh*seq {
		return false
	}
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasTrans,
		C.int(outCh),
		C.int(inCh),
		C.int(seq),
		C.float(1.0),
		(*C.float)(&dy[0]),
		C.int(seq),
		(*C.float)(&x[0]),
		C.int(seq),
		C.float(1.0),
		(*C.float)(&dW[0]),
		C.int(inCh),
	)
	return true
}

func linearBackwardDXCFAccelerate(dx, w, dy []float32, outCh, inCh, seq int) bool {
	if outCh <= 0 || inCh <= 0 || seq <= 0 {
		return false
	}
	if len(dx) < inCh*seq || len(w) < outCh*inCh || len(dy) < outCh*seq {
		return false
	}
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasTrans,
		C.CblasNoTrans,
		C.int(inCh),
		C.int(seq),
		C.int(outCh),
		C.float(1.0),
		(*C.float)(&w[0]),
		C.int(inCh),
		(*C.float)(&dy[0]),
		C.int(seq),
		C.float(0.0),
		(*C.float)(&dx[0]),
		C.int(seq),
	)
	return true
}
