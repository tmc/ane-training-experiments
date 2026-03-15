//go:build darwin && cgo

package storiesane

/*
#cgo darwin CFLAGS: -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

// siluBackward computes dh1[i] = dGate[i] * h3[i] * sig*(1+h1*(1-sig))
// and dh3[i] = dGate[i] * h1[i] * sig
// where sig = 1/(1+exp(-h1[i]))
//
// Processes in chunks to bound stack usage via alloca. Each chunk uses
// vvexpf for the vectorized exp() calls.
static void silu_backward_f32(
	float* restrict dh1,
	float* restrict dh3,
	const float* restrict dGate,
	const float* restrict h1,
	const float* restrict h3,
	int n
) {
	enum { kChunk = 4096 };
	for (int off = 0; off < n; off += kChunk) {
		int cnt = n - off;
		if (cnt > kChunk) cnt = kChunk;
		float negH1[kChunk];
		float minusOne = -1.0f;
		vDSP_vsmul(h1 + off, 1, &minusOne, negH1, 1, cnt);
		float expBuf[kChunk];
		vvexpf(expBuf, negH1, &cnt);
		for (int i = 0; i < cnt; i++) {
			int idx = off + i;
			float sig = 1.0f / (1.0f + expBuf[i]);
			float siluGrad = sig * (1.0f + h1[idx] * (1.0f - sig));
			dh1[idx] = dGate[idx] * h3[idx] * siluGrad;
			dh3[idx] = dGate[idx] * (h1[idx] * sig);
		}
	}
}

// silu_gate_forward_f32 computes gate[i] = silu(h1[i]) * h3[i]
// where silu(x) = x / (1 + exp(-x)).
// Uses chunked vvexpf for vectorized exp().
static void silu_gate_forward_f32(
	float* restrict gate,
	const float* restrict h1,
	const float* restrict h3,
	int n
) {
	enum { kChunk = 4096 };
	for (int off = 0; off < n; off += kChunk) {
		int cnt = n - off;
		if (cnt > kChunk) cnt = kChunk;
		float negH1[kChunk];
		float minusOne = -1.0f;
		vDSP_vsmul(h1 + off, 1, &minusOne, negH1, 1, cnt);
		float expBuf[kChunk];
		vvexpf(expBuf, negH1, &cnt);
		for (int i = 0; i < cnt; i++) {
			int idx = off + i;
			float sig = 1.0f / (1.0f + expBuf[i]);
			gate[idx] = (h1[idx] * sig) * h3[idx];
		}
	}
}

// softmax_strided_ce_f32 computes softmax + cross-entropy loss for a single token
// in channel-first layout. logits are at logits[i*stride+t] for vocab channels i.
// Writes probs back into dLogits. Returns -log(p[target]) or 0 if target is invalid.
// Uses chunked vvexpf for the vectorized exp().
static float softmax_strided_ce_f32(
	float* restrict dLogits,
	const float* restrict logits,
	int target,
	int vocab,
	int stride,
	int t
) {
	// Find max
	float mx = logits[t];
	for (int i = 1; i < vocab; i++) {
		float v = logits[i*stride + t];
		if (v > mx) mx = v;
	}

	// Gather, subtract max, vectorized exp, scatter, accumulate sum
	enum { kChunk = 4096 };
	float sum = 0.0f;
	for (int off = 0; off < vocab; off += kChunk) {
		int cnt = vocab - off;
		if (cnt > kChunk) cnt = kChunk;
		float buf[kChunk];
		for (int i = 0; i < cnt; i++) {
			buf[i] = logits[(off+i)*stride + t] - mx;
		}
		vvexpf(buf, buf, &cnt);
		for (int i = 0; i < cnt; i++) {
			dLogits[(off+i)*stride + t] = buf[i];
			sum += buf[i];
		}
	}

	// Normalize
	float inv = 1.0f / sum;
	for (int i = 0; i < vocab; i++) {
		dLogits[i*stride + t] *= inv;
	}

	// Loss
	if (target < 0 || target >= vocab) {
		for (int i = 0; i < vocab; i++) {
			dLogits[i*stride + t] = 0;
		}
		return 0.0f;
	}
	float p = dLogits[target*stride + t];
	if (p < 1e-10f) p = 1e-10f;
	dLogits[target*stride + t] -= 1.0f;
	return -logf(p);
}

// softmax_row_f32 computes numerically-stable softmax over n elements.
static void softmax_row_f32(float* out, const float* in, int n) {
	float maxv;
	vDSP_maxv(in, 1, &maxv, n);
	float negMax = -maxv;
	vDSP_vsadd(in, 1, &negMax, out, 1, n);
	vvexpf(out, out, &n);
	float sum;
	vDSP_sve(out, 1, &sum, n);
	float inv = 1.0f / sum;
	vDSP_vsmul(out, 1, &inv, out, 1, n);
}
*/
import "C"

import "unsafe"

func scaleSliceAccel(v []float32, scale float32) {
	if len(v) == 0 || scale == 1 {
		return
	}
	C.vDSP_vsmul(
		(*C.float)(unsafe.Pointer(&v[0])), 1,
		(*C.float)(unsafe.Pointer(&scale)),
		(*C.float)(unsafe.Pointer(&v[0])), 1,
		C.vDSP_Length(len(v)),
	)
}

func addSliceAccel(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	C.vDSP_vadd(
		(*C.float)(unsafe.Pointer(&src[0])), 1,
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		C.vDSP_Length(len(dst)),
	)
}

func scaleIntoAccel(dst, src []float32, scale float32) {
	if len(dst) == 0 {
		return
	}
	C.vDSP_vsmul(
		(*C.float)(unsafe.Pointer(&src[0])), 1,
		(*C.float)(unsafe.Pointer(&scale)),
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		C.vDSP_Length(len(dst)),
	)
}

// addScaledResidualAccel: dst[i] = base[i] + scale*branch[i]
func addScaledResidualAccel(dst, base, branch []float32, scale float32) {
	if len(dst) == 0 {
		return
	}
	// dst = scale * branch
	C.vDSP_vsmul(
		(*C.float)(unsafe.Pointer(&branch[0])), 1,
		(*C.float)(unsafe.Pointer(&scale)),
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		C.vDSP_Length(len(dst)),
	)
	// dst = dst + base
	C.vDSP_vadd(
		(*C.float)(unsafe.Pointer(&base[0])), 1,
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		C.vDSP_Length(len(dst)),
	)
}

func siluGateForwardAccel(gate, h1, h3 []float32) {
	n := len(gate)
	if n == 0 {
		return
	}
	C.silu_gate_forward_f32(
		(*C.float)(unsafe.Pointer(&gate[0])),
		(*C.float)(unsafe.Pointer(&h1[0])),
		(*C.float)(unsafe.Pointer(&h3[0])),
		C.int(n),
	)
}

func siluBackwardAccel(dh1, dh3, dGate, h1, h3 []float32) {
	n := len(dh1)
	if n == 0 {
		return
	}
	C.silu_backward_f32(
		(*C.float)(unsafe.Pointer(&dh1[0])),
		(*C.float)(unsafe.Pointer(&dh3[0])),
		(*C.float)(unsafe.Pointer(&dGate[0])),
		(*C.float)(unsafe.Pointer(&h1[0])),
		(*C.float)(unsafe.Pointer(&h3[0])),
		C.int(n),
	)
}

func softmaxStridedCEAccel(dLogits, logits []float32, target, vocab, stride, t int) float32 {
	return float32(C.softmax_strided_ce_f32(
		(*C.float)(unsafe.Pointer(&dLogits[0])),
		(*C.float)(unsafe.Pointer(&logits[0])),
		C.int(target),
		C.int(vocab),
		C.int(stride),
		C.int(t),
	))
}

func softmaxRowAccel(out, in []float32) {
	n := len(in)
	if n == 0 {
		return
	}
	C.softmax_row_f32(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&in[0])),
		C.int(n),
	)
}

