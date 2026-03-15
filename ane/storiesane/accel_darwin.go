//go:build darwin && cgo

package storiesane

/*
#cgo darwin CFLAGS: -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

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
	// Find max using vDSP (handles stride natively).
	float mx;
	vDSP_maxv(logits + t, stride, &mx, vocab);

	// Gather, subtract max, vectorized exp, scatter, accumulate sum.
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

	// Normalize using vDSP (handles stride natively).
	float inv = 1.0f / sum;
	vDSP_vsmul(dLogits + t, stride, &inv, dLogits + t, stride, vocab);

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

// softmax_strided_ce_batch_f32 computes softmax + cross-entropy for tokens [t_start, t_end).
// Returns total loss and valid count. Avoids repeated CGo crossings.
static void softmax_strided_ce_batch_f32(
	float* restrict dLogits,
	const float* restrict logits,
	const unsigned short* restrict targets,
	int vocab,
	int stride,
	int t_start,
	int t_end,
	double* out_loss,
	int* out_valid
) {
	double totalLoss = 0.0;
	int totalValid = 0;
	for (int t = t_start; t < t_end; t++) {
		int target = (int)targets[t];
		float loss = softmax_strided_ce_f32(dLogits, logits, target, vocab, stride, t);
		if (target >= 0 && target < vocab) {
			totalLoss += (double)loss;
			totalValid++;
		}
	}
	*out_loss = totalLoss;
	*out_valid = totalValid;
}

// rms_norm_cf_f32 computes RMS normalization in channel-first layout for all
// tokens in [t_start, t_end). Each token's dim values are at x[d*seq+t] with
// stride=seq. Writes out[d*seq+t] = x[d*seq+t] * scale * w[d] where
// scale = 1/sqrt(mean(x^2) + eps). Optionally writes rrms[t] = scale.
// This replaces seq separate vDSP_svesq CGo calls with a single C function.
static void rms_norm_cf_f32(
	float* restrict out,
	float* restrict rrms,
	const float* restrict x,
	const float* restrict w,
	int dim,
	int seq,
	int t_start,
	int t_end
) {
	float inv_dim = 1.0f / (float)dim;
	for (int t = t_start; t < t_end; t++) {
		float ssq;
		vDSP_svesq(x + t, seq, &ssq, dim);
		float scale = 1.0f / sqrtf(ssq * inv_dim + 1e-5f);
		if (rrms) rrms[t] = scale;
		for (int d = 0; d < dim; d++) {
			out[d * seq + t] = x[d * seq + t] * scale * w[d];
		}
	}
}

// rms_norm_cf_parallel_f32 computes RMS normalization for all seq tokens using
// dispatch_apply for parallelism, avoiding Go goroutine overhead.
static void rms_norm_cf_parallel_f32(
	float* restrict out,
	float* restrict rrms,
	const float* restrict x,
	const float* restrict w,
	int dim,
	int seq
) {
	int workers = 8;
	if (workers > seq) workers = seq;
	if (seq < workers * 4) {
		rms_norm_cf_f32(out, rrms, x, w, dim, seq, 0, seq);
		return;
	}
	int chunk = (seq + workers - 1) / workers;
	dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
	dispatch_apply(workers, queue, ^(size_t wi) {
		int start = (int)wi * chunk;
		int end = start + chunk;
		if (end > seq) end = seq;
		if (start >= seq) return;
		rms_norm_cf_f32(out, rrms, x, w, dim, seq, start, end);
	});
}

// rms_norm_backward_cf_f32 computes both dx and dw for RMS norm backward in
// channel-first layout. Uses dispatch_apply for parallelism, avoiding Go
// goroutine overhead. Each worker computes dx for its token range and
// accumulates dw into a thread-local shard, then the main thread reduces.
static void rms_norm_backward_cf_f32(
	float* restrict dx,
	float* restrict dw,
	const float* restrict dy,
	const float* restrict x,
	const float* restrict w,
	int dim,
	int seq
) {
	int workers = 8;
	if (workers > seq) workers = seq;
	if (seq < workers * 4) workers = 1;
	int chunk = (seq + workers - 1) / workers;

	// Allocate thread-local dw shards.
	float* shards = (float*)calloc(workers * dim, sizeof(float));
	if (!shards) {
		// Fallback: single-threaded.
		for (int t = 0; t < seq; t++) {
			double sum = 0.0;
			for (int d = 0; d < dim; d++) {
				double v = (double)x[d*seq+t];
				sum += v * v;
			}
			double rrms = 1.0 / sqrt(sum / (double)dim + 1e-5);
			double rrms2InvD = (rrms * rrms) / (double)dim;
			double dot = 0.0;
			for (int d = 0; d < dim; d++) {
				dot += (double)(dy[d*seq+t] * x[d*seq+t] * w[d]);
			}
			for (int d = 0; d < dim; d++) {
				double v = (double)dy[d*seq+t] - (double)x[d*seq+t] * dot * rrms2InvD;
				dx[d*seq+t] = (float)(v * rrms * (double)w[d]);
				dw[d] += (float)((double)(dy[d*seq+t] * x[d*seq+t]) * rrms);
			}
		}
		return;
	}

	dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
	dispatch_apply(workers, queue, ^(size_t wi) {
		int start = (int)wi * chunk;
		int end = start + chunk;
		if (end > seq) end = seq;
		if (start >= seq) return;
		float* shard = shards + (int)wi * dim;
		for (int t = start; t < end; t++) {
			double sum = 0.0;
			for (int d = 0; d < dim; d++) {
				double v = (double)x[d*seq+t];
				sum += v * v;
			}
			double rrms = 1.0 / sqrt(sum / (double)dim + 1e-5);
			double rrms2InvD = (rrms * rrms) / (double)dim;
			double dot = 0.0;
			for (int d = 0; d < dim; d++) {
				dot += (double)(dy[d*seq+t] * x[d*seq+t] * w[d]);
			}
			for (int d = 0; d < dim; d++) {
				double v = (double)dy[d*seq+t] - (double)x[d*seq+t] * dot * rrms2InvD;
				dx[d*seq+t] = (float)(v * rrms * (double)w[d]);
				shard[d] += (float)((double)(dy[d*seq+t] * x[d*seq+t]) * rrms);
			}
		}
	});

	// Reduce shards into dw.
	for (int wi = 0; wi < workers; wi++) {
		float* shard = shards + wi * dim;
		vDSP_vadd(shard, 1, dw, 1, dw, 1, dim);
	}
	free(shards);
}

// rms_norm_backward_with_rrms_cf_f32 is like rms_norm_backward_cf_f32 but
// uses precomputed reciprocal-RMS values from the forward pass, skipping
// the sum-of-squares recomputation.
static void rms_norm_backward_with_rrms_cf_f32(
	float* restrict dx,
	float* restrict dw,
	const float* restrict dy,
	const float* restrict x,
	const float* restrict w,
	const float* restrict rrms,
	int dim,
	int seq
) {
	int workers = 8;
	if (workers > seq) workers = seq;
	if (seq < workers * 4) workers = 1;
	int chunk = (seq + workers - 1) / workers;

	float* shards = (float*)calloc(workers * dim, sizeof(float));
	if (!shards) {
		for (int t = 0; t < seq; t++) {
			double r = (double)rrms[t];
			double rrms2InvD = (r * r) / (double)dim;
			double dot = 0.0;
			for (int d = 0; d < dim; d++) {
				dot += (double)(dy[d*seq+t] * x[d*seq+t] * w[d]);
			}
			for (int d = 0; d < dim; d++) {
				double v = (double)dy[d*seq+t] - (double)x[d*seq+t] * dot * rrms2InvD;
				dx[d*seq+t] = (float)(v * r * (double)w[d]);
				dw[d] += (float)((double)(dy[d*seq+t] * x[d*seq+t]) * r);
			}
		}
		return;
	}

	dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
	dispatch_apply(workers, queue, ^(size_t wi) {
		int start = (int)wi * chunk;
		int end = start + chunk;
		if (end > seq) end = seq;
		if (start >= seq) return;
		float* shard = shards + (int)wi * dim;
		for (int t = start; t < end; t++) {
			double r = (double)rrms[t];
			double rrms2InvD = (r * r) / (double)dim;
			double dot = 0.0;
			for (int d = 0; d < dim; d++) {
				dot += (double)(dy[d*seq+t] * x[d*seq+t] * w[d]);
			}
			for (int d = 0; d < dim; d++) {
				double v = (double)dy[d*seq+t] - (double)x[d*seq+t] * dot * rrms2InvD;
				dx[d*seq+t] = (float)(v * r * (double)w[d]);
				shard[d] += (float)((double)(dy[d*seq+t] * x[d*seq+t]) * r);
			}
		}
	});

	for (int wi = 0; wi < workers; wi++) {
		float* shard = shards + wi * dim;
		vDSP_vadd(shard, 1, dw, 1, dw, 1, dim);
	}
	free(shards);
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

import (
	"unsafe"

	"github.com/maderix/ANE/ane/stories"
)

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

// blendResidualInPlaceAccel: sum[i] = base[i] + (sum[i]-base[i])*scale
// Uses vDSP_vintb (vector interpolation): C[i] = A[i] + (B[i]-A[i])*scale
func blendResidualInPlaceAccel(sum, base []float32, scale float32) {
	if len(sum) == 0 {
		return
	}
	// vDSP_vintb: C[i] = A[i] + (B[i] - A[i]) * iScale
	// We want: sum[i] = base[i] + (sum[i] - base[i]) * scale
	// So: A=base, B=sum, C=sum, iScale=scale
	C.vDSP_vintb(
		(*C.float)(unsafe.Pointer(&base[0])), 1,
		(*C.float)(unsafe.Pointer(&sum[0])), 1,
		(*C.float)(unsafe.Pointer(&scale)),
		(*C.float)(unsafe.Pointer(&sum[0])), 1,
		C.vDSP_Length(len(sum)),
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

func softmaxStridedCEBatchAccel(dLogits, logits []float32, targets []uint16, vocab, stride, tStart, tEnd int) (float64, int) {
	var loss C.double
	var valid C.int
	C.softmax_strided_ce_batch_f32(
		(*C.float)(unsafe.Pointer(&dLogits[0])),
		(*C.float)(unsafe.Pointer(&logits[0])),
		(*C.ushort)(unsafe.Pointer(&targets[0])),
		C.int(vocab),
		C.int(stride),
		C.int(tStart),
		C.int(tEnd),
		&loss,
		&valid,
	)
	return float64(loss), int(valid)
}

// rmsNormCFWithRRMSImpl uses a parallel C function with dispatch_apply to
// process all tokens, replacing Go goroutines with GCD threads.
func rmsNormCFWithRRMSImpl(out, rrms, x, w []float32, dim, seq int) {
	var rrmsPtr *C.float
	if rrms != nil {
		rrmsPtr = (*C.float)(unsafe.Pointer(&rrms[0]))
	}
	C.rms_norm_cf_parallel_f32(
		(*C.float)(unsafe.Pointer(&out[0])),
		rrmsPtr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		C.int(dim), C.int(seq),
	)
}

// transposeClassifierForwardTileAccel transposes a size×dim submatrix of embed
// (starting at row `start`) into a dim×size channel-first layout in dst.
// Uses vDSP_mtrans for hardware-accelerated matrix transpose.
func transposeClassifierForwardTileAccel(dst, embed []float32, start, size int) {
	dim := stories.Dim
	src := embed[start*dim : (start+size)*dim]
	// vDSP_mtrans transposes a rows×cols matrix to cols×rows.
	// src is size×dim (rows=size, cols=dim), dst is dim×size.
	C.vDSP_mtrans(
		(*C.float)(unsafe.Pointer(&src[0])), 1,
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		C.vDSP_Length(dim),  // cols of source = rows of destination
		C.vDSP_Length(size), // rows of source = cols of destination
	)
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

// rmsNormBackwardAccel computes both dx and dw for RMS norm backward in a
// single CGo call, using dispatch_apply for parallelism instead of Go goroutines.
// When rrms is non-nil, uses precomputed reciprocal-RMS values from the forward
// pass, skipping the sum-of-squares recomputation.
func rmsNormBackwardAccel(dx, dw, dy, x, w, rrms []float32, dim, seq int) {
	if len(rrms) >= seq {
		C.rms_norm_backward_with_rrms_cf_f32(
			(*C.float)(unsafe.Pointer(&dx[0])),
			(*C.float)(unsafe.Pointer(&dw[0])),
			(*C.float)(unsafe.Pointer(&dy[0])),
			(*C.float)(unsafe.Pointer(&x[0])),
			(*C.float)(unsafe.Pointer(&w[0])),
			(*C.float)(unsafe.Pointer(&rrms[0])),
			C.int(dim),
			C.int(seq),
		)
		return
	}
	C.rms_norm_backward_cf_f32(
		(*C.float)(unsafe.Pointer(&dx[0])),
		(*C.float)(unsafe.Pointer(&dw[0])),
		(*C.float)(unsafe.Pointer(&dy[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		C.int(dim),
		C.int(seq),
	)
}

