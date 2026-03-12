//go:build darwin && arm64 && cgo

package storiesane

/*
#cgo CFLAGS: -O3
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

static void storiesane_cvt_f32_f16(_Float16 *dst, const float *src, size_t n) {
	size_t i = 0;
	for (; i + 7 < n; i += 8) {
		float16x8_t h = vcombine_f16(
			vcvt_f16_f32(vld1q_f32(src + i)),
			vcvt_f16_f32(vld1q_f32(src + i + 4)));
		vst1q_f16((__fp16 *)(dst + i), h);
	}
	for (; i < n; i++) {
		dst[i] = (_Float16)src[i];
	}
}

static void storiesane_write_cf_fp16(
	uint16_t *dst,
	size_t dst_len,
	const float *src,
	size_t src_len,
	size_t plane_stride_elems,
	size_t channel_offset,
	size_t width_offset,
	size_t channels,
	size_t width) {
	if (dst == NULL || src == NULL || channels == 0 || width == 0) {
		return;
	}
	size_t logical = channels * width;
	if (src_len < logical) {
		logical = src_len;
	}
	for (size_t c = 0; c < channels; c++) {
		size_t src_off = c * width;
		if (src_off >= logical) {
			return;
		}
		size_t n = width;
		if (logical - src_off < n) {
			n = logical - src_off;
		}
		size_t dst_off = (channel_offset + c) * plane_stride_elems + width_offset;
		if (dst_off + n > dst_len) {
			return;
		}
		storiesane_cvt_f32_f16((_Float16 *)(dst + dst_off), src + src_off, n);
	}
}
*/
import "C"

import (
	"unsafe"

	xane "github.com/tmc/apple/x/ane"
)

func writeChannelFirstActsOffsetFP16(data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, width int, x []float32) {
	if len(data) == 0 || len(x) == 0 || width <= 0 {
		return
	}
	channels := len(x) / width
	if channels <= 0 {
		return
	}
	C.storiesane_write_cf_fp16(
		(*C.uint16_t)(unsafe.Pointer(unsafe.SliceData(data))),
		C.size_t(len(data)),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(x))),
		C.size_t(len(x)),
		C.size_t(layout.PlaneStride/2),
		C.size_t(channelOffset),
		C.size_t(widthOffset),
		C.size_t(channels),
		C.size_t(width),
	)
}
