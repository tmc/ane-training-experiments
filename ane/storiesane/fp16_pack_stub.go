//go:build !darwin || !arm64 || !cgo

package storiesane

import xane "github.com/tmc/apple/x/ane"

func writeChannelFirstActsOffsetFP16(data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, width int, x []float32) {
	channels := len(x) / width
	for c := 0; c < channels; c++ {
		row := inputRowFP16(data, layout, channelOffset+c)
		writeContiguousFP16(row[widthOffset:widthOffset+width], x[c*width:(c+1)*width])
	}
}
