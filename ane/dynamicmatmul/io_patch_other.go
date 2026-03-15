//go:build !darwin || !cgo

package dynamicmatmul

import "github.com/maderix/ANE/ane/model"

func writeFullTileInput(tile *tile) error {
	return tile.k.WriteInputF32(0, tile.inputPacked)
}

func writeTileRows(tile *tile, rows []int) error {
	return writeFullTileInput(tile)
}

func tileWriteInputF32(tile *tile) error {
	return tile.k.WriteInputF32(0, tile.inputPacked)
}

func tileReadOutputF32(tile *tile) error {
	return tile.k.ReadOutputF32(0, tile.outputPacked)
}

func tileWriteActivationColumnsF32(tile *tile, actCF []float32, batch int) error {
	return writeFullTileInput(tile)
}

func tileCopyOutputToInputFP16(dst *model.Kernel, dstInput, dstChannel int, src *model.Kernel, channels int) error {
	return model.CopyOutputChannelsToInput(dst, dstInput, dstChannel, src, 0, 0, channels)
}
