//go:build !darwin || !cgo

package dynamicmatmul

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
