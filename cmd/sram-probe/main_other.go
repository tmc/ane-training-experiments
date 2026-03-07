//go:build !darwin

package main

import "log"

func main() {
	log.Fatal("sram-probe requires darwin")
}
