//go:build !darwin

package main

import "log"

func main() {
	log.Fatal("inmem-peak requires darwin")
}
