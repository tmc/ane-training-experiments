//go:build !darwin

package main

import "log"

func main() {
	log.Fatal("inmem-basic requires darwin")
}
