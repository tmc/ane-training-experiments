//go:build !darwin

package main

import "log"

func main() {
	log.Fatal("api-exploration requires darwin")
}
