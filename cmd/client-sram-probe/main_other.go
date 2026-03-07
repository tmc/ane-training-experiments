//go:build !darwin

package main

import "fmt"

func main() {
	fmt.Println("client-sram-probe requires darwin")
}
