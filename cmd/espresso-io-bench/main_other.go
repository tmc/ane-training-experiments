//go:build !darwin

package main

import "fmt"

func main() {
	fmt.Println("espresso-io-bench requires darwin")
}
