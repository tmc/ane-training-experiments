//go:build !darwin

package main

import "fmt"

func main() {
	fmt.Println("espresso-ane-eval-bench requires darwin")
}
