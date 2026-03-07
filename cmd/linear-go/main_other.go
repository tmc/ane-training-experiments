//go:build !darwin

package main

import "fmt"

func main() {
	fmt.Println("linear-go requires darwin")
}
