//go:build !darwin

package main

import "fmt"

func main() {
	fmt.Println("client-basic requires darwin")
}
