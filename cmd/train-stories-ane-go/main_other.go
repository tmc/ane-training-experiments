//go:build !darwin

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "train-stories-ane-go is only supported on darwin")
	os.Exit(1)
}
