//go:build !darwin

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "ane-concurrency-probe is only supported on darwin")
	os.Exit(2)
}
