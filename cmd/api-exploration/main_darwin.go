//go:build darwin

package main

import (
	"context"
	"encoding/json"
	"log"
	"os"

	"github.com/maderix/ANE/ane"
)

func main() {
	r := ane.New()
	report, err := r.Probe(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		log.Fatal(err)
	}
}
