package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/maderix/ANE/ane/stories"
)

func main() {
	var (
		modelPath = flag.String("model", "/Volumes/tmc/go/src/github.com/assets/models/stories110M.bin", "path to stories110M.bin")
		ckptPath  = flag.String("ckpt", "", "optional checkpoint path (overrides -model)")
		tokPath   = flag.String("tokenizer", "", "optional tokenizer.bin path for decoded text")
		promptIDs = flag.String("prompt-ids", "1", "comma-separated prompt token IDs")
		maxNew    = flag.Int("max-new", 64, "maximum new tokens to generate")
		temp      = flag.Float64("temperature", 0.8, "sampling temperature (0 for greedy)")
		seed      = flag.Int64("seed", time.Now().UnixNano(), "sampling seed")
		maxSeq    = flag.Int("seq", stories.SeqDefault, "max sequence length")
	)
	flag.Parse()

	prompt, err := parseTokenList(*promptIDs)
	if err != nil {
		fatalf("parse prompt tokens: %v", err)
	}

	mw, src, err := loadWeights(*modelPath, *ckptPath)
	if err != nil {
		fatalf("load weights: %v", err)
	}

	var tok *stories.Tokenizer
	if *tokPath != "" {
		tok, err = stories.LoadTokenizer(*tokPath)
		if err != nil {
			fatalf("load tokenizer: %v", err)
		}
	}

	dec, err := stories.NewDecoder(mw, *maxSeq, *seed)
	if err != nil {
		fatalf("init decoder: %v", err)
	}

	start := time.Now()
	res, err := dec.Decode(tok, stories.DecodeOptions{
		MaxNewTokens: *maxNew,
		Temperature:  float32(*temp),
		Seed:         *seed,
		PromptTokens: prompt,
	})
	if err != nil {
		fatalf("decode: %v", err)
	}
	elapsed := time.Since(start)

	newTokens := len(res.Tokens) - res.PromptLength
	tokPerSec := 0.0
	if elapsed > 0 {
		tokPerSec = float64(newTokens) / elapsed.Seconds()
	}

	fmt.Printf("source=%s prompt_tokens=%d generated_tokens=%d elapsed=%s tok/s=%.2f eos=%v\n",
		src, res.PromptLength, newTokens, elapsed.Round(time.Millisecond), tokPerSec, res.StoppedOnEOS)
	fmt.Printf("tokens=%s\n", joinInts(res.Tokens))
	if tok != nil {
		fmt.Printf("text=%s\n", res.Text)
	} else {
		fmt.Printf("text=(tokenizer not provided)\n")
	}
}

func loadWeights(modelPath, ckptPath string) (*stories.ModelWeights, string, error) {
	if ckptPath != "" {
		mw := stories.NewModelWeights(stories.Vocab)
		opt := stories.NewOptimState(stories.Vocab)
		if _, err := stories.LoadCheckpointV2(ckptPath, mw, opt); err != nil {
			return nil, "", err
		}
		return mw, "checkpoint", nil
	}
	mw, _, err := stories.LoadPretrained(modelPath)
	if err != nil {
		return nil, "", err
	}
	return mw, "pretrained", nil
}

func parseTokenList(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return []int{1}, nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("token[%d]=%q: %w", i, p, err)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		out = append(out, 1)
	}
	return out, nil
}

func joinInts(v []int) string {
	if len(v) == 0 {
		return ""
	}
	b := strings.Builder{}
	for i, x := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(x))
	}
	return b.String()
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
