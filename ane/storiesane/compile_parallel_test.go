package storiesane

import (
	"errors"
	"sync/atomic"
	"testing"
)

func TestCompileParallelPreservesOrder(t *testing.T) {
	values, err := compileParallel(4, func(i int) (int, error) {
		return i * 10, nil
	}, func(int) {})
	if err != nil {
		t.Fatalf("compileParallel: %v", err)
	}
	want := []int{0, 10, 20, 30}
	for i, v := range want {
		if values[i] != v {
			t.Fatalf("values[%d]=%d want %d", i, values[i], v)
		}
	}
}

func TestCompileParallelClosesOnError(t *testing.T) {
	var closed atomic.Int32
	type compiled struct {
		id int
	}
	_, err := compileParallel(4, func(i int) (*compiled, error) {
		if i == 2 {
			return nil, errors.New("boom")
		}
		return &compiled{id: i}, nil
	}, func(v *compiled) {
		if v != nil {
			closed.Add(1)
		}
	})
	if err == nil {
		t.Fatal("compileParallel succeeded; want error")
	}
	if got := closed.Load(); got != 3 {
		t.Fatalf("closed=%d want 3", got)
	}
}
