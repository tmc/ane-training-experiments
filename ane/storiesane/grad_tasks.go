package storiesane

import (
	"runtime"
	"sync"
)

const maxGradTaskConcurrency = 3

type gradTasks struct {
	sem chan struct{}
	wg  sync.WaitGroup
}

func newGradTasks() *gradTasks {
	n := gradTaskConcurrency()
	if n <= 1 {
		return nil
	}
	return &gradTasks{sem: make(chan struct{}, n)}
}

func gradTaskConcurrency() int {
	n := runtime.GOMAXPROCS(0)
	if n < 2 {
		return 1
	}
	if n > maxGradTaskConcurrency {
		n = maxGradTaskConcurrency
	}
	return n
}

func (g *gradTasks) Go(fn func()) {
	if g == nil {
		fn()
		return
	}
	g.sem <- struct{}{}
	g.wg.Add(1)
	go func() {
		defer g.wg.Done()
		defer func() {
			<-g.sem
		}()
		fn()
	}()
}

func (g *gradTasks) Wait() {
	if g == nil {
		return
	}
	g.wg.Wait()
}
