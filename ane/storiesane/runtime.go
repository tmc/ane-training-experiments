package storiesane

import (
	"fmt"
	"time"
)

// Prepare initializes best-effort ANE components so Diagnostics can report
// actual runtime state before the first step.
func (e *Engine) Prepare() {
	if e == nil {
		return
	}
	e.ensureOffload()
	if e.useANE {
		if err := e.ensureLayers(); err != nil {
			e.disableLayerForward(err)
		}
	}
	if e.hybridBackwardRequested {
		_ = e.ensureBackward()
	}
}

func (e *Engine) refreshANERuntimeForWeights() time.Duration {
	if e == nil || !e.useANE {
		return 0
	}
	start := time.Now()
	if e.off == nil {
		e.offDirty = true
		e.ensureOffload()
	} else if err := e.off.refreshWeights(e.mw); err != nil {
		e.offDirty = true
		e.ensureOffload()
	}
	if err := e.refreshLayerWeights(); err != nil {
		e.invalidateLayerForward()
		if ensureErr := e.ensureLayers(); ensureErr != nil {
			e.disableLayerForward(ensureErr)
		}
	}
	if e.hybridBackwardRequested {
		if err := e.refreshBackwardWeights(); err != nil {
			e.invalidateHybridBackward()
			if ensureErr := e.ensureBackward(); ensureErr != nil {
				e.disableHybridBackward(ensureErr)
			}
		}
	}
	dur := time.Since(start)
	e.stepMetrics.addRefresh(dur)
	return dur
}

func (e *Engine) ensureOffload() {
	if e == nil || !e.useANE || e.mw == nil {
		return
	}
	if !e.offDirty && e.off != nil {
		return
	}
	e.off = refreshOffload(e.off, e.mw, e.seq, true)
	e.offDirty = false
}

func (e *Engine) invalidateLayerForward() {
	if e == nil || e.layerInitErr != nil {
		return
	}
	for i := range e.layers {
		if e.layers[i] != nil {
			e.layers[i].close()
		}
	}
	e.layers = nil
	e.layersInit = false
	e.layersDirty = false
}

func (e *Engine) invalidateHybridBackward() {
	if e == nil || e.backwardInitErr != nil {
		return
	}
	for i := range e.backward {
		if e.backward[i] != nil {
			e.backward[i].close()
		}
	}
	e.backward = nil
	e.backwardInit = false
	e.backwardDirty = false
}

func (e *Engine) refreshLayerWeights() error {
	if e == nil || !e.useANE {
		return nil
	}
	if !e.layersInit {
		return e.ensureLayers()
	}
	if e.layerInitErr != nil {
		return e.layerInitErr
	}
	for i := range e.layers {
		if e.layers[i] == nil {
			return fmt.Errorf("refresh layer weights: layer %d is nil", i)
		}
		if err := e.layers[i].refreshWeights(layerForwardWeights{
			RMSAtt: e.mw.Layers[i].RMSAtt,
			Wq:     e.mw.Layers[i].Wq,
			Wk:     e.mw.Layers[i].Wk,
			Wv:     e.mw.Layers[i].Wv,
			Wo:     e.mw.Layers[i].Wo,
			RMSFFN: e.mw.Layers[i].RMSFFN,
			W1:     e.mw.Layers[i].W1,
			W2:     e.mw.Layers[i].W2,
			W3:     e.mw.Layers[i].W3,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (e *Engine) refreshBackwardWeights() error {
	if e == nil || !e.hybridBackwardRequested {
		return nil
	}
	if !e.backwardInit {
		return e.ensureBackward()
	}
	if e.backwardInitErr != nil {
		return e.backwardInitErr
	}
	for i := range e.backward {
		if e.backward[i] == nil {
			return fmt.Errorf("refresh backward weights: layer %d is nil", i)
		}
		if err := e.backward[i].refreshWeights(e.mw.Layers[i]); err != nil {
			return err
		}
	}
	return nil
}
