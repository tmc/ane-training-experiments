package storiesane

import "time"

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
	e.invalidateLayerForward()
	if e.hybridBackwardRequested {
		e.invalidateHybridBackward()
	}
	e.offDirty = true
	e.ensureOffload()
	if err := e.ensureLayers(); err != nil {
		e.disableLayerForward(err)
	}
	if e.hybridBackwardRequested {
		if err := e.ensureBackward(); err != nil {
			e.disableHybridBackward(err)
		}
	}
	return time.Since(start)
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
