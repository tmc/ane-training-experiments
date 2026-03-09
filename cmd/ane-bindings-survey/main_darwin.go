//go:build darwin

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"

	"github.com/maderix/ANE/ane"
	"github.com/maderix/ANE/internal/espressosurface"
	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/private/appleneuralengine"
)

type report struct {
	Probe                                    ane.ProbeReport `json:"probe"`
	SharedConnectionAvailable                bool            `json:"shared_connection_available"`
	SharedPrivateAvailable                   bool            `json:"shared_private_connection_available"`
	SharedConnectionIsVirtualKnown           bool            `json:"shared_connection_is_virtual_known,omitempty"`
	SharedConnectionIsVirtual                bool            `json:"shared_connection_is_virtual,omitempty"`
	SharedConnectionVirtualClientNonNil      bool            `json:"shared_connection_virtual_client_non_nil,omitempty"`
	VirtualClientClassPresent                bool            `json:"virtual_client_class_present"`
	VirtualClientSharedConnectionAvailable   bool            `json:"virtual_client_shared_connection_available"`
	VirtualClientSharedConnectionConnectCode uint32          `json:"virtual_client_shared_connection_connect_code,omitempty"`
	VirtualClientSharedConnectionConnectOK   bool            `json:"virtual_client_shared_connection_connect_ok"`
	VirtualClientInitOK                      bool            `json:"virtual_client_init_ok"`
	VirtualClientConnectCode                 uint32          `json:"virtual_client_connect_code,omitempty"`
	VirtualClientConnectOK                   bool            `json:"virtual_client_connect_ok"`
	Espresso                                 espressoReport  `json:"espresso"`
}

type espressoReport struct {
	FrameworkLoaded bool     `json:"framework_loaded"`
	LoadError       string   `json:"load_error,omitempty"`
	ClassPresent    bool     `json:"class_present"`
	InitOK          bool     `json:"init_ok"`
	Frames          uint64   `json:"frames,omitempty"`
	NonNilFrames    int      `json:"non_nil_frames,omitempty"`
	DistinctFrames  int      `json:"distinct_frames,omitempty"`
	FramePointers   []uint64 `json:"frame_pointers,omitempty"`
}

func main() {
	jsonOut := flag.Bool("json", true, "emit JSON report")
	flag.Parse()

	probe, err := ane.New().Probe(context.Background())
	if err != nil {
		fmt.Printf("probe error: %v\n", err)
		return
	}

	r := report{Probe: probe}

	clientClass := appleneuralengine.GetANEClientClass()
	if obj := clientClass.SharedConnection(); obj.GetID() != 0 {
		r.SharedConnectionAvailable = true
		client := appleneuralengine.ANEClientFromID(obj.GetID())
		r.SharedConnectionIsVirtualKnown = true
		r.SharedConnectionIsVirtual = client.IsVirtualClient()
		if vc := client.VirtualClient(); vc != nil && vc.GetID() != 0 {
			r.SharedConnectionVirtualClientNonNil = true
		}
	}
	if obj := clientClass.SharedPrivateConnection(); obj.GetID() != 0 {
		r.SharedPrivateAvailable = true
	}

	vcClass := appleneuralengine.GetANEVirtualClientClass()
	if objc.GetClass("_ANEVirtualClient") != 0 || objc.GetClass("ANEVirtualClient") != 0 {
		r.VirtualClientClassPresent = true
	}
	if obj := vcClass.SharedConnection(); obj.GetID() != 0 {
		r.VirtualClientSharedConnectionAvailable = true
		vc := appleneuralengine.ANEVirtualClientFromID(obj.GetID())
		code := vc.Connect()
		r.VirtualClientSharedConnectionConnectCode = code
		r.VirtualClientSharedConnectionConnectOK = code == 0
	}

	vc := vcClass.Alloc()
	if vc.GetID() != 0 {
		vc = vc.InitWithSingletonAccess()
		if vc.GetID() != 0 {
			r.VirtualClientInitOK = true
			code := vc.Connect()
			r.VirtualClientConnectCode = code
			r.VirtualClientConnectOK = code == 0
			vc.Release()
		}
	}
	r.Espresso = probeEspresso()

	if *jsonOut {
		enc := json.NewEncoder(flag.CommandLine.Output())
		enc.SetIndent("", "  ")
		_ = enc.Encode(r)
		return
	}

	fmt.Printf("has_ane=%v ne_cores=%d num_anes=%d arch=%q build=%q\n",
		r.Probe.HasANE, r.Probe.NumANECores, r.Probe.NumANEs, r.Probe.Architecture, r.Probe.BuildVersion)
	fmt.Printf("shared_connection=%v shared_private_connection=%v\n", r.SharedConnectionAvailable, r.SharedPrivateAvailable)
	fmt.Printf("shared_connection_is_virtual_known=%v shared_connection_is_virtual=%v shared_connection_virtual_client_non_nil=%v\n",
		r.SharedConnectionIsVirtualKnown, r.SharedConnectionIsVirtual, r.SharedConnectionVirtualClientNonNil)
	fmt.Printf("virtual_shared_connection=%v virtual_shared_connect_ok=%v virtual_shared_connect_code=%d\n",
		r.VirtualClientSharedConnectionAvailable, r.VirtualClientSharedConnectionConnectOK, r.VirtualClientSharedConnectionConnectCode)
	fmt.Printf("virtual_client_class=%v init_ok=%v connect_ok=%v connect_code=%d\n",
		r.VirtualClientClassPresent, r.VirtualClientInitOK, r.VirtualClientConnectOK, r.VirtualClientConnectCode)
	fmt.Printf("espresso_loaded=%v class=%v init=%v frames=%d non_nil_frames=%d distinct_frames=%d\n",
		r.Espresso.FrameworkLoaded, r.Espresso.ClassPresent, r.Espresso.InitOK, r.Espresso.Frames, r.Espresso.NonNilFrames, r.Espresso.DistinctFrames)
}

func probeEspresso() espressoReport {
	rep := espressoReport{
		ClassPresent: objc.GetClass("EspressoANEIOSurface") != 0,
	}
	if !rep.ClassPresent {
		return rep
	}
	rep.FrameworkLoaded = true

	surf, err := espressosurface.Open(4096, 3)
	if err != nil {
		rep.LoadError = err.Error()
		return rep
	}
	defer surf.Cleanup()
	rep.Frames = surf.NFrames()
	if rep.Frames == 0 {
		rep.LoadError = "x/espresso ANESurface init failed"
		return rep
	}
	rep.InitOK = true

	seen := make(map[coregraphics.IOSurfaceRef]struct{}, rep.Frames)
	for i := uint64(0); i < rep.Frames; i++ {
		frame, err := surf.IOSurfaceForFrame(i)
		if err != nil {
			rep.LoadError = err.Error()
			continue
		}
		rep.FramePointers = append(rep.FramePointers, uint64(frame))
		if frame == 0 {
			continue
		}
		rep.NonNilFrames++
		seen[frame] = struct{}{}
	}
	rep.DistinctFrames = len(seen)
	return rep
}
