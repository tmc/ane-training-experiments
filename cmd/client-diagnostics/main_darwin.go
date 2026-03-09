//go:build darwin

package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/maderix/ANE/ane"
	"github.com/maderix/ANE/ane/clientmodel"
	"github.com/tmc/apple/objc"
)

type selectorProbe struct {
	Owner     string `json:"owner"`
	Selector  string `json:"selector"`
	Available bool   `json:"available"`
}

type report struct {
	Probe                               *ane.ProbeReport        `json:"probe,omitempty"`
	ProbeError                          string                  `json:"probe_error,omitempty"`
	ProbeOnly                           bool                    `json:"probe_only,omitempty"`
	HostIsVirtualMachine                bool                    `json:"host_is_virtual_machine,omitempty"`
	HostIsVirtualMachineKnown           bool                    `json:"host_is_virtual_machine_known,omitempty"`
	VirtualClientStatus                 string                  `json:"virtual_client_status,omitempty"`
	CompileError                        string                  `json:"compile_error,omitempty"`
	CompiledModelPath                   string                  `json:"compiled_model_path,omitempty"`
	ForceClientNew                      bool                    `json:"force_client_new"`
	PreferPrivateClient                 bool                    `json:"prefer_private_client"`
	KernelDiagnostics                   clientmodel.Diagnostics `json:"kernel_diagnostics"`
	SelectorAvailability                []selectorProbe         `json:"selector_availability"`
	VirtualSharedConnectionAvailable    bool                    `json:"virtual_shared_connection_available"`
	VirtualSharedConnectionConnectOK    bool                    `json:"virtual_shared_connection_connect_ok"`
	VirtualSharedConnectionConnectTried bool                    `json:"virtual_shared_connection_connect_tried"`
	VirtualSharedConnectionConnectCode  uint32                  `json:"virtual_shared_connection_connect_code,omitempty"`
	VirtualNewProbe                     *virtualNewProbeResult  `json:"virtual_new_probe,omitempty"`
	VirtualNativeProbe                  *nativeVirtualProbe     `json:"virtual_native_probe,omitempty"`
	EvalAttempted                       bool                    `json:"eval_attempted"`
	EvalOK                              bool                    `json:"eval_ok"`
	EvalError                           string                  `json:"eval_error,omitempty"`
	EvalDurationMS                      float64                 `json:"eval_duration_ms,omitempty"`
	OutputSample                        []float32               `json:"output_sample,omitempty"`
}

type virtualNewProbeResult struct {
	Attempted   bool   `json:"attempted"`
	BuildTag    string `json:"build_tag,omitempty"`
	ConnectTry  bool   `json:"connect_tried"`
	ConnectCode uint32 `json:"connect_code,omitempty"`
	ConnectOK   bool   `json:"connect_ok"`
}

type nativeVirtualProbe struct {
	Attempted              bool   `json:"attempted"`
	ClassPresent           bool   `json:"class_present"`
	HasSharedConnection    bool   `json:"has_shared_connection"`
	SharedConnectionNonNil bool   `json:"shared_connection_non_nil"`
	HasNew                 bool   `json:"has_new"`
	NewNonNil              bool   `json:"new_non_nil"`
	HasConnect             bool   `json:"has_connect"`
	ConnectTried           bool   `json:"connect_tried"`
	ConnectCode            uint32 `json:"connect_code,omitempty"`
	ConnectOK              bool   `json:"connect_ok"`
	BuildOrRunError        string `json:"build_or_run_error,omitempty"`
}

type matrixCase struct {
	Name                string
	ForceClientNew      bool
	PreferPrivateClient bool
}

type compileProfile struct {
	ModelType string
	NetPlist  string
}

type ioProfile struct {
	InputBytes  int
	OutputBytes int
}

type summaryStats struct {
	Total int `json:"total"`
	Pass  int `json:"pass"`
	Fail  int `json:"fail"`
}

type matrixEntry struct {
	ModelTarget                      string  `json:"model_target"`
	CaseName                         string  `json:"case"`
	ProbeOnly                        bool    `json:"probe_only,omitempty"`
	VirtualClientStatus              string  `json:"virtual_client_status,omitempty"`
	ModelType                        string  `json:"model_type,omitempty"`
	NetPlist                         string  `json:"net_plist,omitempty"`
	InputBytes                       int     `json:"input_bytes"`
	OutputBytes                      int     `json:"output_bytes"`
	Args                             string  `json:"args,omitempty"`
	ExitOK                           bool    `json:"exit_ok"`
	CompileOK                        bool    `json:"compile_ok"`
	EvalOK                           bool    `json:"eval_ok"`
	EvalMS                           float64 `json:"eval_ms,omitempty"`
	AllowRestrictedAccess            bool    `json:"allow_restricted_access,omitempty"`
	AllowRestrictedKnown             bool    `json:"allow_restricted_known,omitempty"`
	VirtualSharedConnectionAvailable bool    `json:"virtual_shared_connection_available,omitempty"`
	Error                            string  `json:"error,omitempty"`
	Report                           *report `json:"report,omitempty"`
}

type matrixReport struct {
	GeneratedAt string                  `json:"generated_at"`
	ProbeOnly   bool                    `json:"probe_only"`
	Eval        bool                    `json:"eval"`
	Entries     []matrixEntry           `json:"entries"`
	Successes   int                     `json:"successes"`
	Failures    int                     `json:"failures"`
	ByCase      map[string]summaryStats `json:"by_case,omitempty"`
	ByModel     map[string]summaryStats `json:"by_model,omitempty"`
	ByError     map[string]int          `json:"by_error,omitempty"`
}

func main() {
	internalProbeVirtualNew := flag.Bool("internal-probe-virtual-new", false, "internal: subprocess probe for _ANEVirtualClient alloc/new")
	internalProbeVirtualNative := flag.Bool("internal-probe-virtual-native", false, "internal: subprocess probe for _ANEVirtualClient via native ObjC runtime")
	jsonOnError := flag.Bool("json-on-error", false, "emit json report before exiting on compile/eval setup errors")
	matrixMode := flag.Bool("matrix", false, "run diagnostics matrix in subprocess isolation")
	matrixModels := flag.String("matrix-models", "", "comma-separated model targets for matrix mode (compiled paths)")
	matrixModelTypes := flag.String("matrix-model-types", "<empty>", "comma-separated modelType values for matrix mode; use <empty> for empty")
	matrixNetPlists := flag.String("matrix-net-plists", "<empty>", "comma-separated netPlist values for matrix mode; use <empty> for empty")
	matrixIOProfiles := flag.String("matrix-io-profiles", "", "comma-separated io profiles in INxOUT bytes (e.g. 4096x4096,16384x16384)")
	matrixProbeOnly := flag.Bool("matrix-probe-only", false, "run probe-only mode in each matrix case (skip compile/map/eval)")
	matrixEval := flag.Bool("matrix-eval", true, "run eval in matrix mode")
	matrixProbeVirtualNew := flag.Bool("matrix-probe-virtual-new", false, "run virtual-new subprocess probe in each matrix case")
	matrixProbeVirtualNative := flag.Bool("matrix-probe-virtual-native", false, "run native objc virtual-client probe in each matrix case")
	matrixIncludeReport := flag.Bool("matrix-include-report", false, "include full child report payload in matrix entries")
	matrixTimeoutSec := flag.Int("matrix-timeout-sec", 20, "per-case subprocess timeout in matrix mode")
	mlpackage := flag.String("mlpackage", "", "path to source .mlpackage")
	compiled := flag.String("compiled", "", "path to precompiled .mlmodelc")
	modelKey := flag.String("model-key", "s", "_ANEModel key")
	modelType := flag.String("model-type", "", "optional compile option kANEFModelType value")
	netPlist := flag.String("net-plist", "", "optional compile option kANEFNetPlistFilenameKey value")
	forceClientNew := flag.Bool("force-client-new", false, "force dedicated _ANEClient instance instead of shared connection")
	preferPrivateClient := flag.Bool("prefer-private-client", false, "prefer _ANEClient.sharedPrivateConnection over sharedConnection")
	probeOnly := flag.Bool("probe-only", false, "collect selector/client probes without compiling a model")
	probeVirtualNew := flag.Bool("probe-virtual-new", false, "probe _ANEVirtualClient new/initWithSingletonAccess in subprocess isolation")
	probeVirtualNative := flag.Bool("probe-virtual-native", false, "probe _ANEVirtualClient using a native Objective-C subprocess")
	qos := flag.Uint("qos", 21, "ANE QoS")
	inputBytes := flag.Int("input-bytes", 4096, "single input tensor bytes")
	outputBytes := flag.Int("output-bytes", 4096, "single output tensor bytes")
	eval := flag.Bool("eval", true, "run one eval")
	flag.Parse()

	if *internalProbeVirtualNew {
		runInternalVirtualNewProbe()
		return
	}
	if *internalProbeVirtualNative {
		runInternalVirtualNativeProbe()
		return
	}
	if *matrixMode {
		mr, err := runMatrix(matrixOptions{
			models:             parseMatrixModels(*matrixModels),
			compiled:           *compiled,
			mlpackage:          *mlpackage,
			modelKey:           *modelKey,
			modelTypes:         parseMaybeEmptyList(*matrixModelTypes, *modelType),
			netPlists:          parseMaybeEmptyList(*matrixNetPlists, *netPlist),
			ioProfiles:         parseIOProfiles(*matrixIOProfiles, *inputBytes, *outputBytes),
			probeOnly:          *matrixProbeOnly,
			qos:                *qos,
			eval:               *matrixEval,
			probeVirtualNew:    *matrixProbeVirtualNew,
			probeVirtualNative: *matrixProbeVirtualNative,
			includeReport:      *matrixIncludeReport,
			timeoutSecPerRun:   *matrixTimeoutSec,
		})
		if err != nil {
			log.Fatal(err)
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(mr); err != nil {
			log.Fatal(err)
		}
		return
	}

	if !*probeOnly && *mlpackage == "" && *compiled == "" {
		log.Fatal("set -mlpackage or -compiled")
	}
	if !*probeOnly && (*inputBytes <= 0 || *outputBytes <= 0) {
		log.Fatal("input-bytes and output-bytes must be > 0")
	}

	r := report{}
	r.ProbeOnly = *probeOnly
	r.ForceClientNew = *forceClientNew
	r.PreferPrivateClient = *preferPrivateClient
	probe, probeErr := ane.New().Probe(context.Background())
	if probeErr != nil {
		r.ProbeError = probeErr.Error()
	} else {
		r.Probe = &probe
		r.HostIsVirtualMachineKnown = true
		r.HostIsVirtualMachine = probe.IsVirtualMachine
	}
	r.SelectorAvailability, r.VirtualSharedConnectionAvailable, r.VirtualSharedConnectionConnectTried, r.VirtualSharedConnectionConnectOK, r.VirtualSharedConnectionConnectCode = collectSelectorAvailability(*forceClientNew)
	if *probeVirtualNew {
		res, err := runVirtualNewProbeSubprocess()
		if err != nil {
			r.VirtualNewProbe = &virtualNewProbeResult{
				Attempted: true,
				BuildTag:  err.Error(),
			}
		} else {
			r.VirtualNewProbe = &res
		}
	}
	if *probeVirtualNative {
		res, err := runVirtualNativeProbeSubprocess()
		if err != nil {
			r.VirtualNativeProbe = &nativeVirtualProbe{
				Attempted:       true,
				BuildOrRunError: err.Error(),
			}
		} else {
			r.VirtualNativeProbe = &res
		}
	}
	r.VirtualClientStatus = classifyVirtualClientStatus(r)

	if *probeOnly {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(r); err != nil {
			log.Fatal(err)
		}
		return
	}

	k, err := clientmodel.Compile(clientmodel.CompileOptions{
		CompiledModelPath: *compiled,
		ModelPackagePath:  *mlpackage,
		ModelKey:          *modelKey,
		ModelType:         *modelType,
		NetPlistFilename:  *netPlist,
		ForceNewClient:    *forceClientNew,
		PreferPrivateConn: *preferPrivateClient,
		QoS:               uint32(*qos),
		InputBytes:        []int{*inputBytes},
		OutputBytes:       []int{*outputBytes},
	})
	if err != nil {
		if *jsonOnError {
			r.CompileError = err.Error()
			emitReport(r)
			os.Exit(2)
		}
		log.Fatalf("client compile failed: %v", err)
	}
	defer k.Close()

	r.CompiledModelPath = k.CompiledPath()
	r.KernelDiagnostics = k.Diagnostics()

	r.EvalAttempted = *eval
	if *eval {
		in := make([]byte, *inputBytes)
		for i := range in {
			in[i] = byte(i % 251)
		}
		if err := k.WriteInput(0, in); err != nil {
			r.EvalError = fmt.Sprintf("write input: %v", err)
		} else {
			t0 := time.Now()
			err = k.Eval()
			r.EvalDurationMS = float64(time.Since(t0)) / float64(time.Millisecond)
			if err != nil {
				r.EvalError = err.Error()
			} else {
				r.EvalOK = true
				out := make([]byte, *outputBytes)
				if err := k.ReadOutput(0, out); err == nil {
					r.OutputSample = sampleFloats(out, 16)
				}
			}
		}
	}

	emitReport(r)
}

type matrixOptions struct {
	models             []string
	compiled           string
	mlpackage          string
	modelKey           string
	modelTypes         []string
	netPlists          []string
	ioProfiles         []ioProfile
	probeOnly          bool
	qos                uint
	eval               bool
	probeVirtualNew    bool
	probeVirtualNative bool
	includeReport      bool
	timeoutSecPerRun   int
}

func runMatrix(opts matrixOptions) (*matrixReport, error) {
	targets := opts.models
	if len(targets) == 0 && !opts.probeOnly {
		if opts.compiled != "" {
			targets = append(targets, opts.compiled)
		} else if opts.mlpackage != "" {
			targets = append(targets, opts.mlpackage)
		}
	}
	if len(targets) == 0 && !opts.probeOnly {
		return nil, fmt.Errorf("matrix: set -matrix-models or -compiled/-mlpackage")
	}
	if len(targets) == 0 {
		targets = []string{""}
	}
	if opts.timeoutSecPerRun <= 0 {
		opts.timeoutSecPerRun = 20
	}
	if len(opts.modelTypes) == 0 || opts.probeOnly {
		opts.modelTypes = []string{""}
	}
	if len(opts.netPlists) == 0 || opts.probeOnly {
		opts.netPlists = []string{""}
	}
	if len(opts.ioProfiles) == 0 || opts.probeOnly {
		opts.ioProfiles = []ioProfile{{InputBytes: 4096, OutputBytes: 4096}}
	}

	cases := []matrixCase{
		{Name: "shared-default"},
		{Name: "shared-private", PreferPrivateClient: true},
		{Name: "new-default", ForceClientNew: true},
		{Name: "new-private", ForceClientNew: true, PreferPrivateClient: true},
	}
	totalRuns := len(targets) * len(cases) * len(opts.modelTypes) * len(opts.netPlists) * len(opts.ioProfiles)
	if totalRuns > 256 {
		return nil, fmt.Errorf("matrix: refusing to run %d cases (max 256); narrow profiles", totalRuns)
	}

	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("matrix: resolve executable: %w", err)
	}

	out := &matrixReport{
		GeneratedAt: time.Now().Format(time.RFC3339Nano),
		ProbeOnly:   opts.probeOnly,
		Eval:        opts.eval,
		Entries:     make([]matrixEntry, 0, len(targets)*len(cases)),
		ByCase:      make(map[string]summaryStats),
		ByModel:     make(map[string]summaryStats),
		ByError:     make(map[string]int),
	}

	for _, target := range targets {
		target = strings.TrimSpace(target)
		if target == "" && !opts.probeOnly {
			continue
		}
		displayTarget := target
		if displayTarget == "" {
			displayTarget = "<probe-only>"
		}
		for _, mt := range opts.modelTypes {
			for _, np := range opts.netPlists {
				for _, io := range opts.ioProfiles {
					for _, tc := range cases {
						entry := matrixEntry{
							ModelTarget: displayTarget,
							CaseName:    tc.Name,
							ProbeOnly:   opts.probeOnly,
							ModelType:   mt,
							NetPlist:    np,
							InputBytes:  io.InputBytes,
							OutputBytes: io.OutputBytes,
						}
						args := []string{
							"-model-key", opts.modelKey,
							"-qos", fmt.Sprint(opts.qos),
							"-input-bytes", fmt.Sprint(io.InputBytes),
							"-output-bytes", fmt.Sprint(io.OutputBytes),
							fmt.Sprintf("-eval=%t", opts.eval),
							"-json-on-error=true",
							fmt.Sprintf("-force-client-new=%t", tc.ForceClientNew),
							fmt.Sprintf("-prefer-private-client=%t", tc.PreferPrivateClient),
							fmt.Sprintf("-probe-only=%t", opts.probeOnly),
							fmt.Sprintf("-probe-virtual-new=%t", opts.probeVirtualNew),
							fmt.Sprintf("-probe-virtual-native=%t", opts.probeVirtualNative),
						}
						if !opts.probeOnly && mt != "" {
							args = append(args, "-model-type", mt)
						}
						if !opts.probeOnly && np != "" {
							args = append(args, "-net-plist", np)
						}
						if !opts.probeOnly {
							if opts.compiled != "" || strings.HasSuffix(target, ".mlmodelc") {
								args = append(args, "-compiled", target)
							} else {
								args = append(args, "-mlpackage", target)
							}
						}

						ctx, cancel := context.WithTimeout(context.Background(), time.Duration(opts.timeoutSecPerRun)*time.Second)
						cmd := exec.CommandContext(ctx, exe, args...)
						entry.Args = strings.Join(args, " ")
						raw, runErr := cmd.CombinedOutput()
						cancel()

						var child report
						decodeErr := json.Unmarshal(raw, &child)

						if runErr != nil {
							entry.ExitOK = false
							if decodeErr != nil {
								entry.Error = summarizeRunError(raw, ctx.Err() == context.DeadlineExceeded)
								out.Failures++
								updateSummary(out, entry, false)
								out.Entries = append(out.Entries, entry)
								continue
							}
						} else {
							entry.ExitOK = true
							if decodeErr != nil {
								entry.Error = fmt.Sprintf("matrix: decode child json: %v", decodeErr)
								out.Failures++
								updateSummary(out, entry, false)
								out.Entries = append(out.Entries, entry)
								continue
							}
						}
						entry.CompileOK = child.CompiledModelPath != ""
						entry.EvalMS = child.EvalDurationMS
						entry.AllowRestrictedAccess = child.KernelDiagnostics.AllowRestrictedAccess
						entry.AllowRestrictedKnown = child.KernelDiagnostics.AllowRestrictedAccessKnown
						entry.VirtualSharedConnectionAvailable = child.VirtualSharedConnectionAvailable
						entry.VirtualClientStatus = child.VirtualClientStatus
						if child.CompileError != "" {
							entry.Error = child.CompileError
						}
						if child.EvalError != "" && entry.Error == "" {
							entry.Error = child.EvalError
						}
						if opts.includeReport {
							entry.Report = &child
						}
						if opts.eval && !opts.probeOnly {
							entry.EvalOK = child.EvalOK
						} else {
							entry.EvalOK = true
						}
						pass := false
						if opts.probeOnly {
							pass = child.ProbeError == ""
							if !pass {
								entry.Error = child.ProbeError
							}
							if pass && child.VirtualClientStatus == "missing_unexpected_vm" {
								pass = false
								entry.Error = "virtual client missing on VM host"
							}
						} else {
							pass = entry.CompileOK && entry.EvalOK
						}
						if pass {
							out.Successes++
						} else {
							out.Failures++
						}
						updateSummary(out, entry, pass)
						out.Entries = append(out.Entries, entry)
					}
				}
			}
		}
	}

	return out, nil
}

func parseMatrixModels(v string) []string {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parseMaybeEmptyList(value, fallback string) []string {
	if strings.TrimSpace(value) == "" {
		if fallback == "" {
			return []string{""}
		}
		return []string{fallback}
	}
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		switch strings.ToLower(p) {
		case "", "<empty>", "empty", "none", "null":
			out = append(out, "")
		default:
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

func parseIOProfiles(value string, defaultIn, defaultOut int) []ioProfile {
	if strings.TrimSpace(value) == "" {
		return []ioProfile{{InputBytes: defaultIn, OutputBytes: defaultOut}}
	}
	parts := strings.Split(value, ",")
	out := make([]ioProfile, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		chunks := strings.Split(strings.ToLower(p), "x")
		if len(chunks) != 2 {
			continue
		}
		var in, outb int
		if _, err := fmt.Sscanf(chunks[0], "%d", &in); err != nil || in <= 0 {
			continue
		}
		if _, err := fmt.Sscanf(chunks[1], "%d", &outb); err != nil || outb <= 0 {
			continue
		}
		out = append(out, ioProfile{InputBytes: in, OutputBytes: outb})
	}
	if len(out) == 0 {
		return []ioProfile{{InputBytes: defaultIn, OutputBytes: defaultOut}}
	}
	return out
}

func updateSummary(r *matrixReport, entry matrixEntry, pass bool) {
	caseStat := r.ByCase[entry.CaseName]
	caseStat.Total++
	if pass {
		caseStat.Pass++
	} else {
		caseStat.Fail++
	}
	r.ByCase[entry.CaseName] = caseStat

	modelStat := r.ByModel[entry.ModelTarget]
	modelStat.Total++
	if pass {
		modelStat.Pass++
	} else {
		modelStat.Fail++
	}
	r.ByModel[entry.ModelTarget] = modelStat

	if !pass {
		r.ByError[normalizeErrorFingerprint(entry.Error)]++
	}
}

func normalizeErrorFingerprint(err string) string {
	s := strings.TrimSpace(err)
	if s == "" {
		return "failure-without-error-string"
	}
	if strings.Contains(s, "InvalidMILProgram") {
		return "compileModel failed: InvalidMILProgram"
	}
	if strings.Contains(s, "compileModel failed: unknown error") {
		return "compileModel failed: unknown error"
	}
	if strings.Contains(s, "matrix case timed out") {
		return "matrix case timed out"
	}
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if i := strings.Index(s, "client compile failed: "); i >= 0 {
		s = strings.TrimSpace(s[i+len("client compile failed: "):])
	}
	if strings.HasPrefix(s, "compile: ") {
		s = strings.TrimSpace(strings.TrimPrefix(s, "compile: "))
	}
	if i := strings.Index(s, " client "); i >= 0 && i < 30 {
		s = strings.TrimSpace(s[i+1:])
	}
	switch {
	case strings.Contains(s, "compileModel failed:"):
		i := strings.Index(s, "compileModel failed:")
		return strings.TrimSpace(s[i:])
	}
	return s
}

func emitReport(r report) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		log.Fatal(err)
	}
}

func summarizeRunError(raw []byte, timedOut bool) string {
	if timedOut {
		return "matrix case timed out"
	}
	s := strings.TrimSpace(string(raw))
	if s == "" {
		return "subprocess failed without output"
	}
	if strings.Contains(s, "InvalidMILProgram") {
		return "compile: compileModel failed: InvalidMILProgram"
	}
	if strings.Contains(s, "compileModel failed: unknown error") {
		return "compile: compileModel failed: unknown error"
	}
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = strings.TrimSpace(s[:i])
	}
	if i := strings.Index(s, "client compile failed: "); i >= 0 {
		return strings.TrimSpace(s[i+len("client compile failed: "):])
	}
	return s
}

func sampleFloats(buf []byte, max int) []float32 {
	if len(buf) < 4 {
		return nil
	}
	n := len(buf) / 4
	if n > max {
		n = max
	}
	out := make([]float32, 0, n)
	for i := 0; i < n; i++ {
		v := math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		if !isFinite(v) {
			continue
		}
		out = append(out, v)
	}
	return out
}

func isFinite(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func collectSelectorAvailability(forceClientNew bool) ([]selectorProbe, bool, bool, bool, uint32) {
	probes := make([]selectorProbe, 0, 64)
	var (
		virtualSharedAvailable bool
		connectTried           bool
		connectOK              bool
		connectCode            uint32
	)

	clientClass := objc.GetClass("_ANEClient")
	virtualClass := objc.GetClass("_ANEVirtualClient")

	probes = appendSelectorChecks(probes, "_ANEClient(class)", objc.ID(clientClass), []string{
		"sharedConnection",
		"sharedPrivateConnection",
		"new",
	})
	probes = appendSelectorChecks(probes, "_ANEVirtualClient(class)", objc.ID(virtualClass), []string{
		"sharedConnection",
		"new",
	})
	if virtualClass != 0 && objc.RespondsToSelector(objc.ID(virtualClass), objc.Sel("sharedConnection")) {
		virtual := objc.Send[objc.ID](objc.ID(virtualClass), objc.Sel("sharedConnection"))
		virtualSharedAvailable = virtual != 0
		probes = appendSelectorChecks(probes, "_ANEVirtualClient(sharedConnection)", virtual, []string{
			"connect",
			"queue",
			"evaluateWithModel:options:request:qos:error:",
			"doEvaluateWithModel:options:request:qos:completionEvent:error:",
			"doEvaluateWithModelLegacy:options:request:qos:completionEvent:error:",
		})
		if virtual != 0 && objc.RespondsToSelector(virtual, objc.Sel("connect")) {
			connectTried = true
			connectCode = objc.Send[uint32](virtual, objc.Sel("connect"))
			connectOK = connectCode == 0
		}
	}

	client := getANEClientInstance(clientClass, forceClientNew)
	if client != 0 {
		probes = appendSelectorChecks(probes, "_ANEClient(instance)", client, []string{
			"isVirtualClient",
			"virtualClient",
			"mapIOSurfacesWithModel:request:cacheInference:error:",
			"doEvaluateDirectWithModel:options:request:qos:error:",
			"doEvaluateWithModel:options:request:qos:completionEvent:error:",
			"doEvaluateWithModelLegacy:options:request:qos:completionEvent:error:",
		})
		if objc.RespondsToSelector(client, objc.Sel("virtualClient")) {
			virtual := objc.Send[objc.ID](client, objc.Sel("virtualClient"))
			probes = appendSelectorChecks(probes, "_ANEVirtualClient(instance)", virtual, []string{
				"connect",
				"doEvaluateWithModel:options:request:qos:completionEvent:error:",
				"doEvaluateWithModelLegacy:options:request:qos:completionEvent:error:",
				"mapIOSurfacesWithModel:request:cacheInference:error:",
				"doMapIOSurfacesWithModel:request:cacheInference:error:",
			})
		}
	}

	return probes, virtualSharedAvailable, connectTried, connectOK, connectCode
}

func appendSelectorChecks(dst []selectorProbe, owner string, target objc.ID, selectors []string) []selectorProbe {
	for _, name := range selectors {
		dst = append(dst, selectorProbe{
			Owner:     owner,
			Selector:  name,
			Available: target != 0 && objc.RespondsToSelector(target, objc.Sel(name)),
		})
	}
	return dst
}

func getANEClientInstance(class objc.Class, forceNew bool) objc.ID {
	if class == 0 {
		return 0
	}
	selectors := []string{"sharedConnection", "sharedPrivateConnection", "new"}
	if forceNew {
		selectors = []string{"new", "sharedConnection", "sharedPrivateConnection"}
	}
	for _, selName := range selectors {
		sel := objc.Sel(selName)
		if !objc.RespondsToSelector(objc.ID(class), sel) {
			continue
		}
		if id := objc.Send[objc.ID](objc.ID(class), sel); id != 0 {
			return id
		}
	}
	return 0
}

func runVirtualNewProbeSubprocess() (virtualNewProbeResult, error) {
	exe, err := os.Executable()
	if err != nil {
		return virtualNewProbeResult{}, fmt.Errorf("resolve executable: %w", err)
	}
	cmd := exec.Command(exe, "-internal-probe-virtual-new")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return virtualNewProbeResult{}, fmt.Errorf("subprocess failed: %w; output=%s", err, summarizeCommandOutput(out))
	}
	var res virtualNewProbeResult
	if uerr := json.Unmarshal(out, &res); uerr != nil {
		return virtualNewProbeResult{}, fmt.Errorf("decode subprocess json: %w; output=%s", uerr, summarizeCommandOutput(out))
	}
	return res, nil
}

func runVirtualNativeProbeSubprocess() (nativeVirtualProbe, error) {
	exe, err := os.Executable()
	if err != nil {
		return nativeVirtualProbe{}, fmt.Errorf("resolve executable: %w", err)
	}
	cmd := exec.Command(exe, "-internal-probe-virtual-native")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nativeVirtualProbe{}, fmt.Errorf("subprocess failed: %w; output=%s", err, summarizeCommandOutput(out))
	}
	var res nativeVirtualProbe
	if uerr := json.Unmarshal(out, &res); uerr != nil {
		return nativeVirtualProbe{}, fmt.Errorf("decode subprocess json: %w; output=%s", uerr, summarizeCommandOutput(out))
	}
	return res, nil
}

func runInternalVirtualNewProbe() {
	res := virtualNewProbeResult{Attempted: true}
	class := objc.GetClass("_ANEVirtualClient")
	if class == 0 {
		b, _ := json.Marshal(res)
		_, _ = os.Stdout.Write(b)
		_, _ = os.Stdout.WriteString("\n")
		return
	}

	var v objc.ID
	alloc := objc.Send[objc.ID](objc.ID(class), objc.Sel("alloc"))
	if alloc != 0 {
		if objc.RespondsToSelector(alloc, objc.Sel("initWithSingletonAccess")) {
			v = objc.Send[objc.ID](alloc, objc.Sel("initWithSingletonAccess"))
			res.BuildTag = "alloc/initWithSingletonAccess"
		}
		if v == 0 && objc.RespondsToSelector(alloc, objc.Sel("init")) {
			v = objc.Send[objc.ID](alloc, objc.Sel("init"))
			res.BuildTag = "alloc/init"
		}
	}
	if v == 0 && objc.RespondsToSelector(objc.ID(class), objc.Sel("new")) {
		v = objc.Send[objc.ID](objc.ID(class), objc.Sel("new"))
		res.BuildTag = "new"
	}

	if v != 0 && objc.RespondsToSelector(v, objc.Sel("connect")) {
		res.ConnectTry = true
		res.ConnectCode = objc.Send[uint32](v, objc.Sel("connect"))
		res.ConnectOK = res.ConnectCode == 0
	}
	if v != 0 && objc.RespondsToSelector(v, objc.Sel("release")) {
		objc.Send[struct{}](v, objc.Sel("release"))
	}

	enc := json.NewEncoder(os.Stdout)
	_ = enc.Encode(res)
}

func runInternalVirtualNativeProbe() {
	res := nativeVirtualProbe{Attempted: true}
	tmp, err := os.MkdirTemp("", "ane-virtual-probe-*")
	if err != nil {
		res.BuildOrRunError = fmt.Sprintf("mkdtemp: %v", err)
		_ = json.NewEncoder(os.Stdout).Encode(res)
		return
	}
	defer os.RemoveAll(tmp)

	src := filepath.Join(tmp, "probe.m")
	bin := filepath.Join(tmp, "probe")
	if err := os.WriteFile(src, []byte(nativeVirtualProbeSource), 0o644); err != nil {
		res.BuildOrRunError = fmt.Sprintf("write probe source: %v", err)
		_ = json.NewEncoder(os.Stdout).Encode(res)
		return
	}

	clangArgs := []string{
		"-Wall", "-Wextra", "-O2", "-fobjc-arc",
		"-framework", "Foundation",
		"-F/System/Library/PrivateFrameworks",
		"-framework", "AppleNeuralEngine",
		src,
		"-o", bin,
	}
	buildCmd := exec.Command("clang", clangArgs...)
	if out, err := buildCmd.CombinedOutput(); err != nil {
		res.BuildOrRunError = fmt.Sprintf("clang failed: %v; output=%s", err, strings.TrimSpace(string(out)))
		_ = json.NewEncoder(os.Stdout).Encode(res)
		return
	}

	runCmd := exec.Command(bin)
	out, err := runCmd.CombinedOutput()
	if err != nil {
		res.BuildOrRunError = fmt.Sprintf("probe run failed: %v; output=%s", err, strings.TrimSpace(string(out)))
		_ = json.NewEncoder(os.Stdout).Encode(res)
		return
	}
	var child nativeVirtualProbe
	if err := json.Unmarshal(out, &child); err != nil {
		res.BuildOrRunError = fmt.Sprintf("decode probe json: %v; output=%s", err, strings.TrimSpace(string(out)))
		_ = json.NewEncoder(os.Stdout).Encode(res)
		return
	}
	_ = json.NewEncoder(os.Stdout).Encode(child)
}

func summarizeCommandOutput(out []byte) string {
	s := strings.TrimSpace(string(out))
	if s == "" {
		return ""
	}
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return s
}

func classifyVirtualClientStatus(r report) string {
	if r.VirtualSharedConnectionAvailable {
		return "available"
	}
	if r.VirtualNativeProbe != nil {
		if r.VirtualNativeProbe.SharedConnectionNonNil || r.VirtualNativeProbe.NewNonNil {
			return "available"
		}
	}
	if r.HostIsVirtualMachineKnown {
		if r.HostIsVirtualMachine {
			return "missing_unexpected_vm"
		}
		return "skip_expected_non_vm"
	}
	return "missing_unknown_host"
}

const nativeVirtualProbeSource = `#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>

int main(void) {
	@autoreleasepool {
		Class cls = NSClassFromString(@"_ANEVirtualClient");
		BOOL classPresent = (cls != Nil);
		BOOL hasShared = classPresent && class_respondsToSelector(object_getClass(cls), @selector(sharedConnection));
		id shared = nil;
		if (hasShared) {
			shared = ((id (*)(id, SEL))objc_msgSend)(cls, @selector(sharedConnection));
		}
		BOOL hasNew = classPresent && class_respondsToSelector(object_getClass(cls), @selector(new));
		id newObj = nil;
		if (hasNew) {
			newObj = ((id (*)(id, SEL))objc_msgSend)(cls, @selector(new));
		}

		id target = shared ? shared : newObj;
		BOOL hasConnect = target && [target respondsToSelector:@selector(connect)];
		BOOL connectTried = NO;
		unsigned int connectCode = 0;
		BOOL connectOK = NO;
		if (hasConnect) {
			connectTried = YES;
			connectCode = ((unsigned int (*)(id, SEL))objc_msgSend)(target, @selector(connect));
			connectOK = (connectCode == 0);
		}

		printf("{\"attempted\":true,"
		       "\"class_present\":%s,"
		       "\"has_shared_connection\":%s,"
		       "\"shared_connection_non_nil\":%s,"
		       "\"has_new\":%s,"
		       "\"new_non_nil\":%s,"
		       "\"has_connect\":%s,"
		       "\"connect_tried\":%s,"
		       "\"connect_code\":%u,"
		       "\"connect_ok\":%s}\n",
		       classPresent ? "true" : "false",
		       hasShared ? "true" : "false",
		       shared != nil ? "true" : "false",
		       hasNew ? "true" : "false",
		       newObj != nil ? "true" : "false",
		       hasConnect ? "true" : "false",
		       connectTried ? "true" : "false",
		       connectCode,
		       connectOK ? "true" : "false");
	}
	return 0;
}
`
