CC = xcrun clang
CFLAGS = -O2 -Wall -DACCELERATE_NEW_LAPACK -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl
VENV = ../.venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h

HEADERS_ANE = $(HEADERS_LARGE) ane_rmsnorm_bwd.h ane_classifier.h

DEFS =
ifneq ($(strip $(SEQ_OVERRIDE)),)
DEFS += -DSEQ=$(SEQ_OVERRIDE)
endif
ifneq ($(strip $(ACCUM_STEPS_OVERRIDE)),)
DEFS += -DACCUM_STEPS=$(ACCUM_STEPS_OVERRIDE)
endif

OUTBIN = $(if $(strip $(OUT)),$(OUT),$@)

train: train.m ane_runtime.h ane_mil_gen.h model.h forward.h backward.h
	$(CC) $(CFLAGS) $(DEFS) -o $(OUTBIN) train.m $(LDFLAGS)

train_large: train_large.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) $(DEFS) -o $(OUTBIN) train_large.m $(LDFLAGS) -framework Accelerate

train_large_ane: train_large_ane.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) $(DEFS) -o $(OUTBIN) train_large_ane.m $(LDFLAGS) -framework Accelerate

PROBES = test_weight_reload test_perf_stats test_qos_sweep test_ane_advanced

test_chaining_matrix: test_chaining_matrix.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -framework Metal -ldl

test_chaining_suite: test_chaining_suite.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -ldl

test_espresso_bindings: test_espresso_bindings.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -F/System/Library/PrivateFrameworks -ldl

test_metal_ane_bridge: test_metal_ane_bridge.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -framework Metal -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -ldl

test_ane_metal_zero_copy: test_ane_metal_zero_copy.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -framework Metal -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -ldl

test_ane_metal_bidirectional: test_ane_metal_bidirectional.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -framework Metal -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -ldl

test_asymmetric_pipeline: test_asymmetric_pipeline.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -framework Metal -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -ldl

test_bridge: test_bridge.m ../bridge/ane_bridge.h ../bridge/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ $< -I../bridge -L../bridge -lane_bridge -Wl,-rpath,@loader_path/../bridge -framework Foundation -framework IOSurface -framework Metal -ldl

test_symmetric_pipeline: test_symmetric_pipeline.m ../bridge/ane_bridge.h ../bridge/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ $< -I../bridge -L../bridge -lane_bridge -Wl,-rpath,@loader_path/../bridge -framework Foundation -framework IOSurface -framework Metal -ldl

test_request_pool_reuse: test_request_pool_reuse.m ../bridge/ane_bridge.h ../bridge/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ $< -I../bridge -L../bridge -lane_bridge -Wl,-rpath,@loader_path/../bridge -framework Foundation -framework IOSurface -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -ldl

test_ffn_bench: test_ffn_bench.m ../bridge/ane_bridge.h ../bridge/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ $< -I../bridge -L../bridge -lane_bridge -Wl,-rpath,@loader_path/../bridge -framework Foundation -framework IOSurface -ldl

test_espresso_gen: test_espresso_gen.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -ldl

test_draft_autoregressive: test_draft_autoregressive.m ../bridge/ane_bridge.h ../bridge/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ $< -I../bridge -L../bridge -lane_bridge -Wl,-rpath,@loader_path/../bridge -framework Foundation -framework IOSurface -ldl

test_inmemory_vs_filebacked: test_inmemory_vs_filebacked.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -ldl

test_full_draft_transformer: test_full_draft_transformer.m
	$(CC) $(CFLAGS) -o $@ $< -framework Foundation -framework IOSurface -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -ldl

test_rmsnorm_bwd: test_rmsnorm_bwd.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

test_classifier: test_classifier.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

test_weight_reload: test_weight_reload.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_perf_stats: test_perf_stats.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_qos_sweep: test_qos_sweep.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_ane_advanced: test_ane_advanced.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

probes: $(PROBES)

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install blessed psutil numpy

tokenize: venv
	$(PYTHON) tokenize.py

dashboard: venv
	$(PYTHON) dashboard.py

clean:
	rm -f train train_large train_large_ane $(PROBES) test_rmsnorm_bwd test_classifier test_chaining_matrix test_chaining_suite test_espresso_bindings test_metal_ane_bridge test_ane_metal_zero_copy test_ane_metal_bidirectional test_asymmetric_pipeline test_bridge test_symmetric_pipeline test_request_pool_reuse test_ffn_bench test_draft_autoregressive test_inmemory_vs_filebacked test_espresso_gen test_full_draft_transformer

.PHONY: clean tokenize probes venv dashboard
