CC = xcrun clang
CFLAGS = -O2 -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate \
         -isysroot $(shell xcrun --show-sdk-path) -fobjc-arc

# Model selection: make MODEL=qwen3_06b  (default)
#   Available: stories110m, qwen3_06b
MODEL ?= qwen3_06b
MODEL_HDR = models/$(MODEL).h

DEFS =
ifneq ($(strip $(SEQ_OVERRIDE)),)
DEFS += -DSEQ=$(SEQ_OVERRIDE)
endif
ifneq ($(strip $(ACCUM_STEPS_OVERRIDE)),)
DEFS += -DACCUM_STEPS=$(ACCUM_STEPS_OVERRIDE)
endif

OUTBIN = $(if $(strip $(OUT)),$(OUT),$@)

train: train.m config.h io.h cpu_ops.h mil_dynamic.h $(MODEL_HDR)
	@echo "Building for model: $(MODEL)"
	$(CC) $(CFLAGS) $(DEFS) -include $(MODEL_HDR) -o $(OUTBIN) train.m

clean:
	rm -f train

.PHONY: clean
