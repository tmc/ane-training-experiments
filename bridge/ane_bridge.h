// ane_bridge.h — C-callable bridge to ANE private APIs for Python ctypes
// Wraps _ANEInMemoryModel via private AppleNeuralEngine.framework

#ifndef ANE_BRIDGE_H
#define ANE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <mach/mach.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque kernel handle
typedef struct ANEKernelHandle ANEKernelHandle;
typedef struct ANEClientHandle ANEClientHandle;
typedef struct __IOSurface *IOSurfaceRef;
typedef ANEClientHandle *ane_bridge_client_t;
typedef void (*ane_bridge_token_callback_t)(const float *logits, uint32_t logit_count,
                                            float *next_input, uint32_t input_count,
                                            void *ctx);

// Initialize ANE runtime (load private framework, resolve classes)
// Returns 0 on success, -1 on failure
int ane_bridge_init(void);

// Compile a MIL program with weight blobs into an ANE kernel
// mil_text: UTF-8 MIL program text
// mil_len: length of MIL text
// weight_data: raw weight blob (can be NULL)
// weight_len: length of weight blob
// n_inputs: number of input tensors
// input_sizes: array of byte sizes for each input
// n_outputs: number of output tensors
// output_sizes: array of byte sizes for each output
// Returns kernel handle or NULL on failure
ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes);

// Compile with multiple named weight files (for transformer kernels)
// weight_names: array of weight file paths (e.g. "@model_path/weights/wq.bin")
// weight_datas: array of weight data pointers
// weight_lens: array of weight data lengths
// n_weights: number of weight files
ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Evaluate (run) a compiled kernel on ANE
// Returns true on success
bool ane_bridge_eval(ANEKernelHandle *kernel);

// Write data to kernel input tensor
void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes);

// Read data from kernel output tensor
void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes);

// Free a compiled kernel and all associated resources
void ane_bridge_free(ANEKernelHandle *kernel);

// Get compile count (for exec() restart budgeting)
int ane_bridge_get_compile_count(void);

// Reset compile count
void ane_bridge_reset_compile_count(void);

// Build a weight blob in ANE format (128-byte header + fp16 data)
// src: float32 weights [rows x cols]
// Returns allocated buffer and sets out_len. Caller must free().
uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len);

// Build a transposed weight blob in ANE format
uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len);

// Build an int8 weight blob in ANE format (64-byte header + int8 data per chunk)
// src: int8 weights [rows x cols], scale: dequantization scale, zero_point: int8 zero
// For use with constexpr_affine_dequantize in MIL
// Returns allocated buffer and sets out_len. Caller must free().
uint8_t *ane_bridge_build_weight_blob_int8(const int8_t *src, int rows, int cols,
                                            size_t *out_len);

// Quantize float32 weights to int8 and build ANE blob in one step
// Computes per-channel (axis=0) scale = max(abs(row)) / 127
// Returns allocated buffer, sets out_len and out_scale. Caller must free().
uint8_t *ane_bridge_build_weight_blob_quantized(const float *src, int rows, int cols,
                                                 float *out_scale, size_t *out_len);

// Free a blob allocated by ane_bridge_build_weight_blob*
void ane_bridge_free_blob(void *ptr);

// --- Asymmetric Metal->ANE bridge primitives ---

// Open a daemon-backed _ANEClient model for wait-event eval.
// Returns NULL on failure.
ANEClientHandle *ane_bridge_client_open(const char *model_path, const char *model_key,
                                        size_t input_bytes, size_t output_bytes);

// Close and free a client handle.
void ane_bridge_client_close(ANEClientHandle *h);

// Access mapped IO surfaces for zero-copy interop.
IOSurfaceRef ane_bridge_client_input_surface(ANEClientHandle *h);
IOSurfaceRef ane_bridge_client_output_surface(ANEClientHandle *h);

// Convenience surface I/O helpers for client handle.
void ane_bridge_client_write_input(ANEClientHandle *h, const float *data, int count);
void ane_bridge_client_read_output(ANEClientHandle *h, float *data, int count);

// Baseline eval without shared events.
bool ane_bridge_client_eval(ANEClientHandle *h);

// Evaluate N tokens autoregressively while keeping one request mapping active.
// Between tokens, token_callback prepares the next input embedding.
// Returns 0 on success, negative on failure.
int ane_bridge_eval_loopback(ane_bridge_client_t client,
                             const float *initial_input, uint32_t input_count,
                             float *output_logits, uint32_t output_count,
                             int num_tokens,
                             ane_bridge_token_callback_t token_callback,
                             void *callback_ctx);

// Create/release an IOSurfaceSharedEvent object.
// Returned pointer is an Objective-C object retained for C callers.
void *ane_bridge_create_shared_event(void);
void ane_bridge_release_objc(void *obj);

// Extract event port from an IOSurfaceSharedEvent object.
unsigned int ane_bridge_shared_event_port(void *shared_event_obj);

// Evaluate model with a wait-event only sharedEvents payload.
// wait_shared_event_obj must be from ane_bridge_create_shared_event().
bool ane_bridge_eval_with_wait_event(ANEClientHandle *h,
                                     void *wait_shared_event_obj,
                                     uint64_t wait_value,
                                     bool disable_fences_use_shared_events,
                                     bool enable_fw_to_fw_signal);

// Evaluate model with a signal-event only sharedEvents payload.
// signal_shared_event_obj must be from ane_bridge_create_shared_event().
// This is primarily for ANE->Metal signaling probes.
bool ane_bridge_eval_with_signal_event_obj(ANEClientHandle *h,
                                           void *signal_shared_event_obj,
                                           uint64_t signal_value,
                                           bool disable_fences_use_shared_events,
                                           bool enable_fw_to_fw_signal);

// Evaluate with ANE->Metal signal direction.
// Uses FW_SIGNAL=0 and a request completion-handler barrier.
// Returns 0 on success, negative on failure.
int ane_bridge_eval_with_signal_event(ane_bridge_client_t client,
                                      const float *input, uint32_t input_count,
                                      float *output, uint32_t output_count,
                                      mach_port_t signal_event_port,
                                      uint64_t signal_value);

// Full bidirectional evaluation:
// - ANE waits on wait_event_port:wait_value (Metal->ANE)
// - ANE signals signal_event_port:signal_value on completion (ANE->Metal)
// Uses FW_SIGNAL=0 and a request completion-handler barrier.
// Returns 0 on success, negative on failure.
int ane_bridge_eval_bidirectional(ane_bridge_client_t client,
                                  const float *input, uint32_t input_count,
                                  float *output, uint32_t output_count,
                                  mach_port_t wait_event_port,
                                  uint64_t wait_value,
                                  mach_port_t signal_event_port,
                                  uint64_t signal_value);

// Signal an IOSurfaceSharedEvent directly from CPU by Mach port.
// Returns 0 on success, negative on failure.
int ane_bridge_signal_event_cpu(mach_port_t event_port, uint64_t value);

// Wait from CPU until IOSurfaceSharedEvent.signaledValue >= value.
// Returns 0 on success, 1 on timeout, negative on failure.
int ane_bridge_wait_event_cpu(mach_port_t event_port, uint64_t value, uint32_t timeout_ms);

// Create a retained MTLBuffer backed by IOSurface base address using default Metal device.
// Returns Objective-C id<MTLBuffer> as opaque pointer, or NULL on failure.
void *ane_bridge_zero_copy_buffer(IOSurfaceRef surface, size_t bytes);

// Dispatch wait-event eval on a background queue while CPU spin-work runs on caller thread.
// Returns false on eval failure.
bool ane_bridge_gcd_overlap(ANEClientHandle *h,
                            void *wait_shared_event_obj,
                            uint64_t wait_value,
                            bool disable_fences_use_shared_events,
                            bool enable_fw_to_fw_signal,
                            uint32_t cpu_rounds,
                            double *out_total_ms);

// Stories trainer ABI (bridge-backed training loop control).
#include "stories_trainer_bridge.h"

#ifdef __cplusplus
}
#endif

#endif // ANE_BRIDGE_H
