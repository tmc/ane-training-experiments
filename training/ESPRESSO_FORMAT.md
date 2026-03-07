# Espresso format notes for `_ANEModel` FFN generation

This document captures the exact Espresso artifact format observed in:

- `/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/draft125m_ffn.mlmodelc`

and validated against hand-written generation in:

- `/Volumes/tmc/go/src/github.com/maderix/ANE/training/test_espresso_gen.m`

## 1. Raw artifacts from `draft125m_ffn.mlmodelc`

### Directory listing (`ls -la`)

```text
total 55560
drwxr-xr-x@ 12 tmc  staff       384 Mar  3 16:53 .
drwxr-xr-x@ 11 tmc  staff       352 Mar  3 17:06 ..
drwxr-xr-x@  3 tmc  staff        96 Mar  3 16:49 analytics
-rw-r--r--@  1 tmc  staff       343 Mar  3 16:53 coremldata.bin
-rw-r--r--@  1 tmc  staff      1652 Mar  3 16:53 metadata.json
drwxr-xr-x@  3 tmc  staff        96 Mar  3 16:53 model
-rw-r--r--@  1 tmc  staff      2076 Mar  3 16:53 model.espresso.net
-rw-r--r--@  1 tmc  staff       766 Mar  3 16:53 model.espresso.shape
-rw-r--r--@  1 tmc  staff  28422400 Mar  3 16:53 model.espresso.weights
-rw-r--r--@  1 tmc  staff      3094 Mar  3 16:49 model.mil
drwxr-xr-x@  3 tmc  staff        96 Mar  3 16:53 neural_network_optionals
drwxr-xr-x@  3 tmc  staff        96 Mar  3 16:49 weights
```

### `model.espresso.net` (full)

```json
{
  "storage" : "model.espresso.weights",
  "analyses" : {

  },
  "properties" : {

  },
  "format_version" : 200,
  "metadata_in_weights" : [

  ],
  "layers" : [
    {
      "nB" : 768,
      "top" : "linear_0",
      "has_biases" : 1,
      "weights" : {

      },
      "nC" : 3072,
      "blob_weights" : 3,
      "type" : "inner_product",
      "has_relu" : 0,
      "bottom" : "x",
      "blob_biases" : 1,
      "has_tanh" : 0,
      "debug_info" : "linear_0",
      "name" : "linear_0",
      "has_prelu" : 0
    },
    {
      "bottom" : "linear_0",
      "weights" : {

      },
      "mode" : 3,
      "debug_info" : "8__silu_sigmoid__",
      "top" : "8__silu_sigmoid__",
      "type" : "activation",
      "name" : "8__silu_sigmoid__"
    },
    {
      "bottom" : "linear_0,8__silu_sigmoid__",
      "alpha" : 1,
      "operation" : 1,
      "weights" : {

      },
      "fused_relu" : 0,
      "debug_info" : "8",
      "top" : "8",
      "type" : "elementwise",
      "name" : "8",
      "beta" : 0
    },
    {
      "nB" : 768,
      "top" : "linear_1",
      "has_biases" : 1,
      "weights" : {

      },
      "nC" : 3072,
      "blob_weights" : 7,
      "type" : "inner_product",
      "has_relu" : 0,
      "bottom" : "x",
      "blob_biases" : 5,
      "has_tanh" : 0,
      "debug_info" : "linear_1",
      "name" : "linear_1",
      "has_prelu" : 0
    },
    {
      "bottom" : "8,linear_1",
      "alpha" : 1,
      "operation" : 1,
      "weights" : {

      },
      "fused_relu" : 0,
      "debug_info" : "input",
      "top" : "input",
      "type" : "elementwise",
      "name" : "input",
      "beta" : 0
    },
    {
      "has_prelu" : 0,
      "top" : "linear_2",
      "has_biases" : 1,
      "weights" : {

      },
      "nC" : 768,
      "blob_weights" : 11,
      "type" : "inner_product",
      "has_relu" : 0,
      "attributes" : {
        "is_output" : 1
      },
      "bottom" : "input",
      "debug_info" : "linear_2",
      "has_tanh" : 0,
      "blob_biases" : 9,
      "name" : "linear_2",
      "nB" : 3072
    }
  ]
}
```

### `model.espresso.shape` (full)

```json
{
  "layer_shapes" : {
    "linear_1" : {
      "k" : 1,
      "w" : 3072,
      "n" : 1,
      "_rank" : 3,
      "h" : 1
    },
    "input" : {
      "k" : 1,
      "w" : 3072,
      "n" : 1,
      "_rank" : 3,
      "h" : 1
    },
    "x" : {
      "k" : 1,
      "w" : 768,
      "n" : 1,
      "_rank" : 3,
      "h" : 1
    },
    "8" : {
      "k" : 1,
      "w" : 3072,
      "n" : 1,
      "_rank" : 3,
      "h" : 1
    },
    "linear_0" : {
      "k" : 1,
      "w" : 3072,
      "n" : 1,
      "_rank" : 3,
      "h" : 1
    },
    "8__silu_sigmoid__" : {
      "k" : 1,
      "w" : 3072,
      "n" : 1,
      "_rank" : 3,
      "h" : 1
    },
    "linear_2" : {
      "k" : 1,
      "w" : 768,
      "n" : 1,
      "_rank" : 3,
      "h" : 1
    }
  }
}
```

### `model.espresso.weights`

- size: `28422400` bytes
- first 64 bytes (hex):

```text
00000000: 0c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000010: 38 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
00000020: 00 c0 00 00 00 00 00 00 02 00 00 00 00 00 00 00
00000030: 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00
```

## 2. Minimum viable format (for 3-layer SwiGLU FFN)

### File set

For `_ANEModel modelAtURL:key:` + `_ANEClient compileModel/loadModel`, these three files are sufficient for working dimensions:

- `model.espresso.net`
- `model.espresso.shape`
- `model.espresso.weights`

Confirmed by compiling/loading `/tmp/draft_three_*.mlmodelc` containing only those files from draft.

### `model.espresso.net` rules

- `storage` points at the weights filename.
- `format_version` is `200`.
- `layers` is a topologically valid list.
- SwiGLU appears as:
  - `inner_product` (`linear_0`)
  - `activation` with `mode: 3` (sigmoid)
  - `elementwise` with `operation: 1` (multiply) to form SiLU
  - second `inner_product` (`linear_1`)
  - second multiply (`operation: 1`) for gate*up
  - final `inner_product` (`linear_2`) with `attributes.is_output: 1`
- Weight blob IDs for this 3-linear pattern are odd IDs:
  - first linear: bias `1`, weights `3`
  - second linear: bias `5`, weights `7`
  - third linear: bias `9`, weights `11`

### `model.espresso.shape` rules

- Object key: `layer_shapes`.
- Every tensor name referenced by any `bottom`/`top` in net should appear.
- Observed encoding is rank-3 with scalar axes fixed to 1 and feature width on `w`:
  - `_rank = 3`
  - `n = 1`
  - `h = 1`
  - `k = 1`
  - `w = feature_width`
- This is effectively `[1, 1, w]` logical shape for FFN vectors.

### `model.espresso.weights` rules (observed)

The first 26 little-endian `uint64` words are:

1. `12` (blob table count for IDs 0..11)
2. `0`
3. `56`, `1`
4. `bias0_bytes`, `2`
5. `0`, `3`
6. `weight0_bytes`, `4`
7. `0`, `5`
8. `bias1_bytes`, `6`
9. `0`, `7`
10. `weight1_bytes`, `8`
11. `0`, `9`
12. `bias2_bytes`, `10`
13. `0`, `11`
14. `weight2_bytes`, `0`

For CoreML-generated FFN artifacts, byte sizes match:

- `weight*_bytes = nB * nC * 4` (Float32 storage)
- `bias*_bytes = nC * 16`

Payload packing is 256-byte aligned, starting with `bias0` logical offset `56`.

For `dim=768, hidden=3072`, this reproduces exact CoreML offsets:

- `bias0=56`
- `weight0=49408`
- `bias1=9486592`
- `weight1=9535744`
- `bias2=18972928`
- `weight2=18985216`
- total `28422400`

## 3. Hand-written generator and benchmark

Implemented in:

- `/Volumes/tmc/go/src/github.com/maderix/ANE/training/test_espresso_gen.m`

Build:

```bash
make -C /Volumes/tmc/go/src/github.com/maderix/ANE/training test_espresso_gen
```

Run (requested tiny shape):

```bash
/Volumes/tmc/go/src/github.com/maderix/ANE/training/test_espresso_gen
```

Run (working comparison shape):

```bash
ANE_ESPRESSO_DIM=384 \
ANE_ESPRESSO_HIDDEN=1536 \
ANE_ESPRESSO_GEN_MODEL_PATH=/tmp/hand_d384_h1536.mlmodelc \
ANE_ESPRESSO_COREML_MODEL_PATH=/tmp/ane_dim_sweep/ffn_d384_h1536.mlmodelc \
/Volumes/tmc/go/src/github.com/maderix/ANE/training/test_espresso_gen
```

## 4. Comparison results

### Requested tiny model (`dim=64, hidden=256`)

- Hand-written: `_ANEClient compileModel` fails with Espresso error:
  - `Cannot serialize ANEC_IR_repr 9Á@...`
- CoreML-compiled tiny model fails with the same class of error.

Inference: this is a translator/runtime constraint for small FFN dimensions in this `_ANEModel` path, not a hand-written format mismatch.

### Working model (`dim=384, hidden=1536`)

- Hand-written vs CoreML:
  - `avg_us`: `260.510` vs `258.880`
  - `p50_us`: `93.000` vs `93.000`
  - ratio: `avg 1.0063`, `p50 1.0000`
- `model.espresso.net` and `model.espresso.shape` are byte-equivalent except trailing newline.
- `model.espresso.weights` first difference starts exactly at first weight payload offset (header/layout identical).

### Draft-size model (`dim=768, hidden=3072`)

- Hand-written vs draft CoreML:
  - hand `p50_us=172`
  - coreml `p50_us=182`
  - both in ~100-200us range

This confirms the hand-written Espresso JSON + weights layout reproduces the fast fused ANE execution behavior once dimensions are in a translator-supported range.

## 5. Shape-file field annotation

Using draft example entry:

```json
"x": { "k": 1, "w": 768, "n": 1, "_rank": 3, "h": 1 }
```

Field meaning (observed/inferred):

- `layer_shapes`: map from tensor/blob name to shape descriptor.
- keys (`"x"`, `"linear_0"`, `"8"`, etc.) map directly to `bottom`/`top` names in net.
- `_rank`: logical rank (`3` here).
- `n`, `h`, `k`: singleton dimensions for this vector FFN case.
- `w`: feature/channel width consumed/produced by that tensor.

`attributes.is_output = 1` is carried on the output layer in net (`linear_2`) and should be preserved for correct output binding in this flow.
