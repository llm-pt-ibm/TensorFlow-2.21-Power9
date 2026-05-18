# TensorFlow 2.21 on Power9 (PPC64LE)

Build TensorFlow 2.21 from source on IBM POWER9 with NVIDIA Tesla V100 GPU support.

## Status

- ✅ TF 2.21.0 build with CUDA 12.5.1 + cuDNN 9.0.0 (ppc64le)
- ✅ GPU detection, matmul, conv2d, LSTM, mixed-precision FP16 (tensor cores)
- ✅ Multi-GPU MirroredStrategy (NCCL allreduce)
- ✅ SavedModel / Keras / tf.data / tf.function
- 75/76 tests in `gpu/gpu_test_suite.py` PASS (see `Known Limitations` below)

## Quick start

```bash
bash lld-ppc64-fix/build_patched_lld.sh        # ~3h, patched LLD for PPC64LE
bash gpu/tensorflow_gpu.sh                     # main TF build (~3h-6h)
python3 gpu/gpu_test_suite.py                  # functional test suite
```

## Architectural fixes (vs upstream TF)

### 1. PPC64LE multi-TOC `r2` corruption (LLD patch)

`lld-ppc64-fix/build_patched_lld.sh` patches LLD with:
- `PPC64LongBranchThunk` prefixed with `std r2,24(r1)` (multi-TOC fix)
- `setNeedsTocRestore(true)` so LLD auto-patches the trailing `nop` to `ld r2,24(r1)`
- Suppression of "lacks nop, can't restore toc" error for libgcc compatibility

See `lld-ppc64-fix/build_patched_lld.sh` header for patch upstream-ability notes.

### 2. cuDNN 9 headers enforced

The build script overrides the 5 paths Bazel reads `cudnn_version.h` from with v9.0.0 explicitly, since `find | head -1` picks up the older 8.9.7 from conda-forge by default.

### 3. NVCC `.localentry sym, 1` preserved

`gcc_cuda_wrapper.sh` does NOT rewrite `.localentry` directives. The default Clang assembler used by the wrapper accepts the flag correctly (GAS 2.30 rejects it). This preserves the ELFv2 ABI "CLOBBERS_R2" flag that LLD needs to create `PPC64R2SaveStub` for cudart trampolines.

### 4. MLIR generated GPU kernels disabled

`--//tensorflow/core/kernels/mlir_generated:enable_gpu=False` is passed to bazel — see **Known Limitations** for details.

## Known Limitations

### MLIR generated GPU kernels — NOT AVAILABLE on PPC64LE

TF 2.21 ships with two parallel paths for GPU kernel registration:

| Path | How it works | Status here |
|------|--------------|-------------|
| **A. C++ templates** (legacy, since TF 1.x) | `REGISTER3(BinaryOp, GPU, "Add", float, half, double)` in `cwise_op_*.cc` compiled directly | ✅ Used |
| **B. MLIR generated** (since TF 2.12) | Tool `hlo_to_kernel` compiles `*.mlir.tmpl` → LLVM IR → NVPTX PTX → embedded in `.o` | ❌ Disabled |

When `enable_gpu=True` (upstream default), the `MLIR_GENERATED_GPU_KERNELS_ENABLED` macro is defined, which **disables** the path (A) `REGISTER3` macros for `float`/`half`/`double` — relying on (B) to fill in. On PPC64LE that path is broken:

- The `hlo_to_kernel` tool **does build** (with the savres/multiple-def linker fixes documented in `gpu/tensorflow_gpu.sh`)
- The tool **completes the MLIR HLO → LLVM IR conversion** correctly
- But it **segfaults in the LLVM NVPTX backend code generation** when run on a PPC64LE host — this code path is not tested upstream (LLVM/MLIR cross-compile to NVPTX from PPC64LE host)

Consequence in this build: `enable_gpu=False` is forced. The classical (A) C++ templates path is reactivated and every GPU kernel registration works correctly. There is **no functional loss** — only a ~5-10% larger binary because the templates are not bundled by MLIR.

If a future TF/LLVM/MLIR release fixes the NVPTX backend crash on PPC64LE hosts, the flag can be removed.

### Other notes

- cuDNN 9.1+ does not ship ppc64le builds — we pin to 9.0.0.
- XLA GPU JIT compiler is not registered (so `model.fit(..., jit_compile=True)` will fail with "could not find registered compiler"). The test suite uses `jit_compile=False` everywhere.

## Repository layout

```
gpu/
  tensorflow_gpu.sh                    main TF build script (also called build_tf221_power9_gpu_generic.sh)
  gpu_test_suite.py                    76 functional tests covering all TF surfaces
  transfer.sh                          rotating diagnostic / transfer helper
  patch_nv_toc_save.py                 obsolete patcher v6 (kept for reference)
lld-ppc64-fix/
  build_patched_lld.sh                 patched LLD for PPC64LE
  test_lld.sh                          7 LLD validation tests
cpu/
  scripts and docs for CPU-only TF build
```
