# Installing TensorFlow 2.21.0 (GPU) on ppc64le

> Package natively compiled for the **ppc64le (POWER9)** architecture with
> NVIDIA GPU support, available on the `ufcg-ibm` Anaconda channel.

---

## Prerequisites

### Hardware
- **CPU:** IBM POWER9 (ppc64le)
- **GPU:** NVIDIA, compute capability **7.0+** (tested on Tesla V100)

### System software (not bundled in the package)
- **NVIDIA driver** compatible with CUDA 12.x
- **CUDA toolkit 12.x** installed on the host (the package links dynamically
  against `libcudart.so.12`)
- **cuDNN 9.0.0 for ppc64le** (download from NVIDIA — cuDNN 9.1+ is no longer
  distributed for ppc64le)
- **Linux ppc64le** (kernel ≥ 4.18)

### Conda + Python
- Conda (Miniforge, Miniconda or Anaconda)
- Python **3.11** (the package is pinned to CPython 3.11)

If you don't have Conda yet, install Miniforge for ppc64le:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-ppc64le.sh
bash Miniforge3-Linux-ppc64le.sh
```

---

## Installation

### Option 1 — In a new environment (recommended)

```bash
conda create -n tf_gpu python=3.11 -y
conda activate tf_gpu
conda install -c ufcg-ibm -c conda-forge tensorflow-gpu=2.21.0 -y
```

### Option 2 — In an existing environment

```bash
conda activate your_environment
conda install -c ufcg-ibm -c conda-forge tensorflow-gpu=2.21.0 -y
```

The Conda solver will pull the latest build of `tensorflow-gpu=2.21.0`
(currently `py311_2`, which enables native GPU Top-K via RAFT/RMM).

---

## Runtime setup — `LD_LIBRARY_PATH`

The package depends on CUDA 12 runtime libraries and cuDNN 9 that **must
be reachable at import time** via `LD_LIBRARY_PATH`. The package itself
does not ship them.

Typical layout (adjust to your install paths):

```bash
# CUDA toolkit
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/ppc64le-linux/lib:$LD_LIBRARY_PATH

# cuDNN 9 (where you extracted the NVIDIA tarball)
export LD_LIBRARY_PATH=/opt/cudnn-9.0.0/lib:$LD_LIBRARY_PATH

# Conda environment libs (libstdc++ from a newer GCC, NCCL via conda-forge)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

For NCCL multi-GPU support, also make sure `libnccl.so.2` is reachable
(usually pulled in by `nccl` from conda-forge or via a manual symlink).

Persist this in `~/.bashrc` (or your shell's rc) or in the Conda env's
activation hook:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda.sh << 'EOF'
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/ppc64le-linux/lib:/opt/cudnn-9.0.0/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
EOF
```

---

## Verifying the Installation

### Test 1 — Version and CUDA build

```bash
python -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('Build info:', tf.sysconfig.get_build_info())
"
```

**Expected output:**
```
TensorFlow: 2.21.0
Built with CUDA: True
Build info: OrderedDict([('is_cuda_build', True), ...])
```

### Test 2 — GPU detection

```bash
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('GPUs detected:', len(gpus))
for g in gpus:
    print(' -', g)
"
```

**Expected output:**
```
GPUs detected: N
 - PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
 ...
```

### Test 3 — Matmul on GPU

```bash
python -c "
import tensorflow as tf
with tf.device('/GPU:0'):
    a = tf.random.normal([2048, 2048])
    b = tf.random.normal([2048, 2048])
    c = tf.matmul(a, b)
    print('Result shape:', c.shape, '| device:', c.device)
"
```

### Test 4 — Top-K on GPU (RAFT path)

```bash
python -c "
import tensorflow as tf
with tf.device('/GPU:0'):
    x = tf.random.normal([1000, 500])
    values, indices = tf.math.top_k(x, k=5)
    _ = values.numpy()
print('Top-K GPU OK')
"
```

If Top-K returns successfully (instead of `UnimplementedError`), the
RAFT-backed select_k kernel is active.

### Test 5 — Multi-GPU (optional)

```bash
python -c "
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy()
print('Replicas:', strategy.num_replicas_in_sync)
"
```

---

## Notes and Limitations

- **GPU architectures tested:** Tesla V100 (`sm_70`). Ampere and Hopper
  have not been validated on this build.
- **XLA JIT compilation:** `tf.function(..., jit_compile=True)` will fail
  with *"could not find registered compiler"*. Use `jit_compile=False`.
- **MLIR-generated GPU kernels** are disabled in this build (the NVPTX
  backend segfaults on ppc64le hosts). The legacy C++ template path is
  used for all GPU kernel registrations — no functional impact.
- **cuDNN 9.1+** is not available for ppc64le; pin to 9.0.0.

---

## Source and Reproducing the Build

- **Source branch:** [llm-pt-ibm/tensorflow @ `power9-v2.21.0-gpu`](https://github.com/llm-pt-ibm/tensorflow/tree/power9-v2.21.0-gpu)
- **Build scripts and patches:** [llm-pt-ibm/TensorFlow-2.21-Power9](https://github.com/llm-pt-ibm/TensorFlow-2.21-Power9)
- **Conda channel:** [anaconda.org/ufcg-ibm/tensorflow-gpu](https://anaconda.org/ufcg-ibm/tensorflow-gpu)

To rebuild from scratch:

```bash
git clone https://github.com/llm-pt-ibm/TensorFlow-2.21-Power9.git
cd TensorFlow-2.21-Power9/gpu
bash build_tf221_power9_gpu_generic.sh
```

---

## Troubleshooting

### `Could not load dynamic library 'libcudart.so.12'`
CUDA 12 runtime is not in `LD_LIBRARY_PATH`. Make sure the CUDA toolkit
install path (e.g. `/usr/local/cuda-12.2/targets/ppc64le-linux/lib`) is
exported before importing TensorFlow.

### `Could not load dynamic library 'libcudnn.so.9'`
cuDNN 9 is missing or not in `LD_LIBRARY_PATH`. Download the official
NVIDIA cuDNN 9.0.0 ppc64le tarball and add its `lib/` directory to the
path.

### `NCCL: Unable to load NCCL library` (multi-GPU)
`libnccl.so.2` not reachable. Install NCCL via conda-forge:
```bash
conda install -c conda-forge nccl -y
```
or symlink the system NCCL into your environment's `lib/`.

### `GLIBCXX_3.4.29 not found`
The system `libstdc++` is older than required. Make sure
`$CONDA_PREFIX/lib` (which has the newer `libstdc++` from conda-forge)
is **before** `/usr/lib64` in `LD_LIBRARY_PATH`.
