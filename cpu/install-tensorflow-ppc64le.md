# Installing TensorFlow 2.21.0 (CPU) on ppc64le

> Package natively compiled for the **ppc64le (POWER)** architecture and available on the `ufcg-ibm` Anaconda channel.

---

## Prerequisites

- Operating System: **Linux ppc64le**
- Conda installed (Miniforge, Miniconda, or Anaconda)
- Python **3.11**

If you don't have Conda yet, install Miniforge for ppc64le:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-ppc64le.sh
bash Miniforge3-Linux-ppc64le.sh
```

---

## Installation

### Option 1 — In a new environment (recommended)

```bash
# 1. Create a clean environment with Python 3.11
conda create -n my_tf python=3.11 -y

# 2. Activate the environment
conda activate my_tf

# 3. Install TensorFlow from the ufcg-ibm channel
conda install -c ufcg-ibm -c conda-forge tensorflow-cpu=2.21.0 -y
```

### Option 2 — In an existing environment

```bash
conda activate your_environment
conda install -c ufcg-ibm -c conda-forge tensorflow-cpu=2.21.0 -y
```

### Option 3 — Configure channels globally (install once)

If you use TensorFlow frequently, configure the channels once and you'll never need to declare them again:

```bash
conda config --add channels conda-forge
conda config --add channels ufcg-ibm
conda config --set channel_priority flexible
```

After that, simply install with:

```bash
conda install tensorflow-cpu=2.21.0 -y
```

---

## Verifying the Installation

### Test 1 — Version

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Expected output:** `2.21.0`

### Test 2 — Dependencies

```bash
python -c "
import tensorflow as tf
import numpy as np
import keras
import requests
import h5py

print('TensorFlow:', tf.__version__)
print('NumPy:', np.__version__)
print('Keras:', keras.__version__)
print('h5py:', h5py.__version__)
print('All dependencies OK ✅')
"
```

### Test 3 — CPU Operation

```bash
python -c "
import tensorflow as tf

print('Available devices:', tf.config.list_physical_devices())

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print('Matrix multiplication:')
print(tf.matmul(a, b).numpy())
print('CPU OK ✅')
"
```

### Test 4 — Neural Network Training

```bash
python -c "
import tensorflow as tf
import numpy as np

X = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='sgd', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

pred = model.predict(np.array([6.0]), verbose=0)
print(f'Prediction for 6: {pred[0][0]:.2f} (expected ~12.0)')
print('Training OK ✅')
"
```

---

## Confirming the Package Origin

```bash
conda list | grep tensorflow
```

**Expected output:**
```
tensorflow-cpu   2.21.0   py311_0   ufcg-ibm
```

---

## Dependencies Included Automatically

The package automatically resolves and installs all the dependencies below without manual intervention:

| Package | Description |
|---|---|
| numpy >=2.0.0 | Numerical computation |
| keras >=2.0.0 | High-level API for neural networks |
| grpcio | RPC communication |
| h5py | HDF5 file reading/writing |
| protobuf | Data serialization |
| ml_dtypes | Data types for ML |
| absl-py | Google utilities |
| requests | HTTP requests |
| libstdcxx-ng | Standard C++ library |

---

## Common Issues

| Error | Cause | Solution |
|---|---|---|
| `Illegal instruction (core dumped)` | Binary not compatible with CPU | Confirm it is ppc64le with `uname -m` |
| `ImportError: libstdc++.so` | Missing C++ library | `conda install libstdcxx-ng` |
| `PackagesNotFoundError` | Channel not declared | Use `-c ufcg-ibm -c conda-forge` |
| `UserWarning: Do not pass input_shape` | Keras 3.x installed | Use `tf.keras.Input(shape=...)` as the first layer |

---

## References

- Anaconda Channel: [anaconda.org/ufcg-ibm](https://anaconda.org/ufcg-ibm)
- .whl Repository: [github.com/llm-pt-ibm/tensorflow](https://github.com/llm-pt-ibm/tensorflow)
- TensorFlow Documentation: [tensorflow.org](https://www.tensorflow.org)
