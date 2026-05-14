#!/bin/bash
# Validar quais ops do TF GPU funcionam vs crasham. Cada teste em subprocesso
# isolado pra crash de um nao afetar os outros.

export LD_LIBRARY_PATH=/root/cuda_unified/lib64:/root/miniforge3/envs/tf221_build/lib:$LD_LIBRARY_PATH

run_test() {
    local name="$1"
    local code="$2"
    echo -n "  [$name] ... "
    OUT=$(python3 -c "$code" 2>&1)
    RC=$?
    if [ "$RC" = "0" ]; then
        echo "OK ($(echo "$OUT" | tail -1))"
    else
        # Resume erro
        LAST=$(echo "$OUT" | grep -E 'Error|error:|Segmentation' | head -1)
        echo "FAIL rc=$RC ${LAST:-(silent)}"
    fi
}

echo "=== [0] Pre-check: TF importa + GPUs detectadas? ==="
run_test "import + GPU detect" '
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"TF {tf.__version__} GPUs={len(gpus)}")
'
echo ""

# Setup base usado em quase todos: importa + config memory_growth
BASE='
import tensorflow as tf
import numpy as np
gpus = tf.config.list_physical_devices("GPU")
for g in gpus: tf.config.experimental.set_memory_growth(g, True)
'

echo "=== [1] Operacoes CPU (sanity) ==="
run_test "tensor create CPU" "$BASE"'
with tf.device("/CPU:0"):
    x = tf.constant([1.0, 2.0, 3.0])
print(f"shape={x.shape}")
'

run_test "matmul CPU 100x100" "$BASE"'
with tf.device("/CPU:0"):
    A = tf.random.normal([100, 100])
    B = tf.random.normal([100, 100])
    C = tf.matmul(A, B)
print(f"sum={tf.reduce_sum(C).numpy():.2f}")
'

echo ""
echo "=== [2] Tensor creation/movement na GPU ==="
run_test "constant GPU" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.constant([1.0, 2.0, 3.0])
print(f"device={x.device}")
'

run_test "CPU->GPU copy" "$BASE"'
with tf.device("/CPU:0"):
    x = tf.constant([1.0, 2.0, 3.0])
with tf.device("/GPU:0"):
    y = tf.identity(x)
print(f"device={y.device}")
'

run_test "random GPU 100" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.random.normal([100])
print(f"shape={x.shape}")
'

echo ""
echo "=== [3] Operacoes element-wise GPU ==="
run_test "add GPU" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.constant([1.0, 2.0, 3.0])
    y = tf.constant([10.0, 20.0, 30.0])
    z = x + y
print(f"sum={tf.reduce_sum(z).numpy():.2f}")
'

run_test "mul GPU" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.random.normal([1000])
    y = x * 2.0
print(f"sum={tf.reduce_sum(y).numpy():.2f}")
'

run_test "relu GPU" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.random.normal([1000])
    y = tf.nn.relu(x)
print(f"sum={tf.reduce_sum(y).numpy():.2f}")
'

echo ""
echo "=== [4] Reduction ops GPU ==="
run_test "reduce_sum GPU" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.random.normal([1000])
    s = tf.reduce_sum(x).numpy()
print(f"sum={s:.2f}")
'

run_test "argmax GPU" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.random.normal([1000])
    i = tf.argmax(x).numpy()
print(f"idx={i}")
'

echo ""
echo "=== [5] BLAS ops (matmul - usa cuBLAS) ==="
run_test "matmul GPU 10x10" "$BASE"'
with tf.device("/GPU:0"):
    A = tf.random.normal([10, 10])
    B = tf.random.normal([10, 10])
    C = tf.matmul(A, B)
print(f"sum={tf.reduce_sum(C).numpy():.2f}")
'

run_test "matmul GPU 100x100" "$BASE"'
with tf.device("/GPU:0"):
    A = tf.random.normal([100, 100])
    B = tf.random.normal([100, 100])
    C = tf.matmul(A, B)
print(f"sum={tf.reduce_sum(C).numpy():.2f}")
'

run_test "matmul GPU 1000x1000" "$BASE"'
with tf.device("/GPU:0"):
    A = tf.random.normal([1000, 1000])
    B = tf.random.normal([1000, 1000])
    C = tf.matmul(A, B)
print(f"sum={tf.reduce_sum(C).numpy():.2f}")
'

run_test "batch matmul GPU" "$BASE"'
with tf.device("/GPU:0"):
    A = tf.random.normal([4, 100, 100])
    B = tf.random.normal([4, 100, 100])
    C = tf.matmul(A, B)
print(f"shape={C.shape}")
'

echo ""
echo "=== [6] Conv ops (usa cuDNN) ==="
run_test "conv2d GPU 32x32" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.random.normal([1, 32, 32, 3])
    f = tf.random.normal([3, 3, 3, 16])
    y = tf.nn.conv2d(x, f, strides=1, padding="SAME")
print(f"shape={y.shape}")
'

run_test "conv2d GPU 224x224" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.random.normal([1, 224, 224, 3])
    f = tf.random.normal([7, 7, 3, 64])
    y = tf.nn.conv2d(x, f, strides=2, padding="SAME")
print(f"shape={y.shape}")
'

echo ""
echo "=== [7] FFT (usa cuFFT) ==="
run_test "fft GPU" "$BASE"'
with tf.device("/GPU:0"):
    x = tf.complex(tf.random.normal([128]), tf.random.normal([128]))
    y = tf.signal.fft(x)
print(f"shape={y.shape}")
'

echo ""
echo "=== [8] Keras simples (alta-nivel) ==="
run_test "keras dense forward" "$BASE"'
import tensorflow as tf
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
with tf.device("/GPU:0"):
    x = tf.random.normal([2, 5])
    y = model(x)
print(f"shape={y.shape}")
'

echo ""
echo "=== Resumo ==="
echo "OK = funciona | FAIL = quebra. Padroes:"
echo "  Se tudo FAIL: build geral quebrado"
echo "  Se SO matmul/blas FAIL: cuBLAS especifico"
echo "  Se conv FAIL: cuDNN especifico"
echo "  Se element-wise OK + matmul FAIL: kernel launch path quebrou"
