# Teste do pacote tensorflow-gpu=2.21.0 publicado no canal ufcg-ibm.
# Roda no container ppc64le (gaby_workspace) ou em qualquer ppc64le com conda.

source /root/miniforge3/etc/profile.d/conda.sh

# Env limpo (simula usuario final)
conda env remove -n tf_gpu_test -y 2>/dev/null || true
conda create -n tf_gpu_test python=3.11 -y
conda activate tf_gpu_test

# Instala direto do canal publico
conda install -c ufcg-ibm -c conda-forge tensorflow-gpu=2.21.0 -y

# Confirma origem do pacote
conda list | grep -E "tensorflow|numpy|keras"

# CUDA precisa estar no LD_LIBRARY_PATH em runtime (nao vai no pacote)
export LD_LIBRARY_PATH=$HOME/cuda_unified/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# Verificacao funcional
python <<'PYEOF'
import tensorflow as tf
print("TF version :", tf.__version__)
print("Built CUDA :", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print("GPUs       :", len(gpus))
for g in gpus:
    print(" -", g)
# matmul rapido na GPU
if gpus:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1024, 1024])
        b = tf.random.normal([1024, 1024])
        c = tf.matmul(a, b)
        print("matmul OK  :", c.shape, c.dtype)
PYEOF
