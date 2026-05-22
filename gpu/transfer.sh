#!/bin/bash
# Testa install do tensorflow-gpu 2.21.0 py311_2 (com RAFT) direto do canal ufcg-ibm.

source /root/miniforge3/etc/profile.d/conda.sh

conda env remove -n tf_gpu_raft -y 2>/dev/null
conda create -n tf_gpu_raft python=3.11 -y
conda activate tf_gpu_raft

conda install -c ufcg-ibm -c conda-forge tensorflow-gpu=2.21.0 -y
conda list | grep tensorflow-gpu

export LD_LIBRARY_PATH=$HOME/cuda_unified/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

python <<'PYEOF'
import tensorflow as tf
print("TF:", tf.__version__, "| GPUs:", len(tf.config.list_physical_devices('GPU')))
with tf.device('/GPU:0'):
    v, i = tf.math.top_k(tf.random.normal([1000, 500]), k=5)
    _ = v.numpy()
print("OK: Top-K GPU funcional via canal ufcg-ibm")
PYEOF

Limpar versoes antigas (py311_0 sem RAFT + asset errado, py311_1 sem RAFT):
anaconda -t uf-03f4d728-a7fe-4cad-bb54-ef7687be854f remove \
  ufcg-ibm/tensorflow-gpu/2.21.0/linux-ppc64le/tensorflow-gpu-2.21.0-py311_0.conda --force
anaconda -t uf-03f4d728-a7fe-4cad-bb54-ef7687be854f remove \
  ufcg-ibm/tensorflow-gpu/2.21.0/linux-ppc64le/tensorflow-gpu-2.21.0-py311_1.conda --force
