#!/bin/bash
# Recuperar CUDA + testar T11 nativo + suite completo
# (TF_NCCL_VERSION ja esta "2" no cache, so falta GPU recuperar)

source /root/miniforge3/etc/profile.d/conda.sh
conda activate tf221_build || conda activate base || true
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$HOME/cuda_unified/lib64"

echo "=== [1] Status GPU + processos zombies ==="
nvidia-smi | head -25 | sed 's/^/    /'
echo ""
echo ">>> Processos usando /dev/nvidia*:"
fuser /dev/nvidia* 2>&1 | sed 's/^/    /' || echo "    nenhum"
echo ""

echo "=== [2] Tentar reset CUDA (nvidia-smi --gpu-reset) ==="
# Nem sempre funciona, mas tenta
for ID in 0 1; do
    nvidia-smi --gpu-reset -i $ID 2>&1 | head -3 | sed 's/^/    /'
done
echo ""

echo "=== [3] Aguardar 10s e testar detection ==="
sleep 10
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {len(gpus)}')
" 2>&1 | tail -3 | sed 's/^/    /'
echo ""

echo "=== [4] Retry T11 ate 3 vezes (sem pre-load NCCL) ==="
# Garantir pre-load DISABLED
sed -i 's/^_NCCL_PRELOADED = _preload_nccl()/_NCCL_PRELOADED = None  # native test/' /root/tensorflow_gpu/gpu_tests.py

for ATTEMPT in 1 2 3; do
    echo ""
    echo ">>> Tentativa $ATTEMPT:"
    OUT=$(python3 /root/tensorflow_gpu/gpu_tests.py T11 2>&1 | tail -5)
    echo "$OUT" | sed 's/^/    /'
    if echo "$OUT" | grep -q "PASS"; then
        echo ""
        echo "T11 PASSOU NATIVO (sem pre-load)!"
        break
    elif echo "$OUT" | grep -q "only 0 GPU"; then
        echo "    GPU ainda ruim, aguardando 15s antes retry..."
        sleep 15
    else
        echo "    erro diferente de '0 GPU' - parar retry"
        break
    fi
done
echo ""

echo "=== [5] Restaurar gpu_tests.py original (pre-load enabled) ==="
sed -i 's/^_NCCL_PRELOADED = None  # native test/_NCCL_PRELOADED = _preload_nccl()/' /root/tensorflow_gpu/gpu_tests.py
grep "^_NCCL_PRELOADED" /root/tensorflow_gpu/gpu_tests.py | sed 's/^/    /'
