#!/bin/bash
# Rebuild com fix de causa raiz: BFD + --no-multi-toc + --no-plt-localentry.
# Razao: o crash em __nv_cudaEntityRegisterCallback decorre de GCC nao emitir
# `std r2, 24(r1)` antes de bls intra-.so (assume mesmo TOC group), mas LLD
# fez multi-TOC e inseriu stubs inter-TOC que quebram essa suposicao.
# Single-TOC (--no-multi-toc) restaura a invariancia.

set -e

REPO=/home/yalle-silva/TensorFlow-2.21-Power9
TF_DIR=/root/miniforge3/envs/tf221_build/lib/python3.11/site-packages/tensorflow
TF_FW=$TF_DIR/libtensorflow_framework.so.2

echo "=== [1] Diff da configuracao do linker no build script ==="
grep -nE 'BAZEL_FUSE_LD=|--no-multi-toc|--no-plt-localentry|libppc64_savres' \
    $REPO/gpu/build_tf221_power9_gpu_generic.sh | head -20
echo ""

echo "=== [2] Backup do .so atual (caso queira comparar) ==="
if [ -f "$TF_FW" ] && [ ! -f "$TF_FW.lld-broken" ]; then
    cp -p "$TF_FW" "$TF_FW.lld-broken"
    echo "Backup salvo em $TF_FW.lld-broken"
else
    echo "Backup ja existe ou .so ausente, pulando"
fi
echo ""

echo "=== [3] Limpar cache do bazel para o link do framework ==="
# Bazel detecta mudanca em linkopt e refaz a action. Mas pra garantir, podemos
# expurgar apenas as actions de link sem perder os .o compilados.
# Alternativa: bazel clean --expunge (destroi cache inteiro, 2-3h de rebuild)
# Solucao intermediaria: forcar revalidacao
echo "Recomendado: NAO usar 'bazel clean --expunge'. As mudancas sao apenas em"
echo "linkopt, entao o bazel reusa todas as .o e refaz so as link actions."
echo "Tempo esperado: ~30-60 min so pro relink final."
echo ""

echo "=== [4] Lancar o build (script ja com fix aplicado) ==="
echo "Para rodar:"
echo "  cd $REPO/gpu"
echo "  bash build_tf221_power9_gpu_generic.sh 2>&1 | tee /tmp/build_bfd.log"
echo ""
echo "Pra ver progresso especifico de link:"
echo "  tail -f /tmp/build_bfd.log | grep -E 'Linking|libtensorflow_framework|error:'"
echo ""

echo "=== [5] Apos build, validar que o framework agora carrega ==="
echo "Quando o build terminar:"
cat << 'TEST_EOF'

# Teste 1: framework carrega standalone sem segfault
python3 -c "
import os, ctypes
h = ctypes.CDLL('$TF_FW', mode=os.RTLD_LAZY|os.RTLD_LOCAL)
print('Framework load OK:', h)
"

# Teste 2: import tensorflow completo + lista GPUs
python3 -c "
import tensorflow as tf
print('TF', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPUs:', tf.config.list_physical_devices('GPU'))
"

# Teste 3: matmul real na GPU
python3 -c "
import tensorflow as tf
with tf.device('/GPU:0'):
    A = tf.random.normal([2000, 2000])
    B = tf.random.normal([2000, 2000])
    import time
    _ = tf.matmul(A[:10,:10], B[:10,:10])  # warmup
    t0 = time.time()
    C = tf.matmul(A, B)
    soma = tf.reduce_sum(C).numpy()
    print(f'Matmul 2000x2000 em GPU: {time.time()-t0:.3f}s sum={soma:.2f}')
"

TEST_EOF

echo ""
echo "=== [6] Validacao do framework sem rebuild completo (opcional) ==="
echo "Para um teste mais rapido (sem refazer link inteiro), podemos verificar"
echo "se BFD aceita o link com as novas flags via um link test minimo:"
cat << 'LINKTEST_EOF'

# Verifica se ld.bfd entende as flags
cd /tmp
echo 'int main(){return 0;}' > t.c
gcc -fuse-ld=bfd -Wl,--no-multi-toc -Wl,--no-plt-localentry -o /tmp/t /tmp/t.c
echo "ld.bfd aceita flags: exit=$?"
rm -f /tmp/t /tmp/t.c

LINKTEST_EOF
