#!/bin/bash
# =============================================================================
# Diagnóstico e Verificação da Compilação CUDA no Power9
# TensorFlow 2.21 - IBM Power9 (ppc64le) + NVIDIA GPU
#
# Este script contém todos os comandos de diagnóstico e testes para verificar
# se o pipeline NVCC está funcionando corretamente durante o build do TF.
#
# USO:
#   bash diagnose_cuda_build.sh          # Roda todos os testes
#   bash diagnose_cuda_build.sh --quick  # Só verificação rápida
#   bash diagnose_cuda_build.sh --replay # Só replay do último comando NVCC
# =============================================================================

export PATH=/root/miniforge3/nvvm/bin:/root/miniforge3/bin:/root/cuda_unified/bin:$PATH
NVCC="/root/cuda_unified/bin/nvcc"
BAZEL_BASE="/root/.cache/bazel/_bazel_root/cd3e5b24d1c6b8ec7f1ef221724b7b3d"
EXEC_ROOT="$BAZEL_BASE/execroot/org_tensorflow"
TF_WORKSPACE="/home/tensorflow_gpu/tf221_workspace/tensorflow"
NVCC_DUMP="/tmp/nvcc_dump.txt"

MODE="${1:---all}"

echo "======================================================="
echo "  Diagnóstico CUDA Build - Power9"
echo "  $(date)"
echo "======================================================="

# =============================================================================
# [1] VERIFICAÇÃO DO TOOLCHAIN
# =============================================================================
echo ""
echo "━━━ [1] TOOLCHAIN CUDA ━━━"
for bin in nvcc cicc ptxas fatbinary; do
    P=$(which $bin 2>/dev/null)
    if [ -n "$P" ]; then
        ARCH=$(file "$P" 2>/dev/null | grep -oE 'ppc64|x86.64|aarch64' | head -1)
        echo "  ✅ $bin: $P (${ARCH:-unknown})"
    else
        echo "  ❌ $bin: NÃO ENCONTRADO"
    fi
done

# Versão do NVCC
echo ""
$NVCC --version 2>/dev/null | tail -1 || echo "  ❌ NVCC não responde"

# =============================================================================
# [2] VERIFICAÇÃO DE MEMÓRIA E CPU
# =============================================================================
echo ""
echo "━━━ [2] RECURSOS DO SISTEMA ━━━"
echo "  RAM total:  $(free -h | awk '/^Mem/{print $2}')"
echo "  RAM livre:  $(free -h | awk '/^Mem/{print $4}')"
echo "  CPUs:       $(nproc)"
echo "  Processos CUDA ativos: $(pgrep -c -f 'cicc\|ptxas\|nvcc' 2>/dev/null || echo 0)"

# =============================================================================
# [3] VERIFICAÇÃO DOS WRAPPERS
# =============================================================================
echo ""
echo "━━━ [3] WRAPPERS GCC/G++ ━━━"
for wrapper in ~/gcc_cuda_wrapper.sh ~/gxx_cuda_wrapper.sh; do
    if [ -f "$wrapper" ]; then
        echo "  $(basename $wrapper):"
        # Verificar flags críticas
        grep -q 'arch=sm_70' "$wrapper" && echo "    ✅ -arch=sm_70" || echo "    ❌ -arch=sm_70 FALTANDO"
        grep -q 'Xcicc' "$wrapper" && echo "    ✅ -Xcicc -O0" || echo "    ❌ -Xcicc -O0 FALTANDO"
        grep -q 'miniforge3/nvvm/bin' "$wrapper" && echo "    ✅ PATH com cicc" || echo "    ❌ PATH sem cicc"
        grep -q 'mno-float128' "$wrapper" && echo "    ⚠️  -mno-float128 presente (pode causar erro NVCC)" || echo "    ✅ sem -mno-float128"
        grep -q 'local next_arg' "$wrapper" && echo "    ⚠️  'local' keyword fora de function" || echo "    ✅ sem 'local' keyword"
        # Verificar filtro de flags
        grep -q '\-mno-\*' "$wrapper" && echo "    ✅ filtro -m* flags" || echo "    ❌ filtro -m* flags FALTANDO"
    else
        echo "  ❌ $(basename $wrapper) NÃO EXISTE"
    fi
done

# =============================================================================
# [4] VERIFICAÇÃO DO .tf_configure.bazelrc
# =============================================================================
echo ""
echo "━━━ [4] .tf_configure.bazelrc ━━━"
BAZELRC="$TF_WORKSPACE/.tf_configure.bazelrc"
if [ -f "$BAZELRC" ]; then
    if grep -q 'mno-float128' "$BAZELRC"; then
        echo "  ⚠️  -mno-float128 ENCONTRADO no bazelrc (causará erro NVCC!)"
        grep 'mno-float128' "$BAZELRC" | sed 's/^/    /'
    else
        echo "  ✅ sem -mno-float128"
    fi
    if grep -q 'cuda_clang' "$BAZELRC"; then
        echo "  ⚠️  cuda_clang encontrado (deveria ter sido removido)"
    else
        echo "  ✅ sem cuda_clang"
    fi
    echo "  Conteúdo:"
    cat "$BAZELRC" | sed 's/^/    /' | head -15
else
    echo "  ❌ Arquivo não encontrado em $BAZELRC"
fi

[ "$MODE" = "--quick" ] && { echo ""; echo "=== Fim (modo rápido) ==="; exit 0; }

# =============================================================================
# [5] TESTE DE COMPILAÇÃO STANDALONE
# =============================================================================
echo ""
echo "━━━ [5] TESTES DE COMPILAÇÃO STANDALONE ━━━"

# Detectar include dir do CUDA
CUDA_INC_FLAGS=""
for d in /root/cuda_unified/include /root/miniforge3/include /usr/local/cuda/include \
         /root/miniforge3/targets/ppc64le-linux/include /usr/local/include/cuda_stub; do
    if [ -f "$d/cuda_runtime.h" ] || [ -f "$d/cuda.h" ]; then
        CUDA_INC_FLAGS="$CUDA_INC_FLAGS -I$d"
    fi
done

# Teste 1: Kernel trivial
echo "  [5a] Kernel trivial..."
cat > /tmp/diag_trivial.cu << 'EOF'
__global__ void trivial_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i * 2.0f;
}
EOF
START=$(date +%s)
$NVCC -x cu -arch=sm_70 -O0 -Xcicc -O0 --ptxas-options=-O0 \
    -Xcompiler -O0 -std=c++17 $CUDA_INC_FLAGS \
    -c /tmp/diag_trivial.cu -o /tmp/diag_trivial.o 2>/tmp/diag_trivial.log
RC=$?; END=$(date +%s)
if [ $RC -eq 0 ]; then
    echo "    ✅ OK em $((END-START))s"
else
    echo "    ❌ FALHOU (rc=$RC)"
    tail -3 /tmp/diag_trivial.log | sed 's/^/    /'
fi

# Teste 2: Kernel com templates
echo "  [5b] Kernel com templates (4 instanciações)..."
cat > /tmp/diag_templates.cu << 'EOF'
#include <cuda_runtime.h>
template <typename T, int BLOCK_SIZE>
__global__ void reduce_kernel(const T* input, T* output, int n) {
    __shared__ T sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + tid;
    sdata[tid] = (i < n) ? input[i] : T(0);
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
template __global__ void reduce_kernel<float, 256>(const float*, float*, int);
template __global__ void reduce_kernel<double, 256>(const double*, double*, int);
template __global__ void reduce_kernel<float, 128>(const float*, float*, int);
template __global__ void reduce_kernel<int, 256>(const int*, int*, int);
EOF
START=$(date +%s)
$NVCC -x cu -arch=sm_70 -O0 -Xcicc -O0 --ptxas-options=-O0 \
    -Xcompiler -O0 -std=c++17 $CUDA_INC_FLAGS \
    -c /tmp/diag_templates.cu -o /tmp/diag_templates.o 2>/tmp/diag_templates.log
RC=$?; END=$(date +%s)
if [ $RC -eq 0 ]; then
    echo "    ✅ OK em $((END-START))s"
else
    echo "    ❌ FALHOU (rc=$RC)"
    tail -3 /tmp/diag_templates.log | sed 's/^/    /'
fi

# Teste 3: CUB sort (simula o gargalo do TF)
echo "  [5c] CUB DeviceRadixSort (4 tipos)..."
CUB_INC=""
for d in /root/miniforge3/include /usr/local/cuda/include /root/cuda_unified/include; do
    [ -f "$d/cub/cub.cuh" ] && CUB_INC="-I$d" && break
done
if [ -n "$CUB_INC" ]; then
    cat > /tmp/diag_cub.cu << 'CUBEOF'
#include <cub/device/device_radix_sort.cuh>
template <typename KeyT>
void instantiate_sort() {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_bytes, (const KeyT*)nullptr, (KeyT*)nullptr,
        1024, 0, sizeof(KeyT) * 8, 0);
}
template void instantiate_sort<float>();
template void instantiate_sort<double>();
template void instantiate_sort<int>();
template void instantiate_sort<unsigned int>();
CUBEOF
    START=$(date +%s)
    $NVCC -x cu -arch=sm_70 -O0 -Xcicc -O0 --ptxas-options=-O0 \
        -Xcompiler -O0 -std=c++17 $CUDA_INC_FLAGS $CUB_INC \
        -DEIGEN_DONT_VECTORIZE -D__NO_INLINE__ -U__VSX__ -U__ALTIVEC__ \
        -c /tmp/diag_cub.cu -o /tmp/diag_cub.o 2>/tmp/diag_cub.log
    RC=$?; END=$(date +%s)
    if [ $RC -eq 0 ]; then
        echo "    ✅ OK em $((END-START))s ($(ls -lh /tmp/diag_cub.o | awk '{print $5}'))"
    else
        echo "    ❌ FALHOU (rc=$RC)"
        grep -i "error" /tmp/diag_cub.log | tail -3 | sed 's/^/    /'
    fi
else
    echo "    ⏭ CUB não encontrado"
fi

# Limpeza
rm -f /tmp/diag_trivial.{cu,o,log} /tmp/diag_templates.{cu,o,log} /tmp/diag_cub.{cu,o,log}

[ "$MODE" = "--replay" ] || [ "$MODE" = "--all" ] || { echo ""; echo "=== Fim ==="; exit 0; }

# =============================================================================
# [6] REPLAY DO ÚLTIMO COMANDO NVCC DO BAZEL
# =============================================================================
echo ""
echo "━━━ [6] REPLAY DO ÚLTIMO COMANDO NVCC (via Bazel) ━━━"

if [ ! -f "$NVCC_DUMP" ]; then
    echo "  ❌ $NVCC_DUMP não existe — rode o build primeiro"
    exit 0
fi

TOTAL_CMDS=$(grep -c "^NVCC ARGS:" "$NVCC_DUMP" 2>/dev/null || echo 0)
echo "  Total de comandos no dump: $TOTAL_CMDS"

# Tipos CUB compilados
echo "  Variantes CUB compiladas:"
grep "NVCC ARGS:" "$NVCC_DUMP" | grep -oE 'DCUB_TYPE_[A-Z0-9_]+' | sort -u | sed 's/^/    /'

# Último comando
LAST_CMD=$(tail -1 "$NVCC_DUMP" | sed 's/^NVCC ARGS: //')
CU_FILE=$(echo "$LAST_CMD" | grep -oE '[^ ]+\.cu\.cc' | head -1)
echo ""
echo "  Último arquivo: $(basename $CU_FILE)"

# Verificar flags do wrapper
echo "  Flags do wrapper:"
echo "$LAST_CMD" | grep -q "Xcicc" && echo "    ✅ -Xcicc -O0" || echo "    ❌ -Xcicc -O0 FALTANDO"
echo "$LAST_CMD" | grep -q "arch=sm_70" && echo "    ✅ -arch=sm_70" || echo "    ❌ -arch=sm_70 FALTANDO"
echo "$LAST_CMD" | grep -q "mno-float128" && echo "    ⚠️  -mno-float128 presente" || echo "    ✅ sem -mno-float128"

# Replay
if [ -d "$EXEC_ROOT" ]; then
    echo ""
    echo "  >>> Replay: compilando $(basename $CU_FILE) fora do Bazel..."
    cd "$EXEC_ROOT"
    START=$(date +%s)
    $NVCC --verbose \
        -I/root/miniforge3/include \
        -I/root/cuda_unified/include \
        -I/usr/local/cuda/include \
        $LAST_CMD 2>&1 | while IFS= read -r line; do
            NOW=$(date +%s); ELAPSED=$((NOW - START))
            if [[ "$line" == *"/cicc"* ]]; then echo "    [${ELAPSED}s] cicc iniciou (CUDA → PTX)"; fi
            if [[ "$line" == *"/ptxas"* ]]; then echo "    [${ELAPSED}s] ptxas iniciou (PTX → SASS)"; fi
            if [[ "$line" == *"/fatbinary"* ]]; then echo "    [${ELAPSED}s] fatbinary"; fi
            if [[ "$line" == *"error"* ]] && [[ "$line" != *"display_error"* ]]; then echo "    [${ELAPSED}s] ❌ $line"; fi
        done
    END=$(date +%s)
    TOTAL=$((END - START))
    echo "    Tempo total: ${TOTAL}s"
    echo ""
    echo "  Comparação:"
    echo "    Replay (sozinho):      ${TOTAL}s"
    echo "    Dentro do Bazel:       ~150-200s (contensão de 16 jobs)"
    echo "    Razão: 16 processos cicc competindo por CPU"
else
    echo "  ❌ Exec root não encontrado: $EXEC_ROOT"
fi

# =============================================================================
# [7] MONITORAMENTO EM TEMPO REAL (se build estiver rodando)
# =============================================================================
echo ""
echo "━━━ [7] STATUS DO BUILD ━━━"
CICC_PROCS=$(pgrep -c -f cicc 2>/dev/null || echo 0)
PTXAS_PROCS=$(pgrep -c -f ptxas 2>/dev/null || echo 0)
if [ "$CICC_PROCS" -gt 0 ] || [ "$PTXAS_PROCS" -gt 0 ]; then
    echo "  🔨 Build em andamento!"
    echo "  Processos cicc ativos:  $CICC_PROCS"
    echo "  Processos ptxas ativos: $PTXAS_PROCS"
    echo "  RAM livre: $(free -h | awk '/^Mem/{print $4}')"
    echo ""
    echo "  Compilações CUDA ativas:"
    ps aux | grep -E 'cicc|ptxas' | grep -v grep | awk '{
        split($11, a, "/"); 
        printf "    PID %s | CPU %s%% | MEM %s%% | %s\n", $2, $3, $4, a[length(a)]
    }'
else
    echo "  Nenhum processo CUDA ativo (build parado ou finalizado)"
fi

echo ""
echo "======================================================="
echo "  Diagnóstico completo!"
echo "======================================================="
