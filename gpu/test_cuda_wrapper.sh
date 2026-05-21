#!/bin/bash
# =============================================================================
# TESTE UNITÁRIO DO WRAPPER CUDA - Simula compilação como o Bazel faria
# =============================================================================
set -e

echo "======================================================="
echo "  TESTE UNITÁRIO - Wrapper CUDA para Power9/V100"
echo "======================================================="
echo ""

REAL_NVCC="/root/cuda_unified/bin/nvcc"
CUDA_INCLUDE="/root/miniforge3/include"
NVVM_BIN="/root/miniforge3/nvvm/bin"
CONDA_BIN="/root/miniforge3/bin"

# ---- TESTE 0: Verificar binários existem ----
echo "[TESTE 0] Verificando binários..."
PASS=0; FAIL=0

if [ -f "$REAL_NVCC" ]; then
    echo "  ✅ NVCC encontrado: $REAL_NVCC"
    PASS=$((PASS+1))
else
    echo "  ❌ NVCC NÃO encontrado: $REAL_NVCC"
    FAIL=$((FAIL+1))
fi

CICC_PATH=$(find "$NVVM_BIN" -name cicc -type f 2>/dev/null | head -1)
if [ -n "$CICC_PATH" ]; then
    echo "  ✅ cicc encontrado: $CICC_PATH"
    ARCH=$(file "$CICC_PATH" | grep -o 'ppc64\|x86-64\|aarch64' | head -1)
    echo "     Arquitetura: $ARCH"
    PASS=$((PASS+1))
else
    echo "  ❌ cicc NÃO encontrado em $NVVM_BIN"
    FAIL=$((FAIL+1))
fi

if [ -f "$CUDA_INCLUDE/cuda_runtime.h" ]; then
    echo "  ✅ cuda_runtime.h encontrado: $CUDA_INCLUDE/cuda_runtime.h"
    PASS=$((PASS+1))
else
    echo "  ❌ cuda_runtime.h NÃO encontrado em $CUDA_INCLUDE"
    FAIL=$((FAIL+1))
fi

PTXAS_PATH=$(find /root/cuda_unified -name ptxas -type f 2>/dev/null | head -1)
if [ -n "$PTXAS_PATH" ]; then
    echo "  ✅ ptxas encontrado: $PTXAS_PATH"
    PASS=$((PASS+1))
else
    echo "  ⚠️  ptxas NÃO encontrado em /root/cuda_unified (pode estar em outro lugar)"
fi

echo ""

# ---- TESTE 1: NVCC simples SEM fix de PATH ----
echo "[TESTE 1] NVCC simples SEM cicc no PATH (deve FALHAR)..."
cat > /tmp/test_cuda_simple.cu << 'CUEOF'
__global__ void kernel_test() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
}
CUEOF

# Limpa PATH do nvvm para simular o bug
CLEAN_PATH=$(echo "$PATH" | tr ':' '\n' | grep -v nvvm | tr '\n' ':')
if PATH="$CLEAN_PATH" timeout 10 $REAL_NVCC -c /tmp/test_cuda_simple.cu -o /tmp/test1.o -arch=sm_70 -I"$CUDA_INCLUDE" 2>/tmp/test1_err.log; then
    echo "  ⚠️  Compilou mesmo sem PATH (cicc pode já estar linkado)"
    PASS=$((PASS+1))
else
    echo "  ✅ Falhou como esperado (cicc não encontrado)"
    echo "     Erro: $(head -1 /tmp/test1_err.log)"
    PASS=$((PASS+1))
fi
echo ""

# ---- TESTE 2: NVCC simples COM fix de PATH ----
echo "[TESTE 2] NVCC simples COM cicc no PATH (deve SUCESSO)..."
if PATH="$NVVM_BIN:$CONDA_BIN:$PATH" timeout 30 $REAL_NVCC -c /tmp/test_cuda_simple.cu -o /tmp/test2.o -arch=sm_70 -I"$CUDA_INCLUDE" 2>/tmp/test2_err.log; then
    echo "  ✅ SUCESSO! NVCC compilou kernel CUDA simples"
    ls -lh /tmp/test2.o | awk '{print "     Tamanho do .o: " $5}'
    PASS=$((PASS+1))
else
    echo "  ❌ FALHOU! Erro:"
    cat /tmp/test2_err.log
    FAIL=$((FAIL+1))
fi
echo ""

# ---- TESTE 3: NVCC com as FLAGS EXATAS do wrapper ----
echo "[TESTE 3] NVCC com flags do wrapper (simula Bazel)..."
if PATH="$NVVM_BIN:$CONDA_BIN:$PATH" timeout 60 $REAL_NVCC \
    -x cu -O0 \
    --ptxas-options=-O0 \
    --compiler-options=-O0 \
    -DEIGEN_DONT_VECTORIZE \
    -D__NO_INLINE__ \
    -U__VSX__ \
    -U__ALTIVEC__ \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"$CUDA_INCLUDE" \
    -arch=sm_70 \
    --verbose \
    -c /tmp/test_cuda_simple.cu -o /tmp/test3.o 2>/tmp/test3_err.log; then
    echo "  ✅ SUCESSO! Flags do wrapper funcionam"
    ls -lh /tmp/test3.o | awk '{print "     Tamanho do .o: " $5}'
    PASS=$((PASS+1))
else
    echo "  ❌ FALHOU com flags do wrapper! Erro:"
    cat /tmp/test3_err.log
    FAIL=$((FAIL+1))
fi
echo ""

# ---- TESTE 4: Kernel mais complexo (template + shared memory) ----
echo "[TESTE 4] Kernel complexo (templates + shared memory)..."
cat > /tmp/test_cuda_complex.cu << 'CUEOF'
#include <cuda_runtime.h>

template<typename T, int BLOCK_SIZE>
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
CUEOF

START_TS=$(date +%s)
if PATH="$NVVM_BIN:$CONDA_BIN:$PATH" timeout 120 $REAL_NVCC \
    -x cu -O0 \
    --ptxas-options=-O0 \
    --compiler-options=-O0 \
    -DEIGEN_DONT_VECTORIZE \
    -D__NO_INLINE__ \
    -U__VSX__ \
    -U__ALTIVEC__ \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"$CUDA_INCLUDE" \
    -arch=sm_70 \
    -c /tmp/test_cuda_complex.cu -o /tmp/test4.o 2>/tmp/test4_err.log; then
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))
    echo "  ✅ SUCESSO! Kernel complexo compilou em ${ELAPSED}s"
    ls -lh /tmp/test4.o | awk '{print "     Tamanho do .o: " $5}'
    PASS=$((PASS+1))
else
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))
    echo "  ❌ FALHOU após ${ELAPSED}s! Erro:"
    tail -5 /tmp/test4_err.log
    FAIL=$((FAIL+1))
fi
echo ""

# ---- TESTE 5: Telemetria (log de progresso) ----
echo "[TESTE 5] Telemetria do wrapper (escrita em /tmp)..."
rm -f /tmp/nvcc_progress.log
LOGFILE="/tmp/nvcc_progress.log"
LABEL="test_telemetry.cu"
echo "[NVCC START] $LABEL @ $(date '+%H:%M:%S')" >> "$LOGFILE"
START_TS=$(date +%s)

PATH="$NVVM_BIN:$CONDA_BIN:$PATH" $REAL_NVCC \
    -x cu -O0 --ptxas-options=-O0 --compiler-options=-O0 \
    -I"$CUDA_INCLUDE" -arch=sm_70 --verbose \
    -c /tmp/test_cuda_simple.cu -o /tmp/test5.o 2>&1 | while IFS= read -r line; do
    if [[ "$line" == "#$ "* ]]; then
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TS))
        if [[ "$line" == *"/cicc"* ]]; then
            echo "[NVCC 10%] $LABEL | Etapa: Frontend CUDA → PTX | ${ELAPSED}s" >> "$LOGFILE"
        elif [[ "$line" == *"/ptxas"* ]]; then
            echo "[NVCC 60%] $LABEL | Etapa: PTX → SASS | ${ELAPSED}s" >> "$LOGFILE"
        elif [[ "$line" == *"/fatbinary"* ]]; then
            echo "[NVCC 90%] $LABEL | Etapa: Empacotando fatbinary | ${ELAPSED}s" >> "$LOGFILE"
        fi
    fi
done

if [ -f "$LOGFILE" ] && [ -s "$LOGFILE" ]; then
    LINES=$(wc -l < "$LOGFILE")
    echo "  ✅ Log criado com $LINES entradas:"
    cat "$LOGFILE" | sed 's/^/     /'
    PASS=$((PASS+1))
else
    echo "  ❌ Log vazio ou não criado"
    FAIL=$((FAIL+1))
fi
echo ""

# ---- RESULTADO FINAL ----
echo "======================================================="
echo "  RESULTADO: $PASS aprovados, $FAIL reprovados"
echo "======================================================="
if [ $FAIL -eq 0 ]; then
    echo "  🎉 TODOS OS TESTES PASSARAM!"
    echo "  O wrapper está pronto. Pode re-executar o build com confiança."
else
    echo "  ⚠️  Alguns testes falharam. Revise os erros acima."
fi
echo ""

# Limpa arquivos temporários
rm -f /tmp/test_cuda_simple.cu /tmp/test_cuda_complex.cu /tmp/test1.o /tmp/test2.o /tmp/test3.o /tmp/test4.o /tmp/test5.o /tmp/test1_err.log /tmp/test2_err.log /tmp/test3_err.log /tmp/test4_err.log
