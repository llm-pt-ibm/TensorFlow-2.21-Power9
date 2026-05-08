#!/bin/bash
# =============================================================================
# Teste isolado: compila kernels CUDA no Power9 FORA do Bazel
# Testa o pipeline completo: NVCC → cicc → ptxas → fatbinary
# Usando kernels auto-contidos (sem deps do TF/XLA)
#
# USO na VM:
#   chmod +x test_cucc_standalone.sh && ./test_cucc_standalone.sh
# =============================================================================

export PATH=/root/miniforge3/nvvm/bin:/root/miniforge3/bin:/root/cuda_unified/bin:$PATH
NVCC="/root/cuda_unified/bin/nvcc"

# Detectar include dir do CUDA (onde está cuda_runtime.h)
CUDA_INC_FLAGS=""
for d in /root/cuda_unified/include /root/miniforge3/include /usr/local/cuda/include \
         /root/miniforge3/targets/ppc64le-linux/include /usr/local/include/cuda_stub; do
    if [ -f "$d/cuda_runtime.h" ] || [ -f "$d/cuda.h" ]; then
        CUDA_INC_FLAGS="$CUDA_INC_FLAGS -I$d"
    fi
done
echo "CUDA includes: $CUDA_INC_FLAGS"

echo "======================================================="
echo "  Teste Isolado de Compilação CUDA (Power9)"
echo "======================================================="

# [1] Verificar toolchain
echo ""
echo "[1] Verificando toolchain CUDA..."
for bin in nvcc cicc ptxas fatbinary; do
    P=$(which $bin 2>/dev/null)
    if [ -n "$P" ]; then
        ARCH=$(file "$P" 2>/dev/null | grep -oE 'ppc64|x86.64|aarch64' | head -1)
        echo "  ✅ $bin: $P (${ARCH:-unknown})"
    else
        echo "  ❌ $bin: NÃO ENCONTRADO"
        echo "     ERRO FATAL: toolchain CUDA incompleta!"
        exit 1
    fi
done

# [2] Kernel trivial (deve compilar em <5s)
echo ""
echo "[2] Teste 1: Kernel trivial..."
cat > /tmp/test_trivial.cu << 'EOF'
__global__ void trivial_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i * 2.0f;
}
EOF
START=$(date +%s)
$NVCC -x cu -arch=sm_70 -O0 -Xcicc -O0 --ptxas-options=-O0 \
    -Xcompiler -O0 -std=c++17 $CUDA_INC_FLAGS \
    -c /tmp/test_trivial.cu -o /tmp/test_trivial.o 2>/tmp/test_trivial.log
RC=$?
END=$(date +%s)
if [ $RC -eq 0 ]; then
    echo "  ✅ OK em $((END-START))s"
else
    echo "  ❌ FALHOU (rc=$RC) em $((END-START))s"
    tail -5 /tmp/test_trivial.log
    echo "  ERRO FATAL: nem kernel trivial compila!"
    exit 1
fi

# [3] Kernel com templates (simula complexidade média do TF)
echo ""
echo "[3] Teste 2: Kernel com templates..."
cat > /tmp/test_templates.cu << 'EOF'
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

// Instanciações explícitas (como TF faz)
template __global__ void reduce_kernel<float, 256>(const float*, float*, int);
template __global__ void reduce_kernel<double, 256>(const double*, double*, int);
template __global__ void reduce_kernel<float, 128>(const float*, float*, int);
template __global__ void reduce_kernel<int, 256>(const int*, int*, int);
EOF
START=$(date +%s)
$NVCC -x cu -arch=sm_70 -O0 -Xcicc -O0 --ptxas-options=-O0 \
    -Xcompiler -O0 -std=c++17 $CUDA_INC_FLAGS \
    -c /tmp/test_templates.cu -o /tmp/test_templates.o 2>/tmp/test_templates.log
RC=$?
END=$(date +%s)
if [ $RC -eq 0 ]; then
    echo "  ✅ OK em $((END-START))s"
else
    echo "  ❌ FALHOU (rc=$RC) em $((END-START))s"
    tail -10 /tmp/test_templates.log
fi

# [4] Kernel CUB (simula cub_sort_kernel - O MAIS PESADO do TF)
echo ""
echo "[4] Teste 3: Kernel CUB (simula cub_sort_kernel)..."
# Verifica se CUB está disponível
CUB_HEADER=""
for d in /usr/local/cuda/include /root/miniforge3/include /root/cuda_unified/include; do
    if [ -f "$d/cub/cub.cuh" ]; then
        CUB_HEADER="$d"
        break
    fi
done

if [ -z "$CUB_HEADER" ]; then
    # Tenta achar via find
    CUB_HEADER=$(find /usr /root /opt -name "cub.cuh" -path "*/cub/*" 2>/dev/null | head -1)
    [ -n "$CUB_HEADER" ] && CUB_HEADER=$(dirname $(dirname "$CUB_HEADER"))
fi

if [ -n "$CUB_HEADER" ]; then
    echo "  CUB encontrado em: $CUB_HEADER"
    cat > /tmp/test_cub.cu << 'CUBEOF'
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_sort.cuh>

// Simula as instanciações que o TF faz no cub_sort_kernel_cuda_impl.cu.cc
// O TF instancia para: f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

template <typename KeyT>
void instantiate_sort() {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_bytes,
        (const KeyT*)nullptr, (KeyT*)nullptr,
        1024, 0, sizeof(KeyT) * 8, 0);
}

// Instanciações explícitas
template void instantiate_sort<float>();
template void instantiate_sort<double>();
template void instantiate_sort<int>();
template void instantiate_sort<unsigned int>();
CUBEOF
    START=$(date +%s)
    echo "  Compilando... (pode levar 2-10 min)"
    timeout 600 $NVCC -x cu -arch=sm_70 -O0 -Xcicc -O0 --ptxas-options=-O0 \
        -Xcompiler -O0 -std=c++17 $CUDA_INC_FLAGS \
        -I"$CUB_HEADER" \
        -DEIGEN_DONT_VECTORIZE -D__NO_INLINE__ -U__VSX__ -U__ALTIVEC__ \
        --verbose \
        -c /tmp/test_cub.cu -o /tmp/test_cub.o 2>/tmp/test_cub.log &
    PID=$!

    # Monitorar progresso
    while kill -0 $PID 2>/dev/null; do
        NOW=$(date +%s)
        ELAPSED=$((NOW-START))
        # Checar em qual fase está
        if grep -q "/cicc" /tmp/test_cub.log 2>/dev/null && ! grep -q "/ptxas" /tmp/test_cub.log 2>/dev/null; then
            PHASE="cicc (CUDA→PTX)"
        elif grep -q "/ptxas" /tmp/test_cub.log 2>/dev/null && ! grep -q "/fatbinary" /tmp/test_cub.log 2>/dev/null; then
            PHASE="ptxas (PTX→SASS)"
        elif grep -q "/fatbinary" /tmp/test_cub.log 2>/dev/null; then
            PHASE="fatbinary"
        else
            PHASE="preprocessing"
        fi
        printf "\r  [%3ds] Fase: %-25s RAM livre: %s" $ELAPSED "$PHASE" "$(free -h | awk '/^Mem/{print $4}')"
        sleep 5
    done
    echo ""

    wait $PID
    RC=$?
    END=$(date +%s)
    TOTAL=$((END-START))

    if [ $RC -eq 0 ] && [ -f /tmp/test_cub.o ]; then
        SIZE=$(ls -lh /tmp/test_cub.o | awk '{print $5}')
        echo "  ✅ CUB OK em ${TOTAL}s (tamanho: $SIZE)"
    elif [ $RC -eq 124 ]; then
        echo "  ⏰ TIMEOUT após ${TOTAL}s"
        echo "  Últimas linhas:"
        tail -5 /tmp/test_cub.log
    else
        echo "  ❌ CUB FALHOU em ${TOTAL}s (rc=$RC)"
        echo "  Erros:"
        grep -i "error" /tmp/test_cub.log | tail -10
    fi
else
    echo "  ⏭ CUB não encontrado — pulando teste pesado"
fi

# [5] Resumo
echo ""
echo "======================================================="
echo "  RESUMO"
echo "======================================================="
echo "  Teste 1 (trivial):    $([ -f /tmp/test_trivial.o ] && echo '✅ OK' || echo '❌ FALHOU')"
echo "  Teste 2 (templates):  $([ -f /tmp/test_templates.o ] && echo '✅ OK' || echo '❌ FALHOU')"
echo "  Teste 3 (CUB sort):   $([ -f /tmp/test_cub.o ] && echo '✅ OK' || echo '❌/⏭ VER ACIMA')"
echo ""
echo "  Se todos passaram: o pipeline NVCC funciona no Power9!"
echo "  Se CUB falhou/travou: checar RAM e tempo do cicc"
echo "======================================================="

# Limpeza
rm -f /tmp/test_trivial.cu /tmp/test_trivial.o /tmp/test_trivial.log
rm -f /tmp/test_templates.cu /tmp/test_templates.o /tmp/test_templates.log
rm -f /tmp/test_cub.cu /tmp/test_cub.o /tmp/test_cub.log
