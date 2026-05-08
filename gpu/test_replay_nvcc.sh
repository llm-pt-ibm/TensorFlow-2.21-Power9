#!/bin/bash
# =============================================================================
# Replay: Pega o ÚLTIMO comando NVCC do /tmp/nvcc_dump.txt e executa
# diretamente com --verbose para mostrar em qual fase está travando.
#
# USO:
#   bash test_replay_nvcc.sh            # roda o último comando
#   bash test_replay_nvcc.sh 3          # roda o 3º comando do dump
# =============================================================================

export PATH=/root/miniforge3/nvvm/bin:/root/miniforge3/bin:/root/cuda_unified/bin:$PATH

DUMP="/tmp/nvcc_dump.txt"

if [ ! -f "$DUMP" ]; then
    echo "ERRO: $DUMP não existe. O build Bazel precisa ter rodado pelo menos uma vez."
    exit 1
fi

# Contar quantos comandos temos
TOTAL_CMDS=$(grep -c "^NVCC ARGS:" "$DUMP" 2>/dev/null || echo 0)
echo "=== Encontrados $TOTAL_CMDS comandos NVCC no dump ==="
echo ""

# Listar os arquivos .cu.cc compilados
echo "Arquivos .cu.cc compilados:"
grep "^NVCC ARGS:" "$DUMP" | grep -oE '[^ ]+\.cu\.cc' | while read f; do
    echo "  $(basename $f)"
done
echo ""

# Selecionar qual comando rodar
CMD_IDX=${1:-$TOTAL_CMDS}  # último por padrão
echo ">>> Rodando comando #$CMD_IDX de $TOTAL_CMDS"
echo ""

# Extrair o comando
CMD_LINE=$(sed -n "${CMD_IDX}p" "$DUMP" | sed 's/^NVCC ARGS: //')

if [ -z "$CMD_LINE" ]; then
    echo "ERRO: Linha $CMD_IDX está vazia no dump"
    exit 1
fi

# Mostrar o arquivo sendo compilado
CU_FILE=$(echo "$CMD_LINE" | grep -oE '[^ ]+\.cu\.cc' | head -1)
echo "Arquivo: $(basename $CU_FILE)"
echo "Path:    $CU_FILE"
echo ""

# Verificar se -Xcicc -O0 está presente
if echo "$CMD_LINE" | grep -q "Xcicc"; then
    echo "✅ -Xcicc -O0 presente nos args"
else
    echo "❌ -Xcicc -O0 NÃO presente! O wrapper não está injetando."
fi

if echo "$CMD_LINE" | grep -q "arch=sm_70"; then
    echo "✅ -arch=sm_70 presente"
else
    echo "❌ -arch=sm_70 NÃO presente!"
fi

if echo "$CMD_LINE" | grep -q "mno-float128"; then
    echo "⚠️ -mno-float128 ENCONTRADO nos args (não deveria estar!)"
else
    echo "✅ -mno-float128 ausente (correto)"
fi
echo ""

# Montar o comando completo
NVCC="/root/cuda_unified/bin/nvcc"
FULL_CMD="$NVCC --verbose $CMD_LINE"

echo ">>> Executando NVCC direto (com --verbose)..."
echo "    Timeout: 600s"
echo ""

START=$(date +%s)
eval "timeout 600 $FULL_CMD" 2>&1 | while IFS= read -r line; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    
    if [[ "$line" == *"/cicc"* ]]; then
        echo "  [${ELAPSED}s] 🔵 cicc INICIOU (CUDA → PTX) — esta é a fase lenta"
    elif [[ "$line" == *"/ptxas"* ]]; then
        echo "  [${ELAPSED}s] 🟡 ptxas INICIOU (PTX → SASS)"
    elif [[ "$line" == *"/fatbinary"* ]]; then
        echo "  [${ELAPSED}s] 🟢 fatbinary INICIOU"
    elif [[ "$line" == *"/gcc"* ]] || [[ "$line" == *"/g++"* ]]; then
        echo "  [${ELAPSED}s] ⚪ gcc host INICIOU"
    elif [[ "$line" == *"error"* ]] || [[ "$line" == *"Error"* ]] || [[ "$line" == *"fatal"* ]]; then
        echo "  [${ELAPSED}s] ❌ $line"
    fi
done
RC=${PIPESTATUS[0]}

END=$(date +%s)
TOTAL=$((END - START))

echo ""
if [ $RC -eq 0 ]; then
    echo "======================================================="
    echo "  ✅ Compilou em ${TOTAL}s"
    echo "======================================================="
elif [ $RC -eq 124 ]; then
    echo "======================================================="
    echo "  ⏰ TIMEOUT após ${TOTAL}s"
    echo "======================================================="
else
    echo "======================================================="
    echo "  ❌ FALHOU em ${TOTAL}s (exit: $RC)"
    echo "======================================================="
fi

echo ""
echo "Comparação:"
echo "  Standalone CUB (4 tipos):  ~15s"
echo "  Este arquivo via Bazel:    ~${TOTAL}s"
echo "  Diferença = overhead de headers TF/XLA + mais instanciações"
