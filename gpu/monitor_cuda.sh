#!/bin/bash
# =============================================================================
# MONITOR DE COMPILAÇÃO CUDA - Power9/V100
# Lê /tmp/nvcc_progress.log gerado pelo gcc_cuda_wrapper com telemetria
# =============================================================================

LOGFILE="/tmp/nvcc_progress.log"

while true; do
    clear
    echo "======================================================="
    echo "  MONITOR CUDA  |  $(date '+%H:%M:%S')"
    echo "======================================================="

    # Processos NVCC ativos com CPU e tempo
    echo ""
    echo "[ PROCESSOS NVCC ATIVOS ]"
    NVCC_PROCS=$(ps aux | grep -E "[n]vcc|[c]icc|[p]txas" | grep -v monitor)
    if [ -z "$NVCC_PROCS" ]; then
        echo "  Nenhum processo NVCC ativo no momento"
    else
        echo "$NVCC_PROCS" | awk '{
            elapsed = $10
            cpu = $3
            mem = $4
            fname = ""
            for(i=11;i<=NF;i++) {
                if ($i ~ /\.cu\.cc$/) {
                    split($i, parts, "/")
                    fname = parts[length(parts)]
                }
            }
            if (fname == "") fname = $11
            printf "  PID %-8s | CPU: %-5s%% | MEM: %-5s%% | Tempo: %-8s | %s\n", $2, cpu, mem, elapsed, fname
        }'
    fi

    # Progresso por etapa do NVCC
    echo ""
    echo "[ PROGRESSO POR ARQUIVO (últimas 15 entradas) ]"
    if [ -f "$LOGFILE" ]; then
        tail -15 "$LOGFILE" | while IFS= read -r line; do
            if [[ "$line" == *"NVCC START"* ]]; then
                echo "  ░░░░░░░░░░   0% | $line"
            elif [[ "$line" == *"NVCC 10%"* ]]; then
                echo "  ▓░░░░░░░░░  10% | $line"
            elif [[ "$line" == *"NVCC 60%"* ]]; then
                echo "  ▓▓▓▓▓▓░░░░  60% | $line"
            elif [[ "$line" == *"NVCC 90%"* ]]; then
                echo "  ▓▓▓▓▓▓▓▓▓░  90% | $line"
            elif [[ "$line" == *"NVCC 95%"* ]]; then
                echo "  ▓▓▓▓▓▓▓▓▓▓  95% | $line"
            elif [[ "$line" == *"NVCC DONE"* ]]; then
                echo "  ▓▓▓▓▓▓▓▓▓▓ 100% | $line"
            fi
        done
    else
        echo "  Aguardando início da compilação CUDA..."
        echo "  (O arquivo $LOGFILE será criado quando o primeiro .cu.cc for compilado)"
    fi

    # Estatísticas gerais
    echo ""
    echo "[ ESTATÍSTICAS ]"
    if [ -f "$LOGFILE" ]; then
        TOTAL_STARTS=$(grep -c "NVCC START" "$LOGFILE" 2>/dev/null)
        TOTAL_STARTS=${TOTAL_STARTS:-0}
        TOTAL_DONE=$(grep -c "NVCC DONE" "$LOGFILE" 2>/dev/null)
        TOTAL_DONE=${TOTAL_DONE:-0}
        TOTAL_ACTIVE=$((TOTAL_STARTS - TOTAL_DONE))

        # Tempo médio dos concluídos
        AVG_TIME="?"
        if [ "$TOTAL_DONE" -gt 0 ] 2>/dev/null; then
            AVG_TIME=$(grep "NVCC DONE" "$LOGFILE" 2>/dev/null | \
                grep -oP 'Total: \K[0-9]+' | \
                awk '{s+=$1; c++} END {if(c>0) printf "%.0f", s/c; else print "?"}')
        fi

        echo "  Arquivos iniciados : $TOTAL_STARTS"
        echo "  Arquivos concluídos: $TOTAL_DONE"
        echo "  Em compilação agora: $TOTAL_ACTIVE"
        echo "  Tempo médio/arquivo: ${AVG_TIME}s"

        # Mostra o arquivo mais lento
        if [ "$TOTAL_DONE" -gt 0 ] 2>/dev/null; then
            SLOWEST=$(grep "NVCC DONE" "$LOGFILE" 2>/dev/null | \
                grep -oP 'Total: \K[0-9]+' | sort -n | tail -1)
            [ -n "$SLOWEST" ] && echo "  Mais lento até agora: ${SLOWEST}s"
        fi

        # Mostra erros
        ERRORS=$(grep -c "Exit: [^0]" "$LOGFILE" 2>/dev/null)
        ERRORS=${ERRORS:-0}
        [ "$ERRORS" -gt 0 ] 2>/dev/null && echo "  ⚠️  ERROS detectados : $ERRORS"
    else
        echo "  Nenhum dado ainda..."
    fi

    # RAM disponível
    echo ""
    echo "[ MEMÓRIA ]"
    free -h | grep -E "Mem|Swap" | awk '{printf "  %-5s | Total: %-8s | Usado: %-8s | Livre: %s\n", $1, $2, $3, $4}'

    echo ""
    echo "  Atualiza a cada 5s | Ctrl+C para sair"
    sleep 5
done
