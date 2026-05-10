#!/bin/bash
TF_FW=/root/miniforge3/envs/tf221_build/lib/python3.11/site-packages/tensorflow/libtensorflow_framework.so.2

echo "=== Sections .got/.toc/.data (precisa multiplas?) ==="
readelf -SW "$TF_FW" 2>/dev/null | grep -E '\.got|\.toc|\.data\b' | head -10

echo ""
echo "=== Calculo manual: r2 do callback = base + 0xD0CA090 ==="
echo "Stub addis 14 + ld 17208 -> target = r2 + 0xE4338"
echo "Endereco computado relative ao lib base = 0xD0CA090 + 0xE4338 = 0xD1AE3C8"
echo ""
echo "=== Section que contem offset 0xD1AE3C8: ==="
readelf -SW "$TF_FW" 2>/dev/null | awk '
NR>1 && /^[ \t]*\[/ {
    name=$2; addr=strtonum("0x" $4); size=strtonum("0x" $6);
    end = addr + size;
    if (0xD1AE3C8 >= addr && 0xD1AE3C8 < end) print "MATCH: " $0
}'

echo ""
echo "=== Tem multiplas .toc/.got sections? (multi-TOC indicator) ==="
readelf -SW "$TF_FW" 2>/dev/null | grep -cE '\.got\b|\.toc\b|\.got\.plt'

echo ""
echo "=== Quantidade de stubs LLD no total ==="
nm "$TF_FW" 2>/dev/null | grep -c '__long_branch_'

echo ""
echo "=== Range de enderecos dos stubs LLD (vamos ver se sao agrupados) ==="
nm "$TF_FW" 2>/dev/null | grep '__long_branch_' | awk '{print $1}' | sort -u | head -3
echo "..."
nm "$TF_FW" 2>/dev/null | grep '__long_branch_' | awk '{print $1}' | sort -u | tail -3
