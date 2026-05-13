#!/bin/bash
TF_FW=/root/miniforge3/envs/tf221_build/lib/python3.11/site-packages/tensorflow/libtensorflow_framework.so.2

echo "=== [1] Disasm de ToStatusSlow do inicio ate +400 ==="
SYM=$(nm "$TF_FW" 2>/dev/null | grep -E 'ToStatusSlow' | head -1)
echo "$SYM"
VADDR=$(echo "$SYM" | awk '{print "0x"$1}')
gdb -batch -ex "file $TF_FW" -ex 'set pagination off' \
    -ex "disassemble $VADDR, $VADDR + 0x190" \
    2>&1 | grep -vE 'Reading symbols|^$' | head -80
echo ""

echo "=== [2] Disasm de ToStatusSlow continuando ate +1000 (perto do strlen +944) ==="
gdb -batch -ex "file $TF_FW" -ex 'set pagination off' \
    -ex "disassemble $VADDR + 0x300, $VADDR + 0x420" \
    2>&1 | grep -vE 'Reading symbols|^$' | head -60
echo ""

echo "=== [3] Procurar TODAS as ocorrencias de 'std r2, 24(r1)' em ToStatusSlow ==="
gdb -batch -ex "file $TF_FW" -ex 'set pagination off' \
    -ex "disassemble $VADDR, $VADDR + 0x500" \
    2>&1 | grep -E 'std.*r2,24\(r1\)' | head -10
echo ""

echo "=== [4] Procurar bls (calls) DENTRO de ToStatusSlow para ver o que ela chama ==="
gdb -batch -ex "file $TF_FW" -ex 'set pagination off' \
    -ex "disassemble $VADDR, $VADDR + 0x500" \
    2>&1 | grep -E '\sbl\s' | head -20
