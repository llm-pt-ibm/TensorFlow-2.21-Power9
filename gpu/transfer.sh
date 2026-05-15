#!/bin/bash
# Diagnose T4 'lacks nop' mystery + force rebuild se patch nao foi aplicado.

set -o pipefail
source /root/miniforge3/etc/profile.d/conda.sh
conda activate tf221_build || conda activate base || true

LLD_INSTALL="$HOME/tensorflow_gpu/llvm-install"
LLD="$LLD_INSTALL/bin/ld.lld"

for D in \
    "$HOME/tensorflow_gpu/lld-ppc64-fix" \
    "$HOME/tensorflow_gpu" \
    "$(pwd)/lld-ppc64-fix" \
    "$(pwd)"; do
    [ -f "$D/build_patched_lld.sh" ] && SCRIPT_DIR="$D" && break
done
TEST_SCRIPT=""
for P in \
    "$SCRIPT_DIR/test_lld.sh" \
    "$SCRIPT_DIR/teste_lld.sh" \
    "$HOME/tensorflow_gpu/test_lld.sh" \
    "$HOME/tensorflow_gpu/teste_lld.sh" \
    "$(pwd)/test_lld.sh" \
    "$(pwd)/teste_lld.sh"; do
    [ -f "$P" ] && TEST_SCRIPT="$P" && break
done

echo "=== [1] LLD info ==="
ls -la "$LLD"
realpath "$LLD"
file "$LLD"
echo "  ldd $LLD:"
ldd "$LLD" 2>&1 | head -10 | sed 's/^/    /'
echo ""

echo "=== [2] Onde 'lacks nop' realmente esta? ==="
echo ">>> strings (default) em todos arquivos da install:"
find "$LLD_INSTALL" -type f -exec sh -c 'strings "$1" 2>/dev/null | grep -lq "lacks nop" >/dev/null && echo "  $1"' _ {} \; 2>/dev/null
N1=$(find "$LLD_INSTALL" -type f -exec sh -c 'strings "$1" 2>/dev/null | grep -c "lacks nop"' _ {} \; 2>/dev/null | awk '{s+=$1} END{print s}')
echo "  total hits via strings default: $N1"
echo ""
echo ">>> strings -a (all sections) em todos arquivos:"
find "$LLD_INSTALL" -type f 2>/dev/null | while read F; do
    N=$(strings -a "$F" 2>/dev/null | grep -c "lacks nop")
    [ "$N" -gt 0 ] && echo "  $F  ($N hits via -a)"
done
echo ""
echo ">>> grep -a binary direto em todos arquivos:"
find "$LLD_INSTALL" -type f 2>/dev/null | while read F; do
    N=$(grep -aoc "lacks nop" "$F" 2>/dev/null)
    [ "$N" -gt 0 ] && echo "  $F  ($N hits via grep -a)"
done
echo ""
echo ">>> .rodata section direto via objdump:"
objdump -s -j .rodata "$LLD" 2>/dev/null | grep -A1 "lacks" | head -10 || echo "  nada na .rodata do binario principal"
echo ""

echo "=== [3] Confirma source LLD tem a string original ==="
LLD_SRC="$SCRIPT_DIR/llvm-project"
if [ -d "$LLD_SRC" ]; then
    echo "  Procurando em $LLD_SRC/lld/ELF/Arch/PPC64.cpp:"
    grep -n "lacks nop" "$LLD_SRC/lld/ELF/Arch/PPC64.cpp" 2>/dev/null | head -5
    echo ""
    echo "  Lines 1810-1820:"
    sed -n '1808,1822p' "$LLD_SRC/lld/ELF/Arch/PPC64.cpp" 2>/dev/null | sed 's/^/    /'
else
    echo "  $LLD_SRC NAO existe"
fi
echo ""

echo "=== [4] Testar regex python do build script ==="
if [ -f "$LLD_SRC/lld/ELF/Arch/PPC64.cpp" ]; then
python3 << PYEOF
import re
with open("$LLD_SRC/lld/ELF/Arch/PPC64.cpp") as f:
    src = f.read()
p1 = re.compile(r'Err\(ctx\)\s*<<[^;]*?lacks nop[^;]*;', re.DOTALL)
m1 = p1.findall(src)
print(f"  Pattern 1 matches: {len(m1)}")
for m in m1[:3]:
    print("  ---")
    print("  " + m.replace("\n", "\n  ")[:400])
PYEOF
fi
echo ""

echo "=== [5] Force rebuild ==="
echo "  Apagando llvm-build cache + install pra garantir rebuild fresh..."
rm -rf "$SCRIPT_DIR/llvm-build" 2>/dev/null
rm -rf "$LLD_INSTALL/bin/lld" "$LLD_INSTALL/bin/ld.lld" 2>/dev/null
bash "$SCRIPT_DIR/build_patched_lld.sh" 2>&1 | tail -25
echo ""
ls -la "$LLD"
echo ""

echo "=== [6] Pos-rebuild: 'lacks nop' ainda existe? ==="
find "$LLD_INSTALL" -type f 2>/dev/null | while read F; do
    N=$(grep -aoc "lacks nop" "$F" 2>/dev/null)
    [ "$N" -gt 0 ] && echo "  AINDA: $F ($N hits)"
done
echo ""

echo "=== [7] Rodar testes ==="
LLD_PATH="$LLD" bash "$TEST_SCRIPT"
echo ""
echo "exit: $?"
