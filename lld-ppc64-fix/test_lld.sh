#!/bin/bash
# Test suite for LLD PPC64LE patch (v3 - Clang vs GAS detection).
#
# T1: PPC64LongBranchThunk includes `std r2, 24(r1)` (our patch)
# T2: LongBranchThunk has needsTocRestore => nop after bl becomes `ld r2,24(r1)`
# T3: bl to .localentry sym,1 -> LLD creates PPC64R2SaveStub + patches nop
# T4: "lacks nop" error is suppressed when call lacks trailing nop
#
# CRITICAL: GAS old versions reject `.localentry sym, 1` ("not a valid power of 2").
# Clang's integrated assembler may accept it. We try both.

set -o pipefail
# Do not 'set -u' or 'set -e'.

LLD="${LLD_PATH:-$HOME/tensorflow_gpu/llvm-install/bin/ld.lld}"
[ ! -x "$LLD" ] && LLD="$HOME/tensorflow_gpu/llvm-build/bin/ld.lld"
[ ! -x "$LLD" ] && { echo "ERROR: LLD not found"; exit 1; }

echo "=== LLD: $LLD ==="
"$LLD" --version | head -1
if strings "$LLD" 2>/dev/null | grep -q "lacks nop"; then
    echo "  NOTE: 'lacks nop' message STILL in LLD binary."
else
    echo "  NOTE: 'lacks nop' message REMOVED from LLD binary."
fi

# Diagnose assembler support for .localentry sym, 1
echo ""
echo "=== Assembler diagnostics ==="
echo ">>> GCC version: $(gcc --version | head -1)"
echo ">>> GAS version: $(as --version | head -1)"
if command -v clang >/dev/null; then
    echo ">>> Clang version: $(clang --version | head -1)"
    HAVE_CLANG=1
else
    echo ">>> Clang: NOT FOUND"
    HAVE_CLANG=0
fi

# Probe: does $as or clang accept .localentry sym, 1?
PROBE=$(mktemp -d)
cat > "$PROBE/probe.s" << 'EOF'
    .machine power9
    .abiversion 2
    .text
    .globl probe_sym
    .localentry probe_sym, 1
probe_sym:
    blr
EOF
GCC_OK=0; CLANG_OK=0
if gcc -c "$PROBE/probe.s" -o "$PROBE/probe.gcc.o" 2>"$PROBE/gcc.err"; then
    GCC_OK=1
fi
if [ "$HAVE_CLANG" = "1" ]; then
    if clang -c "$PROBE/probe.s" -o "$PROBE/probe.clang.o" 2>"$PROBE/clang.err"; then
        CLANG_OK=1
    fi
fi
echo ""
echo "  GCC/GAS aceita '.localentry sym, 1'?  $([ $GCC_OK = 1 ] && echo SIM || echo NAO)"
[ $GCC_OK = 0 ] && echo "    erro: $(head -2 $PROBE/gcc.err | tail -1)"
echo "  Clang aceita '.localentry sym, 1'?    $([ $CLANG_OK = 1 ] && echo SIM || echo NAO)"
[ $CLANG_OK = 0 ] && [ $HAVE_CLANG = 1 ] && echo "    erro: $(head -2 $PROBE/clang.err | tail -1)"

# Choose assembler that works
if [ "$CLANG_OK" = "1" ]; then
    AS_CMD="clang -c"
    echo "  -> usando: clang"
elif [ "$GCC_OK" = "1" ]; then
    AS_CMD="gcc -c"
    echo "  -> usando: gcc"
else
    AS_CMD=""
    echo "  -> nenhum assembler aceita .localentry sym, 1! Vou usar GCC mas T3 vai falhar."
    AS_CMD="gcc -c"
fi

# Check stOther in probe.o
if [ -f "$PROBE/probe.clang.o" ] && [ "$CLANG_OK" = "1" ]; then
    SO_BYTE=$(python3 -c "
import struct
with open('$PROBE/probe.clang.o','rb') as f:
    data=f.read()
e_shoff=struct.unpack_from('<Q',data,0x28)[0]
e_shentsize=struct.unpack_from('<H',data,0x3a)[0]
e_shnum=struct.unpack_from('<H',data,0x3c)[0]
e_shstrndx=struct.unpack_from('<H',data,0x3e)[0]
sh=[]
for i in range(e_shnum):
    o=e_shoff+i*e_shentsize
    sh.append((struct.unpack_from('<I',data,o)[0], struct.unpack_from('<I',data,o+4)[0],
               struct.unpack_from('<Q',data,o+0x18)[0], struct.unpack_from('<Q',data,o+0x20)[0],
               struct.unpack_from('<Q',data,o+0x38)[0]))
shstr=sh[e_shstrndx][2]
def nm(i):
    o=shstr+sh[i][0]
    return data[o:data.index(b'\x00',o)].decode('latin-1')
sti=None; sst=None
for i,s in enumerate(sh):
    if s[1]==2: sti=i
    if nm(i)=='.strtab' and s[1]==3: sst=i
strt=sh[sst][2]; symt=sh[sti][2]; n=sh[sti][3]//sh[sti][4]; en=sh[sti][4]
for i in range(n):
    o=symt+i*en
    nameoff=struct.unpack_from('<I',data,o)[0]
    name=data[strt+nameoff:data.index(b'\x00',strt+nameoff)].decode('latin-1')
    if name=='probe_sym':
        print(f'{data[o+5]:#04x}'); break
" 2>/dev/null)
    LE=$(( (0x${SO_BYTE#0x} >> 5) & 7 ))
    echo "  Clang probe_sym: stOther=$SO_BYTE  localentry=$LE  $([ $LE = 1 ] && echo '(CLOBBERS_R2 OK)')"
fi
rm -rf "$PROBE"
echo ""

PASS=0; FAIL=0
report() {
    local name="$1"; local ok="$2"; local msg="${3:-}"
    if [ "$ok" = "1" ]; then echo "  [PASS] $name${msg:+ - $msg}"; PASS=$((PASS+1))
    else echo "  [FAIL] $name${msg:+ - $msg}"; FAIL=$((FAIL+1)); fi
}

# ============================================================
# T1+T2: __long_branch_target_func - intra-module, 34MB gap
# Use simple .S without .localentry to avoid GAS issue here
# ============================================================
echo "=== T1+T2: LongBranchThunk std r2,24 + nop patched ==="
TD=$(mktemp -d); cd "$TD"
cat > monolith.s << 'EOF'
    .text
    .globl _start
    .type _start, @function
_start:
    bl target_func
    nop
    blr
    .space 34000000
    # .hidden forces intra-DSO call -> LongBranchThunk (not PLT call stub).
    .hidden target_func
    .globl target_func
    .type target_func, @function
target_func:
    blr
EOF
$AS_CMD monolith.s -o monolith.o 2>/tmp/t12_err.txt
if [ ! -f monolith.o ]; then
    echo "  asm error:"; sed 's/^/    /' /tmp/t12_err.txt
    report "T1" 0 "asm failed"; report "T2" 0 "skipped"
else
    LOG=$("$LLD" --shared monolith.o -o test.so 2>&1)
    if [ ! -f test.so ]; then
        echo "  link error: ${LOG:0:300}"
        report "T1" 0 "link failed"; report "T2" 0 "skipped"
    else
        objdump -d test.so > dump.txt 2>/dev/null
        echo "  stubs encontrados:"
        grep -E "<(__long_branch|__plt|__toc_save)_" dump.txt | head -5 | sed 's/^/    /'
        # T1: thunk has std r2,24(r1)
        thunk_block=$(awk '/<__long_branch_target_func>:/{flag=1; next} flag && /^[0-9a-f]+ <[^>]+>:/{exit} flag' dump.txt)
        first_insn=$(echo "$thunk_block" | head -1)
        if echo "$first_insn" | grep -qE "std *r?2,24\(r?1\)"; then
            report "T1 (Thunk has std r2,24(r1))" 1
        else
            report "T1 (Thunk has std r2,24(r1))" 0 "first: ${first_insn:0:80}"
            echo "    bloco do thunk:"; echo "$thunk_block" | head -5 | sed 's/^/      /'
        fi
        # T2: nop after bl patched
        start_block=$(awk '/<_start>:/{flag=1; next} flag && /^[0-9a-f]+ <[^>]+>:/{exit} flag' dump.txt)
        after_bl=$(echo "$start_block" | sed -n '2p')
        if echo "$after_bl" | grep -qE "ld *r?2,24\(r?1\)"; then
            report "T2 (nop after bl patched)" 1
        else
            report "T2 (nop after bl patched)" 0 "found: ${after_bl:0:80}"
        fi
    fi
fi
cd /; rm -rf "$TD"
echo ""

# ============================================================
# T3: .localentry sym, 1 - depends on assembler support
# ============================================================
echo "=== T3: bl to .localentry sym,1 -> R2SaveStub + nop ==="
TD=$(mktemp -d); cd "$TD"
cat > combo.s << 'EOF'
    .machine power9
    .abiversion 2
    .text
    # .hidden forces intra-DSO call -> R2SaveStub (not PLT call stub).
    .hidden target_clobbers_r2
    .globl target_clobbers_r2
    .type target_clobbers_r2, @function
    .localentry target_clobbers_r2, 1
target_clobbers_r2:
    li 2, 0
    blr
    .globl _start
    .type _start, @function
_start:
    bl target_clobbers_r2
    nop
    blr
EOF
$AS_CMD combo.s -o combo.o 2>/tmp/t3_err.txt
if [ ! -f combo.o ]; then
    echo "  asm error:"; sed 's/^/    /' /tmp/t3_err.txt
    report "T3.0 (asm accepts .localentry 1)" 0
    report "T3.1" 0 "skipped"; report "T3.2" 0 "skipped"; report "T3.3" 0 "skipped"
else
    report "T3.0 (asm accepts .localentry 1)" 1
    echo "  readelf:"
    readelf -sW combo.o 2>/dev/null | grep -E "target_clobbers_r2 *$" | head -1 | sed 's/^/    /'
    # Read raw st_other
    python3 -c "
import struct
with open('combo.o','rb') as f: data=f.read()
e_shoff=struct.unpack_from('<Q',data,0x28)[0]
e_shentsize=struct.unpack_from('<H',data,0x3a)[0]
e_shnum=struct.unpack_from('<H',data,0x3c)[0]
e_shstrndx=struct.unpack_from('<H',data,0x3e)[0]
sh=[]
for i in range(e_shnum):
    o=e_shoff+i*e_shentsize
    sh.append((struct.unpack_from('<I',data,o)[0],struct.unpack_from('<I',data,o+4)[0],
               struct.unpack_from('<Q',data,o+0x18)[0],struct.unpack_from('<Q',data,o+0x20)[0],
               struct.unpack_from('<Q',data,o+0x38)[0]))
shstr=sh[e_shstrndx][2]
def nm(i):
    o=shstr+sh[i][0]; return data[o:data.index(b'\x00',o)].decode('latin-1')
sti=sst=None
for i,s in enumerate(sh):
    if s[1]==2: sti=i
    if nm(i)=='.strtab' and s[1]==3: sst=i
strt=sh[sst][2]; symt=sh[sti][2]; n=sh[sti][3]//sh[sti][4]; en=sh[sti][4]
for i in range(n):
    o=symt+i*en
    no=struct.unpack_from('<I',data,o)[0]
    name=data[strt+no:data.index(b'\x00',strt+no)].decode('latin-1')
    if name=='target_clobbers_r2':
        st=data[o+5]; le=(st>>5)&7
        print(f'  stOther=0x{st:02x} localentry={le} {\"CLOBBERS_R2\" if le==1 else \"\"}')
        break
" 2>/dev/null

    LOG=$("$LLD" --shared combo.o -o test.so 2>&1)
    if [ ! -f test.so ]; then
        echo "  link error: ${LOG:0:300}"
        report "T3.1 (linker built .so)" 0
    else
        report "T3.1 (linker built .so)" 1
        objdump -d test.so > dump.txt 2>/dev/null
        echo "  stubs:"
        grep -E "<(__long_branch|__plt|__toc_save)_" dump.txt | head -5 | sed 's/^/    /'
        if grep -q "__toc_save_" dump.txt; then
            report "T3.2 (R2SaveStub created)" 1
        else
            report "T3.2 (R2SaveStub created)" 0 "no __toc_save_*"
        fi
        start_block=$(awk '/<_start>:/{flag=1; next} flag && /^[0-9a-f]+ <[^>]+>:/{exit} flag' dump.txt)
        after_bl=$(echo "$start_block" | sed -n '2p')
        if echo "$after_bl" | grep -qE "ld *r?2,24\(r?1\)"; then
            report "T3.3 (nop patched after bl)" 1
        else
            report "T3.3 (nop patched after bl)" 0 "found: ${after_bl:0:80}"
        fi
    fi
fi
cd /; rm -rf "$TD"
echo ""

# ============================================================
# T4: bl without trailing nop survives
# ============================================================
echo "=== T4 prep: grep -a 'lacks nop' in LLD bin + libs ==="
# "lacks nop" vive em liblldELF.so (BUILD_SHARED_LIBS=ON), nao em ld.lld.
LLD_DIR=$(dirname "$LLD")
LLD_PARENT=$(dirname "$LLD_DIR")
for F in "$LLD" "$LLD_PARENT/lib"/lib*ELF*.so* "$LLD_PARENT/lib"/lib*Common*.so*; do
    [ -f "$F" ] || continue
    N=$(grep -aoc "lacks nop" "$F" 2>/dev/null)
    [ "$N" -gt 0 ] && echo "  $F: $N hit(s)"
done
echo ""
echo "=== T4: bl without trailing nop survives ==="
TD=$(mktemp -d); cd "$TD"
cat > target.c << 'EOF'
void target_no_nop(void) {}
EOF
gcc -shared -fPIC target.c -o libtarget.so 2>/dev/null

cat > caller.s << 'EOF'
    .text
    .globl _start
    .type _start, @function
_start:
    bl target_no_nop
    addi 1, 1, 0
    blr
EOF
$AS_CMD caller.s -o caller.o 2>/tmp/t4_err.txt
if [ ! -f caller.o ]; then
    echo "  asm error:"; sed 's/^/    /' /tmp/t4_err.txt
    report "T4" 0 "asm failed"
else
    LOG=$("$LLD" caller.o -L. -ltarget -o test_bin 2>&1)
    if [ -n "$LOG" ]; then
        echo "  linker log: ${LOG:0:300}"
    fi
    if echo "$LOG" | grep -q "lacks nop"; then
        report "T4 (no 'lacks nop' error)" 0 "error: lacks nop"
    elif [ -f test_bin ]; then
        report "T4 (no 'lacks nop' error)" 1
    else
        report "T4 (no 'lacks nop' error)" 0 "link failed: ${LOG:0:200}"
    fi
fi
cd /; rm -rf "$TD"
echo ""

echo "=== SUMMARY ==="
echo "  PASS: $PASS / $((PASS+FAIL))"
echo "  FAIL: $FAIL"
exit "$FAIL"
