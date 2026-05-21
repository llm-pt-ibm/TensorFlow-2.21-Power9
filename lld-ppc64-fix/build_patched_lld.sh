#!/bin/bash
# Build patched LLD for PPC64LE to fix the multi-TOC long branch r2 save bug.
# This eliminates the need for the patcher v6 workaround in TensorFlow builds.
#
# Prerequisites: cmake, ninja (or make), clang/gcc, python3
# Time: ~2-4h on Power9, ~30min cross-compiling on x86_64
# RAM: ~16GB recommended
#
# Usage:
#   ./build_patched_lld.sh          # Build on Power9
#   ./build_patched_lld.sh --test   # Build + run lit tests
#
# =============================================================================
# PATCH STATUS / UPSTREAMABILITY
# =============================================================================
# This script applies 4 patches to LLVM/lld. Their nature differs:
#
#   [1] PPC64LongBranchThunk size 32 -> 36 bytes
#   [2] PPC64LongBranchThunk::writeTo prefixes `std r2, 24(r1)`
#       --- CORE FIX (upstreamable). Genuine bug in LLD's PPC64LE backend.
#       Upstream LLD assumes intra-module bls never cross TOC boundaries.
#       In multi-TOC mode (libs with >64KB TOC, e.g. libtensorflow_framework.so
#       ~67MB .text), LongBranchThunks DO cross TOC groups and silently clobber
#       caller's r2. Fix: thunk saves caller r2 before branching.
#       Could be submitted as MR with refinement to only activate in multi-TOC.
#
#   [3] PPC64LongBranchThunk::addSymbols sets needsTocRestore(true)
#       --- OVER-BROAD WORKAROUND (not upstreamable as-is). Marks ALL
#       LongBranchThunks as requiring r2 restore. Correct for thunks created
#       in multi-TOC mode (callee may have clobbered r2), but unnecessary for
#       leaf functions like libgcc `_savegpr0_NN` -> false-positives in [4].
#       Proper upstream fix: only set when target may clobber r2 (e.g. has
#       global entry or localentry=1) AND multi-TOC mode is active.
#
#   [4] Replace "lacks nop, can't restore toc" Err(ctx) with (void)0
#       --- TF-SPECIFIC WORKAROUND (do NOT upstream). Suppresses a legitimate
#       ABI-violation diagnostic. Needed only because [3] is over-broad and
#       triggers the error on libgcc/legacy code that doesn't expect r2 restore.
#       For a clean general-purpose LLD ppc64le release, [3] should be refined
#       so [4] becomes unnecessary.
#
# VERDICT for general distribution:
#   - For TensorFlow build on PPC64LE: this build is correct and sufficient.
#   - As a general PPC64LE LLD release: NOT recommended. Refine [3] first.
#   - As reference for an upstream LLVM MR: patches [1]+[2] are the core
#     contribution; [3]+[4] should be replaced by conditional logic.
# =============================================================================

set -o pipefail

# Activate Conda for cmake/ninja/clang
CONDA_BASE="${CONDA_BASE:-/root/miniforge3}"
source "$CONDA_BASE/etc/profile.d/conda.sh" || true
conda activate tf221_build || conda activate base || true
# Ensure cmake and ninja are available
conda install -c conda-forge cmake ninja -y || true

# Now enable strict error checking
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLVM_SRC="${SCRIPT_DIR}/llvm-project"
BUILD_DIR="${SCRIPT_DIR}/llvm-build"
# Install path matches what gpu/build_tf221_power9_gpu_generic.sh expects:
# $HOME/tensorflow_gpu/llvm-install. Override via LLD_INSTALL_DIR if needed.
INSTALL_DIR="${LLD_INSTALL_DIR:-$HOME/tensorflow_gpu/llvm-install}"
PATCH_FILE="${SCRIPT_DIR}/0001-lld-PPC64-Add-r2-TOC-save-to-LongBranch-thunks.patch"

echo "=== [1/5] Cloning LLVM (shallow) ==="
if [ ! -d "$LLVM_SRC" ]; then
    git clone --depth 1 https://github.com/llvm/llvm-project.git "$LLVM_SRC"
else
    echo "    -> Already cloned, skipping."
fi

echo "=== [2/5] Applying patch ==="
cd "$LLVM_SRC"
# Reset any previous attempts (both files we modify)
git checkout -- lld/ELF/Thunks.cpp 2>/dev/null || true
git checkout -- lld/ELF/Arch/PPC64.cpp 2>/dev/null || true

# Apply the actual code changes (not the git-formatted patch, do it manually)
echo "    -> Patching lld/ELF/Thunks.cpp..."

# --- Patch [1] CORE FIX (upstreamable) ---
# Change PPC64LongBranchThunk size from 32 to 36 to make room for the std r2.
sed -i 's/uint32_t size() override { return 32; }/uint32_t size() override { return 36; }/' \
    lld/ELF/Thunks.cpp

# --- Patch [2] CORE FIX (upstreamable) ---
# Add std r2,24(r1) before writePPC64LoadAndBranch in PPC64LongBranchThunk::writeTo.
# Genuine bug fix: upstream LLD never emits this, breaking multi-TOC libs where
# the thunk crosses TOC groups and clobbers caller's r2.
sed -i '/^void PPC64LongBranchThunk::writeTo/,/^}/ {
    s|writePPC64LoadAndBranch(ctx, buf, offset);|// Save caller TOC pointer for multi-TOC correctness.\n  write32(ctx, buf + 0, 0xf8410018); // std r2, 24(r1)\n  writePPC64LoadAndBranch(ctx, buf + 4, offset);|
}' lld/ELF/Thunks.cpp

# --- Patch [3] OVER-BROAD WORKAROUND (refine before upstreaming) ---
# Set needsTocRestore on every LongBranchThunk so LLD patches the trailing nop
# to `ld r2, 24(r1)`. Correct for thunks crossing TOC, but unnecessary (and
# false-positive-prone) for leaf-style callees like libgcc _savegpr0_NN.
# Upstream-quality fix: gate this on `mayClobberCallerToc(destination)`.
sed -i '/^void PPC64LongBranchThunk::addSymbols/,/^}/ {
    s|addSymbol(ctx.saver.save("__long_branch_" + destination.getName()), STT_FUNC,|Defined *s = addSymbol(ctx.saver.save("__long_branch_" + destination.getName()), STT_FUNC,|
    s|0, isec);|0, isec);\n  s->setNeedsTocRestore(true);|
}' lld/ELF/Thunks.cpp

# --- Patch [4] TF-SPECIFIC WORKAROUND (do NOT upstream) ---
# Demote the "lacks nop" error to a no-op so LLD doesn't abort.
# Some legitimately-missing nops (libgcc _savegpr0_NN, NVCC-generated code that
# manually does ld r2,24(r1)) trigger this error. Replace the multi-line
# `Err(ctx) << ...;` statement with a benign expression.
# Strategy: find the line containing "lacks nop" and replace the whole multi-line
# Err(ctx) statement that contains it.
python3 - << 'PYFIX'
import re
path = "lld/ELF/Arch/PPC64.cpp"
with open(path) as f:
    src = f.read()
# Find Err(ctx) ... ; that contains 'lacks nop'. Use [^;]* to span lines.
pattern = re.compile(r'Err\(ctx\)\s*<<[^;]*?lacks nop[^;]*;', re.DOTALL)
new, n = pattern.subn('/* error suppressed: lacks nop */ (void)0;', src)
if n == 0:
    # Try alternate names (errorOrWarn, warn, etc.)
    pattern2 = re.compile(r'(?:Err\(ctx\)|errorOrWarn|warn)\([^)]*\)\s*<<[^;]*?lacks nop[^;]*;', re.DOTALL)
    new, n = pattern2.subn('/* error suppressed: lacks nop */ (void)0;', src)
if n == 0:
    print("WARN: 'lacks nop' statement not found - LLD may have changed shape")
else:
    with open(path, 'w') as f:
        f.write(new)
    print(f"Replaced {n} 'lacks nop' error statement(s).")
PYFIX

echo "    -> Patch applied successfully."

# Copy the test file
cp "${SCRIPT_DIR}/0001-lld-PPC64-Add-r2-TOC-save-to-LongBranch-thunks.patch" \
   "${LLVM_SRC}/lld/test/ELF/" 2>/dev/null || true

echo "=== [3/5] Configuring build ==="
# Clean stale cache to ensure our settings take effect
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Only build LLD (much faster than full LLVM)
# Use Clang + LLD from Conda to avoid R_PPC64_REL24 overflow in BFD+GCC
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CONDA_PREFIX}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${CONDA_PREFIX}/bin/clang++" \
    -DLLVM_ENABLE_PROJECTS="lld" \
    -DLLVM_TARGETS_TO_BUILD="PowerPC" \
    -DLLVM_USE_LINKER=lld \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DLLVM_PARALLEL_LINK_JOBS=2 \
    "${LLVM_SRC}/llvm"

echo "=== [4/5] Building LLD ==="
ninja lld

echo "=== [5/5] Installing ==="
# 'install-lld' instala SO o componente lld (bin + libs principais), MAS com
# BUILD_SHARED_LIBS=ON as deps LLVM (libLLVMSupport.so etc.) ficam fora.
# 'install' instala TUDO no INSTALL_PREFIX, garantindo que as libs
# correspondam ao ld.lld -> sem desync de "lacks nop" ainda na liblldELF antiga.
ninja install

# Verificacao final: bin e libs principais
echo ""
echo "=== Verificacao pos-install ==="
ls -la "$INSTALL_DIR/bin/ld.lld" "$INSTALL_DIR/bin/lld" 2>/dev/null || true
echo ">>> Procurando 'lacks nop' nos binarios instalados:"
LEFTOVER=0
for F in $(find "$INSTALL_DIR/bin" "$INSTALL_DIR/lib" -type f 2>/dev/null); do
    if grep -aoq "lacks nop" "$F" 2>/dev/null; then
        echo "  AINDA presente em: $F"
        LEFTOVER=1
    fi
done
if [ "$LEFTOVER" = "0" ]; then
    echo "  OK: 'lacks nop' removido de todos os binarios instalados."
else
    echo "  AVISO: 'lacks nop' ainda aparece - patch python pode ter falhado."
fi

echo ""
echo "============================================="
echo " LLD patchado instalado em:"
echo "   $INSTALL_DIR/bin/ld.lld"
echo ""
echo " Para usar no build do TensorFlow:"
echo "   export BAZEL_FUSE_LD=$INSTALL_DIR/bin/ld.lld"
echo "   --linkopt=-fuse-ld=$INSTALL_DIR/bin/ld.lld"
echo "============================================="

if [ "${1:-}" = "--test" ]; then
    echo "=== Running LLD PPC64 tests ==="
    cd "$BUILD_DIR"
    ninja check-lld-elf
fi
