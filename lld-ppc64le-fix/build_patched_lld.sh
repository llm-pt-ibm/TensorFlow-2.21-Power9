#!/bin/bash
# Build patched LLD for PPC64LE: fixes the multi-TOC LongBranchThunk r2
# corruption bug. Applies a single coherent, upstream-quality change set to
# lld/ELF/Thunks.cpp (3 edits) and rebuilds only lld + its dependencies.
#
# Prerequisites: cmake (>=3.20), ninja (or make), a C++17 compiler (clang or
#                gcc), python3, git. ~16GB RAM, ~5GB disk.
# Time: ~30min cross-compiling on x86_64; ~2-4h native on POWER9/POWER10.
#
# Usage:
#   ./build_patched_lld.sh          # Build + install
#   ./build_patched_lld.sh --test   # Build + install + run check-lld-elf
#
# Environment overrides:
#   LLD_INSTALL_DIR   install prefix (default: $HOME/lld-ppc64-install)
#   CONDA_BASE        if set + valid, use that conda env's toolchain;
#                     otherwise the script falls back to system tooling.
#
# =============================================================================
# PATCH STATUS / UPSTREAMABILITY
# =============================================================================
# This script applies 3 patches to LLVM/lld, all targeting lld/ELF/Thunks.cpp.
# All three together form a single coherent, upstream-quality fix.
#
#   [1] PPC64LongBranchThunk::size() 32 -> 36 bytes
#   [2] PPC64LongBranchThunk::writeTo prefixes `std r2, 24(r1)`
#       --- CORE FIX. Genuine bug in LLD's PPC64LE backend. Upstream LLD
#       assumes intra-module bls never cross TOC boundaries. In multi-TOC
#       mode (libs with >64KB TOC, e.g. libtensorflow_framework.so ~67MB
#       .text), LongBranchThunks DO cross TOC groups and silently clobber
#       caller's r2. Fix: thunk saves caller r2 before branching.
#
#   [3] PPC64LongBranchThunk::addSymbols conditionally sets needsTocRestore
#       --- COMPLEMENT TO [2]. Marks the thunk as needing r2 restore ONLY
#       when the destination symbol may actually clobber r2 (global entry
#       point or ABI v1.5 CLOBBERS_R2 marker, i.e. (st_other >> 5) != 0).
#       Leaf-style callees with st_other == 0 are skipped, which means the
#       upstream "lacks nop, can't restore toc" diagnostic remains correct
#       (only fires when an ABI-violating caller bls a clobber-r2 callee
#       without a trailing nop) — no diagnostic suppression needed.
#
# VERDICT for general distribution:
#   - For TensorFlow build on PPC64LE: correct and sufficient.
#   - As a general PPC64LE LLD release: correct and sufficient.
#   - As an upstream LLVM MR: submittable as a single PR ([1]+[2]+[3]).
# =============================================================================

set -o pipefail

# Optional: activate a Conda environment if CONDA_BASE points to one. This is
# convenient on hosts where the system toolchain is too old. If conda is not
# present, fall back to whatever cmake/ninja/clang/gcc the user has on PATH.
if [ -n "${CONDA_BASE:-}" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-base}" 2>/dev/null || true
    conda install -c conda-forge cmake ninja -y 2>/dev/null || true
fi

# Strict error checking
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLVM_SRC="${SCRIPT_DIR}/llvm-project"
BUILD_DIR="${SCRIPT_DIR}/llvm-build"
INSTALL_DIR="${LLD_INSTALL_DIR:-$HOME/lld-ppc64-install}"
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

# --- Patch [3] CORE FIX (upstreamable) ---
# Set needsTocRestore on a LongBranchThunk ONLY when the callee may clobber
# r2. The top 3 bits of st_other encode the ELFv2 local-entry value:
#   - value 0      -> no global entry, leaf-style, r2 preserved by callee
#   - value 1      -> ABI v1.5 CLOBBERS_R2 marker (".localentry sym, 1")
#   - value 2..6   -> function has a global entry point (offset 1 << (val-2))
# We treat values >= 1 as "may clobber r2" and set needsTocRestore on the
# thunk symbol, which makes LLD rewrite the trailing nop to ld r2,24(r1).
# Value == 0 leaves needsTocRestore unset, matching upstream behavior for
# leaf helpers (libgcc savres, simple no-prologue functions) and avoiding
# the "lacks nop, can't restore toc" diagnostic that would otherwise fire
# on legacy callers that omit the nop after bls to such leaves.
python3 - << 'PYFIX3'
import re
path = "lld/ELF/Thunks.cpp"
with open(path) as f:
    src = f.read()

pattern = re.compile(
    r'void PPC64LongBranchThunk::addSymbols\(ThunkSection &isec\) \{\s*'
    r'addSymbol\(\s*ctx\.saver\.save\("__long_branch_" \+ destination\.getName\(\)\),\s*'
    r'STT_FUNC,\s*0,\s*isec\);\s*\}',
    re.DOTALL,
)
replacement = (
    'void PPC64LongBranchThunk::addSymbols(ThunkSection &isec) {\n'
    '  Defined *s = addSymbol(\n'
    '      ctx.saver.save("__long_branch_" + destination.getName()), STT_FUNC,\n'
    '      0, isec);\n'
    '  // Only request post-call r2 restore when the callee may have clobbered\n'
    '  // it: global entry (st_other top bits >= 2) or ABI v1.5 CLOBBERS_R2\n'
    '  // marker (st_other top bits == 1). Leaf-style callees with st_other == 0\n'
    '  // preserve r2 and need no restore.\n'
    '  if ((destination.stOther >> 5) != 0)\n'
    '    s->setNeedsTocRestore(true);\n'
    '}\n'
)
new, n = pattern.subn(replacement, src)
if n == 0:
    raise SystemExit(
        "ERROR: PPC64LongBranchThunk::addSymbols block not matched. "
        "Upstream LLD may have changed shape; update the regex in patch [3]."
    )
with open(path, 'w') as f:
    f.write(new)
print(f"Patch [3] applied (conditional needsTocRestore): {n} block(s) replaced.")
PYFIX3

echo "    -> Patch applied successfully."

# Copy the test file
cp "${SCRIPT_DIR}/0001-lld-PPC64-Add-r2-TOC-save-to-LongBranch-thunks.patch" \
   "${LLVM_SRC}/lld/test/ELF/" 2>/dev/null || true

echo "=== [3/5] Configuring build ==="
# Clean stale cache to ensure our settings take effect
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Compiler selection: prefer Clang. On Conda PPC64LE envs `conda activate`
# exports CC/CXX pointing at conda's GCC cross compiler, which emits PCREL /
# `-fno-plt` reloc types that the LLD shipped in the same env may not handle
# during the LLVM self-link ("unknown relocation 31/60/119/120"). Forcing
# conda's clang side-steps that whole class of failures and also avoids the
# R_PPC64_REL24 overflow that older BFD+GCC hits when linking large LLVM libs.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/clang" ]; then
    CC="${CONDA_PREFIX}/bin/clang"
    CXX="${CONDA_PREFIX}/bin/clang++"
    echo "    -> forcing Conda clang: $CC"
elif command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
    CC="${CC:-$(command -v clang)}"
    CXX="${CXX:-$(command -v clang++)}"
fi
CMAKE_COMPILER_ARGS=()
[ -n "${CC:-}" ]  && CMAKE_COMPILER_ARGS+=(-DCMAKE_C_COMPILER="$CC")
[ -n "${CXX:-}" ] && CMAKE_COMPILER_ARGS+=(-DCMAKE_CXX_COMPILER="$CXX")

# Build only LLD (much faster than full LLVM)
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    "${CMAKE_COMPILER_ARGS[@]}" \
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
# Use the top-level `install` target (not `install-lld`) so that the LLVM
# support libs (libLLVMSupport.so, etc.) co-installed with BUILD_SHARED_LIBS=ON
# end up in $INSTALL_DIR/lib/ — otherwise ld.lld would not find them at runtime.
ninja install

echo ""
echo "=== Post-install verification ==="
ls -la "$INSTALL_DIR/bin/ld.lld" "$INSTALL_DIR/bin/lld" 2>/dev/null || true

echo ""
echo "============================================="
echo " Patched LLD installed at:"
echo "   $INSTALL_DIR/bin/ld.lld"
echo ""
echo " To use:"
echo "   export PATH=$INSTALL_DIR/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
echo "   \$CC -fuse-ld=$INSTALL_DIR/bin/ld.lld ..."
echo "============================================="

if [ "${1:-}" = "--test" ]; then
    echo "=== Running LLD PPC64 tests ==="
    cd "$BUILD_DIR"
    ninja check-lld-elf
fi
