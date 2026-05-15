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
# Reset any previous attempts
git checkout -- lld/ELF/Thunks.cpp 2>/dev/null || true

# Apply the actual code changes (not the git-formatted patch, do it manually)
echo "    -> Patching lld/ELF/Thunks.cpp..."

# 1. Change size from 32 to 36
sed -i 's/uint32_t size() override { return 32; }/uint32_t size() override { return 36; }/' \
    lld/ELF/Thunks.cpp

# 2. Add std r2,24(r1) before writePPC64LoadAndBranch in PPC64LongBranchThunk::writeTo
# The original code:
#   writePPC64LoadAndBranch(ctx, buf, offset);
# Becomes:
#   write32(ctx, buf + 0, 0xf8410018); // std r2, 24(r1)
#   writePPC64LoadAndBranch(ctx, buf + 4, offset);
sed -i '/^void PPC64LongBranchThunk::writeTo/,/^}/ {
    s|writePPC64LoadAndBranch(ctx, buf, offset);|// Save caller TOC pointer for multi-TOC correctness.\n  write32(ctx, buf + 0, 0xf8410018); // std r2, 24(r1)\n  writePPC64LoadAndBranch(ctx, buf + 4, offset);|
}' lld/ELF/Thunks.cpp

# 3. NOTE: We do NOT set needsTocRestore on long branch thunks.
# Reason: libgcc.a has bl calls to _savegpr0_NN/_restgpr0_NN without
# trailing nops (they're leaf functions that never clobber r2).
# Setting needsTocRestore would cause LLD to error on those.
# The patcher v6 handles the RESTORE side (patching nops to ld r2,24(r1))
# for CUDA-related calls in the final .so.

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
mkdir -p "$INSTALL_DIR/bin" "$INSTALL_DIR/lib"
cp bin/ld.lld "$INSTALL_DIR/bin/"
cp lib/*.so* "$INSTALL_DIR/lib/" 2>/dev/null || true
ln -sf ld.lld "$INSTALL_DIR/bin/lld"

echo ""
echo "============================================="
echo " LLD patchado instalado em:"
echo "   $INSTALL_DIR/bin/ld.lld"
echo ""
echo " Para usar no build do TensorFlow:"
echo "   export BAZEL_FUSE_LD=$INSTALL_DIR/bin/ld.lld"
echo "   --linkopt=-fuse-ld=$INSTALL_DIR/bin/ld.lld"
echo ""
echo " Se funcionar, o patcher v6 pode ser REMOVIDO!"
echo "============================================="

if [ "${1:-}" = "--test" ]; then
    echo "=== Running LLD PPC64 tests ==="
    cd "$BUILD_DIR"
    ninja check-lld-elf
fi
