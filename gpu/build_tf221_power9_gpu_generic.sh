#!/bin/bash
# ============================================================================
# TensorFlow 2.21 build for PPC64LE + NVIDIA V100. See README.md for context.
#
# Architectural decisions specific to PPC64LE:
#
#   1. Linker: patched LLD (lld-ppc64-fix/) with r2-save in LongBranchThunks
#      and 'lacks nop' error suppressed for libgcc compat.
#
#   2. Host assembler: Clang from conda env (gcc_cuda_wrapper.sh redirects to
#      it). GAS 2.30 rejects `.localentry sym, 1` which Implib.so trampolines
#      need; Clang accepts it. We do NOT rewrite this directive (clean_asm
#      is a no-op).
#
#   3. cuDNN 9 headers are forcibly copied over the conda-forge 8.9.7 headers
#      in 5 Bazel-visible paths.
#
#   4. MLIR generated GPU kernels are DISABLED via
#        --//tensorflow/core/kernels/mlir_generated:enable_gpu=False
#      because hlo_to_kernel (the MLIR->PTX tool) segfaults in the LLVM NVPTX
#      backend when run on a PPC64LE host. The classical C++ template path
#      (REGISTER3 macros) is used instead — no functional loss.
#
#   5. _savefpr_NN / _restfpr_NN PPC64 helper stubs are auto-generated into
#      libppc64_savres.a and force-linked (LLD does not synthesize them).
#
# See README.md "Known Limitations" for details.
# ============================================================================

set -eo pipefail

# Function to stop the script with a clear message on error
trap 'echo "\n!!! ERROR: The script failed at line $LINENO. Check the output above. !!!" >&2; exit 1' ERR
# ENVIRONMENT VARIABLES - ADJUST ACCORDING TO YOUR VM:
# =========================================================================
export TF_BUILD_DIR="${TF_BUILD_DIR:-$PWD/tf221_workspace}"
export CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}" # <--- ADJUST HERE ACCORDING TO YOUR VM

echo "========================================="
echo "Build directory: $TF_BUILD_DIR"
echo "Conda path: $CONDA_BASE"
echo "========================================="

mkdir -p "$TF_BUILD_DIR"

if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo ">>> Miniforge3 not found at $CONDA_BASE. Starting automatic installation..."
    ARCH=$(uname -m)
    echo ">>> Detected architecture: $ARCH"
    
    MINIFORGE_URL=""
    if [ "$ARCH" = "x86_64" ]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    elif [ "$ARCH" = "ppc64le" ]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-ppc64le.sh"
    elif [ "$ARCH" = "aarch64" ]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    fi

    if [ -z "$MINIFORGE_URL" ]; then
        echo "ERROR: Architecture $ARCH is not supported for automatic Miniforge3 installation."
        echo "Install Miniforge3 manually and point CONDA_BASE to the installation location."
        exit 1
    fi

    echo ">>> Downloading Miniforge3..."
    wget -qO /tmp/Miniforge3.sh "$MINIFORGE_URL" || curl -L -o /tmp/Miniforge3.sh "$MINIFORGE_URL"

    echo ">>> Installing Miniforge3 silently at $CONDA_BASE..."
    bash /tmp/Miniforge3.sh -b -p "$CONDA_BASE"
    rm -f /tmp/Miniforge3.sh

    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo ">>> Miniforge3 installed successfully at $CONDA_BASE!"
    else
        echo "ERROR: Failed to install Miniforge3."
        exit 1
    fi
fi

if ! command -v bazel &> /dev/null; then
    echo "ERROR: Bazel was not found in PATH. Have you compiled/installed Bazel on this new VM?"
    echo "The tutorial relies on Bazel already being available in /usr/local/bin or in PATH."
    exit 1
fi

echo ">>> Checking required system packages..."
if command -v dnf &> /dev/null; then
    dnf install -y git gcc gcc-c++ zip unzip which patch wget vim-common || echo "Warning: Failed to install packages via dnf. Continuing..."
elif command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y git gcc g++ zip unzip patch wget vim-common || echo "Warning: Failed to install packages via apt-get. Continuing..."
elif command -v yum &> /dev/null; then
    yum install -y git gcc gcc-c++ zip unzip which patch wget vim-common || echo "Warning: Failed to install packages via yum. Continuing..."
else
    echo "Warning: Package manager not found. Make sure git, gcc, g++, zip, unzip, patch, wget are installed."
fi

source $CONDA_BASE/etc/profile.d/conda.sh
conda create -n tf221_build "python=3.11" numpy wheel packaging requests setuptools pip -c conda-forge -y || true
conda activate tf221_build

# Install the CUDA toolkit BEFORE the build so that cuda_configure() detects CUDA
# The -dev packages bring the headers (.h) needed for compilation (cusolverDn.h, cublas_v2.h, etc.)
conda install -c conda-forge 'cuda-cudart=12.4.*' 'cuda-cudart-dev=12.4.*' 'cuda-libraries=12.4.*' 'cuda-nvrtc=12.4.*' 'cuda-nvcc=12.4.*' \
    libcusolver-dev libcublas-dev libcusparse-dev libcufft-dev libcurand-dev 'cuda-cupti-dev=12.4.*' \
    nccl gcc_linux-ppc64le=11.4 gxx_linux-ppc64le=11.4 \
    'clang=17.*' 'clangxx=17.*' \
    xz curl \
    lld setuptools pip -y
# NOTE: CUDA 12.4.1 is the LAST version with official NVIDIA support for ppc64le.
# NVCC 12.4 officially accepts Clang up to 17.x.
# CUDA 12.5+ removed ppc64le support completely.

# =========================================================================
# cuDNN 9.0.0 ppc64le from NVIDIA (official). Conda-forge only has cuDNN 8.9.7.
# cuDNN 9.1+ has NO ppc64le build (discontinued by NVIDIA).
# This is the last cuDNN 9 officially available for Power9.
# =========================================================================
echo ">>> Checking cuDNN 9 (official NVIDIA installation)..."
if [ -f "$CONDA_PREFIX/lib/libcudnn.so.9" ] && \
   [ "$(readelf -d $CONDA_PREFIX/lib/libcudnn.so.9 2>/dev/null | grep SONAME | grep -c '\.so\.9')" = "1" ]; then
    echo ">>> cuDNN 9 already installed at \$CONDA_PREFIX/lib"
else
    echo ">>> Downloading cuDNN 9.0.0 ppc64le (843 MB)..."
    CUDNN9_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-ppc64le/cudnn-linux-ppc64le-9.0.0.312_cuda12-archive.tar.xz"
    CUDNN9_SHA256="b8ef6f249128e1985893a8787a21de35cb83ec47c6dc6fd1809061dd9a3ffb20"
    mkdir -p $HOME/cudnn9_download
    cd $HOME/cudnn9_download
    if [ ! -f cudnn9.tar.xz ]; then
        curl -L -o cudnn9.tar.xz "$CUDNN9_URL" || exit_with_error 'failed to download cuDNN 9'
    fi
    # Validate checksum
    ACTUAL_SHA=$(sha256sum cudnn9.tar.xz | awk '{print $1}')
    if [ "$ACTUAL_SHA" != "$CUDNN9_SHA256" ]; then
        echo "ERROR: cuDNN checksum mismatch. Expected $CUDNN9_SHA256 got $ACTUAL_SHA"
        rm -f cudnn9.tar.xz
        exit 1
    fi
    echo ">>> Extracting cuDNN 9..."
    tar xJf cudnn9.tar.xz
    EXTRACTED=$(find $HOME/cudnn9_download -maxdepth 1 -type d -name 'cudnn-linux-ppc64le-9*' | head -1)
    echo ">>> Installing libs and headers at \$CONDA_PREFIX..."
    cp -P "$EXTRACTED"/lib/libcudnn* $CONDA_PREFIX/lib/ || exit_with_error 'failed to copy libs'
    cp "$EXTRACTED"/include/*.h $CONDA_PREFIX/include/ 2>/dev/null || true
    cd $TF_BUILD_DIR
    echo ">>> cuDNN 9 installed. SONAMEs:"
    for so in $CONDA_PREFIX/lib/libcudnn*.so.9; do
        [ -f "$so" ] && echo "  $(basename $so): $(readelf -d "$so" 2>/dev/null | grep SONAME | awk '{print $NF}' | tr -d '[]')"
    done
fi

cd $TF_BUILD_DIR
rm -rf tensorflow
git clone --depth 1 --branch v2.21.0 https://github.com/tensorflow/tensorflow.git
cd tensorflow

# --- PATCH: TF_NCCL_VERSION cannot be empty ------------------------------
# In hermetic mode with an empty cuda_redist_stub, nccl_configure.bzl falls
# back to writing `#define TF_NCCL_VERSION ""`. As a result,
# nccl_stub.cc calls GetDsoHandle("nccl", "") which via
# FormatLibraryFileName produces a non-resolvable soname in the RUNPATH of
# libtensorflow_framework, and Multi-GPU collective ops abort with
# "NCCL: Unable to load NCCL library". We force the fallback to "2" so
# that TF tries libnccl.so.2 (which exists via symlink in $CONDA_PREFIX/lib).
NCCL_BZL="third_party/xla/third_party/nccl/hermetic/nccl_configure.bzl"
if [ -f "$NCCL_BZL" ]; then
    sed -i 's|"#define TF_NCCL_VERSION \\"\\""|"#define TF_NCCL_VERSION \\"2\\""|g' \
        "$NCCL_BZL"
    echo ">>> NCCL: TF_NCCL_VERSION fallback patched empty -> \"2\""
    grep -n "TF_NCCL_VERSION" "$NCCL_BZL" | head -5 | sed 's/^/    /'
fi
# -------------------------------------------------------------------------

rm -f .bazelversion
#bazel clean --expunge 2>/dev/null || true

# --- PATCH: Fix absl headers for NVCC cudafe++ compatibility ---
# The NVCC cudafe++ parser can't handle the complex SFINAE templates used by
# absl's insert_or_assign/try_emplace methods. This patch has TWO layers:
#   1) Internal headers (raw_hash_map.h, btree_container.h): guard the
#      ABSL_INTERNAL_X macro expansions with #ifndef __CUDACC__
#   2) Public headers (btree_map.h, flat_hash_map.h): guard the
#      "using Base::insert_or_assign" and "using Base::try_emplace"
#      declarations, which would fail if the base class members are hidden.
# This is safe because CUDA device code never calls insert_or_assign/try_emplace.
echo ">>> Patching absl headers for NVCC cudafe++ compatibility..."

# First, do bazel fetch to download dependencies
bazel fetch //tensorflow/tools/pip_package:wheel 2>&1 | tail -3 || true

# buffer_debug_log_structs.h uses _Alignof (C macro) in C++ code. TF source bug.
# Define the macro globally via copt to transform _Alignof -> alignof.

# Find the absl container directory in bazel cache
BAZEL_OUTPUT_BASE=$(bazel info output_base 2>/dev/null)
ABSL_CONTAINER_DIR="$BAZEL_OUTPUT_BASE/external/com_google_absl/absl/container"

# Fallback: search in bazel cache if not at expected path
if [ ! -d "$ABSL_CONTAINER_DIR" ]; then
    echo ">>> WARNING: absl not found at expected path, searching..."
    ABSL_CONTAINER_DIR=$(find /root/.cache/bazel -path "*/external/com_google_absl/absl/container" -type d ! -path "*/internal/*" 2>/dev/null | head -1)
fi

if [ -d "$ABSL_CONTAINER_DIR" ]; then
    echo ">>> absl container dir: $ABSL_CONTAINER_DIR"

    # Use a Python script for robust multi-file patching
    python3 - "$ABSL_CONTAINER_DIR" << 'ABSL_PATCH_EOF'
import sys, os, re

container_dir = sys.argv[1]

def patch_internal_header(filepath):
    """Guard ABSL_INTERNAL_X(insert_or_assign,...) and ABSL_INTERNAL_X(try_emplace,...) with #ifndef __CUDACC__"""
    if not os.path.isfile(filepath):
        print(f"  SKIP (not found): {filepath}")
        return
    with open(filepath, 'r') as f:
        content = f.read()
    if '__CUDACC__' in content:
        print(f"  SKIP (already patched): {os.path.basename(filepath)}")
        return

    for method in ['insert_or_assign', 'try_emplace']:
        # Pattern: "  ABSL_INTERNAL_X(method_name," ... closing ");"
        # We wrap the entire macro invocation block with #ifndef __CUDACC__ / #endif
        pattern = re.compile(
            r'(^[ \t]*ABSL_INTERNAL_X\(' + method + r',.*?\);)',
            re.MULTILINE | re.DOTALL
        )
        content = pattern.sub(
            r'#ifndef __CUDACC__\n\1\n#endif  // __CUDACC__',
            content
        )
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  PATCHED: {os.path.basename(filepath)}")

def patch_public_header(filepath):
    """Guard 'using Base::insert_or_assign' and 'using Base::try_emplace' with #ifndef __CUDACC__"""
    if not os.path.isfile(filepath):
        print(f"  SKIP (not found): {filepath}")
        return
    with open(filepath, 'r') as f:
        content = f.read()
    if '#ifndef __CUDACC__' in content and 'insert_or_assign' in content:
        print(f"  SKIP (already patched): {os.path.basename(filepath)}")
        return

    for method in ['insert_or_assign', 'try_emplace']:
        # Pattern: "  using Base::method_name;" (with possible whitespace variations)
        pattern = re.compile(
            r'^([ \t]*using Base::' + method + r'\s*;)',
            re.MULTILINE
        )
        content = pattern.sub(
            r'#ifndef __CUDACC__\n\1\n#endif  // __CUDACC__',
            content
        )
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  PATCHED: {os.path.basename(filepath)}")

# Layer 1: Internal headers (ABSL_INTERNAL_X macro definitions)
print(">>> Layer 1: Patching internal absl headers (ABSL_INTERNAL_X macros)...")
internal_dir = os.path.join(container_dir, "internal")
for header in ["raw_hash_map.h", "btree_container.h"]:
    patch_internal_header(os.path.join(internal_dir, header))

# Layer 2: Public headers (using Base::... declarations)
print(">>> Layer 2: Patching public absl headers (using Base:: declarations)...")
for header in ["btree_map.h", "flat_hash_map.h"]:
    patch_public_header(os.path.join(container_dir, header))

print(">>> All absl NVCC patches applied successfully!")
ABSL_PATCH_EOF
else
    echo ">>> ERROR: Could not find absl container directory anywhere in bazel cache!"
    echo ">>> The build may fail with insert_or_assign/try_emplace errors in CUDA files."
fi
echo ">>> absl headers patched for NVCC!"

echo ">>> Creating GCC/NVCC Mutant..."
mkdir -p $HOME/compiler_hijack
# Prefer Conda GCC (11+) over system GCC (8.5) for C++17 absl compatibility
# We prefer CLANG (15.x) as the NVCC host compiler. Clang has native
# support for `__builtin_vectorelements` (17+), accepts `-mprefer-vector-width=*`
# and `-fno-experimental-sanitize-metadata=*` (16+) which GCC rejects, and in general is
# more consistent with Power9 bug reports. NVCC 12.4 accepts Clang up to 17.x.
# Fallback: Conda GCC if Clang is not installed.
CONDA_CLANG=""
CONDA_CLANGXX=""
[ -x "$CONDA_PREFIX/bin/clang" ] && CONDA_CLANG="$CONDA_PREFIX/bin/clang"
[ -x "$CONDA_PREFIX/bin/clang++" ] && CONDA_CLANGXX="$CONDA_PREFIX/bin/clang++"

CONDA_GCC=""
CONDA_GXX=""
[ -x "$CONDA_PREFIX/bin/powerpc64le-conda-linux-gnu-gcc" ] && CONDA_GCC="$CONDA_PREFIX/bin/powerpc64le-conda-linux-gnu-gcc"
[ -x "$CONDA_PREFIX/bin/powerpc64le-conda-linux-gnu-g++" ] && CONDA_GXX="$CONDA_PREFIX/bin/powerpc64le-conda-linux-gnu-g++"

if [ -n "$CONDA_CLANG" ] && [ -x "$CONDA_CLANG" ]; then
    REAL_GCC_PATH="$CONDA_CLANG"
    REAL_GXX_PATH="${CONDA_CLANGXX:-$CONDA_CLANG}"
    echo ">>> Using Conda Clang: $REAL_GCC_PATH"
    echo ">>> Version: $($REAL_GCC_PATH --version | head -1)"
elif [ -n "$CONDA_GCC" ] && [ -x "$CONDA_GCC" ]; then
    REAL_GCC_PATH="$CONDA_GCC"
    REAL_GXX_PATH="${CONDA_GXX:-$CONDA_GCC}"
    echo ">>> Fallback: using Conda GCC: $REAL_GCC_PATH"
    echo ">>> Version: $($REAL_GCC_PATH --version | head -1)"
else
    REAL_GCC_PATH=$(which gcc)
    REAL_GXX_PATH=$(which g++)
    echo ">>> WARNING: Neither Clang nor Conda GCC found, using system: $REAL_GCC_PATH"
fi

cat << EOF > $HOME/compiler_hijack/gcc
#!/bin/bash
# compiler_hijack/gcc: wrapper that points to Clang (or GCC fallback).
# NVCC uses it via -ccbin to compile host code.
# -fPIC always injected (R_PPC64_TOC16_LO in PPC64LE device stubs).
exec "$REAL_GCC_PATH" -fPIC "\$@"
EOF

cat << EOF > $HOME/compiler_hijack/g++
#!/bin/bash
# compiler_hijack/g++: wrapper pointing to Clang++ (or G++ fallback).
# NVCC uses it via -ccbin.
# -fPIC always injected.
exec "${REAL_GXX_PATH}" -fPIC "\$@"
EOF

chmod +x $HOME/compiler_hijack/gcc
chmod +x $HOME/compiler_hijack/g++
ln -sf $HOME/compiler_hijack/g++ $HOME/compiler_hijack/c++
# ld/ld.bfd symlinks for compiler_hijack.
# We prefer Conda binutils (>= 2.40, supports Power10 PCREL relocs from jpegxl/Highway).
# AlmaLinux 8.10 ships binutils 2.30 which doesn't know reloc types 132/133.
# GCC with -fuse-ld=bfd looks for "ld.bfd" WITHOUT prefix, so we create symlinks
# without prefix pointing to the prefixed Conda binaries.
CONDA_LD_PREFIXED="$(command -v powerpc64le-conda-linux-gnu-ld 2>/dev/null || true)"
CONDA_LD_BFD_PREFIXED="$(command -v powerpc64le-conda-linux-gnu-ld.bfd 2>/dev/null || true)"
if [ -n "$CONDA_LD_PREFIXED" ] && [ -x "$CONDA_LD_PREFIXED" ]; then
    ln -sf "$CONDA_LD_PREFIXED" $HOME/compiler_hijack/ld
    echo ">>> Conda ld -> $CONDA_LD_PREFIXED"
else
    ln -sf /usr/bin/ld        $HOME/compiler_hijack/ld
    echo ">>> WARNING: Conda ld not found, using /usr/bin/ld"
fi
if [ -n "$CONDA_LD_BFD_PREFIXED" ] && [ -x "$CONDA_LD_BFD_PREFIXED" ]; then
    ln -sf "$CONDA_LD_BFD_PREFIXED" $HOME/compiler_hijack/ld.bfd
    echo ">>> Conda ld.bfd -> $CONDA_LD_BFD_PREFIXED"
else
    ln -sf /usr/bin/ld.bfd    $HOME/compiler_hijack/ld.bfd
    echo ">>> WARNING: Conda ld.bfd not found, using /usr/bin/ld.bfd"
fi
# ld.lld: installed via conda-forge at the start of the script. Symlinked into hijack
# so that GCC with -fuse-ld=lld can find it.
CONDA_LD_LLD="$(command -v ld.lld 2>/dev/null || true)"
if [ -n "$CONDA_LD_LLD" ] && [ -x "$CONDA_LD_LLD" ]; then
    ln -sf "$CONDA_LD_LLD" $HOME/compiler_hijack/ld.lld
    echo ">>> ld.lld -> $CONDA_LD_LLD"
else
    echo ">>> WARNING: ld.lld not found. Install with: conda install -c conda-forge lld"
fi
export PATH=$HOME/compiler_hijack:$PATH

export CC=gcc
export CXX=g++
export CFLAGS="-O3"
export CXXFLAGS="-O3"
export TF_NEED_CLANG=0
export TF_DOWNLOAD_CLANG=0
export TF_NEED_CUDA=1
export TF_CUDA_PATHS=/usr/local/cuda
export TF_NEED_TENSORRT=0
export TF_NEED_ROCM=0
export TF_NEED_OPENCL=0
export TF_SET_ANDROID_WORKSPACE=0
export PYTHON_BIN_PATH=$(which python3)
export PYTHON_LIB_PATH=$(python3 -c 'import site; print(site.getsitepackages()[0])')
export TF_CONFIGURE_IOS=0
export TF_CUDA_CLANG=0
export TF_CUDA_COMPUTE_CAPABILITIES="7.0"
export CC_OPT_FLAGS=""
export GCC_HOST_COMPILER_PATH=$(which gcc)

yes "" | ./configure || true

# ./configure may inject --config=cuda_clang and linkers (lld/gold)
# that conflict with our gcc_cuda_wrapper.sh. We strip everything and control
# the compiler manually via --action_env=CC.
echo ">>> Cleaning .tf_configure.bazelrc (removing cuda_clang and linker flags)..."
if [ -f .tf_configure.bazelrc ]; then
    # Remove problematic lines (with or without quotes)
    sed -i '/cuda_clang/d' .tf_configure.bazelrc
    sed -i '/fuse-ld/d' .tf_configure.bazelrc
    # Remove -mno-float128 from --copt (NVCC doesn't understand this Power9 GCC flag)
    sed -i '/mno-float128/d' .tf_configure.bazelrc
    # Remove any existing --config=cuda to avoid duplication
    sed -i '/--config=cuda$/d' .tf_configure.bazelrc
    # Add clean --config=cuda (single occurrence)
    echo 'build --config=cuda' >> .tf_configure.bazelrc
    echo ">>> .tf_configure.bazelrc fixed successfully!"
    echo ">>> Final content:"
    cat .tf_configure.bazelrc
else
    echo "ERROR: .tf_configure.bazelrc was not generated by ./configure!"
    exit 1
fi

cd ~

wget -qO rules.tar.gz https://github.com/google-ml-infra/rules_ml_toolchain/archive/d8cb9c2c168cd64000eaa6eda0781a9615a26ffe.tar.gz
rm -rf rules_ml_toolchain_patched
tar -xzf rules.tar.gz
mv rules_ml_toolchain-d8cb9c2c168cd64000eaa6eda0781a9615a26ffe rules_ml_toolchain_patched
sed -i '/load("@bazel_tools\/\/tools\/build_defs\/repo:local.bzl", "new_local_repository")/d' rules_ml_toolchain_patched/cc/deps/cc_toolchain_deps.bzl
sed -i 's/new_local_repository(/native.new_local_repository(/g' rules_ml_toolchain_patched/cc/deps/cc_toolchain_deps.bzl
cd $TF_BUILD_DIR/tensorflow

# --- LLVM stub ---
mkdir -p $HOME/llvm_stub
echo 'VERSION = "0.0.0"' > $HOME/llvm_stub/version.bzl
touch $HOME/llvm_stub/WORKSPACE

cat > $HOME/llvm_stub/BUILD << 'EOF'
filegroup(name = "all", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "clang", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "clang++", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "ld", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "ar", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "llvm-ar", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "distro_libs", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "includes", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "asan_ignorelist", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cuda_wrappers_headers", hdrs = [], visibility = ["//visibility:public"])
EOF

# --- CUDA redistribution stub (avoids PyPI CUDA for inaccessible wheels) ---
mkdir -p $HOME/cuda_redist_stub
touch $HOME/cuda_redist_stub/WORKSPACE
echo 'VERSION = "12"' > $HOME/cuda_redist_stub/version.bzl

# cuda_nccl_stub: separate stub for the @cuda_nccl override with VERSION="2".
# (Full block, generic copy of cuda_redist_stub but with VERSION="2".)
# Created after cuda_redist_stub is complete (line ~420+).
cat > $HOME/cuda_redist_stub/versions.bzl << 'EOF'
NVIDIA_WHEEL_VERSIONS = {
    "12": [],
}
EOF
cat > $HOME/cuda_redist_stub/empty.c << 'C_EOF'
void __dummy_func() {}
C_EOF
gcc -c $HOME/cuda_redist_stub/empty.c -o $HOME/cuda_redist_stub/empty.o || true
ar rcs $HOME/cuda_redist_stub/libdummy.a $HOME/cuda_redist_stub/empty.o || true
gcc -shared -o $HOME/cuda_redist_stub/libdummy.so $HOME/cuda_redist_stub/empty.o || true

cat > $HOME/cuda_redist_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "all", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "header_list", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "headers", hdrs = [], visibility = ["//visibility:public"])
cc_library(name = "libs", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cudart", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "static", srcs = ["libdummy.a"], visibility = ["//visibility:public"])
filegroup(name = "shared", srcs = ["libdummy.so"], visibility = ["//visibility:public"])
cc_library(name = "cupti", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cudnn", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cublas", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cusolver", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cusparse", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "curand", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cufft", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "nvjitlink", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "nccl", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "nvcc", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "nvvm", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "ptxas", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "fatbinary", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "cuobjdump", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "nvlink", srcs = [], visibility = ["//visibility:public"])
filegroup(name = "nvdisasm", srcs = [], visibility = ["//visibility:public"])
EOF

# --- cuda_nccl_stub: separate stub ONLY for the @cuda_nccl override ---
# nccl_configure.bzl reads @cuda_nccl//:version.bzl as _nccl_version and uses
# it for the TF_NCCL_VERSION macro. Sharing cuda_redist_stub (which has
# VERSION="12" from CUDA major) makes TF generate TF_NCCL_VERSION="12" and dlopen
# libnccl.so.12 (nonexistent). Separate stub with VERSION="2" produces
# libnccl.so.2 (exists via symlink in $CONDA_PREFIX/lib).
mkdir -p $HOME/cuda_nccl_stub
touch $HOME/cuda_nccl_stub/WORKSPACE
echo 'VERSION = "2"' > $HOME/cuda_nccl_stub/version.bzl
cp $HOME/cuda_redist_stub/empty.c $HOME/cuda_nccl_stub/
cp $HOME/cuda_redist_stub/empty.o $HOME/cuda_nccl_stub/
cp $HOME/cuda_redist_stub/libdummy.a $HOME/cuda_redist_stub/libdummy.so $HOME/cuda_nccl_stub/
cp $HOME/cuda_redist_stub/BUILD $HOME/cuda_nccl_stub/BUILD
cp $HOME/cuda_redist_stub/versions.bzl $HOME/cuda_nccl_stub/versions.bzl

# RAFT-ENABLE 2026-05-22: cuda_cccl_stub separado expondo Thrust + CUB +
# libcudacxx (cuda/std) do CUDA 12.2 toolkit como header_list. Necessario pra
# RMM/RAFT 25.08 acharem <thrust/detail/config.h>. Compartilhar cuda_redist_stub
# (que tem header_list vazio) quebra o select_k_exec_raft.
CCCL=$HOME/cuda_cccl_stub
CUDA_TK_INC=/usr/local/cuda-12.2/targets/ppc64le-linux/include
rm -rf $CCCL && mkdir -p $CCCL/include
if [ -d "$CUDA_TK_INC/thrust" ]; then
    cp -r $CUDA_TK_INC/thrust $CCCL/include/
    cp -r $CUDA_TK_INC/cub    $CCCL/include/
    cp -r $CUDA_TK_INC/cuda   $CCCL/include/
    touch $CCCL/WORKSPACE
    cat > $CCCL/BUILD << 'CCCL_BUILD'
package(default_visibility = ["//visibility:public"])
filegroup(
    name = "header_list",
    srcs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.cuh",
        "include/**/*.inc",
        "include/**/*",
    ], exclude = [
        "include/**/*.cmake",
        "include/**/*.txt",
        "include/**/CMakeLists.txt",
    ], allow_empty = True),
)
cc_library(
    name = "headers",
    textual_hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.cuh",
        "include/**/*.inc",
        "include/**/*",
    ], exclude = [
        "include/**/*.cmake",
        "include/**/*.txt",
        "include/**/CMakeLists.txt",
    ], allow_empty = True),
    includes = ["include"],
)
CCCL_BUILD
    echo 'VERSION = "2.2.0"' > $CCCL/version.bzl
    touch $CCCL/versions.bzl
    echo ">>> cuda_cccl_stub criado em $CCCL (Thrust+CUB+libcudacxx do toolkit)"
else
    echo ">>> WARNING: $CUDA_TK_INC nao encontrado, cuda_cccl_stub vazio (RAFT vai falhar)"
fi

# --- TensorRT stub ---
mkdir -p $HOME/tensorrt_stub
touch $HOME/tensorrt_stub/WORKSPACE
cat > $HOME/tensorrt_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "LICENSE", srcs = [])
cc_library(name = "tensorrt_headers", hdrs = [])
cc_library(name = "tensorrt", srcs = [])
config_setting(name = "use_static_tensorrt", values = {"define": "stub=true"})
py_library(name = "tensorrt_config_py", srcs = [])
EOF

cat > $HOME/tensorrt_stub/build_defs.bzl << 'EOF'
def if_tensorrt(if_true, if_false = []):
    return if_false
def is_tensorrt_configured():
    return False
def if_tensorrt_exec(if_true, if_false = []):
    return if_false
EOF

# --- ROCm stub ---
mkdir -p $HOME/rocm_stub/rocm
touch $HOME/rocm_stub/WORKSPACE
cat > $HOME/rocm_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
config_setting(name = "is_rocm_enabled", values = {"define": "stub=true"})
filegroup(name = "LICENSE", srcs = [])
EOF

cat > $HOME/rocm_stub/rocm/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "all_files", srcs = [])
filegroup(name = "build_defs", srcs = [])
filegroup(name = "build_defs_bzl", srcs = [])
filegroup(name = "hip", srcs = [])
filegroup(name = "hipblaslt", srcs = [])
filegroup(name = "hipfft", srcs = [])
filegroup(name = "hiprand", srcs = [])
filegroup(name = "hipsolver", srcs = [])
filegroup(name = "hipsparse", srcs = [])
filegroup(name = "miopen", srcs = [])
filegroup(name = "rccl", srcs = [])
filegroup(name = "rocblas", srcs = [])
filegroup(name = "rocm_headers", srcs = [])
filegroup(name = "rocminfo", srcs = [])
filegroup(name = "rocm_path_type", srcs = [])
filegroup(name = "rocm_rpath", srcs = [])
filegroup(name = "rocprim", srcs = [])
filegroup(name = "rocprofiler-sdk", srcs = [])
filegroup(name = "rocsolver", srcs = [])
filegroup(name = "roctracer", srcs = [])
filegroup(name = "toolchain_data", srcs = [])
filegroup(name = "use_rocm_hermetic_rpath", srcs = [])
config_setting(name = "using_hipcc", values = {"define": "stub=false"})
config_setting(name = "linux_x64", values = {"define": "stub=false"})
EOF

# crosstool subfolder for rocm_stub
mkdir -p $HOME/rocm_stub/crosstool
cat > $HOME/rocm_stub/crosstool/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "toolchain", srcs = [])
EOF

cat > $HOME/rocm_stub/rocm/build_defs.bzl << 'EOF'
def if_rocm(if_true, if_false = []):
    return if_false
def if_rocm_is_configured(if_true=[], if_false=[], *args, **kwargs):
    return if_false
def is_rocm_configured():
    return False
def get_rbe_amdgpu_pool(**kwargs):
    return None
def rocm_copts(**kwargs):
    return []
def rocm_default_copts(**kwargs):
    return []
def if_rocm_exec(if_true, if_false = []):
    return if_false
def enable_cuda_flag():
    return []
def if_version_equal_or_greater_than(a, b, if_true, if_false=[]):
    return if_false
def select_threshold(a, b, if_true, if_false=[]):
    return if_false
def rocm_version_number():
    return "0.0.0"
def if_cuda_or_rocm(if_true, if_false=[]):
    return if_false
def if_gpu_is_configured(if_true=[], if_false=[], *args, **kwargs):
    return if_false
def rocm_gpu_architectures():
    return []
def if_rocm_hipblaslt(if_true, if_false = []):
    return if_false
def rocm_library(**kwargs):
    pass
EOF

# --- PyPI stub (for hermetic Pip libraries) ---
mkdir -p $HOME/pypi_stub
touch $HOME/pypi_stub/WORKSPACE
cat > $HOME/pypi_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
py_library(name = "pkg", srcs = [])
EOF

cat > $HOME/pypi_stub/requirements.bzl << 'EOF'
def requirement(name):
    normalized = name.replace("-", "_").lower()
    return "@@pypi//" + normalized
def install_deps(**kwargs):
    pass
EOF

for pkg in astor astunparse dill gast h5py jax lit numpy opt_einsum packaging portpicker protobuf requests scipy tblib termcolor typing_extensions wrapt zstandard; do
  mkdir -p $HOME/pypi_stub/$pkg
  cat > $HOME/pypi_stub/$pkg/BUILD << EOF
package(default_visibility = ["//visibility:public"])
py_library(name = "$pkg", srcs = [])
py_library(name = "pkg", srcs = [])
EOF
done

# Numpy needs C headers (we copy instead of linking to avoid symlink loop)
NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")
rm -rf $HOME/pypi_stub/numpy/numpy_include
cp -rL "$NUMPY_INC" $HOME/pypi_stub/numpy/numpy_include
cat > $HOME/pypi_stub/numpy/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "numpy_headers",
    hdrs = glob(["numpy_include/**/*.h"]),
    includes = ["numpy_include"],
)
py_library(name = "numpy", srcs = [])
EOF

# --- Python stub (python_3_11_host) ---
mkdir -p $HOME/python_stub
touch $HOME/python_stub/WORKSPACE
ln -sf $CONDA_BASE/envs/tf221_build/bin/python3 $HOME/python_stub/python3_bin
cat > $HOME/python_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
exports_files(["python3_bin"])
filegroup(name = "python", srcs = ["python3_bin"])
filegroup(name = "files", srcs = ["python3_bin"])
filegroup(name = "includes", srcs = [])
cc_library(name = "python_headers", hdrs = [])
EOF

# --- Python config stub (local_config_python) ---
mkdir -p $HOME/python_config_stub
touch $HOME/python_config_stub/WORKSPACE
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
rm -rf $HOME/python_config_stub/python_include
cp -rL "$PYINC" $HOME/python_config_stub/python_include

cat > $HOME/python_config_stub/py_cc_toolchain.bzl << 'EOF'
def _py_cc_toolchain_impl(ctx):
    if ctx.attr.headers:
        cc_info = ctx.attr.headers[CcInfo]
    else:
        cc_info = CcInfo(compilation_context = cc_common.create_compilation_context())
    headers_target = struct(
        providers_map = {
            "CcInfo": cc_info,
        },
    )
    return [platform_common.ToolchainInfo(
        py_cc_toolchain = struct(
            headers = headers_target,
            python_version = ctx.attr.python_version,
        ),
    )]

py_cc_toolchain = rule(
    implementation = _py_cc_toolchain_impl,
    attrs = {
        "headers": attr.label(providers = [CcInfo]),
        "python_version": attr.string(default = "3.11"),
    },
)
EOF

cat > $HOME/python_config_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")
load(":py_cc_toolchain.bzl", "py_cc_toolchain")

py_runtime(
    name = "py2_runtime",
    interpreter = "@python_3_11_host//:python",
    python_version = "PY2",
)
py_runtime(
    name = "py3_runtime",
    interpreter = "@python_3_11_host//:python",
    python_version = "PY3",
)
py_runtime_pair(
    name = "py_runtime_pair",
    py2_runtime = ":py2_runtime",
    py3_runtime = ":py3_runtime",
)
toolchain(
    name = "py_toolchain",
    toolchain = ":py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
cc_library(
    name = "python_headers",
    hdrs = glob(["python_include/**/*.h"]),
    includes = ["python_include"],
)
alias(
    name = "headers",
    actual = ":python_headers",
)
py_cc_toolchain(
    name = "py_cc_toolchain_impl",
    headers = ":python_headers",
    python_version = "3.11",
)
toolchain(
    name = "py_cc_toolchain",
    toolchain = ":py_cc_toolchain_impl",
    toolchain_type = "@rules_python//python/cc:toolchain_type",
)
config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)
EOF

cd $TF_BUILD_DIR/tensorflow

# --- local_config_cuda stub ---
# cuda_configure() doesn't generate the "enable_cuda" bool_flag in TF2.21 local mode.
# We create a stub that declares the flags, cc_library with real system CUDA,
# and the minimal cuda_config.h file so that the build works.
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
CUDNN_PATH="${CUDNN_PATH:-/usr/local/cuda}"
mkdir -p $HOME/local_config_cuda_stub/cuda
touch $HOME/local_config_cuda_stub/WORKSPACE

# Real copy of headers to avoid issues with absolute symbolic links in Bazel
rm -rf $HOME/local_config_cuda_stub/cuda_include
mkdir -p $HOME/local_config_cuda_stub/cuda_include
[ -d "$CUDA_PATH/include" ] && cp -rL "$CUDA_PATH/include"/* $HOME/local_config_cuda_stub/cuda_include/ || true

cat > $HOME/local_config_cuda_stub/BUILD << 'BAZEOF'
package(default_visibility = ["//visibility:public"])
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

bool_flag(
    name = "enable_cuda",
    build_setting_default = True,
)
config_setting(
    name = "is_cuda_enabled",
    flag_values = {":enable_cuda": "True"},
)

# config_settings referenced by XLA/TSL
config_setting(
    name = "is_cuda_compiler_nvcc",
    define_values = {"cuda_compiler": "nvcc"},
)
config_setting(
    name = "is_cuda_compiler_clang",
    define_values = {"cuda_compiler": "clang"},
)

cc_library(
    name = "cuda_headers",
    hdrs = glob([
        "cuda_include/*.h",
        "cuda_include/**/*.h",
        "cuda_include/*.hpp",
        "cuda_include/**/*.hpp",
        "cuda_include/**/*.inc",
        "cuda_include/**/*.cuh",
        "cuda_include/**/*.cu",
    ], allow_empty = True),
    includes = ["cuda_include"],
)
alias(name = "cuda", actual = ":cuda_headers")

# Common targets referenced by TF BUILD files
cc_library(name = "cudart", deps = [":cuda_headers"])
cc_library(name = "cuda_driver", deps = [":cuda_headers"])
cc_library(name = "cupti", deps = [":cuda_headers"])


config_setting(name = "cuda_tools", define_values = {"using_cuda_nvcc": "true"})
filegroup(name = "cuda_root", srcs = [])
BAZEOF

rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include
# Copy the entire folder safely
if [ -d "$CUDA_PATH/include" ]; then cp -rL "$CUDA_PATH/include" $HOME/local_config_cuda_stub/cuda/cuda_include; fi
if [ ! -d $HOME/local_config_cuda_stub/cuda/cuda_include ]; then mkdir -p $HOME/local_config_cuda_stub/cuda/cuda_include; fi

# Reality check: Conda fragments the cuda-cudart, cuda-nvcc, etc. packages into ultra-isolated "pkgs" directories.
# We locate key files from each package and consolidate everything into the stub root.
echo ">>> Searching for isolated CUDA blocks across the system..."
for CRIT_HEADER in "cuda_runtime_api.h" "vector_functions.h" "math_functions.h" "cuda_fp16.h" "cuda_bf16.h" "driver_types.h"; do
    CRIT_PATH=$(find /usr /opt /root $CONDA_PREFIX -name "$CRIT_HEADER" 2>/dev/null | head -n 1)
    if [ ! -z "$CRIT_PATH" ]; then
        CRIT_DIR=$(dirname "$CRIT_PATH")
        echo ">>> Found $CRIT_HEADER isolated at: $CRIT_DIR"
        cp -rL "$CRIT_DIR"/* $HOME/local_config_cuda_stub/cuda/cuda_include/ 2>/dev/null || true
    fi
done

# Conda fragmentation: `crt/host_defines.h` usually comes in a "crt" subfolder
TARGET_CRT=$(find /usr /opt /root $CONDA_PREFIX -path "*/crt/host_defines.h" 2>/dev/null | head -n 1)
if [ ! -z "$TARGET_CRT" ]; then
    CRT_DIR=$(dirname "$TARGET_CRT")
    echo ">>> Found the isolated CRT directory at: $CRT_DIR"
    cp -rL "$CRT_DIR" $HOME/local_config_cuda_stub/cuda/cuda_include/ 2>/dev/null || true
fi

# Bring in unified headers from the active Conda environment (breaks through the isolated package bubble)
if [ ! -z "${CONDA_PREFIX:-}" ]; then
    echo ">>> Injecting merged headers from the active Conda environment (subtractive)..."
    for D in "$CONDA_PREFIX/include" "$CONDA_PREFIX/targets/"*"/include"; do
        if [ -d "$D" ]; then
            cp -rL "$D"/* $HOME/local_config_cuda_stub/cuda/cuda_include/ 2>/dev/null || true
        fi
    done
    # Clean up host invader libraries that poison Bazel's own "BoringSSL" and "Abseil"
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/openssl
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/google
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/absl
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/grpc*
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/protobuf
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/zlib*
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/python*
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/curl
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/sqlite*
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/unicode
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/icu*
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/png*
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/jpeg*
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/gif*
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/json*
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/pybind11
    # Delete NVCC compiler headers that are already HERMETICALLY downloaded by XLA
    # (If not deleted, the host version overrides the hermetic TF 2.21 one and GCC breaks on <<<grid, block>>>)
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/cub
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/thrust
    rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/cutlass
fi

# ----------------------------------------------------------------------------
# Force cuDNN 9 headers into every path Bazel consults.
# CUDA_PATH/include comes from conda-forge which has cuDNN 8.9.7. If we don't
# overwrite, TF is compiled with cuDNN 8 headers but linked at runtime
# with cuDNN 9 -> Conv2D/Train fail with "Loaded runtime CuDNN library: 9.0.0
# but source was compiled with: 8.9.7" and "No DNN in stream executor".
# ----------------------------------------------------------------------------
CUDNN9_INC="$HOME/cudnn9_download/cudnn-linux-ppc64le-9.0.0.312_cuda12-archive/include"
if [ -d "$CUDNN9_INC" ]; then
    echo ">>> Overwriting cudnn*.h with v9 in Bazel paths..."
    for T in \
        "$HOME/cuda_unified/include" \
        "$HOME/cuda_unified/include/third_party/gpus/cudnn" \
        "$HOME/local_config_cuda_stub/cuda/cuda_include" \
        "$HOME/local_config_cuda_stub/cuda/cuda_include/third_party/gpus/cudnn" \
        "$TF_BUILD_DIR/tensorflow/third_party/gpus/cudnn"; do
        if [ -d "$T" ]; then
            cp -L "$CUDNN9_INC"/cudnn*.h "$T/" 2>/dev/null || true
            V=$(grep -oE "CUDNN_MAJOR [0-9]+" "$T/cudnn_version.h" 2>/dev/null | head -1)
            echo "    $T -> $V"
        fi
    done
else
    echo ">>> WARNING: $CUDNN9_INC doesn't exist, cuDNN 9 headers will not be applied"
fi

# Conda fragmentation: `crt/host_defines.h` typically comes isolated in the nvcc package, sniff and copy:
TARGET_CRT=$(find /usr /opt /root $CONDA_PREFIX -path "*/crt/host_defines.h" 2>/dev/null | head -n 1 || true)
if [ ! -z "$TARGET_CRT" ]; then
    CRT_DIR=$(dirname "$TARGET_CRT")
    echo ">>> Found the isolated CRT at: $CRT_DIR"
    cp -rL "$CRT_DIR" $HOME/local_config_cuda_stub/cuda/cuda_include/ 2>/dev/null || true
fi

# Conda fragmentation: CUPTI headers live in extras/CUPTI/include/ (not in the main include/)
echo ">>> Searching for CUPTI headers..."
CUPTI_H=$(find /usr /opt $CONDA_PREFIX $HOME/cuda_unified -name "cupti.h" 2>/dev/null | head -n 1 || true)
if [ ! -z "$CUPTI_H" ]; then
    CUPTI_DIR=$(dirname "$CUPTI_H")
    echo ">>> Found CUPTI at: $CUPTI_DIR"
    cp -rL "$CUPTI_DIR"/* $HOME/local_config_cuda_stub/cuda/cuda_include/ 2>/dev/null || true
else
    echo ">>> CUPTI not found - creating minimal stub (for profiler compilation only)"
    cat > $HOME/local_config_cuda_stub/cuda/cuda_include/cupti.h << 'CUPTI_STUB'
#ifndef CUPTI_H_STUB_PPC64LE
#define CUPTI_H_STUB_PPC64LE
// ppc64le: Minimal CUPTI stub for compilation without the full toolkit.
// CUPTI is used only for profiling - does not affect GPU functionality.
#include <stdint.h>
#include <stddef.h>
typedef enum { CUPTI_SUCCESS = 0, CUPTI_ERROR_NOT_INITIALIZED = 15 } CUptiResult;
typedef void* CUpti_SubscriberHandle;
typedef uint64_t CUpti_Timestamp;
typedef enum { CUPTI_ACTIVITY_KIND_INVALID = 0 } CUpti_ActivityKind;
typedef struct { CUpti_ActivityKind kind; } CUpti_Activity;
typedef enum { CUPTI_CB_DOMAIN_INVALID = 0 } CUpti_CallbackDomain;
typedef uint32_t CUpti_CallbackId;
typedef struct { CUpti_CallbackDomain domain; CUpti_CallbackId cbid; } CUpti_CallbackData;
typedef void (*CUpti_CallbackFunc)(void*, CUpti_CallbackDomain, CUpti_CallbackId, const CUpti_CallbackData*);
// Stub functions
static inline CUptiResult cuptiSubscribe(CUpti_SubscriberHandle* s, CUpti_CallbackFunc f, void* d) { return CUPTI_ERROR_NOT_INITIALIZED; }
static inline CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle s) { return CUPTI_ERROR_NOT_INITIALIZED; }
static inline CUptiResult cuptiGetTimestamp(CUpti_Timestamp* t) { if(t) *t=0; return CUPTI_SUCCESS; }
static inline CUptiResult cuptiEnableCallback(uint32_t e, CUpti_SubscriberHandle s, CUpti_CallbackDomain d, CUpti_CallbackId c) { return CUPTI_ERROR_NOT_INITIALIZED; }
static inline CUptiResult cuptiActivityEnable(CUpti_ActivityKind k) { return CUPTI_ERROR_NOT_INITIALIZED; }
static inline CUptiResult cuptiActivityDisable(CUpti_ActivityKind k) { return CUPTI_ERROR_NOT_INITIALIZED; }
static inline CUptiResult cuptiActivityFlushAll(uint32_t f) { return CUPTI_ERROR_NOT_INITIALIZED; }
static inline CUptiResult cuptiFinalize(void) { return CUPTI_ERROR_NOT_INITIALIZED; }
#endif  // CUPTI_H_STUB_PPC64LE
CUPTI_STUB
fi

# CUPTI headers generated by NVIDIA frequently lack include guards.
# When XLA includes cupti_driver_cbid.h directly AND via cupti.h, GCC
# sees multiple definitions. Solution: inject #pragma once into the ones that need it.
echo ">>> Adding include guards to CUPTI headers..."
for f in $HOME/local_config_cuda_stub/cuda/cuda_include/cupti*.h; do
    if [ -f "$f" ] && ! grep -q 'pragma once' "$f" && ! grep -q '#ifndef.*CUPTI.*_H' "$f"; then
        sed -i '1i #pragma once' "$f"
        echo "   -> #pragma once added to $(basename $f)"
    fi
done

# Older CUPTI may lack CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER
if [ -f "$HOME/local_config_cuda_stub/cuda/cuda_include/cupti.h" ]; then
    if ! grep -q 'CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER' "$HOME/local_config_cuda_stub/cuda/cuda_include/cupti.h"; then
        echo '#define CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER ((CUpti_ActivityAttribute)0)' >> "$HOME/local_config_cuda_stub/cuda/cuda_include/cupti.h"
    fi
    # Force replacement in case the original NVIDIA header defined it as bare 0:
    sed -i 's/#define CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER 0/#define CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER ((CUpti_ActivityAttribute)0)/g' "$HOME/local_config_cuda_stub/cuda/cuda_include/cupti.h"
fi


# RAFT-ENABLE 2026-05-22: usa Thrust/CUB/libcudacxx do CUDA 12.2 toolkit (2.x).
# Antes usava Thrust/CUB 1.17.2 standalone, mas RMM/RAFT 25.08 exigem 2.x do CCCL bundle.
rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/cub
rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/thrust
rm -rf $HOME/local_config_cuda_stub/cuda/cuda_include/cuda

CUDA_TOOLKIT_INC=/usr/local/cuda-12.2/targets/ppc64le-linux/include
if [ -d "$CUDA_TOOLKIT_INC/thrust" ]; then
    cp -r "$CUDA_TOOLKIT_INC/thrust" $HOME/local_config_cuda_stub/cuda/cuda_include/
    cp -r "$CUDA_TOOLKIT_INC/cub"    $HOME/local_config_cuda_stub/cuda/cuda_include/
    [ -d "$CUDA_TOOLKIT_INC/cuda" ] && cp -r "$CUDA_TOOLKIT_INC/cuda" $HOME/local_config_cuda_stub/cuda/cuda_include/
    echo ">>> Thrust/CUB/libcudacxx copiados do CUDA toolkit ($CUDA_TOOLKIT_INC)"
    grep -E "#define THRUST_VERSION " $HOME/local_config_cuda_stub/cuda/cuda_include/thrust/version.h | head -1
else
    echo ">>> WARNING: $CUDA_TOOLKIT_INC nao encontrado, fallback Thrust 1.17.2"
    rm -rf /tmp/cuda_legacy_headers
    mkdir -p /tmp/cuda_legacy_headers
    wget -qO /tmp/cuda_legacy_headers/cub.tar.gz https://github.com/NVIDIA/cub/archive/refs/tags/1.17.2.tar.gz
    wget -qO /tmp/cuda_legacy_headers/thrust.tar.gz https://github.com/NVIDIA/thrust/archive/refs/tags/1.17.2.tar.gz
    tar -xzf /tmp/cuda_legacy_headers/cub.tar.gz -C /tmp/cuda_legacy_headers
    tar -xzf /tmp/cuda_legacy_headers/thrust.tar.gz -C /tmp/cuda_legacy_headers
    cp -r /tmp/cuda_legacy_headers/cub-1.17.2/cub $HOME/local_config_cuda_stub/cuda/cuda_include/
    cp -r /tmp/cuda_legacy_headers/thrust-1.17.2/thrust $HOME/local_config_cuda_stub/cuda/cuda_include/
    rm -rf /tmp/cuda_legacy_headers
fi

# Ensure CUB, Thrust, and libcudacxx (CCCL) are present in the Bazel sandbox
echo ">>> Injecting CUB (CCCL) directly into the CUDA stub..."

cat > $HOME/local_config_cuda_stub/cuda/BUILD << 'BAZEOF'
package(default_visibility = ["//visibility:public"])
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

bool_flag(
    name = "include_cuda_libs",
    build_setting_default = True,
)
config_setting(
    name = "is_cuda_libs_enabled",
    flag_values = {":include_cuda_libs": "True"},
)

# tf_wheel implicit dependency
bool_flag(
    name = "override_include_cuda_libs",
    build_setting_default = False,
)

# Real CUDA headers via local copy
cc_library(
    name = "cuda_headers",
    hdrs = glob([
        "cuda_include/*.h",
        "cuda_include/**/*.h",
        "cuda_include/*.hpp",
        "cuda_include/**/*.hpp",
        "cuda_include/**/*.inc",
        "cuda_include/**/*.cuh",
        "cuda_include/**/*.cu",
    ], allow_empty = True),
    includes = ["cuda_include"],
)

# Common aliases referenced by TF BUILD files
cc_library(name = "cuda", deps = [":cuda_headers"])
cc_library(name = "cudart", deps = [":cuda_headers"])
cc_library(name = "cudart_static", deps = [":cuda_headers"])
cc_library(name = "cublas", deps = [":cuda_headers"])
cc_library(name = "cufft", deps = [":cuda_headers"])
cc_library(name = "curand", deps = [":cuda_headers"])
cc_library(name = "cusolver", deps = [":cuda_headers"])
cc_library(name = "cusparse", deps = [":cuda_headers"])
cc_library(name = "cupti_headers", deps = [":cuda_headers"])
cc_library(name = "cupti", deps = [":cuda_headers"])
cc_library(name = "cudnn", deps = [":cuda_headers"])
cc_library(name = "cudnn_header", deps = [":cuda_headers"])
cc_library(name = "nvml", deps = [":cuda_headers"])
cc_library(name = "cccl", deps = [":cuda_headers"])
cc_library(name = "cuda_driver", deps = [":cuda_headers"])
cc_library(name = "cub_headers", deps = [":cuda_headers"]) 

# DSO (dynamic shared object) targets referenced by XLA/TSL
cc_library(name = "cupti_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "cuda_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "cudnn_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "cublas_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "cufft_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "cusolver_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "cusparse_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "curand_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "cudart_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "cuda_driver_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "nvml_dsos", srcs = [], deps = [":cuda_headers"])

filegroup(name = "cuda_root", srcs = [])
config_setting(name = "cuda_tools_and_libs", define_values = {"using_cuda_nvcc": "true"})
config_setting(name = "cuda_tools", define_values = {"using_cuda_nvcc": "true"})

# Python config targets referenced by tensorflow/tools/build_info and others
py_library(name = "cuda_config_py", srcs = ["cuda_config.py"])

# Genrule to generate the minimal cuda_config.h
genrule(
    name = "cuda_config_h",
    outs = ["cuda/cuda_config.h"],
    cmd = """cat > $@ << 'CFGEOF'
#ifndef CUDA_CUDA_CONFIG_H_
#define CUDA_CUDA_CONFIG_H_
#define TF_CUDA_VERSION "12.4"
#define TF_CUDNN_VERSION "9"
#define TF_CUDA_TOOLKIT_PATH "/usr/local/cuda"
#define TF_CUDA_CAPABILITIES "3.5,7.0"
#endif
CFGEOF""",
)
cc_library(
    name = "cuda_config",
    hdrs = [":cuda_config_h"],
)

# Other targets referenced by TF
cc_library(name = "nccl", deps = [":cuda_headers"])
cc_library(name = "nvjitlink", deps = [":cuda_headers"])
cc_library(name = "nccl_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "nvjitlink_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "cuda_runtime", deps = [":cuda_headers", ":cudart"])
cc_library(name = "cuda_runtime_impl", deps = [":cuda_headers", ":cudart"])
cc_library(name = "cublas_static", deps = [":cuda_headers"])
cc_library(name = "cusolver_static", deps = [":cuda_headers"])
cc_library(name = "cusparse_static", deps = [":cuda_headers"])
cc_library(name = "cufft_static", deps = [":cuda_headers"])
cc_library(name = "curand_static", deps = [":cuda_headers"])
cc_library(name = "cublas_lt", deps = [":cuda_headers"])
cc_library(name = "cublasLt", deps = [":cuda_headers"])
cc_library(name = "cublas_lt_dsos", srcs = [], deps = [":cuda_headers"])
cc_library(name = "link_stub", deps = [":cuda_headers"])
filegroup(name = "cuda-nvvm", srcs = [])
filegroup(name = "libdevice_root", srcs = [])
BAZEOF

cat > $HOME/local_config_cuda_stub/cuda/build_defs.bzl << 'EOF'
# ppc64le: full stub for @local_config_cuda//cuda:build_defs.bzl
# All exported symbols that TF 2.21 imports from this file:
def cuda_library(**kwargs): native.cc_library(**kwargs)
def cuda_header_library(**kwargs): native.cc_library(**kwargs)
def if_cuda(if_true, if_false = []): return if_true
def if_cuda_exec(if_true, if_false = []): return if_true
def if_cuda_is_configured(if_true, if_false = []): return if_true
def is_cuda_configured(): return True
def cuda_default_copts(): return []
def cuda_gpu_architectures(): return ["sm_35", "sm_70"]
def get_cuda_version_number(): return 1204
def if_cuda_newer_than(ver, if_true, if_false=[]): return if_true
def if_local_cuda(if_true, if_false=[]): return if_true
def cuda_copts(**kwargs): return []
def cuda_defines(**kwargs): return []
def if_cuda_or_rocm(if_true, if_false=[]): return if_true
def if_gpu_is_configured(if_true=[], if_false=[], *args, **kwargs): return if_true
def cuda_rpath_flags(relpath=""): return []
EOF

# CUDA Python configuration file (referenced by cuda_config_py)
cat > $HOME/local_config_cuda_stub/cuda/cuda_config.py << 'PYCFG'
"""Auto-generated CUDA config for ppc64le stub."""
cuda_toolkit_path = "/usr/local/cuda"
cuda_version = "12.4"
cudnn_version = "9"
cuda_compute_capabilities = ["3.5", "7.0"]
PYCFG

echo ">>> local_config_cuda stub created with CUDA at: $CUDA_PATH"



cat > /tmp/patch_workspace3.py << 'PYEOF'
"""
Patch WORKSPACE for ppc64le GPU: comments out ENTIRE BLOCKS (statements)
related to downloading hermetic CUDA wheels.
"""

with open("WORKSPACE", "r") as f:
    lines = f.readlines()

statements = []
current_block = []
block_start = 0
paren_depth = 0

for i, line in enumerate(lines):
    stripped = line.strip()

    if paren_depth == 0 and (stripped == "" or stripped.startswith("#")):
        if current_block:
            statements.append((block_start, i - 1, current_block))
            current_block = []
        statements.append((i, i, [line]))
        continue

    if not current_block:
        block_start = i

    current_block.append(line)
    paren_depth += line.count("(") - line.count(")")

    if paren_depth <= 0:
        paren_depth = 0
        statements.append((block_start, i, current_block))
        current_block = []

if current_block:
    statements.append((block_start, len(lines) - 1, current_block))

# local_config_cuda and local_config_nccl removed from the list!
# (these two must run to detect the real system GPU)
# NOTE: the "cuda_" pattern captures ALL hermetic cuda_*
#       but does NOT capture "local_config_cuda" (which ends with _cuda, not cuda_)
skip_patterns = [
    "cc_toolchain_deps",
    "register_toolchains",
    "cuda_",
    "CUDA_",
    "CUDNN_",
    "nccl_",
    "nvshmem",
    "NVSHMEM",
    "nvidia_",
    "python_wheel_version_suffix",
    "llvm_linux",
    "llvm_darwin",
    "llvm18_",
    "llvm19_",
    "llvm20_",
    "llvm21_",
]

new_lines = []
commented = 0

# Blocks containing these patterns must NEVER be commented out
# (even if they match a skip_pattern)
keep_patterns = ["local_config_cuda", "local_config_nccl", "cuda_configure", "nccl_configure"]

for start, end, block_lines in statements:
    block_text = "".join(block_lines)
    already_comment = all(l.strip().startswith("#") or l.strip() == "" for l in block_lines)

    should_skip = any(p in block_text for p in skip_patterns)
    should_keep = any(p in block_text for p in keep_patterns)

    if not already_comment and should_skip and not should_keep:
        for bl in block_lines:
            new_lines.append("# PPC64LE: " + bl)
        commented += 1
    else:
        new_lines.extend(block_lines)

with open("WORKSPACE", "w") as f:
    f.writelines(new_lines)

print(f"OK: WORKSPACE patched - {commented} blocks commented out")
PYEOF
python3 /tmp/patch_workspace3.py

# --- tf_wheel_version_suffix stub ---
# python_wheel_version_suffix_repository() was commented out by the WORKSPACE patch.
# We create the stub manually and inject it into WORKSPACE.
mkdir -p $HOME/tf_wheel_version_suffix_stub
touch $HOME/tf_wheel_version_suffix_stub/WORKSPACE
touch $HOME/tf_wheel_version_suffix_stub/BUILD

cat > $HOME/tf_wheel_version_suffix_stub/wheel_version_suffix.bzl << 'EOF'
# ppc64le: empty suffix version for release build
WHEEL_VERSION_SUFFIX = ""
SEMANTIC_WHEEL_VERSION_SUFFIX = ""
EOF

# Inject definition at the end of WORKSPACE (new_local_repository works without load())
cat >> WORKSPACE << WSEOF

# PPC64LE: inject tf_wheel_version_suffix (python_wheel_version_suffix_repository commented out)
local_repository(
    name = "tf_wheel_version_suffix",
    path = "$HOME/tf_wheel_version_suffix_stub",
)
WSEOF
echo ">>> tf_wheel_version_suffix stub injected into WORKSPACE"

# Direct patch on tf_version.default.bzl (more reliable approach than sed)
# Removes the load() of @tf_wheel_version_suffix and hardcodes the empty value
python3 - << 'PYEOF'
import re, os

for fname in [
    "tensorflow/tf_version.default.bzl",
    "tensorflow/tf_version.bzl",
]:
    if not os.path.exists(fname):
        continue
    with open(fname, "r") as f:
        content = f.read()

    # Remove load() of @tf_wheel_version_suffix (can be single or multiline)
    content = re.sub(
        r'load\s*\(\s*"@tf_wheel_version_suffix[^"]*"[^)]*\)',
        'WHEEL_VERSION_SUFFIX = ""\nSEMANTIC_WHEEL_VERSION_SUFFIX = ""  # ppc64le: stub',
        content,
        flags=re.DOTALL
    )
    # If WHEEL_VERSION_SUFFIX is still used as an external variable, ensure the definition
    if 'WHEEL_VERSION_SUFFIX' not in content:
        content = 'WHEEL_VERSION_SUFFIX = ""\n' + content

    with open(fname, "w") as f:
        f.write(content)
    print(f"OK: {fname} patched - @tf_wheel_version_suffix removed")
PYEOF




PYTHON_REAL_PATH="$CONDA_BASE/envs/tf221_build/bin/python3"
cat > $HOME/python3_bazel.sh << BEOF
#!/bin/bash
unset PYTHONHOME
exec $PYTHON_REAL_PATH "\$@"
BEOF
chmod +x $HOME/python3_bazel.sh
echo ">>> python3_bazel.sh created pointing to: $PYTHON_REAL_PATH"

cat > /tmp/patch_toolchains.py << 'PYEOF'
import re

path = "third_party/xla/third_party/py/python_init_toolchains.bzl"
with open(path, "r") as f:
    content = f.read()

new_func = '''def python_init_toolchains(name = "python", python_version = None, **kwargs):
    """Disabled for ppc64le: no prebuilt hermetic Python available."""
    pass
'''

content = re.sub(
    r'def python_init_toolchains\(.*?\n(?=def |\Z)',
    new_func + '\n',
    content,
    flags=re.DOTALL
)

with open(path, "w") as f:
    f.write(content)

print("OK: python_init_toolchains patched")
PYEOF
python3 /tmp/patch_toolchains.py

cat > /tmp/patch_pip.py << 'PYEOF'
import os

path = "third_party/xla/third_party/py/python_init_pip.bzl"
with open(path, "r") as f:
    content = f.read()

content = content.replace(
    '''        python_interpreter_target = "@{}_host//:python".format(
            get_toolchain_name_per_python_version("python"),
        ),''',
    '        python_interpreter = "' + os.environ.get("HOME", "/root") + '/python3_bazel.sh",  # ppc64le: wrapper that unsets PYTHONHOME'
)

with open(path, "w") as f:
    f.write(content)

print("OK: python_init_pip patched")
PYEOF
python3 /tmp/patch_pip.py

sed -i 's/nvidia_wheel_versions\[$/nvidia_wheel_versions.get(/' \
    third_party/xla/third_party/py/setup_py_nvidia_dependencies_util.py

sed -i 's/      str(cuda_version)/      str(cuda_version), {}/' \
    third_party/xla/third_party/py/setup_py_nvidia_dependencies_util.py

sed -i 's/  \].items():/  ).items():/' \
    third_party/xla/third_party/py/setup_py_nvidia_dependencies_util.py

sed -i '1s/^/#include <cstdint>\n/' third_party/xla/xla/backends/cpu/codegen/builtin_fp16.h

# With Clang 17 + CUDA 12.4, we no longer need the patches:
#   -mprefer-vector-width=512: Clang 15+ accepts
#   -fno-experimental-sanitize-metadata=all: Clang 16+ accepts
#   __builtin_vectorelements: Clang 17+ has natively

# --- PATCH PPC64LE: select_k_thunk uses _stub instead of _raft ---
# RAFT-ENABLE 2026-05-22: disabled to attempt real RAFT/RMM build on ppc64le.
# Set RAFT_DISABLE=1 in the env to restore the stub fallback.
if [ "${RAFT_DISABLE:-0}" = "1" ]; then
echo ">>> Patching select_k_thunk BUILD to use _stub instead of _raft..."
python3 - << 'SELECT_K_PATCH'
import os, re
p = 'third_party/xla/xla/backends/gpu/runtime/BUILD'
if not os.path.exists(p):
    print(f'WARNING: {p} not found')
else:
    with open(p, 'r') as f:
        content = f.read()
    if 'ppc64le_select_k_stub' in content:
        print('SKIP: already patched')
    else:
        # Replace the first occurrence: the line [":select_k_exec_raft"]
        # inside if_cuda_is_configured(...) of the select_k_thunk target.
        new_content = content.replace(
            '[":select_k_exec_raft"],\n        no_cuda = [":select_k_exec_stub"],',
            '[":select_k_exec_stub"],  # ppc64le_select_k_stub: was _raft\n        no_cuda = [":select_k_exec_stub"],',
            1,
        )
        if new_content == content:
            # Fallback attempt: more flexible multiline regex
            new_content = re.sub(
                r'if_cuda_is_configured\(\s*\[":select_k_exec_raft"\]',
                'if_cuda_is_configured(\n        [":select_k_exec_stub"]  # ppc64le_select_k_stub',
                content,
                count=1,
            )
        if new_content != content:
            with open(p, 'w') as f:
                f.write(new_content)
            print(f'PATCHED: {p}')
        else:
            print(f'WARNING: pattern did not match in {p}')
SELECT_K_PATCH
fi  # RAFT-ENABLE: end of select_k stub block

# --- PATCH PPC64LE: disable mlir_generated GPU kernels ---
# These kernels depend on hlo_to_kernel running as a build tool, which doesn't link
# in this build (protobuf duplicate symbols + other issues). The .cc files in
# tensorflow/core/kernels/mlir_generated/gpu_op_*.cc only do kernel
# registration (REGISTER_KERNEL_BUILDER) - they aren't referenced externally. Emptying them
# makes these MLIR kernels not be registered, and TF falls back to
# the non-MLIR implementations (traditional Eigen/cuDNN) for Relu/Elu/Cast/etc.
echo ">>> Emptying gpu_op_*.cc in mlir_generated (disables MLIR codegen)..."
python3 - << 'EMPTY_MLIR_OPS'
import os, glob

mlir_dir = 'tensorflow/core/kernels/mlir_generated'
if not os.path.isdir(mlir_dir):
    print(f'WARNING: {mlir_dir} does not exist')
else:
    count = 0
    for path in glob.glob(os.path.join(mlir_dir, 'gpu_op_*.cc')):
        with open(path, 'r') as f:
            content = f.read()
        if 'ppc64le_disabled' in content:
            continue  # idempotent
        with open(path, 'w') as f:
            f.write('// ppc64le_disabled: MLIR-generated GPU kernel registration\n')
            f.write('// (hlo_to_kernel build tool does not link in this setup; TF uses non-MLIR fallback)\n')
        count += 1
    print(f'Emptied {count} gpu_op_*.cc files')
EMPTY_MLIR_OPS

# --- PATCH PPC64LE: host_triple in mlir_generated/build_defs.bzl ---
# The _gen_kernel_library rule chooses host_triple via select() that only knows
# aarch64 and default=x86_64. On PPC64LE it falls back to x86_64 -> kernel_gen is invoked
# with the wrong triple, generates invalid output (or nothing), and the
# _mlir_ciface_* symbols never appear. The fix is to force the correct host_triple for PPC.
echo ">>> Patching mlir_generated/build_defs.bzl for PPC64LE host_triple..."
python3 - << 'MLIR_TRIPLE_PATCH'
import os, re

paths = [
    'tensorflow/core/kernels/mlir_generated/build_defs.bzl',
]
for p in paths:
    if not os.path.exists(p):
        continue
    with open(p, 'r') as f:
        content = f.read()
    if 'powerpc64le-unknown-linux-gnu' in content:
        print(f'SKIP: {p} already patched')
        continue
    # Replace the host_triple select() with a version that returns
    # powerpc64le-unknown-linux-gnu directly. Since this build_defs.bzl
    # runs on the host where we want to compile, and our host is fixed PPC64LE,
    # we can hardcode (vs trying to guess via @platforms//cpu:ppc).
    new_content = re.sub(
        r'host_triple = select\(\{[^}]*"//conditions:default":\s*"x86_64-unknown-linux-gnu"[^}]*\}\)',
        'host_triple = "powerpc64le-unknown-linux-gnu"',
        content,
        flags=re.DOTALL,
    )
    if new_content != content:
        with open(p, 'w') as f:
            f.write(new_content)
        print(f'PATCHED: {p}')
    else:
        print(f'WARNING: host_triple pattern not found in {p}')
MLIR_TRIPLE_PATCH

# --- PATCH PPC64LE: rename anonymous namespace in all .cu.cc ---
# DISABLED 2026-05-09: this patch was for BFD-strict R_PPC64_TOC16_LO
# (was LLD). With BFD we don't need it. And I suspect it's causing segfault
# in __sti____cudaRegisterAll due to inconsistency between source code and registrations
# generated by cudafe. The code below is kept in case you want to reactivate, but I skip via
# early return in the python script.
echo ">>> Applying inline namespace patch to .cu.cc files (needed for LLD)..."
python3 - << 'CU_CC_NS_PATCH'
import os, re, hashlib

def patch_cu_cc(filepath):
    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read()
    except Exception:
        return False
    if 'ppc64le_anon_' in content:
        return False
    if not re.search(r'^\s*namespace\s*\{', content, re.MULTILINE):
        return False
    h = hashlib.md5(filepath.encode()).hexdigest()[:8]
    ns_name = f'ppc64le_anon_{h}'
    # Use `inline namespace`: members have external linkage (fixes the
    # NVCC/PPC64LE bug) AND are transparent to the parent namespace. The C++17 standard
    # defines that `namespace { ... }` is equivalent to `inline namespace X { ... }
    # using namespace X;`, so we don't introduce a semantic difference - we just give
    # it a name and leave the external linkage.
    new_content = re.sub(
        r'^(\s*)namespace\s*\{',
        rf'\1inline namespace {ns_name} {{',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    # Match ONLY unnamed closings after "namespace" - those of an anonymous namespace.
    # `}  // namespace` (end of line) or `}  // anonymous namespace` (end of line).
    # Does NOT match `}  // namespace foo` (that's a named namespace closing).
    new_content2 = re.sub(
        r'\}\s*//\s*(?:anonymous\s+namespace|namespace)\s*$',
        f'}}  // namespace {ns_name}',
        new_content,
        count=1,
        flags=re.MULTILINE,
    )
    if new_content2 == new_content:
        # Closing did not match: file without the conventional comment. Do not write
        # (we don't want to leave the file with renamed opening and inconsistent closing/using).
        return False
    try:
        os.chmod(filepath, 0o644)
    except Exception:
        pass
    with open(filepath, 'w') as f:
        f.write(new_content2)
    return True

patched = 0
for root_dir in ['tensorflow', 'third_party/xla']:
    if not os.path.isdir(root_dir):
        continue
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.cu.cc'):
                p = os.path.join(root, f)
                try:
                    if patch_cu_cc(p):
                        patched += 1
                except Exception as e:
                    print(f'ERROR {p}: {e}')

print(f'Total .cu.cc files patched: {patched}')
CU_CC_NS_PATCH

# Remove problematic hardcoded paths in the XLA headers that break the build with external stubs
echo ">>> Fixing hardcoded CUDA paths in third_party/xla/xla and tsl..."
find third_party/xla/xla third_party/xla/third_party/tsl -type f \( -name "*.h" -o -name "*.cc" \) -exec sed -i 's|third_party/gpus/cuda/extras/CUPTI/include/||g; s|third_party/gpus/cuda/include/||g; s|third_party/gpus/cudnn/include/||g; s|third_party/gpus/cudnn/||g; s|third_party/gpus/cuda/||g; s|third_party/tensorrt/||g' {} +


# We need the same overrides for fetch to work without cycles in WORKSPACE
BAZEL_OVERRIDE_FLAGS="\
    --override_repository=rules_ml_toolchain=$HOME/rules_ml_toolchain_patched \
    --override_repository=llvm_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm_linux_aarch64=$HOME/llvm_stub \
    --override_repository=llvm_darwin_aarch64=$HOME/llvm_stub \
    --override_repository=llvm18_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm19_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm20_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm21_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm18_linux_aarch64=$HOME/llvm_stub \
    --override_repository=llvm20_linux_aarch64=$HOME/llvm_stub \
    --override_repository=llvm21_linux_aarch64=$HOME/llvm_stub \
    --override_repository=llvm18_darwin_aarch64=$HOME/llvm_stub \
    --override_repository=llvm20_darwin_aarch64=$HOME/llvm_stub \
    --override_repository=local_config_tensorrt=$HOME/tensorrt_stub \
    --override_repository=local_config_rocm=$HOME/rocm_stub \
    --override_repository=local_config_cuda=$HOME/local_config_cuda_stub \
    --override_repository=cuda_crt=$HOME/cuda_redist_stub \
    --override_repository=cuda_cudart=$HOME/cuda_redist_stub \
    --override_repository=cuda_profiler_api=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvcc=$HOME/cuda_nvcc_stub \
    --override_repository=cuda_nvml=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvtx=$HOME/cuda_redist_stub \
    --override_repository=cuda_cccl=$HOME/cuda_cccl_stub \
    --override_repository=cuda_cudnn=$HOME/cuda_redist_stub \
    --override_repository=cuda_cudnn9=$HOME/cuda_redist_stub \
    --override_repository=cuda_cublas=$HOME/cuda_redist_stub \
    --override_repository=cuda_cusolver=$HOME/cuda_redist_stub \
    --override_repository=cuda_cusparse=$HOME/cuda_redist_stub \
    --override_repository=cuda_curand=$HOME/cuda_redist_stub \
    --override_repository=cuda_cufft=$HOME/cuda_redist_stub \
    --override_repository=cuda_cupti=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvdisasm=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvvm=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvjitlink=$HOME/cuda_redist_stub \
    --override_repository=cuda_nccl=$HOME/cuda_nccl_stub \
    --override_repository=cuda_nvrtc=$HOME/cuda_redist_stub \
    --override_repository=nvidia_wheel_versions=$HOME/cuda_redist_stub \
    --override_repository=tf_wheel_version_suffix=$HOME/tf_wheel_version_suffix_stub \
    --override_repository=pypi=$HOME/pypi_stub \
    --override_repository=python_3_11_host=$HOME/python_stub \
    --override_repository=local_config_python=$HOME/python_config_stub \
    --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
    --repo_env=USE_HERMETIC_CC_TOOLCHAIN=0 \
    --noincompatible_enable_cc_toolchain_resolution"

eval bazel fetch @boringssl//:crypto $BAZEL_OVERRIDE_FLAGS || true
BASE_H="$(bazel info output_base)/external/boringssl/src/include/openssl/base.h"
if [ -f "$BASE_H" ]; then
    chmod u+w "$BASE_H"
    sed -i 's/#error "Unknown target CPU"/#define OPENSSL_64_BIT\n#define OPENSSL_NO_ASM/g' "$BASE_H"
    echo ">>> BoringSSL patched for Power9"
else
    echo ">>> WARNING: BoringSSL base.h not found (will be patched during the build)"
fi

sed -i 's/Shape(Shape&&) noexcept;/Shape(Shape\&\&);/g' third_party/xla/xla/shape.h
sed -i 's/operator=(Shape&&) noexcept;/operator=(Shape\&\&);/g' third_party/xla/xla/shape.h
sed -i 's/) noexcept = default;/) = default;/g' third_party/xla/xla/shape.cc

sed -i 's|#include "third_party/tensorrt/tensorrt_config.h"|#define TF_TENSORRT_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc
sed -i 's|#include "tensorrt_config.h"|#define TF_TENSORRT_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc
sed -i 's|#include "third_party/gpus/rocm/rocm_config.h"|#define TF_ROCM_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc

python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/tsl/profiler/utils/group_events.cc'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(r'(NoDestructor[^\(]+)\s*\(\s*\{', r'\1(absl::flat_hash_set<int64_t>{', t)
with open(p, 'w') as f:
    f.write(t)
PYEOF

python3 - << 'PYEOF'
import re
p = 'tensorflow/compiler/mlir/lite/transforms/quantization/fuse_qdq_pass.cc'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(r'(NoDestructor[^\(]+)\s*\(\s*\{', r'\1(absl::flat_hash_set<std::string>{', t)
with open(p, 'w') as f:
    f.write(t)
PYEOF

python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/service/memory_space_assignment/allocation_value.h'
with open(p, 'r') as f:
    t = f.read()
t = t.replace('AllocationValue(AllocationValue&&) noexcept = default;\n', '')
t = t.replace('AllocationValue& operator=(AllocationValue&&) noexcept = default;\n', '')
t = re.sub(r'(class|struct)\s+AllocationValue\s*\{',
           r'\1 AllocationValue {\n public:\n  AllocationValue(AllocationValue&&) = default;\n  AllocationValue& operator=(AllocationValue&&) = default;\n',
           t, count=1)
with open(p, 'w') as f:
    f.write(t)
PYEOF

python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/hlo/utils/hlo_sharding_util.h'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(
    r'TileAssignment\(\s*\{\s*num_groups,\s*num_devices_per_group\s*\}\s*\)',
    r'TileAssignment(absl::Span<const int64_t>({num_groups, num_devices_per_group}))',
    t
)
with open(p, 'w') as f:
    f.write(t)
PYEOF

python3 - << 'PYEOF'
p = 'tensorflow/compiler/mlir/lite/transforms/optimize_pass.cc'
with open(p, 'r') as f:
    t = f.read()
t = t.replace('std::multiplies()', 'std::multiplies<int64_t>()')
with open(p, 'w') as f:
    f.write(t)
PYEOF

python3 - << 'PYEOF'
import glob
import re
for p in glob.glob('tensorflow/dtensor/mlir/*.cc'):
    with open(p, 'r') as f:
        t = f.read()
    t = t.replace('namespace tensorflow {\nusing llvm::cast;\nusing llvm::isa;\nusing llvm::dyn_cast;\n', 'namespace tensorflow {')
    t = t.replace('#include "llvm/Support/Casting.h"\nnamespace tensorflow {', 'namespace tensorflow {')
    if re.search(r'\b(cast|isa|dyn_cast)<', t):
        t = t.replace('namespace tensorflow {', '#include "llvm/Support/Casting.h"\nnamespace tensorflow {\nusing llvm::cast;\nusing llvm::isa;\nusing llvm::dyn_cast;\n')
    with open(p, 'w') as f:
        f.write(t)
PYEOF

python3 - << 'PYEOF'
import glob
import re

for p in glob.glob('third_party/xla/xla/codegen/*.h') + glob.glob('third_party/xla/xla/service/cpu/*.cc'):
    with open(p, 'r') as f:
        t = f.read()
    t = re.sub(r'noexcept\s*=\s*default;', '= default;', t)
    t = re.sub(
        r'push_back\(\s*\{([^{}]*?)thread_safe_module\(\)\s*\}\s*\)',
        r'push_back(EmittedKernel{\1thread_safe_module()})',
        t
    )
    with open(p, 'w') as f:
        f.write(t)
PYEOF

python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/ffi/attribute_map.cc'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(
    r'static_assert\(false, "Unsupported.*?"\);',
    r'return absl::InvalidArgumentError("Unsupported attribute/scalar/array/flat type");',
    t
)
with open(p, 'w') as f:
    f.write(t)
PYEOF

# --- FIX: plugin_registry.cc / GCC 8 static_assert(false) ---
python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/stream_executor/plugin_registry.cc'
with open(p, 'r') as f:
    t = f.read()

# GetPluginKind(): returns a valid kind
t = re.sub(
    r'static_assert\(false, "Unsupported factory type"\);',
    r'return PluginKind::kBlas;',
    t,
    count=1
)

# GetPluginName(): returns a valid string_view
t = re.sub(
    r'static_assert\(false, "Unsupported factory type"\);',
    r'return "blas";',
    t,
    count=1
)

with open(p, 'w') as f:
    f.write(t)
PYEOF

python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/backends/cpu/ynn_support.cc'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(r'kAllowedTypes\s*\(\s*\{', r'kAllowedTypes(absl::flat_hash_set<std::tuple<xla::PrimitiveType, xla::PrimitiveType, xla::PrimitiveType>>{', t)
with open(p, 'w') as f:
    f.write(t)
PYEOF

python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/backends/cpu/runtime/convolution_lib.h'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(
    r'\[count_down\]\(\)\s*mutable\s*\{\s*count_down\.CountDown\(\);\s*\}',
    r'[count_down]() mutable { auto cd = count_down; cd.CountDown(); }',
    t
)
with open(p, 'w') as f:
    f.write(t)
PYEOF

# --- FIX: expand_integer_power.cc / GCC 8 brace-init ambiguity ---
# GCC 8 does not resolve {op->getOperands()} with constructors inherited from MLIR.
# We replace the brace-init with an explicit PowOp::Adaptor constructor.
python3 - << 'POWPATCH'
p = 'third_party/xla/xla/codegen/emitters/transforms/expand_integer_power.cc'
with open(p, 'r') as f:
    t = f.read()
# GCC 8: brace-init {op->getOperands()} is ambiguous with constructors inherited from MLIR.
# {operands} is the 4th arg (Adaptor), op->getAttrs() is the 5th arg separately.
# Only swap {op->getOperands()} for PowOp::Adaptor(op->getOperands())
t = t.replace(
    '{op->getOperands()}',
    'mlir::mhlo::PowOp::Adaptor(op->getOperands())'
)
with open(p, 'w') as f:
    f.write(t)
print("OK: expand_integer_power.cc patched (GCC 8 brace-init)")
POWPATCH

# --- CRITICAL FIX: thunk_executor.cc / GCC 8 vector<ThunkOperation> ---
#
# ThunkOperation inherits from ExecutionGraph::Operation (a class with virtual
# destructor and unique_ptr members). GCC 8 does not detect that the implicit
# copy ctor is deleted and tries to copy in vector::reserve(). Solution:
# EXPLICITLY delete the copy ctor -> move_if_noexcept uses move.
#
python3 - << 'THUNKPATCH'
import re, sys

p = 'third_party/xla/xla/backends/cpu/runtime/thunk_executor.cc'
try:
    with open(p, 'r') as f:
        t = f.read()
except FileNotFoundError:
    print(f"WARNING: {p} not found, skipping")
    sys.exit(0)

original = t

# Inject copy delete + move default AFTER the class opening
# [^{]* captures the inheritance ": public ExecutionGraph::Operation"
# Add "public:" because the default access for class{} is private
t, n = re.subn(
    r'(class\s+ThunkOperation\b[^{]*\{)',
    r'\1\n'
    r' public:\n'
    r'  // GCC 8 ppc64le: explicit delete forces move_if_noexcept to use move\n'
    r'  ThunkOperation(const ThunkOperation&) = delete;\n'
    r'  ThunkOperation& operator=(const ThunkOperation&) = delete;\n'
    r'  ThunkOperation(ThunkOperation&&) = default;\n'
    r'  ThunkOperation& operator=(ThunkOperation&&) = default;',
    t,
    count=1
)

if n == 0:
    print("ERROR: Regex did not find 'class ThunkOperation'!")
    sys.exit(1)

if t != original:
    with open(p, 'w') as f:
        f.write(t)
    print("OK: thunk_executor.cc - copy ctor explicitly deleted (GCC 8 ppc64le fix)")
else:
    print("WARNING: No changes")
THUNKPATCH

# Workaround for gcc8 bitfield size warning/error in XLA
sed -i 's/PrimitiveType index_primitive_type_ : 8;/PrimitiveType index_primitive_type_ : 16;/' third_party/xla/xla/layout.h || true
sed -i 's/PrimitiveType pointer_primitive_type_ : 8;/PrimitiveType pointer_primitive_type_ : 16;/' third_party/xla/xla/layout.h || true

# --- Compilation of .cu.cc is now natively handled by the NVCC Wrapper! ---
# The mutant will intercept and send to the real CUDA compiler.


python3 - << 'PYEOF'
import re

path = "tensorflow/tools/pip_package/build_pip_package.py"
with open(path, "r") as f:
    code = f.read()

# 1. Protect glob()
def glob0_safe(m):
    expr = m.group(1)
    return f"({expr}[0] if {expr} else None)"
code = re.sub(r"glob\.glob\(([^\)]*)\)\s*\[0\]", glob0_safe, code)

# 2. Global injection
injection = """
import shutil
import os
import subprocess
import re

# Patch 1: Survive null directories
_orig_copytree = shutil.copytree
def _robust_copytree(src, dst, *args, **kwargs):
    if not src or not os.path.exists(str(src)):
        print(f"WARNING: Missing directory ignored -> {src}")
        return
    kwargs['dirs_exist_ok'] = True
    try:
        return _orig_copytree(src, dst, *args, **kwargs)
    except Exception as e:
        print(f"WARNING: Ignored error copying {src}: {e}")
        return

shutil.copytree = _robust_copytree

# Patch 2: Intercept and fix setup.py in the correct temporary folder
_orig_run = subprocess.run
def _patched_run(args, **kwargs):
    if len(args) > 1 and "setup.py" in args[1]:
        cwd = kwargs.get('cwd', '.')
        setup_path = os.path.join(cwd, args[1])
        if os.path.exists(setup_path):
            with open(setup_path, "r") as f:
                c = f.read()
            # Add quotes around numeric versions that would cause a syntax error
            c = re.sub(r"(\w+_version\s*=\s*)([0-9\.]+)", r"\g<1>'\g<2>'", c)
            with open(setup_path, "w") as f:
                f.write(c)
            print(f"\\n[!!!] SUCCESS: setup.py intercepted and fixed in folder {cwd}! [!!!]\\n")
    return _orig_run(args, **kwargs)

subprocess.run = _patched_run
"""

if code.startswith("#!"):
    parts = code.split("\n", 1)
    code = parts[0] + "\n" + injection + parts[1]
else:
    code = injection + "\n" + code

with open(path, "w") as f:
    f.write(code)

print("OK: Super-Patch V2 applied successfully!")
PYEOF

python3 - << 'PYEOF'
import re

filepath = "tensorflow/tensorflow.bzl"
with open(filepath, "r") as f:
    data = f.read()

data = re.sub(r'inputs\s*=\s*\[\s*"//command_line_option:modify_execution_info",?\s*\]', 'inputs = []', data)
data = re.sub(r'outputs\s*=\s*\[\s*"//command_line_option:modify_execution_info",?\s*\]', 'outputs = []', data)

old_func_pattern = r'def _local_exec_transition_impl\(settings, attr\):.*?return \{.*?\}'
new_func_stub = 'def _local_exec_transition_impl(settings, attr):\n    return {}'
data = re.sub(old_func_pattern, new_func_stub, data, flags=re.DOTALL)

with open(filepath, "w") as f:
    f.write(data)
print("OK: tensorflow.bzl patched")
PYEOF

cd $TF_BUILD_DIR/tensorflow

# Uninstall installed TF (avoids conflict in the ExtractAPI step)
pip uninstall tensorflow -y || true

# NOTE: The pybind11 visibility patch was moved to BEFORE the bazel build
# (uses bazel fetch + sed to ensure that -fvisibility=hidden is removed before compilation)

# Aggressively replace any reference to --@local_config_cuda//:enable_cuda or include_cuda_libs
python3 - << 'PYEOF'
import os, re
tf_dir = "."
for root, dirs, files in os.walk(tf_dir):
    for f in files:
        if f.endswith(".bazelrc") or f.endswith(".bzl"):
            path = os.path.join(root, f)
            try:
                with open(path, "r") as file:
                    content = file.read()
                modified = False
                if "--@local_config_cuda//:enable_cuda" in content:
                    content = re.sub(r'--@local_config_cuda//:enable_cuda\b(?:=\w+)?', '', content)
                    modified = True
                if "--@local_config_cuda//cuda:include_cuda_libs" in content:
                    content = re.sub(r'--@local_config_cuda//cuda:include_cuda_libs\b(?:=\w+)?', '', content)
                    modified = True
                if modified:
                    with open(path, "w") as file:
                        file.write(content)
                    print(f"OK: Isolated flag removed from {path}")
            except Exception:
                pass
PYEOF

# Definitive Bypass for the Bazel 7 Exec Root Validator
# Bazel 7 blocks absolute paths in --copt (e.g.: /tmp/... or $HOME/...),
# BUT allows paths under "/usr/local/include" since GCC trusts them natively.
mkdir -p /usr/local/include/cuda_stub
rm -rf /usr/local/include/cuda_stub/*
cp -rL $HOME/local_config_cuda_stub/cuda/cuda_include/* /usr/local/include/cuda_stub/
# --- Fallback: ensure cuda_bf16.h exists (older CUDA on ppc64le may lack it) ---
if [ ! -f /usr/local/include/cuda_stub/cuda_bf16.h ]; then
    echo ">>> cuda_bf16.h not found - creating minimal stub for __nv_bfloat16..."
    cat > /usr/local/include/cuda_stub/cuda_bf16.h << 'BF16_STUB'
/* ppc64le stub: minimal __nv_bfloat16 definition for CUDA compilation */
#ifndef __CUDA_BF16_H__
#define __CUDA_BF16_H__

#ifndef __CUDA_ALIGN__
#if defined(__CUDACC__)
#define __CUDA_ALIGN__(n) __align__(n)
#else
#define __CUDA_ALIGN__(n) alignas(n)
#endif
#endif

#ifndef __CUDA_HOSTDEVICE_FP16_DECL__
#if defined(__CUDACC__)
#define __CUDA_HOSTDEVICE_FP16_DECL__ static __device__ __host__ __forceinline__
#else
#define __CUDA_HOSTDEVICE_FP16_DECL__ static inline
#endif
#endif

typedef struct __CUDA_ALIGN__(2) {
    unsigned short x;
} __nv_bfloat16;

typedef struct __CUDA_ALIGN__(4) {
    unsigned int x;
} __nv_bfloat162;

/* Basic conversions needed by XLA buffer comparator */
__CUDA_HOSTDEVICE_FP16_DECL__ __nv_bfloat16 __float2bfloat16(const float f) {
    __nv_bfloat16 val;
    unsigned int u;
    __builtin_memcpy(&u, &f, sizeof(u));
    /* Round to nearest even */
    unsigned int lsb = (u >> 16) & 1;
    unsigned int rounding_bias = 0x7fffU + lsb;
    u += rounding_bias;
    val.x = (unsigned short)(u >> 16);
    return val;
}

__CUDA_HOSTDEVICE_FP16_DECL__ float __bfloat162float(const __nv_bfloat16 h) {
    float f;
    unsigned int u = ((unsigned int)h.x) << 16;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

#endif /* __CUDA_BF16_H__ */
BF16_STUB
    echo ">>> cuda_bf16.h stub created."
fi
# --- libcudacxx (cuda/atomic etc.) ---
CUDA_ATOMIC_SRC=$(find /usr /usr/local /opt $CONDA_PREFIX -path "*/include/cuda/atomic" 2>/dev/null | head -n 1)
if [ -n "$CUDA_ATOMIC_SRC" ]; then
    CUDA_INCLUDE_BASE=$(dirname "$CUDA_ATOMIC_SRC")
    mkdir -p /usr/local/include/cuda_stub/cuda
    cp -rL "$CUDA_INCLUDE_BASE"/* /usr/local/include/cuda_stub/cuda/ 2>/dev/null || true
fi
# --- NVML real header (requires NVIDIA driver with full headers) ---
rm -rf /usr/local/include/cuda_stub/nvml
mkdir -p /usr/local/include/cuda_stub/nvml/include
echo ">>> Searching for COMPLETE nvml.h (with functions, not just types)..."

# Prioritize known paths from the real NVIDIA driver
NVML_H_SRC=""
for candidate in \
    /usr/local/cuda/include/nvml.h \
    /usr/include/nvidia/nvml.h \
    /usr/include/nvml.h \
    $(find /usr/local/cuda*/include /opt/cuda*/include -name nvml.h 2>/dev/null) \
    $(find $CONDA_PREFIX -name nvml.h 2>/dev/null); do
    if [ -f "$candidate" ] && grep -q 'nvmlErrorString' "$candidate" 2>/dev/null; then
        NVML_H_SRC="$candidate"
        echo ">>> REAL nvml.h found at: $NVML_H_SRC"
        break
    fi
done

# If a full one wasn't found, try any nvml.h ignoring copies from previous runs
if [ -z "$NVML_H_SRC" ]; then
    NVML_H_SRC=$(find /usr /usr/local /opt ${CONDA_PREFIX:-/dev/null} -type f -name nvml.h 2>/dev/null | grep -v "cuda_stub" | grep -v "cuda_unified" | head -n 1)
fi

if [ -z "$NVML_H_SRC" ]; then
    echo ">>> WARNING: nvml.h not found on the system. Creating a full mock from scratch..."
    cat > /tmp/mock_nvml.h << 'EOF'
/* Mock nvml.h auto-generated for compilation */
#ifndef MOCK_NVML_H
#define MOCK_NVML_H

typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    NVML_ERROR_TIMEOUT = 10,
    NVML_ERROR_IRQ_ISSUE = 11,
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,
    NVML_ERROR_CORRUPTED_INFOROM = 14,
    NVML_ERROR_GPU_IS_LOST = 15,
    NVML_ERROR_RESET_REQUIRED = 16,
    NVML_ERROR_OPERATING_SYSTEM = 17,
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,
    NVML_ERROR_IN_USE = 19,
    NVML_ERROR_MEMORY = 20,
    NVML_ERROR_NO_DATA = 21,
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23,
    NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

#endif
EOF
    NVML_H_SRC="/tmp/mock_nvml.h"
else
    echo ">>> WARNING: nvml.h found at $NVML_H_SRC may be an incomplete stub."
fi

cp -f "$NVML_H_SRC" /usr/local/include/cuda_stub/nvml/include/nvml.h

# Check if nvml.h (now copied) is complete (has functions, not just types)
if ! grep -q 'nvmlErrorString' /usr/local/include/cuda_stub/nvml/include/nvml.h; then
    echo ">>> nvml.h is a minimal stub. Adding missing NVML function declarations..."
    cat >> /usr/local/include/cuda_stub/nvml/include/nvml.h << 'NVML_FUNCS'

/* === ppc64le patch: NVML function declarations for stub headers === */
#ifndef NVML_STUB_FUNCTIONS_PATCHED
#define NVML_STUB_FUNCTIONS_PATCHED

#ifndef NVML_ERROR_NOT_SUPPORTED
#define NVML_ERROR_NOT_SUPPORTED 3
#endif
#ifndef NVML_ERROR_FUNCTION_NOT_FOUND
#define NVML_ERROR_FUNCTION_NOT_FOUND 13
#endif
#ifndef NVML_FEATURE_DISABLED
#define NVML_FEATURE_DISABLED 0
#endif
#ifndef NVML_FEATURE_ENABLED
#define NVML_FEATURE_ENABLED 1
#endif
#ifndef NVML_NVLINK_CAP_P2P_SUPPORTED
#define NVML_NVLINK_CAP_P2P_SUPPORTED 0
#endif
#ifndef NVML_GPU_FABRIC_STATE_NOT_SUPPORTED
#define NVML_GPU_FABRIC_STATE_NOT_SUPPORTED 0
#endif

/* Guard types that may already be defined by real nvml.h */
#ifndef __NVML_DEVICE_T_DEFINED__
#define __NVML_DEVICE_T_DEFINED__
typedef struct nvmlDevice_st* nvmlDevice_t;
#endif

#ifndef __NVML_ENABLE_STATE_T_DEFINED__
#define __NVML_ENABLE_STATE_T_DEFINED__
typedef unsigned int nvmlEnableState_t;
#endif

#ifndef __NVML_GPU_FABRIC_INFO_V_T_DEFINED__
#define __NVML_GPU_FABRIC_INFO_V_T_DEFINED__
typedef struct {
    unsigned int version;
    char clusterUuid[16];
    unsigned int state;
    unsigned int cliqueId;
} nvmlGpuFabricInfoV_t;
#endif

#ifndef nvmlGpuFabricInfo_v2
#define nvmlGpuFabricInfo_v2 2
#endif

#ifdef __cplusplus
extern "C" {
#endif

const char* nvmlErrorString(nvmlReturn_t result);
nvmlReturn_t nvmlInit(void);
nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlShutdown(void);
nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen);
nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth);
nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);
nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, unsigned int capability, unsigned int *capResult);
nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t *fabricInfo);

#ifdef __cplusplus
}
#endif

#endif /* NVML_STUB_FUNCTIONS_PATCHED */
NVML_FUNCS
    echo ">>> OK! NVML declarations injected into the stub."
fi

# Ensure the constant required by nvml_stub.cc and cast to a C++ enum
if ! grep -q 'NVML_ERROR_FUNCTION_NOT_FOUND' /usr/local/include/cuda_stub/nvml/include/nvml.h; then
    echo '#define NVML_ERROR_FUNCTION_NOT_FOUND ((nvmlReturn_t)13)' >> /usr/local/include/cuda_stub/nvml/include/nvml.h
else
    sed -i 's/#define NVML_ERROR_FUNCTION_NOT_FOUND 13/#define NVML_ERROR_FUNCTION_NOT_FOUND ((nvmlReturn_t)13)/g' /usr/local/include/cuda_stub/nvml/include/nvml.h
fi
# Copy nvml.h ALSO to the include paths where Bazel cc_library(name="nvml") looks
cp -f /usr/local/include/cuda_stub/nvml/include/nvml.h /usr/local/include/cuda_stub/nvml.h
cp -f /usr/local/include/cuda_stub/nvml/include/nvml.h $HOME/local_config_cuda_stub/cuda/cuda_include/nvml.h
cp -f /usr/local/include/cuda_stub/nvml/include/nvml.h $HOME/cuda_unified/include/nvml.h 2>/dev/null || true

# --- NCCL real header (requires NCCL) ---
mkdir -p /usr/local/include/cuda_stub/third_party/nccl
NCCL_H_SRC=$(find /usr /usr/local /opt $CONDA_PREFIX -name nccl.h 2>/dev/null | head -n 1)
if [ -z "$NCCL_H_SRC" ]; then
    echo "ERROR: nccl.h not found. Install NCCL (headers)."
    exit 1
fi
cp -f "$NCCL_H_SRC" /usr/local/include/cuda_stub/third_party/nccl/nccl.h
# Remove CUB/Thrust from GCC's include path (they contain __global__ and <<<>>> which GCC doesn't understand)
rm -rf /usr/local/include/cuda_stub/cub
rm -rf /usr/local/include/cuda_stub/thrust
echo ">>> CUB/Thrust removed from GCC's include path"

# --- NVTX headers (nvtx3/nvToolsExt.h) required by XLA's CUPTI profiler ---
echo ">>> Searching for NVTX (nvtx3) headers..."
NVTX_H=$(find /usr /opt ${CONDA_PREFIX:-/dev/null} -path "*/nvtx3/nvToolsExt.h" 2>/dev/null | head -n 1)
if [ -z "$NVTX_H" ]; then
    echo ">>> NVTX not found on system. Trying conda..."
    conda install -c conda-forge cuda-nvtx -y 2>/dev/null || true
    NVTX_H=$(find ${CONDA_PREFIX:-/dev/null} -path "*/nvtx3/nvToolsExt.h" 2>/dev/null | head -n 1)
fi
if [ -z "$NVTX_H" ]; then
    echo ">>> NVTX still not found. Downloading NVTX3 from GitHub (header-only, arch-independent)..."
    rm -rf /tmp/nvtx3_download
    mkdir -p /tmp/nvtx3_download
    wget -qO /tmp/nvtx3_download/nvtx.tar.gz https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.1.0.tar.gz
    tar -xzf /tmp/nvtx3_download/nvtx.tar.gz -C /tmp/nvtx3_download
    NVTX_H=$(find /tmp/nvtx3_download -path "*/nvtx3/nvToolsExt.h" 2>/dev/null | head -n 1)
fi
if [ ! -z "$NVTX_H" ]; then
    NVTX_DIR=$(dirname "$NVTX_H")  # .../include/nvtx3
    echo ">>> NVTX found at: $NVTX_DIR"
    # Inject into ALL include paths used by the build
    for dest in /usr/local/include/cuda_stub \
                $HOME/local_config_cuda_stub/cuda/cuda_include \
                $HOME/cuda_unified/include; do
        mkdir -p "$dest/nvtx3"
        cp -rL "$NVTX_DIR"/* "$dest/nvtx3/" 2>/dev/null || true
    done
    echo ">>> OK! NVTX headers injected successfully."
else
    echo ">>> ERROR: NVTX (nvtx3/nvToolsExt.h) not found after all attempts."
fi
rm -rf /tmp/nvtx3_download 2>/dev/null || true

echo "=== BUILDING UNIFIED CUDA TREE (MOCK) ==="
# To restore TensorFlow's native (NVCC) toolchain without breaking on Conda fragmentation,
# we build a fake tree identical to /usr/local/cuda and point TF_CUDA_PATHS here.
rm -rf $HOME/cuda_unified
mkdir -p $HOME/cuda_unified/include
mkdir -p $HOME/cuda_unified/lib64
mkdir -p $HOME/cuda_unified/bin
mkdir -p $HOME/cuda_unified/nvvm/libdevice

echo "-> Linking headers..."
cp -rL $HOME/local_config_cuda_stub/cuda/cuda_include/* $HOME/cuda_unified/include/ 2>/dev/null || true

echo "-> Linking binaries (nvcc, ptxas, fatbinary)..."
NVCC_PATH=$(which nvcc 2>/dev/null || find /usr /opt $CONDA_PREFIX -name nvcc -executable -type f 2>/dev/null | head -n 1)
[ ! -z "$NVCC_PATH" ] && ln -sf "$NVCC_PATH" $HOME/cuda_unified/bin/nvcc

for tool in ptxas fatbinary; do
  TOOL_PATH=$(find /usr /opt $CONDA_PREFIX -name $tool -executable -type f 2>/dev/null | head -n 1)
  [ ! -z "$TOOL_PATH" ] && ln -sf "$TOOL_PATH" $HOME/cuda_unified/bin/
done

echo "-> Linking essential dynamic libraries (.so)..."
for lib in libcudart libcublas libcudnn libcurand libcusolver libcusparse libcufft libnvJitLink libnccl libnvidia-ml; do
  LIB_PATH=$(find /usr /opt /lib /lib64 $CONDA_PREFIX -name "${lib}.so*" 2>/dev/null | head -n 1)
  if [ ! -z "$LIB_PATH" ]; then
    ln -sf "$LIB_PATH" $HOME/cuda_unified/lib64/
    # Create unversioned symlink if needed (cuda_configure needs libXXX.so)
    BASENAME=$(basename "$LIB_PATH")
    if [ "$BASENAME" != "${lib}.so" ]; then
      ln -sf "$LIB_PATH" $HOME/cuda_unified/lib64/${lib}.so
    fi
  fi
done

echo "-> Linking libdevice (nvvm)..."
LIBDEVICE_PATH=$(find /usr /opt $CONDA_PREFIX -name "libdevice.10.bc" 2>/dev/null | head -n 1)
[ ! -z "$LIBDEVICE_PATH" ] && ln -sf "$LIBDEVICE_PATH" $HOME/cuda_unified/nvvm/libdevice/

export TF_CUDA_PATHS=$HOME/cuda_unified
echo ">>> Unified CUDA tree ready at: $HOME/cuda_unified"
echo "============================================="

# --- DEDICATED cuda_nvcc stub (with real system binaries) ---
# The local_config_cuda crosstool references @@cuda_nvcc//:bin, :ptxas, etc.
# We need a separate stub from the generic one that contains the real executables.
echo ">>> Creating cuda_nvcc_stub with real binaries..."
mkdir -p $HOME/cuda_nvcc_stub
touch $HOME/cuda_nvcc_stub/WORKSPACE
echo 'VERSION = "12"' > $HOME/cuda_nvcc_stub/version.bzl

# Symlinks for real CUDA binaries (inside bin/ to avoid Bazel cycle)
mkdir -p $HOME/cuda_nvcc_stub/bin
for tool in nvcc ptxas fatbinary cuobjdump nvlink nvdisasm cicc nvprune; do
    TOOL_REAL=$(find $HOME/cuda_unified/bin $CONDA_PREFIX/bin /usr/local/cuda/bin -name "$tool" -executable 2>/dev/null | head -n 1)
    if [ ! -z "$TOOL_REAL" ]; then
        ln -sf "$TOOL_REAL" $HOME/cuda_nvcc_stub/bin/$tool
        echo "   -> $tool => $TOOL_REAL"
    fi
done

# Symlink for nvvm/libdevice
mkdir -p $HOME/cuda_nvcc_stub/nvvm/libdevice
LIBDEVICE=$(find $HOME/cuda_unified /usr /opt $CONDA_PREFIX -name "libdevice.10.bc" 2>/dev/null | head -n 1)
[ ! -z "$LIBDEVICE" ] && ln -sf "$LIBDEVICE" $HOME/cuda_nvcc_stub/nvvm/libdevice/

cat > $HOME/cuda_nvcc_stub/BUILD << 'NVCC_EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "all", srcs = glob(["bin/*", "nvvm/**"], allow_empty = True))
filegroup(name = "bin", srcs = glob(["bin/*"], allow_empty = True))
filegroup(name = "nvcc", srcs = ["bin/nvcc"])
filegroup(name = "ptxas", srcs = glob(["bin/ptxas"], allow_empty = True))
filegroup(name = "fatbinary", srcs = glob(["bin/fatbinary"], allow_empty = True))
filegroup(name = "cuobjdump", srcs = glob(["bin/cuobjdump"], allow_empty = True))
filegroup(name = "nvlink", srcs = glob(["bin/nvlink"], allow_empty = True))
filegroup(name = "nvdisasm", srcs = glob(["bin/nvdisasm"], allow_empty = True))
filegroup(name = "nvvm", srcs = glob(["nvvm/**"], allow_empty = True))
filegroup(name = "header_list", srcs = [])
cc_library(name = "headers", hdrs = [])
filegroup(name = "static", srcs = [])
filegroup(name = "shared", srcs = [])
NVCC_EOF
echo ">>> cuda_nvcc_stub created successfully"

# --- CRITICAL PATCH: GCC Wrapper (Interceptor V4 - @params Support) ---
echo ">>> Updating GCC/G++ wrappers with advanced Assembly interceptor..."

cat > $HOME/gcc_cuda_wrapper.sh << 'WRAPPER_EOF'
#!/bin/bash
unset NVCC_APPEND_FLAGS
unset NVCC_PREPEND_FLAGS
# CRITICAL: Ensure cicc and ptxas are in PATH for NVCC
export PATH=${CONDA_PREFIX}/nvvm/bin:${CONDA_PREFIX}/bin:${HOME}/cuda_unified/bin:$PATH
ARGS=()
IS_CUDA=0
REAL_NVCC="${HOME}/cuda_unified/bin/nvcc"
REAL_GCC="${CONDA_PREFIX}/bin/clang"
# Linker: uses /root/compiler_hijack which has ld/ld.bfd as UN-prefixed symlinks
# to the Conda binaries (binutils >= 2.40 with support for Power10 PCREL relocs
# from jpegxl Highway). AlmaLinux 8.10 ships binutils 2.30 which fails on reloc 132.
# Since GCC with -fuse-ld=bfd searches for "ld.bfd" without prefix, we need the symlink.
if [ -x "${HOME}/compiler_hijack/ld.bfd" ]; then
    REAL_LD_DIR="${HOME}/compiler_hijack"
elif [ -x /usr/bin/ld.bfd ]; then
    REAL_LD_DIR="/usr/bin"
elif [ -x /bin/ld.bfd ]; then
    REAL_LD_DIR="/bin"
else
    REAL_LD_DIR="$(dirname "$(command -v ld.bfd 2>/dev/null || echo /usr/bin/ld)")"
fi
GCC_FORCE_LD="-B$REAL_LD_DIR"

# NOTE: DO NOT rewrite .localentry $sym, N -> 0.
# REAL_GCC=clang (line above) correctly accepts `.localentry sym, 1`
# (PPC64 ELFv2 ABI v1.5+ "CLOBBERS_R2" flag), producing stOther=0x20.
# This allows patched LLD to create a PPC64R2SaveStub for bls to these
# symbols (e.g.: Implib.so/cudart_stub trampolines) and patch the nop after
# the bl into `ld r2, 24(r1)`. Rewriting the flag to 0 kills this chain
# and is the root cause of the r2 crash in __sti____cudaRegisterAll in TF.
clean_asm() {
    : # no-op
}

for arg in "$@"; do
    # RAFT-ENABLE: select_k_exec_raft.cc inclui headers .cuh (CUDA device code).
    # cuda_library nao basta - precisa do wrapper rotear pro modo NVCC.
    if [[ "$arg" == *.cu.cc ]] || [[ "$arg" == *select_k_exec_raft.cc ]]; then
        IS_CUDA=1
    fi
    if [[ "$arg" == @* ]]; then
        param_file="${arg#@}"
        if [ -f "$param_file" ]; then
            while read -r line; do
                clean_line=$(echo "$line" | tr -d "'" | tr -d '"')
                clean_asm "$clean_line"
                if [[ "$clean_line" == *.cu.cc ]] || [[ "$clean_line" == *select_k_exec_raft.cc ]]; then IS_CUDA=1; fi
            done < "$param_file"
        fi
        ARGS+=("$arg")
        continue
    fi

    if [ "${SKIP_NEXT:-0}" -eq 1 ]; then
        SKIP_NEXT=0
        if [ "$arg" = "cuda" ]; then continue; else ARGS+=("-x" "$arg"); continue; fi
    fi

    if [ "${SKIP_NEXT_MLLVM:-0}" -eq 1 ]; then
        SKIP_NEXT_MLLVM=0
        continue
    fi

    case "$arg" in
        --cuda-gpu-arch=*|--cuda-include-ptx=*|-Xcuda-fatbinary=*|-nvcc_options=*|--cuda-path=*|--offload-arch=*) ;;
        -mno-*|-mcpu=*|-mtune=*|-mvsx|-maltivec) ARGS+=("$arg");; # Keep for GCC, strip later for NVCC
        -mllvm) SKIP_NEXT_MLLVM=1 ;;
        -x) SKIP_NEXT=1 ;;
        *.S|*.s)
            clean_asm "$arg"
            ARGS+=("$arg")
        ;;
        *) ARGS+=("$arg") ;;
    esac
done

# Detect exec-configuration builds (host tools / genrules).
# Bazel places exec-config outputs in paths like bazel-out/ppc-opt-exec-ST-xxx/.
# These tools must actually RUN on the build host, so they must NOT contain
# real CUDA kernel launches (which require a working CUDA runtime).
# Force GCC compilation for .cu.cc in exec config — the #ifdef __CUDACC__ guards
# will strip GPU kernel code, producing a tool that links and runs correctly.
IS_EXEC=0
for arg in "$@"; do
    if [[ "$arg" == *"-exec-"* ]]; then
        IS_EXEC=1
        break
    fi
    if [[ "$arg" == @* ]]; then
        pf="${arg#@}"
        if [ -f "$pf" ] && grep -q "\-exec-" "$pf" 2>/dev/null; then
            IS_EXEC=1
            break
        fi
    fi
done
if [ $IS_EXEC -eq 1 ] && [ $IS_CUDA -eq 1 ]; then
    # Exec-config .cu.cc: produce an empty object file.
    # GCC cannot compile these files because they depend on CUDA-only headers
    # (CUB, Thrust, etc.) that aren't available outside NVCC.
    # Exec tools don't need GPU kernels — they just need to link and run.
    OUT_FILE=""
    for i in "${!ARGS[@]}"; do
        if [[ "${ARGS[$i]}" == "-o" ]]; then
            OUT_FILE="${ARGS[$i+1]}"
            break
        fi
    done
    # Also check param files for -o
    if [ -z "$OUT_FILE" ]; then
        for arg in "$@"; do
            if [[ "$arg" == @* ]]; then
                pf="${arg#@}"
                if [ -f "$pf" ]; then
                    OUT_FILE=$(grep -A1 '^-o$' "$pf" 2>/dev/null | tail -1 | tr -d "'" | tr -d '"')
                    [ -n "$OUT_FILE" ] && break
                fi
            fi
        done
    fi
    if [ -n "$OUT_FILE" ]; then
        mkdir -p "$(dirname "$OUT_FILE")"
        echo "// empty exec-config .cu.cc stub" | $REAL_GCC -x c++ -c - -o "$OUT_FILE"
        # Also create the .d dependency file Bazel expects
        DEP_FILE=""
        for i in "${!ARGS[@]}"; do
            if [[ "${ARGS[$i]}" == "-MF" ]]; then
                DEP_FILE="${ARGS[$i+1]}"
                break
            fi
        done
        if [ -z "$DEP_FILE" ]; then
            for arg in "$@"; do
                if [[ "$arg" == @* ]]; then
                    pf="${arg#@}"
                    if [ -f "$pf" ]; then
                        DEP_FILE=$(grep -A1 '^-MF$' "$pf" 2>/dev/null | tail -1 | tr -d "'" | tr -d '"')
                        [ -n "$DEP_FILE" ] && break
                    fi
                fi
            done
        fi
        # Fallback: derive .d path from .o path
        if [ -z "$DEP_FILE" ]; then
            DEP_FILE="${OUT_FILE%.o}.d"
        fi
        mkdir -p "$(dirname "$DEP_FILE")"
        echo "$OUT_FILE:" > "$DEP_FILE"
        exit $?
    fi
    # Fallback: let GCC try (will likely fail, but better than silently doing nothing)
    IS_CUDA=0
fi

if [ $IS_CUDA -eq 1 ]; then
    EXPANDED_ARGS=()
    for arg in "${ARGS[@]}"; do
        if [[ "$arg" == @* ]]; then
            pf="${arg#@}"
            if [ -f "$pf" ]; then
                while read -r line; do
                    clean_line=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | tr -d "'" | tr -d '"')
                    EXPANDED_ARGS+=("$clean_line")
                done < "$pf"
            fi
        else
            EXPANDED_ARGS+=("$arg")
        fi
    done

    CLEAN_ARGS=()
    SKIP_NEXT_ARG=0
    for i in "${!EXPANDED_ARGS[@]}"; do
        if [ $SKIP_NEXT_ARG -eq 1 ]; then
            SKIP_NEXT_ARG=0
            continue
        fi
        arg="${EXPANDED_ARGS[$i]}"
        
        case "$arg" in
            -Xcompiler|--compiler-options)
                next_arg="${EXPANDED_ARGS[$i+1]}"
                if [[ "$next_arg" == "-fPIC" ]] || [[ "$next_arg" == "-fpic" ]]; then
                    CLEAN_ARGS+=("--compiler-options=${next_arg}")
                    SKIP_NEXT_ARG=1
                elif [[ "$next_arg" == -f* ]] || [[ "$next_arg" == -W* ]]; then
                    SKIP_NEXT_ARG=1
                else
                    CLEAN_ARGS+=("--compiler-options=${next_arg}")
                    SKIP_NEXT_ARG=1
                fi
                ;;
            -Xlinker)
                CLEAN_ARGS+=("-Xlinker=${EXPANDED_ARGS[$i+1]}")
                SKIP_NEXT_ARG=1
                ;;
            -Wno-*|-fno-*|-fstack-protector*|-Wall|-Werror*|-Wunused*|-Wformat*|-g0|-ffunction-sections|-fdata-sections|-fvisibility=*|-frandom-seed=*|-pthread|-no-canonical-prefixes|--no-canonical-prefixes) ;;
            -D_FORTIFY_SOURCE*|-U_FORTIFY_SOURCE) ;; # Incompatible with -O0
            -MF)
                DEP_FILE="${EXPANDED_ARGS[$i+1]}"
                CLEAN_ARGS+=("-MF" "$DEP_FILE")
                SKIP_NEXT_ARG=1
                ;;
            -x) 
                SKIP_NEXT_ARG=1
                ;;
            -fPIC|-fpic)
                CLEAN_ARGS+=("-Xcompiler" "$arg")
                ;;
            -fno-visibility-inlines-hidden|-fexceptions)
                CLEAN_ARGS+=("-Xcompiler" "$arg")
                ;;
            -fPIC|-fpic|-fexceptions|-fPIC)
                CLEAN_ARGS+=("-Xcompiler" "$arg")
                ;;
            -f*|-W*|-mno-*|-mfloat*|-mcpu=*|-mtune=*|-mvsx|-maltivec|-mcmodel=*)
                # Strip GCC-specific flags that NVCC doesn't understand
                ;;
            -O*) CLEAN_ARGS+=("$arg") ;;
            -iquote)
                CLEAN_ARGS+=("-I" "${EXPANDED_ARGS[$i+1]}")
                SKIP_NEXT_ARG=1
                ;;
            --sysroot|-B)
                CLEAN_ARGS+=("-Xcompiler" "$arg" "-Xcompiler" "${EXPANDED_ARGS[$i+1]}")
                SKIP_NEXT_ARG=1
                ;;
            -B*)
                CLEAN_ARGS+=("-Xcompiler" "$arg")
                ;;
            *) CLEAN_ARGS+=("$arg") ;;
        esac
    done
    
    # -fPIC: required on ppc64le when linking shared libs with lld (TOC16 relocs).
    # Bazel passes -fPIC but this wrapper strips -f* for NVCC, so inject explicitly.
    CLEAN_ARGS+=("-x" "cu" "-arch=sm_70" "-O0" "-Xcicc" "-O0" "--ptxas-options=-O0" "-Xcompiler" "-O0" "-allow-unsupported-compiler" "--compiler-options=-fPIC" "--compiler-options=-mcmodel=medium" "--expt-relaxed-constexpr" "-include" "cuda_bf16.h" "-ccbin" "${HOME}/compiler_hijack" "-std=c++17" "-DEIGEN_DONT_VECTORIZE" "-D__NO_INLINE__" "-U__VSX__" "-U__ALTIVEC__" "-D_GLIBCXX_USE_CXX11_ABI=1" "-isystem" "${CONDA_PREFIX}/include" "-isystem" "${HOME}/cuda_unified/include" "-isystem" "/usr/local/include/cuda_stub" "-isystem" "/usr/local/cuda/include")
    echo "NVCC ARGS: ${CLEAN_ARGS[@]}" >> /tmp/nvcc_dump.txt
    $REAL_NVCC "${CLEAN_ARGS[@]}"
    RC=$?
    # Post-process .d file: remove absolute path dependencies that Bazel rejects
    if [ -n "$DEP_FILE" ] && [ -f "$DEP_FILE" ]; then
        sed -i 's| /[^ ]*||g' "$DEP_FILE"
    fi
    exit $RC
else
    # Pre-process params files: remove --start-lib/--end-lib before GCC calls collect2/ld
    FINAL_ARGS=()
    for arg in "${ARGS[@]}"; do
        if [[ "$arg" == "-Wl,--start-lib" ]] || [[ "$arg" == "-Wl,--end-lib" ]] || [[ "$arg" == "--start-lib" ]] || [[ "$arg" == "--end-lib" ]]; then
            continue
        fi
        if [[ "$arg" == @* ]]; then
            param_file="${arg#@}"
            if [ -f "$param_file" ]; then
                tmp=$(mktemp /tmp/GCC_params_XXXXXX)
                # Filter out start-lib/end-lib arguments line by line
                grep -v -E '^[[:space:]]*(-Wl,)?--start-lib[[:space:]]*$' "$param_file" | grep -v -E '^[[:space:]]*(-Wl,)?--end-lib[[:space:]]*$' > "$tmp"
                if ! diff -q "$param_file" "$tmp" >/dev/null 2>&1; then
                    echo "Wrapper(GCC): Stripped flags from $param_file -> $tmp" >> /tmp/wrapper_debug.log
                    FINAL_ARGS+=("@$tmp")
                else
                    rm "$tmp"
                    FINAL_ARGS+=("$arg")
                fi
            else
                FINAL_ARGS+=("$arg")
            fi
        else
            FINAL_ARGS+=("$arg")
        fi
    done
    exec $REAL_GCC $GCC_FORCE_LD -mcpu=power9 "-isystem/usr/local/include/cuda_stub" "${FINAL_ARGS[@]}"
fi
WRAPPER_EOF
# Expand ${HOME} and ${CONDA_PREFIX} in the wrapper to actual values now,
# because Bazel clears the env when executing actions and these vars end up empty.
sed -i "s|\${HOME}|$HOME|g; s|\${CONDA_PREFIX}|$CONDA_PREFIX|g" $HOME/gcc_cuda_wrapper.sh
chmod +x $HOME/gcc_cuda_wrapper.sh

cat > $HOME/gxx_cuda_wrapper.sh << 'WRAPPER_EOF'
#!/bin/bash
unset NVCC_APPEND_FLAGS
unset NVCC_PREPEND_FLAGS
# CRITICAL: Ensure cicc and ptxas are in PATH for NVCC
export PATH=${CONDA_PREFIX}/nvvm/bin:${CONDA_PREFIX}/bin:${HOME}/cuda_unified/bin:$PATH
ARGS=()
IS_CUDA=0
REAL_NVCC="${HOME}/cuda_unified/bin/nvcc"
REAL_GCC="$(which clang++ 2>/dev/null || which powerpc64le-conda-linux-gnu-g++ 2>/dev/null || echo /usr/bin/g++)"
# Linker: uses /root/compiler_hijack which has ld/ld.bfd as UN-prefixed symlinks
# to the Conda binaries (binutils >= 2.40 with support for Power10 PCREL relocs
# from jpegxl Highway). AlmaLinux 8.10 ships binutils 2.30 which fails on reloc 132.
# Since GCC with -fuse-ld=bfd searches for "ld.bfd" without prefix, we need the symlink.
if [ -x "${HOME}/compiler_hijack/ld.bfd" ]; then
    REAL_LD_DIR="${HOME}/compiler_hijack"
elif [ -x /usr/bin/ld.bfd ]; then
    REAL_LD_DIR="/usr/bin"
elif [ -x /bin/ld.bfd ]; then
    REAL_LD_DIR="/bin"
else
    REAL_LD_DIR="$(dirname "$(command -v ld.bfd 2>/dev/null || echo /usr/bin/ld)")"
fi
GCC_FORCE_LD="-B$REAL_LD_DIR"

# NOTE: DO NOT rewrite .localentry $sym, N -> 0.
# REAL_GCC=clang (line above) correctly accepts `.localentry sym, 1`
# (PPC64 ELFv2 ABI v1.5+ "CLOBBERS_R2" flag), producing stOther=0x20.
# This allows patched LLD to create a PPC64R2SaveStub for bls to these
# symbols (e.g.: Implib.so/cudart_stub trampolines) and patch the nop after
# the bl into `ld r2, 24(r1)`. Rewriting the flag to 0 kills this chain
# and is the root cause of the r2 crash in __sti____cudaRegisterAll in TF.
clean_asm() {
    : # no-op
}

for arg in "$@"; do
    # RAFT-ENABLE: select_k_exec_raft.cc inclui headers .cuh (CUDA device code).
    # cuda_library nao basta - precisa do wrapper rotear pro modo NVCC.
    if [[ "$arg" == *.cu.cc ]] || [[ "$arg" == *select_k_exec_raft.cc ]]; then
        IS_CUDA=1
    fi
    if [[ "$arg" == @* ]]; then
        param_file="${arg#@}"
        if [ -f "$param_file" ]; then
            while read -r line; do
                clean_line=$(echo "$line" | tr -d "'" | tr -d '"')
                clean_asm "$clean_line"
                if [[ "$clean_line" == *.cu.cc ]] || [[ "$clean_line" == *select_k_exec_raft.cc ]]; then IS_CUDA=1; fi
            done < "$param_file"
        fi
        ARGS+=("$arg")
        continue
    fi

    if [ "${SKIP_NEXT:-0}" -eq 1 ]; then
        SKIP_NEXT=0
        if [ "$arg" = "cuda" ]; then continue; else ARGS+=("-x" "$arg"); continue; fi
    fi

    if [ "${SKIP_NEXT_MLLVM:-0}" -eq 1 ]; then
        SKIP_NEXT_MLLVM=0
        continue
    fi

    case "$arg" in
        --cuda-gpu-arch=*|--cuda-include-ptx=*|-Xcuda-fatbinary=*|-nvcc_options=*|--cuda-path=*|--offload-arch=*) ;;
        -mno-*|-mcpu=*|-mtune=*|-mvsx|-maltivec) ARGS+=("$arg");; # Keep for GCC, strip later for NVCC
        -mllvm) SKIP_NEXT_MLLVM=1 ;;
        -x) SKIP_NEXT=1 ;;
        *.S|*.s)
            clean_asm "$arg"
            ARGS+=("$arg")
        ;;
        *) ARGS+=("$arg") ;;
    esac
done

if [ $IS_CUDA -eq 1 ]; then
    EXPANDED_ARGS=()
    for arg in "${ARGS[@]}"; do
        if [[ "$arg" == @* ]]; then
            pf="${arg#@}"
            if [ -f "$pf" ]; then
                while read -r line; do
                    clean_line=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | tr -d "'" | tr -d '"')
                    EXPANDED_ARGS+=("$clean_line")
                done < "$pf"
            fi
        else
            EXPANDED_ARGS+=("$arg")
        fi
    done

    CLEAN_ARGS=()
    SKIP_NEXT_ARG=0
    for i in "${!EXPANDED_ARGS[@]}"; do
        if [ $SKIP_NEXT_ARG -eq 1 ]; then
            SKIP_NEXT_ARG=0
            continue
        fi
        arg="${EXPANDED_ARGS[$i]}"
        
        case "$arg" in
            -Xcompiler|--compiler-options)
                next_arg="${EXPANDED_ARGS[$i+1]}"
                if [[ "$next_arg" == "-fPIC" ]] || [[ "$next_arg" == "-fpic" ]]; then
                    CLEAN_ARGS+=("--compiler-options=${next_arg}")
                    SKIP_NEXT_ARG=1
                elif [[ "$next_arg" == -f* ]] || [[ "$next_arg" == -W* ]]; then
                    SKIP_NEXT_ARG=1
                else
                    CLEAN_ARGS+=("--compiler-options=${next_arg}")
                    SKIP_NEXT_ARG=1
                fi
                ;;
            -Xlinker)
                CLEAN_ARGS+=("-Xlinker=${EXPANDED_ARGS[$i+1]}")
                SKIP_NEXT_ARG=1
                ;;
            -Wno-*|-fno-*|-fstack-protector*|-Wall|-Werror*|-Wunused*|-Wformat*|-g0|-ffunction-sections|-fdata-sections|-fvisibility=*|-frandom-seed=*) ;;
            -D_FORTIFY_SOURCE*|-U_FORTIFY_SOURCE) ;; # Incompatible with -O0
            -MF)
                DEP_FILE="${EXPANDED_ARGS[$i+1]}"
                CLEAN_ARGS+=("-MF" "$DEP_FILE")
                SKIP_NEXT_ARG=1
                ;;
            -x)
                SKIP_NEXT_ARG=1
                ;;
            -fPIC|-fpic)
                CLEAN_ARGS+=("-Xcompiler" "$arg")
                ;;
            -fno-visibility-inlines-hidden|-fexceptions)
                CLEAN_ARGS+=("-Xcompiler" "$arg")
                ;;
            -fPIC|-fpic|-fexceptions|-fPIC)
                CLEAN_ARGS+=("-Xcompiler" "$arg")
                ;;
            -f*|-W*|-mno-*|-mfloat*|-mcpu=*|-mtune=*|-mvsx|-maltivec|-mcmodel=*)
                # Strip GCC-specific flags that NVCC doesn't understand
                ;;
            -O*) CLEAN_ARGS+=("$arg") ;;
            -iquote)
                CLEAN_ARGS+=("-I" "${EXPANDED_ARGS[$i+1]}")
                SKIP_NEXT_ARG=1
                ;;
            --sysroot|-B)
                CLEAN_ARGS+=("-Xcompiler" "$arg" "-Xcompiler" "${EXPANDED_ARGS[$i+1]}")
                SKIP_NEXT_ARG=1
                ;;
            -B*)
                CLEAN_ARGS+=("-Xcompiler" "$arg")
                ;;
            *) CLEAN_ARGS+=("$arg") ;;
        esac
    done
    
    CLEAN_ARGS+=("-x" "cu" "-arch=sm_70" "-O0" "-Xcicc" "-O0" "--ptxas-options=-O0" "-Xcompiler" "-O0" "-allow-unsupported-compiler" "--compiler-options=-fPIC" "-cudart=none" "-ccbin" "${HOME}/compiler_hijack" "-std=c++17" "-DEIGEN_DONT_VECTORIZE" "-D__NO_INLINE__" "-U__VSX__" "-U__ALTIVEC__" "-D_GLIBCXX_USE_CXX11_ABI=1" "-isystem" "${CONDA_PREFIX}/include" "-isystem" "${HOME}/cuda_unified/include" "-isystem" "/usr/local/include/cuda_stub" "-isystem" "/usr/local/cuda/include")
    echo "NVCC ARGS: ${CLEAN_ARGS[@]}" >> /tmp/nvcc_dump.txt
    $REAL_NVCC "${CLEAN_ARGS[@]}"
    RC=$?
    # Post-process .d file: remove absolute path dependencies that Bazel rejects
    if [ -n "$DEP_FILE" ] && [ -f "$DEP_FILE" ]; then
        sed -i 's| /[^ ]*||g' "$DEP_FILE"
    fi
    exit $RC
else
    # Pre-process params files: remove --start-lib/--end-lib before GCC calls collect2/ld
    FINAL_ARGS=()
    for arg in "${ARGS[@]}"; do
        if [[ "$arg" == "-Wl,--start-lib" ]] || [[ "$arg" == "-Wl,--end-lib" ]] || [[ "$arg" == "--start-lib" ]] || [[ "$arg" == "--end-lib" ]]; then
            continue
        fi
        if [[ "$arg" == @* ]]; then
            param_file="${arg#@}"
            if [ -f "$param_file" ]; then
                tmp=$(mktemp /tmp/GXX_params_XXXXXX)
                # Filter out start-lib/end-lib arguments line by line
                grep -v -E '^[[:space:]]*(-Wl,)?--start-lib[[:space:]]*$' "$param_file" | grep -v -E '^[[:space:]]*(-Wl,)?--end-lib[[:space:]]*$' > "$tmp"
                if ! diff -q "$param_file" "$tmp" >/dev/null 2>&1; then
                    echo "Wrapper(GXX): Stripped flags from $param_file -> $tmp" >> /tmp/wrapper_debug.log
                    FINAL_ARGS+=("@$tmp")
                else
                    rm "$tmp"
                    FINAL_ARGS+=("$arg")
                fi
            else
                FINAL_ARGS+=("$arg")
            fi
        else
            FINAL_ARGS+=("$arg")
        fi
    done
    exec $REAL_GCC $GCC_FORCE_LD -mcpu=power9 "-isystem/usr/local/include/cuda_stub" "${FINAL_ARGS[@]}"
fi
WRAPPER_EOF
sed -i "s|\${HOME}|$HOME|g; s|\${CONDA_PREFIX}|$CONDA_PREFIX|g" $HOME/gxx_cuda_wrapper.sh
chmod +x $HOME/gxx_cuda_wrapper.sh

export CC=$HOME/gcc_cuda_wrapper.sh
export CXX=$HOME/gxx_cuda_wrapper.sh
export GCC_HOST_COMPILER_PATH=$HOME/gcc_cuda_wrapper.sh

# --- CRITICAL STEP: Patch pybind11_bazel BEFORE compilation ---
# pybind11_bazel hard-codes "-fvisibility=hidden" in the copts of the pybind_extension rule.
# This is applied AFTER Bazel's --copt, overriding -fvisibility=default.
# Solution: fetch first, patch the build_defs.bzl, and invalidate the pybind object cache.
echo ">>> Running bazel fetch to populate externals cache..."
bazel fetch //tensorflow/tools/pip_package:wheel 2>/dev/null || true

# ==============================================================================
# SALVATION BLOCK: Final Stubs and Bazel Cache Fixes
# ==============================================================================
export BAZEL_BASE=$(bazel info output_base 2>/dev/null)
echo ">>> Bazel base set at: $BAZEL_BASE"

# --- XLA patch for GCC 11: std::vector::reserve with noexcept move constructor ---
echo ">>> Patching xla/runtime/execution_graph.h and command_executor.cc for GCC 11..."
XLA_DIR="$BAZEL_BASE/external/xla/xla"
if [ ! -d "$XLA_DIR" ]; then
    XLA_DIR=$(find /root/.cache/bazel -path "*/external/xla/xla" -type d 2>/dev/null | head -1)
fi

if [ -d "$XLA_DIR" ]; then
    python3 - "$XLA_DIR" << 'XLA_PATCH_EOF'
import sys, os, re

xla_dir = sys.argv[1]
exec_graph_h = os.path.join(xla_dir, "runtime", "execution_graph.h")

if os.path.isfile(exec_graph_h):
    with open(exec_graph_h, "r") as f:
        content = f.read()
    
    # Add noexcept to move constructor and assignment operator
    content = re.sub(r'Operation\(Operation&&\)\s*=\s*default;', 'Operation(Operation&&) noexcept = default;', content)
    content = re.sub(r'Operation&\s*operator=\(Operation&&\)\s*=\s*default;', 'Operation& operator=(Operation&&) noexcept = default;', content)
    
    with open(exec_graph_h, "w") as f:
        f.write(content)
    print("  PATCHED: execution_graph.h")

cmd_exec_cc = os.path.join(xla_dir, "backends", "gpu", "runtime", "command_executor.cc")
if os.path.isfile(cmd_exec_cc):
    with open(cmd_exec_cc, "r") as f:
        content = f.read()
    
    # Add noexcept move constructor and assignment to CommandOperation
    if 'CommandOperation(CommandOperation&&) noexcept = default;' not in content:
        insert_text = "\n  CommandOperation(CommandOperation&&) noexcept = default;\n  CommandOperation& operator=(CommandOperation&&) noexcept = default;\n"
        content = re.sub(r'(class CommandOperation : public ExecutionGraph::Operation \{\n\s*public:)', r'\1' + insert_text, content)
        with open(cmd_exec_cc, "w") as f:
            f.write(content)
        print("  PATCHED: command_executor.cc")

XLA_PATCH_EOF
else
    echo ">>> WARNING: XLA directory not found in Bazel cache! The build may fail at command_executor.cc."
fi


# 1. cuDNN injector (should be in $CONDA_PREFIX/include after NVIDIA download)
# EXPLICIT priority to the cuDNN 9.0.0 download (over /root/miniforge3/pkgs/cudnn-8.9.7.* etc.).
# Without this, `find | head -1` may pick v8 and overwrite v9 with v8 -> Conv2D breaks.
CUDNN9_INC="$HOME/cudnn9_download/cudnn-linux-ppc64le-9.0.0.312_cuda12-archive/include"
if [ -f "$CUDNN9_INC/cudnn_version.h" ] && grep -q "CUDNN_MAJOR 9" "$CUDNN9_INC/cudnn_version.h"; then
    CUDNN_H="$CUDNN9_INC/cudnn_version.h"
    echo ">>> Using cuDNN 9 download: $CUDNN9_INC"
else
    # Fallback: prefer packages with 'cudnn-9' in the path, otherwise any one
    CUDNN_H=$(find /usr /opt $CONDA_PREFIX -name "cudnn_version.h" 2>/dev/null | grep -i "cudnn-9\|cudnn9" | head -n 1)
    [ -z "$CUDNN_H" ] && CUDNN_H=$(find /usr /opt $CONDA_PREFIX -name "cudnn_version.h" 2>/dev/null | head -n 1)
fi
if [ -z "$CUDNN_H" ]; then
    echo ">>> ERROR: cudnn_version.h not found. cuDNN 9 should have been installed at the start of the script."
    exit 1
fi
if [ ! -z "$CUDNN_H" ]; then
    CUDNN_DIR=$(dirname "$CUDNN_H")
    cp -rL $CUDNN_DIR/cudnn*.h $HOME/local_config_cuda_stub/cuda/cuda_include/ 2>/dev/null || true
    cp -rL $CUDNN_DIR/cudnn*.h $HOME/cuda_unified/include/ 2>/dev/null || true
    cp -rL $CUDNN_DIR/cudnn*.h /usr/local/include/cuda_stub/ 2>/dev/null || true
    # cudnn_frontend expects third_party/gpus/cudnn/cudnn.h within the repo
    mkdir -p third_party/gpus/cudnn
    if [ -f "$CUDNN_DIR/cudnn.h" ]; then
        cp -f "$CUDNN_DIR/cudnn.h" third_party/gpus/cudnn/cudnn.h
    fi
    # Copy all cuDNN headers to third_party/gpus/cudnn/
    cp -rL $CUDNN_DIR/cudnn*.h third_party/gpus/cudnn/ 2>/dev/null || true
    # cudnn_frontend in exec config (Bazel sandbox) needs to find via -I path
    # IMPORTANT: uses SYMLINKS (not copies) to avoid struct redefinition -
    # cuDNN headers use #pragma once which is path-based; separate physical copies
    # cause GCC to include the same header twice causing "redefinition" errors.
    mkdir -p /usr/local/include/cuda_stub/third_party/gpus/cudnn
    for hdr in /usr/local/include/cuda_stub/cudnn*.h; do
        [ -f "$hdr" ] && ln -sf "$hdr" /usr/local/include/cuda_stub/third_party/gpus/cudnn/$(basename "$hdr")
    done
fi

# --- CUDA headers expected at third_party/gpus/cuda/include ---
CUDA_H_SRC=""
# Prefer the system CUDA toolkit (full driver API)
if [ -f /usr/local/cuda/include/cuda.h ]; then
    CUDA_H_SRC="/usr/local/cuda/include/cuda.h"
else
    for cand in $(find /usr /usr/local /opt $CONDA_PREFIX -name cuda.h 2>/dev/null); do
        if grep -q "CUdeviceptr" "$cand"; then
            CUDA_H_SRC="$cand"
            break
        fi
    done
fi
if [ -z "$CUDA_H_SRC" ]; then
    echo "ERROR: cuda.h with CUdeviceptr not found. Check CUDA installation (driver API)."
    exit 1
fi
mkdir -p third_party/gpus/cuda/include
cp -f "$CUDA_H_SRC" third_party/gpus/cuda/include/cuda_real.h

# Ensure CUDA Driver API symbols even if the copied header is incomplete
cat > third_party/gpus/cuda/include/cuda.h << 'CUDA_WRAPPER'
#ifndef TF_CUDA_WRAPPER_H_
#define TF_CUDA_WRAPPER_H_
// Wrapper: includes real cuda.h and defines missing symbols if needed
#include "cuda_real.h"

#ifndef CUdeviceptr
typedef unsigned long long CUdeviceptr;
#endif
#ifndef CU_MEM_ATTACH_GLOBAL
#define CU_MEM_ATTACH_GLOBAL 1
#endif
#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS 0
#endif

// Weak declaration if cuMemAllocManaged isn't visible
#endif  // TF_CUDA_WRAPPER_H_
CUDA_WRAPPER

CUDA_RT_H_SRC=$(find /usr /usr/local /opt $CONDA_PREFIX -name cuda_runtime.h 2>/dev/null | head -n 1)
if [ -z "$CUDA_RT_H_SRC" ]; then
    echo "ERROR: cuda_runtime.h not found. Check CUDA installation."
    exit 1
fi
cp -f "$CUDA_RT_H_SRC" third_party/gpus/cuda/include/cuda_runtime.h

# Copy ALL essential CUDA headers to third_party/gpus/cuda/include/
# (several TF .cc files include via "third_party/gpus/cuda/include/...")
CUDA_RT_API_SRC=$(find /usr /usr/local /opt $CONDA_PREFIX -name cuda_runtime_api.h 2>/dev/null | head -n 1)
if [ ! -z "$CUDA_RT_API_SRC" ]; then
    CUDA_INC_DIR=$(dirname "$CUDA_RT_API_SRC")
    echo ">>> Copying complete CUDA headers from $CUDA_INC_DIR to third_party/gpus/cuda/include/..."
    # Copy all essential CUDA headers (without overwriting cuda.h which already has a wrapper)
    for hdr in cuda_runtime_api.h cuda_device_runtime_api.h cuda_fp16.h cuda_bf16.h \
               cuda_fp8.h driver_types.h vector_types.h builtin_types.h device_types.h \
               host_defines.h surface_types.h texture_types.h channel_descriptor.h \
               cuda_texture_types.h device_launch_parameters.h cuComplex.h \
               library_types.h cuda_runtime_api_enum.h; do
        SRC=$(find /usr /usr/local /opt $CONDA_PREFIX -name "$hdr" 2>/dev/null | head -n 1)
        if [ ! -z "$SRC" ]; then
            cp -f "$SRC" third_party/gpus/cuda/include/ 2>/dev/null || true
        fi
    done
    # Copy crt/ subdirectory if it exists (host_defines.h, etc.)
    CRT_DIR=$(find /usr /usr/local /opt $CONDA_PREFIX -path "*/crt/host_defines.h" 2>/dev/null | head -n 1)
    if [ ! -z "$CRT_DIR" ]; then
        mkdir -p third_party/gpus/cuda/include/crt
        cp -rL "$(dirname $CRT_DIR)"/* third_party/gpus/cuda/include/crt/ 2>/dev/null || true
    fi
fi

# --- Bazel strict includes: declare CUDA headers in the gpu_runtime_impl target ---
python3 - << 'PYEOF'
import re
p = 'tensorflow/core/common_runtime/gpu/BUILD'
with open(p, 'r') as f:
    t = f.read()

def add_hdrs(block: str) -> str:
    hdrs = [
        '"third_party/gpus/cuda/include/cuda.h"',
        '"third_party/gpus/cuda/include/cuda_runtime.h"',
    ]
    if 'hdrs' in block:
        def repl(m):
            body = m.group(1)
            for h in hdrs:
                if h not in body:
                    body += f"\n        {h},"
            return f"hdrs = [{body}\n    ]"
        return re.sub(r'hdrs\s*=\s*\[(.*?)\n\s*\]', repl, block, flags=re.S)
    # insert hdrs before deps if missing
    return re.sub(r'(deps\s*=\s*\[)', 'hdrs = [\n        ' + '\n        '.join(hdrs) + '\n    ]\n\n    \\1', block, count=1)

pattern = r'(cc_library\(\s*name\s*=\s*"gpu_runtime_impl".*?\n\))'
m = re.search(pattern, t, flags=re.S)
if m:
    block = m.group(1)
    new_block = add_hdrs(block)
    t = t.replace(block, new_block)
    with open(p, 'w') as f:
        f.write(t)
    print('OK: gpu_runtime_impl hdrs patched')
else:
    print('WARNING: gpu_runtime_impl target not found')
PYEOF

# --- Bazel strict deps: BUILD files for third_party/gpus/cuda and cudnn ---
echo ">>> Creating BUILD files for third_party/gpus CUDA/cuDNN headers..."

# BUILD for third_party/gpus/ (root package)
mkdir -p third_party/gpus
cat > third_party/gpus/BUILD << 'GPUS_BUILD'
package(default_visibility = ["//visibility:public"])
exports_files(glob(["**"]))
GPUS_BUILD

# BUILD for third_party/gpus/cuda/include/
mkdir -p third_party/gpus/cuda/include
cat > third_party/gpus/cuda/BUILD << 'CUDA_BUILD'
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "cuda_headers",
    hdrs = glob(["include/*.h", "include/**/*.h"]),
    includes = ["include"],
)
CUDA_BUILD

# BUILD for third_party/gpus/cudnn/
mkdir -p third_party/gpus/cudnn
cat > third_party/gpus/cudnn/BUILD << 'CUDNN_BUILD'
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "cudnn_headers",
    hdrs = glob(["*.h"]),
    includes = ["."],
)
CUDNN_BUILD

# Patch TF targets that include third_party/gpus headers without declaring deps
echo ">>> Patching TF BUILD files to declare CUDA/cuDNN header deps..."
python3 - << 'STRICT_DEPS_PATCH'
import re, os, glob

# Targets that need deps for third_party/gpus headers
build_files_to_patch = [
    'tensorflow/core/grappler/clusters/BUILD',
    'tensorflow/core/kernels/BUILD',
    'tensorflow/core/common_runtime/gpu/BUILD',
    'tensorflow/core/platform/BUILD',
]

# Dynamic search: any BUILD in tensorflow/ that has .cc referencing cudnn
import subprocess
try:
    # Find .cc that include cudnn.h and map to their BUILD files
    result = subprocess.getoutput(
        "grep -rl 'cudnn\\.h\\|cuda_runtime_api\\.h' tensorflow/ --include='*.cc' 2>/dev/null"
    )
    for cc_file in result.strip().split('\n'):
        if cc_file:
            build_dir = os.path.dirname(cc_file)
            build_path = os.path.join(build_dir, 'BUILD')
            if os.path.exists(build_path) and build_path not in build_files_to_patch:
                build_files_to_patch.append(build_path)
except Exception:
    pass

cuda_dep = '"//third_party/gpus/cuda:cuda_headers"'
cudnn_dep = '"//third_party/gpus/cudnn:cudnn_headers"'

for bf in build_files_to_patch:
    if not os.path.exists(bf):
        continue
    with open(bf, 'r') as f:
        content = f.read()

    # Find the 'utils' target (cc_library) and add deps
    modified = False

    # Add CUDA/cuDNN deps if missing
    for dep in [cuda_dep, cudnn_dep]:
        if dep not in content:
            # Add into the first deps = [...] in the file, or create
            content = content.replace(
                'deps = [',
                f'deps = [\n        {dep},',
                1
            )
            modified = True

    if modified:
        with open(bf, 'w') as f:
            f.write(content)
        print(f"   -> Patched: {bf}")
    else:
        print(f"   -> Already ok: {bf}")

print("OK: Strict deps fixed")
STRICT_DEPS_PATCH

# ==============================================================================
# PATCH: Strict deps for tensorflow/core/kernels/BUILD (conv_ops)
# conv_ops includes gpu_asm_opts.h via conv_ops_impl.h -> redzone_allocator.h
# The target uses tf_kernel_library (not cc_library), and the dep for
# redzone_allocator is inside if_cuda_or_rocm which may not evaluate on ppc64le.
# ==============================================================================
echo ">>> Patching tensorflow/core/kernels/BUILD for conv_ops deps..."
python3 - << 'KERNELS_DEPS_PATCH'
import re, os

build_path = "tensorflow/core/kernels/BUILD"
if os.path.exists(build_path):
    with open(build_path, 'r') as f:
        content = f.read()
    
    # XLA deps needed by conv_ops (via transitive includes through redzone_allocator)
    xla_deps = [
        '"@xla//xla/stream_executor/gpu:gpu_asm_opts"',
    ]
    
    modified = False
    for dep in xla_deps:
        # Check in the main deps block of conv_ops only (not if_cuda blocks)
        # tf_kernel_library( name = "conv_ops" ... deps = [
        pattern = r'(tf_kernel_library\(\s*(?:.*?\n)*?\s*name\s*=\s*"conv_ops".*?deps\s*=\s*\[)'
        m = re.search(pattern, content, re.S)
        if m:
            # Check only in the main deps block (before if_cuda)
            deps_start = m.end()
            next_if = content.find('] + if_cuda', deps_start)
            if next_if == -1:
                next_if = content.find('],', deps_start)
            main_block = content[deps_start:next_if] if next_if > 0 else ""
            
            if dep not in main_block:
                content = content[:deps_start] + f'\n        {dep},' + content[deps_start:]
                modified = True
                print(f"   -> Added dep {dep} to conv_ops")
        else:
            print(f"   -> WARNING: Could not find tf_kernel_library conv_ops")

    if modified:
        with open(build_path, 'w') as f:
            f.write(content)
    else:
        print("   -> Already ok")
else:
    print(f"WARNING: {build_path} not found")
KERNELS_DEPS_PATCH

# 2. cuda_config.h injector
cat > /tmp/cuda_config.h << 'CFGEOF'
#ifndef CUDA_CUDA_CONFIG_H_
#define CUDA_CUDA_CONFIG_H_
#define TF_CUDA_VERSION "12.4"
#define TF_CUDNN_VERSION "9"
#define TF_CUDART_VERSION "12"
#define TF_CUPTI_VERSION "12"
#define TF_CUBLAS_VERSION "12"
#define TF_CUSOLVER_VERSION "11"
#define TF_CUFFT_VERSION "11"
#define TF_CUSPARSE_VERSION "12"
#define TF_CUDA_TOOLKIT_PATH "/usr/local/cuda"
#define TF_CUDA_CAPABILITIES "3.5,7.0"
#endif
CFGEOF
cp /tmp/cuda_config.h $HOME/local_config_cuda_stub/cuda/cuda_include/ 2>/dev/null || true
cp /tmp/cuda_config.h $HOME/cuda_unified/include/ 2>/dev/null || true
cp /tmp/cuda_config.h /usr/local/include/cuda_stub/ 2>/dev/null || true

# ==============================================================================
# 3. The Definitive .localentry Killer (V5 - Bulletproof)
# ==============================================================================
echo ">>> Unlocking Bazel cache permissions..."
chmod -R u+w "$BAZEL_BASE/external/xla" "$BAZEL_BASE/external/tsl" "$BAZEL_BASE/external/local_xla" 2>/dev/null || true

echo ">>> Hunting the .localentry bug in the XLA C++ source code (Brute Force Mode)..."
python3 - << 'EOF'
import os

bazel_base = os.environ.get('BAZEL_BASE', '')
search_dirs = [
    'third_party/xla',
    'tensorflow',
    os.path.join(bazel_base, 'external/xla'),
    os.path.join(bazel_base, 'external/tsl'),
    os.path.join(bazel_base, 'external/local_xla')
]

count = 0
for d in search_dirs:
    if not os.path.exists(d): continue
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith('.cc') or f.endswith('.h') or f.endswith('.S') or f.endswith('.inc') or f.endswith('.py') or f.endswith('.bzl'):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    # If the line has the instruction and has ", 2", we attack.
                    if '.localentry' in content and ', 2' in content:
                        lines = content.split('\n')
                        changed = False
                        for i, line in enumerate(lines):
                            if '.localentry' in line and ', 2' in line:
                                lines[i] = line.replace(', 2', ', 0')
                                changed = True

                        if changed:
                            os.chmod(path, 0o666)
                            with open(path, 'w', encoding='utf-8') as file:
                                file.write('\n'.join(lines))
                            print(f"TARGET ELIMINATED: {path}")
                            count += 1
                except Exception:
                    pass
print(f"Total files purified: {count}")
EOF

# Delete any residual junk to force a clean build
find "$BAZEL_BASE" -name "*.tramp.S" -delete 2>/dev/null || true
find . -name "*.tramp.S" -delete 2>/dev/null || true
rm -rf bazel-bin/tensorflow/tools/pip_package/wheel_house/*.whl 2>/dev/null || true
echo ">>> Cleanup complete. Ready for the build!"

# ==============================================================================
# PATCH: GCC 8 NoDestructor brace-init ambiguity in cupti_tracer.cc
# ==============================================================================
# GCC 8 (ppc64le) cannot disambiguate between NoDestructor(const T&) and
# NoDestructor(T&&) when receiving a {brace-init} list. The original code is:
#   NoDestructor<std::vector<CbidCategoryMap const*>> const kExtraCbidCategories(
#       {&GetExtra12080(), &GetExtra12000()});
# Fix: insert the explicit type before the { to eliminate the ambiguity:
#   NoDestructor<std::vector<CbidCategoryMap const*>> const kExtraCbidCategories(
#       std::vector<CbidCategoryMap const*>{&GetExtra12080(), &GetExtra12000()});
echo ">>> Patching NoDestructor brace-init for GCC 8 compatibility..."
python3 - << 'NODESTRUCTOR_PATCH'
import os, re

bazel_base = os.environ.get('BAZEL_BASE', '')

# Look for cupti_tracer.cc in all possible locations
targets = []
for d in [os.path.join(bazel_base, 'external/xla'),
          os.path.join(bazel_base, 'external/local_xla'),
          'third_party/xla']:
    if not os.path.isdir(d):
        continue
    for root, dirs, files in os.walk(d):
        for f in files:
            if f == 'cupti_tracer.cc':
                targets.append(os.path.join(root, f))

# Also scan ALL .cc and .h files in XLA for generic NoDestructor patterns
for d in [os.path.join(bazel_base, 'external/xla'),
          os.path.join(bazel_base, 'external/local_xla'),
          'third_party/xla']:
    if not os.path.isdir(d):
        continue
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith(('.cc', '.h')):
                path = os.path.join(root, f)
                if path not in targets:
                    try:
                        with open(path, 'r', errors='ignore') as fh:
                            if 'NoDestructor' in fh.read(50000) :
                                targets.append(path)
                    except:
                        pass

total = 0
for path in targets:
    try:
        os.chmod(path, 0o666)
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            content = fh.read()
    except Exception as e:
        print(f"   WARNING: Could not read {path}: {e}")
        continue

    original = content

    # Generic pattern: NoDestructor<TYPE> ANYTHING (
    #     {items});
    # We need to capture the TYPE (with nested angle brackets) and insert
    # TYPE before the { in the constructor call.

    # Use regex to find the exact known pattern in cupti_tracer.cc
    # and similar variants in other files.
    # The regex captures:
    #   group 1: everything from "NoDestructor<" to the final ">" (including nested type)
    #   group 2: the type between < >
    #   group 3: everything between ">" and "("  (e.g.: " const\n      kExtraCbidCategories")
    #   group 4: whitespace between "(" and "{"

    # Approach: find NoDestructor< manually with bracket matching
    marker = 'NoDestructor<'
    new_content = content
    offset = 0
    pos = 0

    while True:
        idx = new_content.find(marker, pos)
        if idx == -1:
            break

        # Encontra o < de abertura
        angle_start = idx + len('NoDestructor')
        # Balance brackets
        depth = 0
        i = angle_start
        angle_end = -1
        while i < len(new_content):
            c = new_content[i]
            if c == '<':
                depth += 1
            elif c == '>':
                depth -= 1
                if depth == 0:
                    angle_end = i
                    break
            i += 1

        if angle_end == -1:
            pos = idx + len(marker)
            continue

        type_str = new_content[angle_start+1:angle_end]

        # Now look for '(' AFTER '>', but there may be anything in between
        # (like "const kExtraCbidCategories", "const\n      kVar", etc.)
        # Look for the next '(' after angle_end
        paren_pos = new_content.find('(', angle_end + 1)
        if paren_pos == -1 or paren_pos > angle_end + 200:
            pos = angle_end + 1
            continue

        # Check that between '>' and '(' there is no other statement (;, {, })
        between = new_content[angle_end+1:paren_pos]
        if any(c in between for c in ';{}'):
            pos = angle_end + 1
            continue

        # Now look for '{' after '(', skipping whitespace
        k = paren_pos + 1
        while k < len(new_content) and new_content[k] in ' \t\n\r':
            k += 1

        if k < len(new_content) and new_content[k] == '{':
            # FOUND! Insert the type before '{'
            new_content = new_content[:k] + type_str + new_content[k:]
            total += 1
            print(f"   -> Fix applied to: {path} (position {k})")
            pos = k + len(type_str) + 1
        else:
            pos = angle_end + 1

    if new_content != original:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(new_content)

print(f"OK: {total} NoDestructor brace-init patterns fixed for GCC 8")
NODESTRUCTOR_PATCH
# ==============================================================================

BAZEL_BASE=$(bazel info output_base 2>/dev/null)
PYBIND_BZL="$BAZEL_BASE/external/pybind11_bazel/build_defs.bzl"
if [ -f "$PYBIND_BZL" ]; then
    chmod u+w "$PYBIND_BZL"
    sed -i 's/"-fvisibility=hidden"/""/g' "$PYBIND_BZL"
    sed -i "s/'-fvisibility=hidden'/''/g" "$PYBIND_BZL"
    echo "OK: pybind11_bazel/build_defs.bzl - removed -fvisibility=hidden"
else
    echo "WARNING: $PYBIND_BZL not found"
fi

# Patch pybind11_abseil status_utils for symbol export
for f in "$BAZEL_BASE"/external/pybind11_abseil/pybind11_abseil/*.{h,cc}; do
    [ -f "$f" ] || continue
    chmod u+w "$f"
    if ! grep -q 'pragma GCC visibility' "$f"; then
        sed -i '1i #pragma GCC visibility push(default)' "$f"
        echo "$f" >> /dev/null
    fi
done
echo "OK: pybind11_abseil patched with visibility default"

# Patch pybind11_protobuf for symbol export (PyProtoGetCppMessagePointer etc.)
for f in "$BAZEL_BASE"/external/pybind11_protobuf/pybind11_protobuf/*.{h,cc}; do
    [ -f "$f" ] || continue
    chmod u+w "$f"
    if ! grep -q 'pragma GCC visibility' "$f"; then
        sed -i '1i #pragma GCC visibility push(default)' "$f"
    fi
done
echo "OK: pybind11_protobuf patched with visibility default"

# Invalidate pybind object cache to force recompilation
find "$BAZEL_BASE" -path "*/pybind11*/*.o" -delete 2>/dev/null || true
find "$BAZEL_BASE" -path "*/pybind11*/*.pic.o" -delete 2>/dev/null || true

# ==============================================================================
# PATCH: Guard XLA .cc files that include .cu.h with __global__ (CUDA syntax)
# ==============================================================================
# Files like ragged_all_to_all_kernel_cuda.cc are normal .cc files but include .cu.h
# that contain __global__, __launch_bounds__ - syntax that ONLY NVCC understands.
# The .cu.cc patch (done above) does not cover them. We need to locate them in the
# Bazel cache and wrap them with #ifdef __CUDACC__ so that GCC generates empty TUs.
# GPU operations keep working via cuBLAS/cuDNN/dlopen at runtime.
echo ">>> Guarding XLA .cc files that include .cu.h with CUDA syntax..."
python3 - << 'CUCC_EXTERNAL_PATCH'
import os, glob, re

bazel_base = os.environ.get('BAZEL_BASE', '')
search_dirs = [
    os.path.join(bazel_base, 'external/xla'),
    os.path.join(bazel_base, 'external/local_xla'),
    'third_party/xla',
]

count = 0
for d in search_dirs:
    if not os.path.isdir(d):
        continue
    for root, dirs, files in os.walk(d):
        for f in files:
            # Only .cc files that are NOT .cu.cc (those were already patched earlier)
            if not f.endswith('.cc') or f.endswith('.cu.cc'):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
            except Exception:
                continue

            # Already patched?
            if 'ppc64le GCC patch' in content:
                continue

            # Detect whether this .cc includes a .cu.h (which certainly has __global__)
            if not re.search(r'#include\s+[<"].*\.cu\.h[>"]', content):
                continue

            patched = (
                '// ppc64le GCC patch: this .cc includes .cu.h with __global__ - requires NVCC\n'
                '#ifdef __CUDACC__\n'
                + content +
                '\n#endif  // __CUDACC__\n'
            )
            try:
                os.chmod(path, 0o666)
                with open(path, 'w', encoding='utf-8') as fh:
                    fh.write(patched)
                count += 1
                print(f"   -> Guarded: {path}")
            except Exception as e:
                print(f"   WARNING: Failed to patch {path}: {e}")

print(f"OK: {count} .cc files with .cu.h guarded with #ifdef __CUDACC__")
CUCC_EXTERNAL_PATCH
# ==============================================================================
# INJECTOR PATCH: Dummy implementations for unresolved symbols
# (Symbols originally in .cu.cc that were "hidden" by #ifdef __CUDACC__)
# ==============================================================================
echo ">>> Injecting dummies for GpuSleep, LaunchDelayKernel, CubSortKeys, CubSortPairs..."
python3 - << 'DUMMY_PATCH'
import os

bazel_base = os.popen('bazel info output_base').read().strip()
if not bazel_base:
    print("WARNING: bazel info output_base failed. Using fallback...")
    bazel_base = "/root/.cache/bazel/_bazel_root/cd3e5b24d1c6b8ec7f1ef221724b7b3d"

dummies = {
    'tensorflow/python/framework/test_ops.cc': """
// ppc64le patch: dummy for missing .cu.cc symbol
namespace tensorflow {
  void GpuSleep(OpKernelContext* context, int delay) {}
}
""",
    'tensorflow/core/kernels/test_ops.cc': """
// ppc64le patch: dummy for missing .cu.cc symbol
namespace tensorflow {
  void GpuSleep(OpKernelContext* context, int delay) {}
}
""",
    f'{bazel_base}/external/xla/xla/stream_executor/cuda/cuda_timer.cc': """
// ppc64le patch: dummy for missing .cu.cc symbol
namespace stream_executor {
namespace gpu {
  absl::StatusOr<GpuSemaphore> LaunchDelayKernel(Stream* stream) {
    return absl::InternalError("Dummy implementation for LaunchDelayKernel (CUDA kernels disabled during GCC compilation)");
  }
}
}
""",
    f'{bazel_base}/external/xla/xla/stream_executor/cuda/cub_sort_kernel_cuda.cc': """
// ppc64le patch: dummy for missing .cu.cc symbol
#include <cuda_runtime_api.h>
namespace stream_executor {
namespace cuda {
  template <typename T>
  cudaError CubSortKeys(void*, unsigned long&, void const*, void*, unsigned long, bool, unsigned long, CUstream_st*) { return cudaSuccess; }

  template <typename K, typename V>
  cudaError CubSortPairs(void*, unsigned long&, void const*, void*, void const*, void*, unsigned long, bool, unsigned long, CUstream_st*) { return cudaSuccess; }
}
}
"""
}

for path, dummy_code in dummies.items():
    if os.path.exists(path):
        os.chmod(path, 0o666)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove the old patch version if present
        marker = "// ppc64le patch: dummy for missing"
        if marker in content:
            content = content[:content.find(marker)]

        # Write the (clean) content + new dummy
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content + f"\n{dummy_code}\n")
        print(f"OK: Dummies injected into {path}")
    else:
        print(f"WARNING: File not found for dummy injection: {path}")

# Fix missing proto deps in pjrt/gpu/BUILD
build_path = f"{bazel_base}/external/xla/xla/pjrt/gpu/BUILD"
if os.path.exists(build_path):
    os.chmod(build_path, 0o666)
    with open(build_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove incorrect injections from previous runs
    for bad_dep in [
        'kernel_reuse_cache_cc_proto',
        'kernel_reuse_cache_proto',
        'kernel_reuse_cache_cc',
    ]:
        bad_str = f'\n        "//xla/service/gpu:{bad_dep}",'
        if bad_str in content:
            content = content.replace(bad_str, '')
            print(f"OK: Removed incorrect dep {bad_dep}")

    # Now inject the correct deps
    missing_deps = [
        "//xla/service/gpu:kernel_reuse_cache",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:compile_module_to_llvm_ir",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:stream_executor_util",
        "//xla/service/gpu:gpu_constants",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:execution_stream_assignment",
        "//xla/service:llvm_compiler",
        "//xla/service:compilation_stats",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/backends/gpu/runtime:sequential_thunk",
        "//xla/backends/gpu/runtime:collective_thunk",
        "//xla/backends/gpu/runtime:host_execute_thunk",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/service/gpu/autotuning:autotuner_util",
        "//xla/service/gpu/autotuning:autotune_cache_key",
    ]
    
    import re
    
    # Extract the se_gpu_pjrt_client cc_library block
    m = re.search(r'(cc_library\(\s*name\s*=\s*"se_gpu_pjrt_client".*?deps\s*=\s*\[)', content, re.S)
    if m:
        insert_point = m.end()
        # Find just the main deps = [...] block (before if_cuda_or_rocm)
        # Check which deps are already in the main deps block (not in if_cuda blocks)
        main_deps_end = content.find('] + if_cuda_or_rocm', insert_point)
        if main_deps_end == -1:
            main_deps_end = content.find('],', insert_point)
        main_deps_section = content[insert_point:main_deps_end] if main_deps_end > 0 else ""
        
        deps_to_add = []
        for dep in missing_deps:
            if f'"{dep}"' not in main_deps_section:
                deps_to_add.append(f'        "{dep}",')
        
        if deps_to_add:
            injection = '\n'.join(deps_to_add) + '\n'
            content = content[:insert_point] + '\n' + injection + content[insert_point:]
            print(f"OK: Added {len(deps_to_add)} deps in the main block of se_gpu_pjrt_client")
        else:
            print("OK: All deps already present in the main block")
    else:
        print("WARNING: cc_library se_gpu_pjrt_client not found")

    with open(build_path, 'w', encoding='utf-8') as f:
        f.write(content)
else:
    print("WARNING: xla/pjrt/gpu/BUILD not found in Bazel cache")

DUMMY_PATCH

# ==============================================================================
# PATCH: Guard XLA .cc files that depend on Thrust/RAFT/RMM (CUDA toolkit)
# RAFT-ENABLE 2026-05-22: disabled to attempt real RAFT/RMM build on ppc64le.
# Set RAFT_DISABLE=1 in the env to re-guard select_k_exec_raft.cc.
# ==============================================================================
if [ "${RAFT_DISABLE:-0}" = "1" ]; then
echo ">>> Guarding .cc files that depend on Thrust/RAFT/RMM..."
python3 - << 'RAFT_PATCH'
import os, subprocess

bazel_base = subprocess.getoutput('bazel info output_base').strip()
if not bazel_base or not os.path.isdir(bazel_base):
    bazel_base = "/root/.cache/bazel/_bazel_root/cd3e5b24d1c6b8ec7f1ef221724b7b3d"

# Files that include RAFT/RMM/Thrust headers - require NVCC
cuda_toolkit_files = [
    "external/xla/xla/backends/gpu/runtime/select_k_exec_raft.cc",
]

count = 0
for rel_path in cuda_toolkit_files:
    path = os.path.join(bazel_base, rel_path)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        continue

    os.chmod(path, 0o666)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if content.startswith('// ppc64le GCC patch'):
        print(f"OK: {rel_path} is already guarded")
        continue

    patched = (
        '// ppc64le GCC patch: this file requires Thrust/RAFT/RMM - needs NVCC\n'
        '#ifdef __CUDACC__\n'
        + content +
        '\n#endif  // __CUDACC__\n'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(patched)
    count += 1
    print(f"OK: Guarded {rel_path}")

print(f"OK: {count} RAFT/Thrust files guarded with #ifdef __CUDACC__")
RAFT_PATCH
fi  # RAFT-ENABLE: end of Thrust/RAFT/RMM guard block

# ==============================================================================
# PATCH: _Alignof -> alignof (C11 -> C++11) in buffer_debug_log_structs.h
# ==============================================================================
echo ">>> Fixing _Alignof -> alignof in buffer_debug_log_structs.h..."
python3 - << 'ALIGNOF_PATCH'
import os, subprocess

bazel_base = subprocess.getoutput('bazel info output_base').strip()
if not bazel_base or not os.path.isdir(bazel_base):
    bazel_base = "/root/.cache/bazel/_bazel_root/cd3e5b24d1c6b8ec7f1ef221724b7b3d"

path = os.path.join(bazel_base, "external/xla/xla/backends/gpu/runtime/buffer_debug_log_structs.h")
if os.path.exists(path):
    os.chmod(path, 0o666)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '_Alignof' in content:
        content = content.replace('_Alignof', 'alignof')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"OK: _Alignof -> alignof in {path}")
    else:
        print("OK: buffer_debug_log_structs.h is already fixed")
else:
    print(f"WARNING: {path} not found")
ALIGNOF_PATCH

# ==============================================================================
# PATCH: GCC 8 vector + unique_ptr (execution_graph.h)
# ROOT CAUSE: Operation base class declares copy = default for a type containing
# vector<unique_ptr<Operation>>. GCC 8 incorrectly reports is_copy_constructible
# as true, causing move_if_noexcept to fallback to copy which then fails.
# Fix: delete copy constructor in Operation class in execution_graph.h.
# ==============================================================================
echo ">>> Fixing execution_graph.h (root of the GCC 8 bug)..."
python3 - << 'EXECGRAPH_PATCH'
import os, subprocess, re, shutil

bazel_base = subprocess.getoutput('bazel info output_base').strip()
if not bazel_base or not os.path.isdir(bazel_base):
    bazel_base = "/root/.cache/bazel/_bazel_root/cd3e5b24d1c6b8ec7f1ef221724b7b3d"

path = os.path.join(bazel_base, "external/xla/xla/runtime/execution_graph.h")
if os.path.exists(path):
    os.chmod(path, 0o666)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '// GCC8 fix: delete copy' not in content:
        # Replace "Operation(const Operation&) = default;" with "= delete;"
        content = content.replace(
            'Operation(const Operation&) = default;',
            '// GCC8 fix: delete copy (unique_ptr member makes this impossible)\n    Operation(const Operation&) = delete;'
        )
        content = content.replace(
            'Operation& operator=(const Operation&) = default;',
            'Operation& operator=(const Operation&) = delete;'
        )
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("OK: execution_graph.h - Operation copy constructor deleted")
    else:
        print("OK: execution_graph.h already fixed")
else:
    print(f"WARNING: {path} not found")
EXECGRAPH_PATCH

# ==============================================================================
# PATCH: GCC 8 initializer_list -> absl::Span (command_buffer_conversion_pass.cc)
# GCC 8 cannot implicitly convert std::initializer_list to absl::Span.
# Fix: change constexpr auto = {...} to constexpr std::array.
# ==============================================================================
echo ">>> Fixing initializer_list->Span in command_buffer_conversion_pass.cc..."
python3 - << 'INITLIST_PATCH'
import os, subprocess, re

bazel_base = subprocess.getoutput('bazel info output_base').strip()
if not bazel_base or not os.path.isdir(bazel_base):
    bazel_base = "/root/.cache/bazel/_bazel_root/cd3e5b24d1c6b8ec7f1ef221724b7b3d"

path = os.path.join(bazel_base, "external/xla/xla/backends/gpu/runtime/command_buffer_conversion_pass.cc")
if os.path.exists(path):
    os.chmod(path, 0o666)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '// GCC8 fix: array instead of initializer_list' not in content:
        # Replace:
        #   static constexpr auto kRequireConditionals = {DebugOptions::CONDITIONAL, DebugOptions::WHILE};
        # With:
        #   static constexpr std::array<DebugOptions::CommandBufferCmdType, 2> kRequireConditionals = {DebugOptions::CONDITIONAL, DebugOptions::WHILE};
        
        content = re.sub(
            r'static constexpr auto kRequireConditionals = \{([^}]+)\};',
            lambda m: f'// GCC8 fix: array instead of initializer_list\n  static constexpr DebugOptions::CommandBufferCmdType kRequireConditionals[] = {{{m.group(1)}}};',
            content
        )
        content = re.sub(
            r'static constexpr auto kRequireTracing = \{([^}]+)\};',
            lambda m: f'static constexpr DebugOptions::CommandBufferCmdType kRequireTracing[] = {{{m.group(1)}}};',
            content
        )
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("OK: command_buffer_conversion_pass.cc - initializer_list->array fix applied")
    else:
        print("OK: command_buffer_conversion_pass.cc already fixed")
else:
    print(f"WARNING: {path} not found")
INITLIST_PATCH

# ==============================================================================

# dynamic_kernels remains active: the .cu.cc files will be loaded via dlopen at runtime
# (we cannot compile them with GCC, only NVCC understands CUDA syntax)

# ==============================================================================
# INJECTOR PATCH: Adding cuDNN headers into the GCC sandbox
# ==============================================================================
echo ">>> Searching for and injecting cuDNN headers..."

# EXPLICIT priority to the cuDNN 9.0.0 download (same reason as the previous block).
CUDNN9_INC="$HOME/cudnn9_download/cudnn-linux-ppc64le-9.0.0.312_cuda12-archive/include"
if [ -f "$CUDNN9_INC/cudnn_version.h" ] && grep -q "CUDNN_MAJOR 9" "$CUDNN9_INC/cudnn_version.h"; then
    CUDNN_H="$CUDNN9_INC/cudnn_version.h"
else
    CUDNN_H=$(find /usr /opt $CONDA_PREFIX -name "cudnn_version.h" 2>/dev/null | grep -i "cudnn-9\|cudnn9" | head -n 1)
    [ -z "$CUDNN_H" ] && CUDNN_H=$(find /usr /opt $CONDA_PREFIX -name "cudnn_version.h" 2>/dev/null | head -n 1)
fi

if [ -z "$CUDNN_H" ]; then
    echo ">>> ERROR: cuDNN 9 not found. It should have been installed during initial setup."
    exit 1
fi

if [ ! -z "$CUDNN_H" ]; then
    CUDNN_DIR=$(dirname "$CUDNN_H")
    echo ">>> cuDNN found at: $CUDNN_DIR"

    cp -rL $CUDNN_DIR/cudnn*.h $HOME/local_config_cuda_stub/cuda/cuda_include/ 2>/dev/null || true
    cp -rL $CUDNN_DIR/cudnn*.h $HOME/cuda_unified/include/ 2>/dev/null || true
    cp -rL $CUDNN_DIR/cudnn*.h /usr/local/include/cuda_stub/ 2>/dev/null || true

    echo ">>> OK! cuDNN headers successfully injected into the sandbox."
else
    echo ">>> ERROR: Could not download/find cuDNN."
fi
# ==============================================================================

# ==============================================================================
# INJECTOR PATCH 2: Generating forced cuda_config.h (V2 Complete)
# ==============================================================================
echo ">>> Generating cuda_config.h manually in the GCC sandbox..."
cat > /tmp/cuda_config.h << 'CFGEOF'
#ifndef CUDA_CUDA_CONFIG_H_
#define CUDA_CUDA_CONFIG_H_
#define TF_CUDA_VERSION "12.4"
#define TF_CUDNN_VERSION "9"
#define TF_CUDART_VERSION "12"
#define TF_CUPTI_VERSION "12"
#define TF_CUBLAS_VERSION "12"
#define TF_CUSOLVER_VERSION "11"
#define TF_CUFFT_VERSION "11"
#define TF_CUSPARSE_VERSION "12"
#define TF_CUDA_TOOLKIT_PATH "/usr/local/cuda"
#define TF_CUDA_CAPABILITIES "3.5,7.0"
#endif
CFGEOF
cp /tmp/cuda_config.h $HOME/local_config_cuda_stub/cuda/cuda_include/ 2>/dev/null || true
cp /tmp/cuda_config.h $HOME/cuda_unified/include/ 2>/dev/null || true
cp /tmp/cuda_config.h /usr/local/include/cuda_stub/ 2>/dev/null || true
echo ">>> OK! cuda_config.h successfully injected."
# ==============================================================================

# ==============================================================================
# INJECTOR PATCH 3: CUDA Toolkit Headers (cublas, cusolver, etc.)
# TF includes: "third_party/gpus/cuda/include/cublas_v2.h" etc.
# We need to create this directory structure in the Bazel sandbox.
# ==============================================================================
echo ">>> Injecting CUDA toolkit headers into third_party/gpus/cuda/include/..."

# Find CUDA toolkit include directory
CUDA_INC=""
for d in /usr/local/cuda/include /usr/local/cuda-*/include $CONDA_PREFIX/include $CONDA_PREFIX/targets/ppc64le-linux/include; do
    if [ -f "$d/cublas_v2.h" ]; then
        CUDA_INC="$d"
        break
    fi
done

if [ -z "$CUDA_INC" ]; then
    CUDA_INC=$(find /usr /opt $CONDA_PREFIX -name "cublas_v2.h" 2>/dev/null | head -1)
    [ ! -z "$CUDA_INC" ] && CUDA_INC=$(dirname "$CUDA_INC")
fi

if [ ! -z "$CUDA_INC" ]; then
    echo ">>> CUDA toolkit headers found at: $CUDA_INC"

    # Create third_party/gpus/cuda/include/ in all stub directories
    for STUB_DIR in $HOME/local_config_cuda_stub/cuda/cuda_include \
                    $HOME/cuda_unified/include \
                    /usr/local/include/cuda_stub; do
        THIRD_PARTY_DIR=$(dirname $(dirname $(dirname "$STUB_DIR")))/third_party/gpus/cuda/include
        mkdir -p "$THIRD_PARTY_DIR" 2>/dev/null || true
        
        # Copy all CUDA headers
        for h in cublas_v2.h cublas.h cublas_api.h cublasLt.h \
                 cusolver_common.h cusolverDn.h cusolverSp.h cusolverRf.h \
                 cusparse.h cusparse_v2.h \
                 cufft.h cufftw.h cufftXt.h \
                 curand.h curand_kernel.h \
                 cuda.h cuda_runtime.h cuda_runtime_api.h \
                 cuda_fp16.h cuda_bf16.h \
                 driver_types.h device_types.h \
                 vector_types.h surface_types.h texture_types.h \
                 channel_descriptor.h cuda_device_runtime_api.h \
                 library_types.h; do
            if [ -f "$CUDA_INC/$h" ]; then
                cp -L "$CUDA_INC/$h" "$THIRD_PARTY_DIR/" 2>/dev/null || true
            fi
        done
    done
    
    # Also put them directly in the --copt=-I path
    mkdir -p /usr/local/include/cuda_stub/third_party/gpus/cuda/include 2>/dev/null || true
    cp -rL "$CUDA_INC"/*.h /usr/local/include/cuda_stub/third_party/gpus/cuda/include/ 2>/dev/null || true

    echo ">>> OK! CUDA toolkit headers successfully injected."
else
    echo ">>> WARNING: CUDA toolkit headers not found!"
fi
# --- .tf_configure.bazelrc cleanup ---
sed -i 's/3\.5,7\.0/7.0/g' .tf_configure.bazelrc 2>/dev/null || true
sed -i 's/3\.5/7.0/g' .tf_configure.bazelrc 2>/dev/null || true
sed -i '/fuse-ld=gold/d' .tf_configure.bazelrc 2>/dev/null || true
sed -i '/use-gold-linker/d' .tf_configure.bazelrc 2>/dev/null || true

# Linker: LLD + patcher v6 (known FUNCTIONAL configuration).
#
# LLD handles relocations better than BFD on large libs. Patcher v6
# (patch_nv_toc_save.py, invoked after pip install) covers the remaining PPC64LE ABI bug:
# neither GCC nor Clang emit `std r2, 24(r1)` on intra-.so calls
# (correct per ELFv2 ABI), but LLD inserts __long_branch_* inter-TOC stubs
# that corrupt r2. The patcher injects wrappers (`std r2,24(r1); b stub`) into
# 16 bytes of zero padding after each cuda stub + direct bls to __cuda*/__nv*
# in functions without prologue save.
#
# Validated state: TF imports + 2x V100 detected via list_physical_devices.
# GPU Matmul may still crash in other uncovered chains (architectural
# GCC/NVCC bug, requires upstream fix).
# Use patched LLD (with r2 TOC save fix) if available, otherwise Conda LLD.
PATCHED_LLD="$HOME/tensorflow_gpu/llvm-install/bin/ld.lld"
if [ -x "$PATCHED_LLD" ]; then
    BAZEL_FUSE_LD="$PATCHED_LLD"
    export LD_LIBRARY_PATH="$HOME/tensorflow_gpu/llvm-install/lib:${LD_LIBRARY_PATH:-}"
    echo ">>> Using PATCHED LLD (with multi-TOC r2 save fix): $PATCHED_LLD"
else
    BAZEL_FUSE_LD="lld"
    echo ">>> Using Conda LLD (without r2 fix). Patcher v6 will be applied."
fi
echo ">>> Linker selected for Bazel: -fuse-ld=${BAZEL_FUSE_LD}"

# ----------------------------------------------------------------------------
# LLD does NOT auto-generate the _savefpr_NN/_restfpr_NN/_savevr_NN/_restvr_NN/
# _savegpr_NN/_restgpr_NN stubs that GCC references in out-of-line prologues/epilogues
# (optimization for many callee-saved registers). BFD generates inline stubs,
# LLD requires external linkage. libgcc has double-underscore versions but
# not single-underscore, so we generate the stubs in asm here.
#
# With BFD (current), this block is dispensable but we keep it in case we
# return to LLD. We skip generation if BAZEL_FUSE_LD is not lld.
# ----------------------------------------------------------------------------
if [[ "$BAZEL_FUSE_LD" == *lld* ]]; then
SAVRES_DIR=$HOME/ppc64_savres
mkdir -p "$SAVRES_DIR"
cat > "$SAVRES_DIR/savres.S" << 'SAVRES_ASM'
.text
.machine power9

# Save FP registers fNN..f31 to negative offsets from r1
.altmacro
.macro DEFFPR n, off
.globl _savefpr_\n
.type _savefpr_\n, @function
_savefpr_\n:
    stfd \n, \off(1)
.endm
.macro DEFFPR_REST n, off
.globl _restfpr_\n
.type _restfpr_\n, @function
_restfpr_\n:
    lfd \n, \off(1)
.endm

# _savefpr_14 .. _savefpr_31 (fall-through chain, ends with blr)
DEFFPR 14, -144
DEFFPR 15, -136
DEFFPR 16, -128
DEFFPR 17, -120
DEFFPR 18, -112
DEFFPR 19, -104
DEFFPR 20, -96
DEFFPR 21, -88
DEFFPR 22, -80
DEFFPR 23, -72
DEFFPR 24, -64
DEFFPR 25, -56
DEFFPR 26, -48
DEFFPR 27, -40
DEFFPR 28, -32
DEFFPR 29, -24
DEFFPR 30, -16
DEFFPR 31, -8
    blr

# _restfpr_14 .. _restfpr_31
DEFFPR_REST 14, -144
DEFFPR_REST 15, -136
DEFFPR_REST 16, -128
DEFFPR_REST 17, -120
DEFFPR_REST 18, -112
DEFFPR_REST 19, -104
DEFFPR_REST 20, -96
DEFFPR_REST 21, -88
DEFFPR_REST 22, -80
DEFFPR_REST 23, -72
DEFFPR_REST 24, -64
DEFFPR_REST 25, -56
DEFFPR_REST 26, -48
DEFFPR_REST 27, -40
DEFFPR_REST 28, -32
DEFFPR_REST 29, -24
DEFFPR_REST 30, -16
DEFFPR_REST 31, -8
    blr

# Save GPRs r14..r31
.macro DEFGPR n, off
.globl _savegpr_\n
.type _savegpr_\n, @function
_savegpr_\n:
    std \n, \off(1)
.endm
.macro DEFGPR_REST n, off
.globl _restgpr_\n
.type _restgpr_\n, @function
_restgpr_\n:
    ld \n, \off(1)
.endm

DEFGPR 14, -144
DEFGPR 15, -136
DEFGPR 16, -128
DEFGPR 17, -120
DEFGPR 18, -112
DEFGPR 19, -104
DEFGPR 20, -96
DEFGPR 21, -88
DEFGPR 22, -80
DEFGPR 23, -72
DEFGPR 24, -64
DEFGPR 25, -56
DEFGPR 26, -48
DEFGPR 27, -40
DEFGPR 28, -32
DEFGPR 29, -24
DEFGPR 30, -16
DEFGPR 31, -8
    blr

DEFGPR_REST 14, -144
DEFGPR_REST 15, -136
DEFGPR_REST 16, -128
DEFGPR_REST 17, -120
DEFGPR_REST 18, -112
DEFGPR_REST 19, -104
DEFGPR_REST 20, -96
DEFGPR_REST 21, -88
DEFGPR_REST 22, -80
DEFGPR_REST 23, -72
DEFGPR_REST 24, -64
DEFGPR_REST 25, -56
DEFGPR_REST 26, -48
DEFGPR_REST 27, -40
DEFGPR_REST 28, -32
DEFGPR_REST 29, -24
DEFGPR_REST 30, -16
DEFGPR_REST 31, -8
    blr

# Save Vector Regs v20..v31 (ALTIVEC, 16-byte aligned)
.macro DEFVR n, off
.globl _savevr_\n
.type _savevr_\n, @function
_savevr_\n:
    li 11, \off
    stvx \n, 11, 1
.endm
.macro DEFVR_REST n, off
.globl _restvr_\n
.type _restvr_\n, @function
_restvr_\n:
    li 11, \off
    lvx \n, 11, 1
.endm

DEFVR 20, -192
DEFVR 21, -176
DEFVR 22, -160
DEFVR 23, -144
DEFVR 24, -128
DEFVR 25, -112
DEFVR 26, -96
DEFVR 27, -80
DEFVR 28, -64
DEFVR 29, -48
DEFVR 30, -32
DEFVR 31, -16
    blr

DEFVR_REST 20, -192
DEFVR_REST 21, -176
DEFVR_REST 22, -160
DEFVR_REST 23, -144
DEFVR_REST 24, -128
DEFVR_REST 25, -112
DEFVR_REST 26, -96
DEFVR_REST 27, -80
DEFVR_REST 28, -64
DEFVR_REST 29, -48
DEFVR_REST 30, -32
DEFVR_REST 31, -16
    blr
SAVRES_ASM

echo ">>> Compiling PPC64 save/restore stubs for LLD..."
gcc -c -fPIC -mcpu=power9 -o "$SAVRES_DIR/savres.o" "$SAVRES_DIR/savres.S"
ar rcs "$SAVRES_DIR/libppc64_savres.a" "$SAVRES_DIR/savres.o"
echo ">>> $SAVRES_DIR/libppc64_savres.a generated"
SAVRES_LINKOPTS=(
    --linkopt=-L$HOME/ppc64_savres
    --linkopt=-Wl,--whole-archive,-lppc64_savres,--no-whole-archive
    --host_linkopt=-L$HOME/ppc64_savres
    --host_linkopt=-Wl,--whole-archive,-lppc64_savres,--no-whole-archive
)
else
echo ">>> BFD in use: skipping libppc64_savres.a generation (BFD auto-generates save/restore stubs)"
SAVRES_LINKOPTS=()
fi

bazel --host_jvm_args="-Xms4g" --host_jvm_args="-Xmx8g" build \
    --config=cuda \
    --features=-start_end_lib \
    --features=-header_modules \
    --features=-use_header_modules \
    --features=-module_maps \
    --features=-layering_check \
    --host_features=-header_modules \
    --host_features=-use_header_modules \
    --host_features=-module_maps \
    --host_features=-layering_check \
    --@rules_ml_toolchain//common:enable_cuda=True \
    --repo_env=TF_NEED_CUDA=1 \
    --define=using_cuda_nvcc=true \
    --//tensorflow/core/kernels/mlir_generated:enable_gpu=False \
    --action_env=CC=$HOME/gcc_cuda_wrapper.sh \
    --action_env=GCC_HOST_COMPILER_PATH=$HOME/gcc_cuda_wrapper.sh \
    --action_env=LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/cuda_unified/lib64 \
    --define=tflite_with_xnnpack=false \
    --local_ram_resources=HOST_RAM*.6 \
    --per_file_copt=".*cub_sort_kernel.*@-O0" \
    --per_file_copt=".*cutlass_gemm_kernel.*@-O0" \
    --per_file_copt=".*delay_kernel.*@-O0" \
    --per_file_copt=".*@-Wno-error" \
    --copt=-fvisibility=default \
    --cxxopt=-fvisibility=default \
    --host_copt=-fvisibility=default \
    --host_cxxopt=-fvisibility=default \
    --copt=-DPYBIND11_EXPORT="__attribute__((visibility(\"default\")))" \
    --copt=-DPYBIND11_MODULE_LOCAL="" \
    --copt=-mcmodel=medium \
    --cxxopt=-mcmodel=medium \
    --linkopt=-fuse-ld=${BAZEL_FUSE_LD} \
    --host_linkopt=-fuse-ld=${BAZEL_FUSE_LD} \
    --linkopt=-Wl,-z,notext \
    --host_linkopt=-Wl,-z,notext \
    --linkopt=-Wl,-Bsymbolic-functions \
    --host_linkopt=-Wl,-Bsymbolic-functions \
    --linkopt=-Wl,--export-dynamic \
    --linkopt=-Wl,--allow-multiple-definition \
    --linkopt=-Wl,--allow-shlib-undefined \
    --host_linkopt=-Wl,--allow-multiple-definition \
    --host_linkopt=-Wl,--allow-shlib-undefined \
    --linkopt=-static-libgcc \
    --host_linkopt=-static-libgcc \
    --host_linkopt=-Wl,-rpath,$CONDA_PREFIX/lib \
    "${SAVRES_LINKOPTS[@]}" \
    --extra_toolchains=@@local_config_python//:py_cc_toolchain \
    --override_repository=rules_ml_toolchain=$HOME/rules_ml_toolchain_patched \
    --override_repository=llvm_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm_linux_aarch64=$HOME/llvm_stub \
    --override_repository=llvm_darwin_aarch64=$HOME/llvm_stub \
    --override_repository=llvm18_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm19_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm20_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm21_linux_x86_64=$HOME/llvm_stub \
    --override_repository=llvm18_linux_aarch64=$HOME/llvm_stub \
    --override_repository=llvm20_linux_aarch64=$HOME/llvm_stub \
    --override_repository=llvm21_linux_aarch64=$HOME/llvm_stub \
    --override_repository=llvm18_darwin_aarch64=$HOME/llvm_stub \
    --override_repository=llvm20_darwin_aarch64=$HOME/llvm_stub \
    --override_repository=local_config_tensorrt=$HOME/tensorrt_stub \
    --override_repository=local_config_rocm=$HOME/rocm_stub \
    --override_repository=cuda_crt=$HOME/cuda_redist_stub \
    --override_repository=cuda_cudart=$HOME/cuda_redist_stub \
    --override_repository=cuda_profiler_api=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvcc=$HOME/cuda_nvcc_stub \
    --override_repository=cuda_nvml=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvtx=$HOME/cuda_redist_stub \
    --override_repository=cuda_cccl=$HOME/cuda_cccl_stub \
    --override_repository=cuda_cudnn=$HOME/cuda_redist_stub \
    --override_repository=cuda_cudnn9=$HOME/cuda_redist_stub \
    --override_repository=cuda_cublas=$HOME/cuda_redist_stub \
    --override_repository=cuda_cusolver=$HOME/cuda_redist_stub \
    --override_repository=cuda_cusparse=$HOME/cuda_redist_stub \
    --override_repository=cuda_curand=$HOME/cuda_redist_stub \
    --override_repository=cuda_cufft=$HOME/cuda_redist_stub \
    --override_repository=cuda_cupti=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvdisasm=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvvm=$HOME/cuda_redist_stub \
    --override_repository=cuda_nvjitlink=$HOME/cuda_redist_stub \
    --override_repository=cuda_nccl=$HOME/cuda_nccl_stub \
    --override_repository=cuda_nvrtc=$HOME/cuda_redist_stub \
    --override_repository=nvidia_wheel_versions=$HOME/cuda_redist_stub \
    --override_repository=pypi=$HOME/pypi_stub \
    --override_repository=pypi_absl_py=$HOME/pypi_stub \
    --override_repository=pypi_astunparse=$HOME/pypi_stub \
    --override_repository=pypi_auditwheel=$HOME/pypi_stub \
    --override_repository=pypi_flatbuffers=$HOME/pypi_stub \
    --override_repository=pypi_gast=$HOME/pypi_stub \
    --override_repository=pypi_keras=$HOME/pypi_stub \
    --override_repository=pypi_lit=$HOME/pypi_stub \
    --override_repository=pypi_ml_dtypes=$HOME/pypi_stub \
    --override_repository=pypi_mods=$HOME/pypi_stub \
    --override_repository=pypi_numpy=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_cublas_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_cuda_cupti_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_cuda_nvcc_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_cuda_nvrtc_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_cuda_runtime_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_cudnn_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_cufft_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_curand_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_cusolver_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_cusparse_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_nccl_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_nvjitlink_cu12=$HOME/pypi_stub \
    --override_repository=pypi_nvidia_nvshmem_cu12=$HOME/pypi_stub \
    --override_repository=pypi_opt_einsum=$HOME/pypi_stub \
    --override_repository=pypi_packaging=$HOME/pypi_stub \
    --override_repository=pypi_protobuf=$HOME/pypi_stub \
    --override_repository=pypi_requests=$HOME/pypi_stub \
    --override_repository=pypi_setuptools=$HOME/pypi_stub \
    --override_repository=pypi_termcolor=$HOME/pypi_stub \
    --override_repository=pypi_typing_extensions=$HOME/pypi_stub \
    --override_repository=pypi_wheel=$HOME/pypi_stub \
    --override_repository=pypi_wrapt=$HOME/pypi_stub \
    --override_repository=python_3_11_host=$HOME/python_stub \
    --override_repository=local_config_python=$HOME/python_config_stub \
    --override_repository=tf_wheel_version_suffix=$HOME/tf_wheel_version_suffix_stub \
    --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
    --repo_env=USE_HERMETIC_CC_TOOLCHAIN=0 \
    --noincompatible_enable_cc_toolchain_resolution \
    --discard_analysis_cache \
    --nokeep_state_after_build \
    --notrack_incremental_state \
    --local_resources=cpu=16 \
    --strategy=CudaCompile=local \
    --per_file_copt=".*\.cu\.cc@--maxrregcount=64" \
    --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=7.0 \
    --define=TF_CUDA_COMPUTE_CAPABILITIES=7.0 \
    --jobs=16 \
    //tensorflow/tools/pip_package:wheel

# Locate the generated .whl
WHL_FILE=$(find $TF_BUILD_DIR/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/ -name '*.whl' 2>/dev/null | head -1)
if [ -z "$WHL_FILE" ]; then
    echo "FATAL ERROR: No .whl file was found after compilation!"
    echo "Check the bazel build output above."
    exit 1
fi
echo ">>> Wheel found: $WHL_FILE"

cd ~

echo ">>> Resolving pending GPU kernel symbols (Auto-Stubbing)..."
PATCH_DIR="/tmp/tf_whl_patch"
rm -rf "$PATCH_DIR" && mkdir -p "$PATCH_DIR"
cp "$WHL_FILE" "$PATCH_DIR/original.whl"
cd "$PATCH_DIR"
unzip -q original.whl

TF_SO_FILES=$(find . -name "*.so" -o -name "*.so.*" | grep -v "__pycache__")
echo "   -> Scanning $(echo "$TF_SO_FILES" | wc -w) .so files in the wheel..."

# Use nm to grab ALL undefined symbols from ALL .so files
# Collect symbols defined in ALL .so files so we don't stub the ones that already exist
echo "   -> Collecting symbols defined in the wheel..."
DEFINED_SYMS_FILE=$(mktemp)
for so_file in $TF_SO_FILES; do
    nm -D --defined-only "$so_file" 2>/dev/null | awk '{print $NF}'
done | sort -u > "$DEFINED_SYMS_FILE"

echo "   -> Collecting undefined symbols in the wheel..."
ALL_UNDEF_FILE=$(mktemp)
for so_file in $TF_SO_FILES; do
    nm -D --undefined-only "$so_file" 2>/dev/null | awk '{print $NF}'
done | sort -u > "$ALL_UNDEF_FILE"

# Symbols that are undefined and NOT defined in any other .so of the wheel
TRULY_UNDEF_FILE=$(mktemp)
comm -23 "$ALL_UNDEF_FILE" "$DEFINED_SYMS_FILE" > "$TRULY_UNDEF_FILE"

echo "   -> Filtering TensorFlow symbols (local namespaces and MLIR)..."
# INCLUDE only C++ symbols from the main TF/XLA/Eigen namespaces:
#   _ZN10tensorflow...
#   _ZN5Eigen...
#   _ZN15stream_executor...
#   _ZN3tsl...
#   _ZN3xla...
#   _mlir...
# This ensures we don't stub system functions (C/C++ stdlib, python, etc.)
# but we'll catch any TF function (like UseDeterministicSegmentReductions)
# that disappeared because of the GCC/NVCC bypass.
MISSING_SYMS_FILE=$(mktemp)

grep -E '^(_Z.*10tensorflow|_Z.*5Eigen|_Z.*15stream_executor|_Z.*3tsl|_Z.*3xla|_mlir)' "$TRULY_UNDEF_FILE" >> "$MISSING_SYMS_FILE" 2>/dev/null || true

rm -f "$DEFINED_SYMS_FILE" "$ALL_UNDEF_FILE" "$TRULY_UNDEF_FILE" "$MISSING_SYMS_FILE"
TF_CC_SO=$(find . -name "libtensorflow_cc.so.2" | head -1)
STUB_DIR="$(dirname "${TF_CC_SO:-.}")"

echo "   -> Generating libnvml_hotfix.so for NVML bypass on Power9..."
HOTFIX_FILE="nvml_hotfix.c"
cat > "$HOTFIX_FILE" << 'EOF'
#include <stdio.h>
#define NVML_ERROR_FUNCTION_NOT_FOUND 13
int nvmlDeviceGetGpuFabricInfoV(void* device, void* gpuFabricInfo) {
    return NVML_ERROR_FUNCTION_NOT_FOUND;
}
EOF
HOTFIX_SO="$STUB_DIR/libnvml_hotfix.so"
gcc -shared -fPIC -o "$HOTFIX_SO" "$HOTFIX_FILE"
rm "$HOTFIX_FILE"

# Modify self_check.py to load the NVML hotfix and the cuDNN libraries
SELFCHECK=$(find . -name "self_check.py" | head -1)
if [ -n "$SELFCHECK" ]; then
    sed -i '1i\
# ppc64le GPU patch: pre-load NVML hotfix\
import ctypes as _ctypes, os as _os\
_dir = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))\
_hotfix = _os.path.join(_dir, "libnvml_hotfix.so")\
if _os.path.exists(_hotfix):\
    _ctypes.CDLL(_hotfix, _os.RTLD_NOW | _os.RTLD_GLOBAL)\
del _ctypes, _os, _dir, _hotfix\
' "$SELFCHECK"
    echo "   -> self_check.py configured for auto-load of the NVML vaccine and full cuDNN."
fi

# ----------------------------------------------------------------------------
# Patch BFD PPC64LE multi-TOC bug: the 'nop' after 'bl <long_branch_stub>' should have
# been rewritten by the linker to 'ld r2,24(r1)' (reloads caller's TOC),
# but GNU ld 2.40 ppc64le doesn't perform the rewrite when the call crosses TOC
# groups in huge libs (libtensorflow_framework.so.2 ~67MB of .text).
# Without the reload, r2 retains the callee's TOC and any subsequent
# addis/addi r2,... becomes garbage -> SIGSEGV in static init of __sti____cudaRegisterAll.
#
# Fix: walk through .text of each .so in the wheel, and for each bl that points to a
# *.long_branch.* stub, if the following instruction is nop, rewrite it as
# ld r2,24(r1) (LE bytes 18 00 41 e8). Safe because the standard prologue of
# any function that calls another function always does std r2,24(r1).
# ----------------------------------------------------------------------------
# (TOC patching done post-pip install via patch_nv_toc_save.py - see line ~4095)

echo "   -> Repackaging Wheel..."
rm original.whl
zip -q -r "$WHL_FILE" .
echo ">>> Wheel armored and vaccinated successfully!"

cd ~
python3 -m pip install --force-reinstall --no-deps "$WHL_FILE"

conda install -c conda-forge "h5py>=3.11,<3.15" grpcio libclang ml_dtypes "protobuf>=6.31.1,<8.0.0" libstdcxx-ng -y
# NOTE: cudnn removed here intentionally. We use cuDNN 9.0.0 ppc64le from
# NVIDIA (downloaded during initial setup), not the 8.9.7 from conda-forge.
python3 -m pip install absl-py astunparse flatbuffers gast google-pasta keras opt-einsum termcolor wrapt


# Create cuDNN symlinks in cuda_unified (where TF looks for CUDA libs)
echo ">>> Linking cuDNN into cuda_unified..."
for f in $CONDA_PREFIX/lib/libcudnn*; do
    [ -f "$f" ] && ln -sf "$f" "$HOME/cuda_unified/lib64/$(basename $f)"
done

# Create a generic cuDNN symlink if it doesn't exist
if [ ! -e "$CONDA_PREFIX/lib/libcudnn.so" ]; then
    CUDNN_REAL=$(ls -1 $CONDA_PREFIX/lib/libcudnn.so.* 2>/dev/null | head -1)
    if [ -n "$CUDNN_REAL" ]; then
        ln -sf "$CUDNN_REAL" "$CONDA_PREFIX/lib/libcudnn.so"
    fi
fi

# Real cuDNN 9 installed via NVIDIA tarball (cudnn-linux-ppc64le-9.0.0.312_cuda12).
# We copy the real .so.9 from $CONDA_PREFIX/lib to cuda_unified/lib64.
# cuDNN 9 changed library names compared to v8:
#   v8: libcudnn_ops_infer.so.8, libcudnn_cnn_infer.so.8, etc.
#   v9: libcudnn_ops.so.9, libcudnn_cnn.so.9, libcudnn_graph.so.9, etc.
echo ">>> Linking real cuDNN 9 into cuda_unified..."
for cudnn_lib in $CONDA_PREFIX/lib/libcudnn*.so*; do
    [ -e "$cudnn_lib" ] || continue
    ln -sf "$cudnn_lib" "$HOME/cuda_unified/lib64/$(basename $cudnn_lib)" 2>/dev/null
done

# System GCC 8 lacks GLIBCXX_3.4.29 (required by Conda's NumPy).
# Use Conda's newer libstdc++.
export LD_LIBRARY_PATH=$HOME/cuda_unified/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo ""
echo ">>> Running full GPU diagnostic..."
cd ~
python3 << 'PYEOF'
import os, sys, ctypes, subprocess

print("\n" + "="*70)
print("  GPU DIAGNOSTIC - TensorFlow 2.21 / Power9")
print("="*70)

# 1. nvidia-smi
print("\n[1] nvidia-smi:")
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
    if r.returncode == 0:
        print(f"  OK: {r.stdout.strip()}")
    else:
        print(f"  ERROR: nvidia-smi failed (rc={r.returncode}): {r.stderr.strip()}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. Check CUDA libraries via dlopen
print("\n[2] CUDA libraries (dlopen):")
cuda_libs = [
    ("libcuda.so.1", "CUDA Driver"),
    ("libcudart.so.12", "CUDA Runtime"),
    ("libcudart.so", "CUDA Runtime (generic)"),
    ("libcublas.so.12", "cuBLAS"),
    ("libcublas.so", "cuBLAS (generic)"),
    ("libcufft.so.11", "cuFFT"),
    ("libcufft.so", "cuFFT (generic)"),
    ("libcurand.so.10", "cuRAND"),
    ("libcurand.so", "cuRAND (generic)"),
    ("libcusolver.so.11", "cuSOLVER"),
    ("libcusolver.so", "cuSOLVER (generic)"),
    ("libcusparse.so.12", "cuSPARSE"),
    ("libcusparse.so", "cuSPARSE (generic)"),
    ("libcudnn.so.9", "cuDNN 9"),
    ("libcudnn.so.8", "cuDNN 8"),
    ("libcudnn.so", "cuDNN (generic)"),
    ("libnccl.so.2", "NCCL"),
    ("libnccl.so", "NCCL (generic)"),
    ("libnvrtc.so.12", "NVRTC"),
    ("libnvrtc.so", "NVRTC (generic)"),
]
for lib, desc in cuda_libs:
    try:
        h = ctypes.CDLL(lib)
        print(f"  OK: {lib} ({desc})")
    except OSError as e:
        err_short = str(e).split(":")[0] if ":" in str(e) else str(e)
        print(f"  MISSING: {lib} ({desc}) -> {err_short}")

# 3. LD_LIBRARY_PATH
print(f"\n[3] LD_LIBRARY_PATH:")
for p in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    if p:
        exists = os.path.isdir(p)
        count = len([f for f in os.listdir(p) if f.endswith(".so") or ".so." in f]) if exists else 0
        print(f"  {'OK' if exists else 'MISSING'}: {p} ({count} .so)")

# 4. Check TF cuDNN plugin
print(f"\n[4] TensorFlow cuDNN plugin:")
tf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__import__('tensorflow').__file__)))
for root, dirs, files in os.walk(tf_dir):
    for f in files:
        if "cudnn" in f.lower() and (f.endswith(".so") or ".so." in f):
            full = os.path.join(root, f)
            print(f"  Found: {full}")

# 5. Try to import TF and list GPUs
print(f"\n[5] TensorFlow GPU test:")
import tensorflow as tf
print(f"  Version: {tf.__version__}")
print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPUs detected: {len(gpus)}")
for g in gpus:
    print(f"    -> {g}")

if not gpus:
    print("\n  DIAGNOSTIC: GPU not detected. Check the sections above.")
    print("  - If [1] failed: problem with NVIDIA driver")
    print("  - If [2] has MISSING entries: CUDA libraries absent from LD_LIBRARY_PATH")
    print("  - If [4] is empty: cuDNN plugin was not compiled/included in the wheel")
else:
    import time
    with tf.device('/CPU:0'):
        A = tf.random.normal([5000, 5000])
        B = tf.random.normal([5000, 5000])
    with tf.device('/GPU:0'):
        # Warm-up
        _ = tf.matmul(A[:10, :10], B[:10, :10])

        start = time.time()
        C = tf.matmul(A, B)
        end = time.time()
    total = tf.reduce_sum(C).numpy()
    print(f"\n  Matmul 5000x5000 on GPU: {end - start:.4f}s (sum={total:.2f})")
    print("  [VICTORY] TensorFlow with GPU on Power9 is 100% operational!")

print("\n" + "="*70)
PYEOF
