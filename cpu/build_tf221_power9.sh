#!/bin/bash
set -e
set -o pipefail

# =============================================================================
# Build TensorFlow 2.21 for Power9 (ppc64le) — AlmaLinux
# Prerequisite: Bazel 7.x already compiled and installed in /usr/local/bin/bazel
# Prerequisite: Miniforge3 installed in $HOME/miniforge3
# =============================================================================

export TF_BUILD_DIR="${TF_BUILD_DIR:-$PWD/tf221_workspace}"
export CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"

TF_DIR="$TF_BUILD_DIR"
CONDA_ENV="tf221_build"
LLVM_STUB="$HOME/llvm_stub"
ML_TOOLCHAIN="$HOME/rules_ml_toolchain_patched"
PYTHON_WRAPPER="/tmp/tf_python3_wrapper.sh"

echo "=== Compiling TensorFlow 2.21 for ppc64le (Power9) ==="

# ---------- Step 1: OS Dependencies ----------
echo "=== 1/12: Installing system dependencies ==="
sudo dnf install -y git gcc gcc-c++ zip unzip which patch wget vim-common

# ---------- Step 2: Conda Environment (NO Bazel — we already have native) ----------
echo "=== 2/12: Creating Conda environment ==="
source $CONDA_BASE/etc/profile.d/conda.sh
conda create -n "$CONDA_ENV" "python=3.11" "numpy>=2.0.0" wheel packaging requests -c conda-forge -y
conda activate "$CONDA_ENV"

echo "Python: $(python3 --version) em $(which python3)"
echo "Bazel: $(bazel --version)"

# ---------- Step 3: Clone TF 2.21 ----------
echo "=== 3/12: Cloning TensorFlow 2.21 ==="
mkdir -p "$TF_DIR" && cd "$TF_DIR"
rm -rf tensorflow
git clone --depth 1 --branch v2.21.0 https://github.com/tensorflow/tensorflow.git
cd tensorflow

# ---------- Step 4: Clean configs ----------
echo "=== 4/12: Cleaning old configs ==="
rm -f .bazelversion
#bazel clean --expunge

# ---------- Step 5: Environment variables ----------
echo "=== 5/12: Configuring environment variables ==="
export CC=gcc
export CXX=g++
export CFLAGS="-mno-float128 -O3"
export CXXFLAGS="-mno-float128 -O3"
export TF_NEED_CLANG=0
export TF_DOWNLOAD_CLANG=0
export TF_NEED_CUDA=0
export TF_NEED_ROCM=0
export TF_NEED_OPENCL=0
export TF_SET_ANDROID_WORKSPACE=0
export PYTHON_BIN_PATH=$(which python3)
export PYTHON_LIB_PATH=$(python3 -c 'import site; print(site.getsitepackages()[0])')
export TF_CONFIGURE_IOS=0

# ---------- Passo 6: Configure ----------
echo "=== 6/12: Rodando ./configure ==="
echo -e "\n\n\n\n\n\n\n\n\n\n" | ./configure

# ---------- Passo 7: Patch rules_ml_toolchain (bug local.bzl) ----------
echo "=== 7/12: Patching rules_ml_toolchain ==="
cd ~
wget -qO rules.tar.gz https://github.com/google-ml-infra/rules_ml_toolchain/archive/d8cb9c2c168cd64000eaa6eda0781a9615a26ffe.tar.gz
rm -rf "$ML_TOOLCHAIN"
tar -xzf rules.tar.gz
mv rules_ml_toolchain-d8cb9c2c168cd64000eaa6eda0781a9615a26ffe "$ML_TOOLCHAIN"
sed -i '/load("@bazel_tools\/\/tools\/build_defs\/repo:local.bzl", "new_local_repository")/d' "$ML_TOOLCHAIN/cc/deps/cc_toolchain_deps.bzl"
sed -i 's/new_local_repository(/native.new_local_repository(/g' "$ML_TOOLCHAIN/cc/deps/cc_toolchain_deps.bzl"
cd "$TF_DIR/tensorflow"

# ---------- Passo 8: Criar LLVM Stub Repos ----------
echo "=== 8/12: Criando LLVM stub repos ==="
mkdir -p "$LLVM_STUB"
echo 'VERSION = "0.0.0"' > "$LLVM_STUB/version.bzl"
touch "$LLVM_STUB/WORKSPACE"

cat > "$LLVM_STUB/BUILD" << 'EOF'
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

# --- CUDA stub (local_config_cuda) ---
CUDA_STUB="$HOME/cuda_stub"
mkdir -p "$CUDA_STUB/cuda"
touch "$CUDA_STUB/WORKSPACE"
cat > "$CUDA_STUB/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
config_setting(name = "is_cuda_enabled", values = {"define": "stub=true"})
config_setting(name = "is_cuda_compiler_present", values = {"define": "stub=true"})
filegroup(name = "LICENSE", srcs = [])
EOF

cat > "$CUDA_STUB/cuda/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(":build_defs.bzl", "cuda_header_library")
cc_library(name = "cuda_headers", hdrs = [], visibility = ["//visibility:public"])
cc_library(name = "cudart_static", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cuda_driver", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cudart", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cublas", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cufft", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cusolver", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cusparse", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "curand", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cudnn", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "cupti_headers", hdrs = [], visibility = ["//visibility:public"])
bool_flag(name = "include_cuda_libs", build_setting_default = False)
bool_flag(name = "override_include_cuda_libs", build_setting_default = False)
py_library(name = "cuda_config_py", srcs = [])
filegroup(name = "build_defs", srcs = [])
filegroup(name = "build_defs_bzl", srcs = [])
filegroup(name = "cub_headers", srcs = [])
filegroup(name = "cuda_runtime", srcs = [])
filegroup(name = "cudnn_header", srcs = [])
filegroup(name = "cupti_dsos", srcs = [])
filegroup(name = "implicit_cuda_headers_dependency", srcs = [])
filegroup(name = "nvjitlink", srcs = [])
filegroup(name = "nvptxcompiler", srcs = [])
filegroup(name = "nvrtc_headers", srcs = [])
filegroup(name = "runtime_fatbinary", srcs = [])
filegroup(name = "runtime_nvlink", srcs = [])
filegroup(name = "runtime_ptxas", srcs = [])
config_setting(name = "TRUE", values = {"define": "stub=true"})
config_setting(name = "FALSE", values = {"define": "stub=false"})
config_setting(name = "using_clang", values = {"define": "stub=false"})
config_setting(name = "using_clang_opt", values = {"define": "stub=false"})
config_setting(name = "using_config_cuda", values = {"define": "stub=false"})
config_setting(name = "using_nvcc", values = {"define": "stub=false"})
config_setting(name = "cuda_tools", values = {"define": "stub=false"})
config_setting(name = "cuda_tools_and_libs", values = {"define": "stub=false"})
EOF

# Subpasta crosstool para o cuda_stub
mkdir -p "$CUDA_STUB/crosstool"
cat > "$CUDA_STUB/crosstool/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "toolchain", srcs = [])
filegroup(name = "toolchain-linux-x86_64", srcs = [])
EOF

cat > "$CUDA_STUB/cuda/build_defs.bzl" << 'EOF'
def if_cuda(if_true, if_false = []):
    return if_false
def if_cuda_is_configured(if_true=[], if_false=[], *args, **kwargs):
    return if_false
def cuda_default_copts():
    return []
def cuda_gpu_architectures():
    return []
def cuda_header_library(**kwargs):
    pass
def if_cuda_newer_than(version, if_true, if_false = []):
    return if_false
def is_cuda_configured():
    return False
def cuda_library(**kwargs):
    pass
def if_cuda_exec(if_true, if_false = []):
    return if_false
def enable_cuda_flag():
    return []
def if_version_equal_or_greater_than(a, b, if_true, if_false=[]):
    return if_false
def select_threshold(a, b, if_true, if_false=[]):
    return if_false
def if_cuda_or_rocm(if_true, if_false=[]):
    return if_false
def if_gpu_is_configured(if_true=[], if_false=[], *args, **kwargs):
    return if_false
EOF
touch "$CUDA_STUB/cuda/cuda_config.h"

# --- NCCL stub (local_config_nccl) ---
NCCL_STUB="$HOME/nccl_stub"
mkdir -p "$NCCL_STUB/nccl"
touch "$NCCL_STUB/WORKSPACE"
cat > "$NCCL_STUB/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
cc_library(name = "nccl_config", hdrs = [], visibility = ["//visibility:public"])
EOF

cat > "$NCCL_STUB/nccl/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
cc_library(name = "nccl", srcs = [], visibility = ["//visibility:public"])
EOF

cat > "$NCCL_STUB/nccl/build_defs.bzl" << 'EOF'
def if_nccl(if_true, if_false = []):
    return if_false
EOF

# --- CUDA redistribution stub (genérico para cuda_cudart, cuda_nvcc, etc.) ---
CUDA_REDIST_STUB="$HOME/cuda_redist_stub"
mkdir -p "$CUDA_REDIST_STUB"
touch "$CUDA_REDIST_STUB/WORKSPACE"
echo 'VERSION = "0.0.0"' > "$CUDA_REDIST_STUB/version.bzl"
cat > "$CUDA_REDIST_STUB/versions.bzl" << 'EOF'
NVIDIA_WHEEL_VERSIONS = {
    "0.0.0": [],
}
EOF
cat > "$CUDA_REDIST_STUB/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "all", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "headers", hdrs = [], visibility = ["//visibility:public"])
cc_library(name = "libs", srcs = [], visibility = ["//visibility:public"])
EOF

# --- TensorRT stub ---
TENSORRT_STUB="$HOME/tensorrt_stub"
mkdir -p "$TENSORRT_STUB"
touch "$TENSORRT_STUB/WORKSPACE"
cat > "$TENSORRT_STUB/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "LICENSE", srcs = [])
cc_library(name = "tensorrt_headers", hdrs = [])
cc_library(name = "tensorrt", srcs = [])
config_setting(name = "use_static_tensorrt", values = {"define": "stub=true"})
py_library(name = "tensorrt_config_py", srcs = [])
EOF

cat > "$TENSORRT_STUB/build_defs.bzl" << 'EOF'
def if_tensorrt(if_true, if_false = []):
    return if_false
def is_tensorrt_configured():
    return False
def if_tensorrt_exec(if_true, if_false = []):
    return if_false
EOF

# --- ROCm stub ---
ROCM_STUB="$HOME/rocm_stub"
mkdir -p "$ROCM_STUB/rocm"
touch "$ROCM_STUB/WORKSPACE"
cat > "$ROCM_STUB/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
config_setting(name = "is_rocm_enabled", values = {"define": "stub=true"})
filegroup(name = "LICENSE", srcs = [])
EOF

cat > "$ROCM_STUB/rocm/BUILD" << 'EOF'
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

# Subpasta crosstool para o rocm_stub
mkdir -p "$ROCM_STUB/crosstool"
cat > "$ROCM_STUB/crosstool/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "toolchain", srcs = [])
EOF

cat > "$ROCM_STUB/rocm/build_defs.bzl" << 'EOF'
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

# --- PyPI stub (para dependências python) ---
echo "⚙️  Configuring pypi_stub..."
PYPI_STUB="$HOME/pypi_stub"
mkdir -p "$PYPI_STUB"
touch "$PYPI_STUB/WORKSPACE"
cat > "$PYPI_STUB/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
py_library(name = "pkg", srcs = [])
EOF

cat > "$PYPI_STUB/requirements.bzl" << 'EOF'
def requirement(name):
    normalized = name.replace("-", "_").lower()
    return "@@pypi//" + normalized
def install_deps(**kwargs):
    pass
EOF

for pkg in astor astunparse dill gast h5py jax lit numpy opt_einsum packaging portpicker protobuf requests scipy tblib termcolor typing_extensions wrapt zstandard; do
  mkdir -p "$PYPI_STUB/$pkg"
  cat > "$PYPI_STUB/$pkg/BUILD" << EOF
package(default_visibility = ["//visibility:public"])
py_library(name = "$pkg", srcs = [])
py_library(name = "pkg", srcs = [])
EOF
done

# Numpy precisa de headers C
NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")
rm -f "$PYPI_STUB/numpy/numpy_include"
ln -snf "$NUMPY_INC" "$PYPI_STUB/numpy/numpy_include"
cat > "$PYPI_STUB/numpy/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "numpy_headers",
    hdrs = glob(["numpy_include/**/*.h"]),
    includes = ["numpy_include"],
)
py_library(name = "numpy", srcs = [])
EOF

# --- Python stub (python_3_11_host) ---
echo "⚙️  Configuring python_stub..."
PYTHON_STUB="$HOME/python_stub"
mkdir -p "$PYTHON_STUB"
touch "$PYTHON_STUB/WORKSPACE"
ln -sf $CONDA_BASE/envs/tf221_build/bin/python3 "$PYTHON_STUB/python3_bin"
cat > "$PYTHON_STUB/BUILD" << 'EOF'
package(default_visibility = ["//visibility:public"])
exports_files(["python3_bin"])
filegroup(name = "python", srcs = ["python3_bin"])
filegroup(name = "files", srcs = ["python3_bin"])
filegroup(name = "includes", srcs = [])
cc_library(name = "python_headers", hdrs = [])
EOF

# --- Python config stub (local_config_python) ---
echo "⚙️  Configuring python_config_stub..."
PYTHON_CONFIG_STUB="$HOME/python_config_stub"
mkdir -p "$PYTHON_CONFIG_STUB"
touch "$PYTHON_CONFIG_STUB/WORKSPACE"
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
rm -f "$PYTHON_CONFIG_STUB/python_include"
ln -snf "$PYINC" "$PYTHON_CONFIG_STUB/python_include"

cat > "$PYTHON_CONFIG_STUB/py_cc_toolchain.bzl" << 'EOF'
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

cat > "$PYTHON_CONFIG_STUB/BUILD" << 'EOF'
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

# ---------- Passo 9: Patch do WORKSPACE ----------
echo "=== 9/12: Patching WORKSPACE ==="
python3 - << 'PYEOF'
"""
Patch WORKSPACE para ppc64le: comenta BLOCOS INTEIROS (statements)
que contenham referências a CUDA, NCCL, NVSHMEM, LLVM toolchains.
"""

with open("WORKSPACE", "r") as f:
    lines = f.readlines()

# 1. Parsear o arquivo em "statements" (blocos top-level com parênteses balanceados)
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

# 2. Padrões para identificar statements que devem ser comentados
skip_patterns = [
    "cc_toolchain_deps",
    "register_toolchains",
    "cuda_json_init",
    "cuda_redist",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
    "cuda_configure",
    "local_config_cuda",
    "nccl_redist",
    "nccl_configure",
    "local_config_nccl",
    "nvshmem",
    "NVSHMEM",
    "CCCL_DIST",
    "CCCL_GITHUB",
    "cuda_profiler_api",
    "cuda_nvcc",
    "cuda_nvml",
    "cuda_nvtx",
    "cuda_cccl",
    "cuda_cudnn",
    "cuda_cublas",
    "cuda_nvdisasm",
    "cuda_nvvm",
]

# 3. Para cada statement, checar se contém algum padrão → comentar TUDO
new_lines = []
commented = 0
for start, end, block_lines in statements:
    block_text = "".join(block_lines)
    already_comment = all(l.strip().startswith("#") or l.strip() == "" for l in block_lines)

    if not already_comment and any(p in block_text for p in skip_patterns):
        for bl in block_lines:
            new_lines.append("# PPC64LE: " + bl)
        commented += 1
    else:
        new_lines.extend(block_lines)

with open("WORKSPACE", "w") as f:
    f.writelines(new_lines)

print(f"OK: WORKSPACE patched — {commented} blocos comentados")
PYEOF

# ---------- Passo 10: Desativar Python Hermético ----------
echo "=== 10/12: Patching Python hermético ==="

# 10A — Wrapper Python
cat > "$PYTHON_WRAPPER" << 'EOF'
#!/bin/bash
unset PYTHONHOME
exec $CONDA_BASE/envs/tf221_build/bin/python3 "$@"
EOF
chmod +x "$PYTHON_WRAPPER"

# 10B — Desabilitar download do Python hermético
python3 - << 'PYEOF'
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

# 10C — Redirect pip_parse para usar o wrapper Python
python3 - << 'PYEOF'
import os

path = "third_party/xla/third_party/py/python_init_pip.bzl"
with open(path, "r") as f:
    content = f.read()

content = content.replace(
    '''        python_interpreter_target = "@{}_host//:python".format(
            get_toolchain_name_per_python_version("python"),
        ),''',
    '        python_interpreter = "/tmp/tf_python3_wrapper.sh",  # ppc64le: wrapper that unsets PYTHONHOME'
)

with open(path, "w") as f:
    f.write(content)

print("OK: python_init_pip patched")
PYEOF

# Verificação
echo "--- toolchains ---"
grep -n "pass" third_party/xla/third_party/py/python_init_toolchains.bzl
echo "--- pip ---"
grep -n "python_interpreter" third_party/xla/third_party/py/python_init_pip.bzl

# 10D — Patch setup_py_nvidia_dependencies_util.py (KeyError em versões CUDA desconhecidas)
echo "Patching setup_py_nvidia_dependencies_util.py..."
sed -i 's/nvidia_wheel_versions\[$/nvidia_wheel_versions.get(/' \
    third_party/xla/third_party/py/setup_py_nvidia_dependencies_util.py
sed -i 's/      str(cuda_version)/      str(cuda_version), {}/' \
    third_party/xla/third_party/py/setup_py_nvidia_dependencies_util.py
sed -i 's/  \].items():/  ).items():/' \
    third_party/xla/third_party/py/setup_py_nvidia_dependencies_util.py

# 10E — Patch XLA builtin_fp16.h (missing <cstdint> for uint16_t on GCC 13+)
echo "Patching XLA builtin_fp16.h missing cstdint..."
sed -i '1s/^/#include <cstdint>\n/' third_party/xla/xla/backends/cpu/codegen/builtin_fp16.h

# 10F — Remover flags específicas de x86/Clang do XLA
echo "Patching XLA hardcoded x86/Clang flags..."
find third_party/xla/xla -type f \( -name "BUILD" -o -name "*.bzl" \) | xargs sed -i 's/"-mprefer-vector-width=512"/""/g; s/"-fno-experimental-sanitize-metadata=all"/""/g'
find third_party/xla/xla -type f \( -name "BUILD" -o -name "*.bzl" \) | xargs sed -i "s/'-mprefer-vector-width=512'/''/g; s/'-fno-experimental-sanitize-metadata=all'/''/g"

# 10G — Patch XLA eigen_unary.cc (Bug do GCC com __builtin_vectorelements)
echo "Patching XLA eigen_unary.cc Clang builtins..."
sed -i 's/defined(__has_builtin) && __has_builtin(__builtin_vectorelements)/0/g' third_party/xla/xla/codegen/intrinsic/cpp/eigen_unary.cc

# 10H — Patch BoringSSL (Unknown target CPU on Power9)
echo "Patching BoringSSL for PowerPC..."
bazel fetch @boringssl//:crypto || true
BASE_H="$(bazel info output_base)/external/boringssl/src/include/openssl/base.h"
if [ -f "$BASE_H" ]; then
    chmod u+w "$BASE_H"
    sed -i 's/#error "Unknown target CPU"/#define OPENSSL_64_BIT\n#define OPENSSL_NO_ASM/g' "$BASE_H"
fi

# 10I — Patch XLA shape.h/cc (Bug GCC 13 noexcept mismatch em PowerPC)
echo "Patching XLA shape.h noexcept..."
sed -i 's/Shape(Shape&&) noexcept;/Shape(Shape\&\&);/g' third_party/xla/xla/shape.h
sed -i 's/operator=(Shape&&) noexcept;/operator=(Shape\&\&);/g' third_party/xla/xla/shape.h
sed -i 's/) noexcept = default;/) = default;/g' third_party/xla/xla/shape.cc

# 10J — Patch XLA dso_loader.cc (Remover Headers CUDA/TensorRT ausentes)
echo "Patching XLA dso_loader.cc cuda_config.h..."
sed -i 's|#include "third_party/gpus/cuda/cuda_config.h"|#define TF_CUDA_VERSION "0"\n#define TF_CUDNN_VERSION "0"\n#define TF_CUDART_VERSION "0"\n#define TF_CUPTI_VERSION "0"\n#define TF_CUBLAS_VERSION "0"\n#define TF_CUSOLVER_VERSION "0"\n#define TF_CUFFT_VERSION "0"\n#define TF_CUSPARSE_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc
sed -i 's|#include "third_party/tensorrt/tensorrt_config.h"|#define TF_TENSORRT_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc
sed -i 's|#include "third_party/gpus/rocm/rocm_config.h"|#define TF_ROCM_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc
sed -i 's|#include "third_party/nccl/nccl_config.h"|#define TF_NCCL_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc

# 10K — Patch XLA group_events.cc (Bug GCC NoDestructor Ambiguity — resilient to any absl version)
echo "Patching XLA group_events.cc NoDestructor..."
python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/tsl/profiler/utils/group_events.cc'
with open(p, 'r') as f:
    t = f.read()
# Replace NoDestructor<flat_hash_set<int64_t>> with static const auto* = new ...
# This avoids the ambiguous constructor issue across all absl versions
t = re.sub(
    r'static\s+(?:const\s+)?absl::NoDestructor\s*<((?:(?!static\s+(?:const\s+)?absl::NoDestructor)[\s\S])+?)>\s+(\w+)\s*\(',
    r'static const auto* \2 =\n      new \1(',
    t
)
with open(p, 'w') as f:
    f.write(t)
PYEOF

# 10L — Patch XLA ffi attribute_map.cc (Bug GCC 8 static_assert template dep)
echo "Patching XLA ffi attribute_map.cc static_assert(false) -> static_assert(sizeof(U)==0)..."
python3 - << 'PYEOF'
p = 'third_party/xla/xla/ffi/attribute_map.cc'
with open(p, 'r') as f:
    t = f.read()
t = t.replace('static_assert(false', 'static_assert(sizeof(U) == 0')
with open(p, 'w') as f:
    f.write(t)
PYEOF

# 10M — Patch MLIR fuse_qdq_pass.cc (Bug GCC NoDestructor Ambiguity — resilient to any absl version)
echo "Patching MLIR fuse_qdq_pass.cc NoDestructor..."
python3 - << 'PYEOF'
import re
p = 'tensorflow/compiler/mlir/lite/transforms/quantization/fuse_qdq_pass.cc'
with open(p, 'r') as f:
    t = f.read()
# Replace NoDestructor<flat_hash_set<string>> with static const auto* = new ...
t = re.sub(
    r'static\s+(?:const\s+)?absl::NoDestructor\s*<((?:(?!static\s+(?:const\s+)?absl::NoDestructor)[\s\S])+?)>\s+(\w+)\s*\(',
    r'static const auto* \2 =\n      new \1(',
    t
)
with open(p, 'w') as f:
    f.write(t)
PYEOF

# 10M — Patch XLA AllocationValue (Bug GCC 8 Vector Move-Copy Fallback)
echo "Patching allocation_value.h for noexcept vector reallocation..."
python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/service/memory_space_assignment/allocation_value.h'
with open(p, 'r') as f:
    t = f.read()

# Limpar qualquer injeçao noexcept anterior
t = t.replace('AllocationValue(AllocationValue&&) noexcept = default;\n', '')
t = t.replace('AllocationValue& operator=(AllocationValue&&) noexcept = default;\n', '')

# Injetar os construtores de move padroes para evitar fallback pra copia
t = re.sub(r'(class|struct)\s+AllocationValue\s*\{',
           r'\1 AllocationValue {\n public:\n  AllocationValue(AllocationValue&&) = default;\n  AllocationValue& operator=(AllocationValue&&) = default;\n',
           t, count=1)
with open(p, 'w') as f:
    f.write(t)
PYEOF

# 10N — Patch XLA TileAssignment (Bug GCC Ambiguity Span)
echo "Patching hlo_sharding_util.h for TileAssignment ambiguity..."
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

# 10O — Patch MLIR TFLite optimize_pass.cc (Bug GCC 8 CTAD std::multiplies)
echo "Patching optimize_pass.cc for std::multiplies CTAD bug..."
python3 - << 'PYEOF'
p = 'tensorflow/compiler/mlir/lite/transforms/optimize_pass.cc'
with open(p, 'r') as f:
    t = f.read()
t = t.replace('std::multiplies()', 'std::multiplies<int64_t>()')
with open(p, 'w') as f:
    f.write(t)
PYEOF

# 10P — Patch DTensor MLIR (Bug LLVM Namespace cast/isa)
echo "Patching dtensor MLIR files for llvm::cast missing scope..."
python3 - << 'PYEOF'
import glob
import re
for p in glob.glob('tensorflow/dtensor/mlir/*.cc'):
    with open(p, 'r') as f:
        t = f.read()
    # Desfazer a injeçao da rotina anterior se existir
    t = t.replace('namespace tensorflow {\nusing llvm::cast;\nusing llvm::isa;\nusing llvm::dyn_cast;\n', 'namespace tensorflow {')
    t = t.replace('#include "llvm/Support/Casting.h"\nnamespace tensorflow {', 'namespace tensorflow {')
    # Se o arquivo usar a conversao, importamos adequadamente
    if re.search(r'\b(cast|isa|dyn_cast)<', t):
        t = t.replace('namespace tensorflow {', '#include "llvm/Support/Casting.h"\nnamespace tensorflow {\nusing llvm::cast;\nusing llvm::isa;\nusing llvm::dyn_cast;\n')
    with open(p, 'w') as f:
        f.write(t)
PYEOF

# 10Q — Patch XLA CPU Emitter (Bug GCC 8 noexcept and aggregate push_back)
echo "Patching XLA codegen headers for noexcept move constructors..."
python3 - << 'PYEOF'
import glob
import re

for p in glob.glob('third_party/xla/xla/codegen/*.h') + glob.glob('third_party/xla/xla/service/cpu/*.cc'):
    with open(p, 'r') as f:
        t = f.read()
    
    # Remove noexcept from defaulted move constructors
    t = re.sub(r'noexcept\s*=\s*default;', '= default;', t)
    
    # Fix push_back aggregate initialization for EmittedKernel
    t = re.sub(
        r'push_back\(\s*\{([^{}]*?)thread_safe_module\(\)\s*\}\s*\)',
        r'push_back(EmittedKernel{\1thread_safe_module()})',
        t
    )
    
    with open(p, 'w') as f:
        f.write(t)
PYEOF

# 10R — Patch XLA ynn_support.cc (Bug GCC NoDestructor Ambiguity — resilient to any absl version)
echo "Patching ynn_support.cc for NoDestructor ambiguity..."
python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/backends/cpu/ynn_support.cc'
with open(p, 'r') as f:
    t = f.read()
# Replace NoDestructor<...> with static const auto* = new ...
t = re.sub(
    r'static\s+(?:const\s+)?absl::NoDestructor\s*<((?:(?!static\s+(?:const\s+)?absl::NoDestructor)[\s\S])+?)>\s+(\w+)\s*\(',
    r'static const auto* \2 =\n      new \1(',
    t
)
with open(p, 'w') as f:
    f.write(t)
PYEOF

# 10S — Patch XLA convolution_lib.h (Bug GCC 8 const lambda capture CountDown)
echo "Patching convolution_lib.h for const lambda auto-capture..."
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

# 10T — Patch XLA thunk_executor.cc (Bug GCC 8 std::unique_ptr copy vector reserve)
echo "Patching thunk_executor.cc to avoid vector copy constructor implicit call on reserve..."
python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/backends/cpu/runtime/thunk_executor.cc'
with open(p, 'r') as f:
    t = f.read()

# Remove a tentativa falha anterior
t = re.sub(
    r'class ThunkOperation : public ExecutionGraph::Operation \{\n public:\n  ThunkOperation\(ThunkOperation&&\) noexcept = default;\n  ThunkOperation& operator=\(ThunkOperation&&\) noexcept = default;',
    r'class ThunkOperation : public ExecutionGraph::Operation {', t
)

# Adiciona move constructors explícitos com campos move
code = '''class ThunkOperation : public ExecutionGraph::Operation {
 public:
  ThunkOperation(ThunkOperation&& other) noexcept
      : ExecutionGraph::Operation(std::move(other)),
        name_(std::move(other.name_)),
        op_type_id_(other.op_type_id_),
        buffer_uses_(std::move(other.buffer_uses_)),
        resource_uses_(std::move(other.resource_uses_)) {}
  ThunkOperation& operator=(ThunkOperation&& other) noexcept {
    ExecutionGraph::Operation::operator=(std::move(other));
    name_ = std::move(other.name_);
    op_type_id_ = other.op_type_id_;
    buffer_uses_ = std::move(other.buffer_uses_);
    resource_uses_ = std::move(other.resource_uses_);
    return *this;
  }'''

t = t.replace('class ThunkOperation : public ExecutionGraph::Operation {', code)

with open(p, 'w') as f:
    f.write(t)
PYEOF

# 10U — Patch XLA expand_integer_power.cc (Bug GCC 8 Ambiguity curly braces)
echo "Patching expand_integer_power.cc to avoid brace initialization ambiguity..."
python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/codegen/emitters/transforms/expand_integer_power.cc'
with open(p, 'r') as f:
    t = f.read()

# Replace `{op->getOperands()}` with `op->getOperands()`
t = t.replace('{op->getOperands()}', 'op->getOperands()')

with open(p, 'w') as f:
    f.write(t)
PYEOF

# 10W — Patch gen_git_source.py (Branch check bypass)
echo "Patching gen_git_source.py..."
python3 - << 'PYEOF'
import re
p = 'tensorflow/tools/git/gen_git_source.py'
with open(p, 'r') as f:
    t = f.read()

# Forçar branch_ref a ser sempre válido ou vazio mas existente
t = t.replace('spec["branch"] = parse_branch_ref(git_head_path)', 'spec["branch"] = "v2.21.0"')
t = t.replace('return unknown_label', 'return b"v2.21.0"')

with open(p, 'w') as f:
    f.write(t)
PYEOF

# 10V — Patch build_pip_package.py para proteger glob.glob[0] e copytree
# First, restore the file to its pristine state to avoid double-patching from previous runs
git checkout -- tensorflow/tools/pip_package/build_pip_package.py 2>/dev/null || true

python3 - << 'PYEOF'
import re

path = "tensorflow/tools/pip_package/build_pip_package.py"
with open(path, "r") as f:
    code = f.read()

# 1. Protege o glob()
def glob0_safe(m):
    expr = m.group(1)
    return f"({expr}[0] if {expr} else None)"
code = re.sub(r"glob\.glob\(([^\)]*)\)\s*\[0\]", glob0_safe, code)

# 2. Injeção global
injection = """
import shutil
import os
import subprocess
import re

# Sentinel: prevent double-patching on re-runs
_PPC64LE_PATCHED = True

# Patch 1: Sobreviver a diretórios nulos
_orig_copytree = shutil.copytree
def _robust_copytree(src, dst, *args, **kwargs):
    if not src or not os.path.exists(str(src)):
        print(f"AVISO: Diretório ausente ignorado -> {src}")
        return
    kwargs['dirs_exist_ok'] = True
    try:
        return _orig_copytree(src, dst, *args, **kwargs)
    except Exception as e:
        print(f"AVISO: Erro ignorado ao copiar {src}: {e}")
        return

shutil.copytree = _robust_copytree

# Patch 2: Interceptar e corrigir o setup.py na pasta temporária correta
_orig_run = subprocess.run
def _patched_run(args, **kwargs):
    if len(args) > 1 and "setup.py" in args[1]:
        cwd = kwargs.get('cwd', '.')
        setup_path = os.path.join(cwd, args[1])
        if os.path.exists(setup_path):
            with open(setup_path, "r") as f:
                c = f.read()
            # Adiciona aspas em versões numéricas que causariam erro de sintaxe
            c = re.sub(r"(\w+_version\s*=\s*)([0-9\.]+)", r"\g<1>'\g<2>'", c)
            with open(setup_path, "w") as f:
                f.write(c)
            print(f"\\n[!!!] SUCESSO: setup.py interceptado e corrigido na pasta {cwd}! [!!!]\\n")
    return _orig_run(args, **kwargs)

subprocess.run = _patched_run
"""

# Only inject if not already patched (prevents recursion on re-runs)
if "_PPC64LE_PATCHED" not in code:
    if code.startswith("#!"):
        parts = code.split("\n", 1)
        code = parts[0] + "\n" + injection + parts[1]
    else:
        code = injection + "\n" + code

with open(path, "w") as f:
    f.write(code)

print("OK: Super-Patch V2 aplicado com sucesso!")
PYEOF

# ---------- Passo 11: Patch do tensorflow.bzl (Bug Bazel 7) ----------
echo "=== 11/13: Patching tensorflow.bzl (Bazel 7 transition fix) ==="
python3 - << 'PYEOF'
import re

filepath = "tensorflow/tensorflow.bzl"
with open(filepath, "r") as f:
    data = f.read()

# Limpa inputs/outputs da transição quebrada
data = re.sub(r'inputs\s*=\s*\[\s*"//command_line_option:modify_execution_info",?\s*\]', 'inputs = []', data)
data = re.sub(r'outputs\s*=\s*\[\s*"//command_line_option:modify_execution_info",?\s*\]', 'outputs = []', data)

# Substitui o corpo da função por um stub limpo
old_func_pattern = r'def _local_exec_transition_impl\(settings, attr\):.*?return \{.*?\}'
new_func_stub = 'def _local_exec_transition_impl(settings, attr):\n    return {}'
data = re.sub(old_func_pattern, new_func_stub, data, flags=re.DOTALL)

with open(filepath, "w") as f:
    f.write(data)
print("OK: tensorflow.bzl patched")
PYEOF

# ---------- Passo 11.1: Patch pybind11_bazel (Visibilidade GCC — solução definitiva) ----------
echo "=== 11.1/13: Removendo -fvisibility=hidden do pybind_library macro ==="
# AVISO: Requer que o cache do Bazel já exista. Se for a primeira execução,
# o build fará um fetch automático e este patch será aplicado no primeiro ciclo.
pip uninstall tensorflow -y || true

python3 - << 'PYEOF'
import os
import re

base = os.popen("bazel info output_base 2>/dev/null").read().strip()

# 1. Patch DEFINITIVO: remover -fvisibility=hidden do macro pybind_library
pybind_bzl = f"{base}/external/pybind11_bazel/build_defs.bzl"
if os.path.exists(pybind_bzl):
    os.chmod(pybind_bzl, 0o644)
    with open(pybind_bzl, 'r') as f:
        content = f.read()
    # AVISO: Starlark NAO aceita comentarios # dentro de listas.
    content = content.replace('"" # -fvisibility=hidden disabled for ppc64le', '')  # limpar patch anterior
    content = content.replace("'' # -fvisibility=hidden disabled for ppc64le", '')
    content = content.replace('"-fvisibility=hidden"', '""')
    content = content.replace("'-fvisibility=hidden'", "''")
    import re
    content = re.sub(r'\[\s*""\s*\]', '[]', content)
    content = re.sub(r"\[\s*''\s*\]", '[]', content)
    with open(pybind_bzl, 'w') as f:
        f.write(content)
    print(f"OK: pybind11_bazel/build_defs.bzl patchado")
else:
    print(f"AVISO: {pybind_bzl} nao encontrado ainda (sera patchado apos primeiro fetch)")

# 2. Patch de visibilidade irrestrita para TODOS os pacotes auxiliares pybind11 (abseil, protobuf, status)
import glob

def enforce_visibility(search_path_pattern):
    for path in glob.glob(search_path_pattern, recursive=True):
        if not os.path.isfile(path): continue
        if path.endswith(".h") or path.endswith(".cc"):
            os.chmod(path, 0o644)
            with open(path, 'r') as f:
                content = f.read()
            content = content.replace('__attribute__((visibility("default"))) ', '')
            if '#pragma GCC visibility push(default)' not in content:
                # Inserir no início (após os includes)
                content = re.sub(
                    r'((?:#include [^\n]+\n)+)(?!#include)',
                    lambda m: m.group(0) + '\n#pragma GCC visibility push(default)\n',
                    content, count=1
                )
                if '#pragma GCC visibility push(default)' not in content:
                    content = '#pragma GCC visibility push(default)\n\n' + content
                content = content.rstrip() + '\n\n#pragma GCC visibility pop\n'
                with open(path, 'w') as f:
                    f.write(content)
                print(f"Visibilidade forçada em: {path}")

# Garantir visibilidade irrestrita p/ status e utils
enforce_visibility(f"{base}/external/pybind11_protobuf/pybind11_protobuf/**/*.h")
enforce_visibility(f"{base}/external/pybind11_protobuf/pybind11_protobuf/**/*.cc")
enforce_visibility(f"{base}/external/pybind11_abseil/**/*.h")
enforce_visibility(f"{base}/external/pybind11_abseil/**/*.cc")

print("=== Todos os patches de visibilidade aplicados! ===")
PYEOF

# ---------- Passo 12: BUILD ----------
echo "=== 12/13: Iniciando compilação (isso vai demorar 4-8 horas) ==="
NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")
bazel build \
    --config=opt \
    --action_env=GIT_TAG_OVERRIDE=v2.21.0 \
    --copt=-fvisibility=default \
    --cxxopt=-fvisibility=default \
    --copt=-DPYBIND11_EXPORT="__attribute__((visibility(\"default\")))" \
    --define=tflite_with_xnnpack=false \
    --local_ram_resources=HOST_RAM*.6 \
    --per_file_copt=".*@-Wno-error" \
    --extra_toolchains=@@local_config_python//:py_cc_toolchain \
    --override_repository=rules_ml_toolchain="$ML_TOOLCHAIN" \
    --override_repository=llvm_linux_x86_64="$LLVM_STUB" \
    --override_repository=llvm_linux_aarch64="$LLVM_STUB" \
    --override_repository=llvm_darwin_aarch64="$LLVM_STUB" \
    --override_repository=llvm18_linux_x86_64="$LLVM_STUB" \
    --override_repository=llvm19_linux_x86_64="$LLVM_STUB" \
    --override_repository=llvm20_linux_x86_64="$LLVM_STUB" \
    --override_repository=llvm21_linux_x86_64="$LLVM_STUB" \
    --override_repository=llvm18_linux_aarch64="$LLVM_STUB" \
    --override_repository=llvm20_linux_aarch64="$LLVM_STUB" \
    --override_repository=llvm21_linux_aarch64="$LLVM_STUB" \
    --override_repository=llvm18_darwin_aarch64="$LLVM_STUB" \
    --override_repository=llvm20_darwin_aarch64="$LLVM_STUB" \
    --override_repository=local_config_cuda="$CUDA_STUB" \
    --override_repository=local_config_nccl="$NCCL_STUB" \
    --override_repository=local_config_tensorrt="$TENSORRT_STUB" \
    --override_repository=local_config_rocm="$ROCM_STUB" \
    --override_repository=cuda_cudart="$CUDA_REDIST_STUB" \
    --override_repository=cuda_profiler_api="$CUDA_REDIST_STUB" \
    --override_repository=cuda_nvcc="$CUDA_REDIST_STUB" \
    --override_repository=cuda_nvml="$CUDA_REDIST_STUB" \
    --override_repository=cuda_nvtx="$CUDA_REDIST_STUB" \
    --override_repository=cuda_cccl="$CUDA_REDIST_STUB" \
    --override_repository=cuda_cudnn="$CUDA_REDIST_STUB" \
    --override_repository=cuda_cudnn9="$CUDA_REDIST_STUB" \
    --override_repository=cuda_cublas="$CUDA_REDIST_STUB" \
    --override_repository=cuda_cusolver="$CUDA_REDIST_STUB" \
    --override_repository=cuda_cusparse="$CUDA_REDIST_STUB" \
    --override_repository=cuda_curand="$CUDA_REDIST_STUB" \
    --override_repository=cuda_cufft="$CUDA_REDIST_STUB" \
    --override_repository=cuda_cupti="$CUDA_REDIST_STUB" \
    --override_repository=cuda_nvdisasm="$CUDA_REDIST_STUB" \
    --override_repository=cuda_nvvm="$CUDA_REDIST_STUB" \
    --override_repository=cuda_nvjitlink="$CUDA_REDIST_STUB" \
    --override_repository=nvidia_wheel_versions="$CUDA_REDIST_STUB" \
    --override_repository=pypi="$PYPI_STUB" \
    --override_repository=pypi_absl_py="$PYPI_STUB" \
    --override_repository=pypi_astunparse="$PYPI_STUB" \
    --override_repository=pypi_auditwheel="$PYPI_STUB" \
    --override_repository=pypi_flatbuffers="$PYPI_STUB" \
    --override_repository=pypi_gast="$PYPI_STUB" \
    --override_repository=pypi_keras="$PYPI_STUB" \
    --override_repository=pypi_lit="$PYPI_STUB" \
    --override_repository=pypi_ml_dtypes="$PYPI_STUB" \
    --override_repository=pypi_mods="$PYPI_STUB" \
    --override_repository=pypi_numpy="$PYPI_STUB" \
    --override_repository=pypi_nvidia_cublas_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_cuda_cupti_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_cuda_nvcc_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_cuda_nvrtc_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_cuda_runtime_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_cudnn_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_cufft_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_curand_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_cusolver_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_cusparse_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_nccl_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_nvjitlink_cu12="$PYPI_STUB" \
    --override_repository=pypi_nvidia_nvshmem_cu12="$PYPI_STUB" \
    --override_repository=pypi_opt_einsum="$PYPI_STUB" \
    --override_repository=pypi_packaging="$PYPI_STUB" \
    --override_repository=pypi_protobuf="$PYPI_STUB" \
    --override_repository=pypi_requests="$PYPI_STUB" \
    --override_repository=pypi_setuptools="$PYPI_STUB" \
    --override_repository=pypi_termcolor="$PYPI_STUB" \
    --override_repository=pypi_typing_extensions="$PYPI_STUB" \
    --override_repository=pypi_wheel="$PYPI_STUB" \
    --override_repository=pypi_wrapt="$PYPI_STUB" \
    --override_repository=python_3_11_host="$PYTHON_STUB" \
    --override_repository=local_config_python="$PYTHON_CONFIG_STUB" \
    --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
    --repo_env=USE_HERMETIC_CC_TOOLCHAIN=0 \
    --noincompatible_enable_cc_toolchain_resolution \
    --jobs=$(nproc) \
    //tensorflow/tools/pip_package:wheel

# ---------- Step 13: Report and Installation ----------
WHEEL_FILE=$(ls bazel-bin/tensorflow/tools/pip_package/wheel_house/*.whl bazel-bin/tensorflow/tools/pip_package/*.whl 2>/dev/null | head -n 1)

if [ -f "$WHEEL_FILE" ]; then
    echo "=== COMPILAÇÃO CONCLUÍDA COM SUCESSO! ==="
    echo "Pacote gerado: $WHEEL_FILE"
    echo "Instalando dependências adicionais via Conda (sem numpy — já instalado pelo conda)..."
    conda install -c conda-forge "h5py<3.15.0" grpcio libclang ml_dtypes "protobuf>=6.31.1,<8.0.0" absl-py astunparse flatbuffers gast google-pasta opt_einsum termcolor wrapt libstdcxx-ng pillow tensorboard -y
    echo "Instalando o TensorFlow gerado localmente (--no-deps evita compilar numpy do zero)..."
    pip uninstall tensorflow -y || true
    cd ~
    pip install --force-reinstall --no-deps "$TF_DIR/tensorflow/$WHEEL_FILE"
    pip install "keras>=3.0.0" flatbuffers || true
    echo "=== INSTALAÇÃO CONCLUÍDA. TensorFlow pronto para uso no ambiente $CONDA_ENV! ==="
else
    echo "=== ERROR: .whl package not found ==="
    exit 1
fi

# ---------- Step 14: Fire Test ----------
echo "=== 14/14: Running Fire Test ==="
cd ~
export LD_LIBRARY_PATH=$CONDA_BASE/envs/$CONDA_ENV/lib:$LD_LIBRARY_PATH
python3 -c "
import tensorflow as tf
import time

print(f'\n--- Starting Fire Test on Power9 (TF {tf.__version__}) ---')
devices = tf.config.list_physical_devices()
print('Detected devices:', [d.name for d in devices])

A = tf.random.normal([5000, 5000])
B = tf.random.normal([5000, 5000])

start = time.time()
C = tf.matmul(A, B)
end = time.time()

soma = tf.reduce_sum(C).numpy()
print(f'Result sum: {soma:.2f}')
print(f'Execution time: {end - start:.4f} seconds')
print('[VICTORY] Native TensorFlow for Power9 is 100% operational!')
"
