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
export PY_VER="${PY_VER:-3.11}"

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
conda create -n "$CONDA_ENV" "python=$PY_VER" numpy wheel packaging requests -c conda-forge -y
conda activate "$CONDA_ENV"

# Detect actual Python version from the environment
export ACTUAL_PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export PY_VER_FLAT=$(echo $ACTUAL_PY_VER | tr -d '.')


echo "Python: $(python3 --version) em $(which python3)"
echo "Bazel: $(bazel --version)"

# ---------- Step 3: Clone TF 2.21 ----------
echo "=== 3/12: Cloning TensorFlow 2.21 ==="
mkdir -p "$TF_DIR" && cd "$TF_DIR"
rm -rf tensorflow
git clone --branch power9-v2.21.0-cpu-only https://github.com/llm-pt-ibm/tensorflow.git
cd tensorflow

# ---------- Step 4: Clean configs ----------
echo "=== 4/12: Cleaning old configs ==="
rm -f .bazelversion
bazel clean --expunge

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

# --- Python stub (python_${PY_VER_FLAT}_host) ---
echo "⚙️  Configuring python_stub..."
PYTHON_STUB="$HOME/python_stub"
mkdir -p "$PYTHON_STUB"
touch "$PYTHON_STUB/WORKSPACE"
ln -sf $CONDA_BASE/envs/tf221_build/bin/python3 "$PYTHON_STUB/python3_bin"
cat > "$PYTHON_STUB/BUILD" << EOF
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
ln -sf "$PYINC" "$PYTHON_CONFIG_STUB/python_include"

cat > "$PYTHON_CONFIG_STUB/py_cc_toolchain.bzl" << EOF
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
        "python_version": attr.string(default = "${ACTUAL_PY_VER}"),
    },
)
EOF

cat > "$PYTHON_CONFIG_STUB/BUILD" << EOF
package(default_visibility = ["//visibility:public"])
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")
load(":py_cc_toolchain.bzl", "py_cc_toolchain")

py_runtime(
    name = "py2_runtime",
    interpreter = "@python_${PY_VER_FLAT}_host//:python",
    python_version = "PY2",
)
py_runtime(
    name = "py3_runtime",
    interpreter = "@python_${PY_VER_FLAT}_host//:python",
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
    python_version = "${ACTUAL_PY_VER}",
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


# ---------- Passo 10: Criando python3_bazel wrapper ----------
echo "=== 10/12: Criando python_wrapper ==="
cat > "$PYTHON_WRAPPER" << EOF
#!/bin/bash
unset PYTHONHOME
exec $CONDA_BASE/envs/tf221_build/bin/python3 "\$@"
EOF
chmod +x "$PYTHON_WRAPPER"

# ---------- Passo 12: BUILD ----------
echo "=== 12/13: Iniciando compilação (isso vai demorar 4-8 horas) ==="
NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")
bazel build \
    --config=opt \
    --copt=-fvisibility=default \
    --cxxopt=-fvisibility=default \
    --copt=-DPYBIND11_EXPORT="__attribute__((visibility(\"default\")))" \
    --define=tflite_with_xnnpack=false \
    --local_ram_resources=HOST_RAM*.6 \
    --per_file_copt=".*@-Wno-error" \
    --extra_toolchains=@@local_config_python//:py_cc_toolchain \
    --override_repository=python_${PY_VER_FLAT}_host="$PYTHON_STUB" \
    --override_repository=local_config_python="$PYTHON_CONFIG_STUB" \
    --repo_env=HERMETIC_PYTHON_VERSION=$ACTUAL_PY_VER \
    --repo_env=USE_HERMETIC_CC_TOOLCHAIN=0 \
    --noincompatible_enable_cc_toolchain_resolution \
    --jobs=$(nproc) \
    //tensorflow/tools/pip_package:wheel

# ---------- Step 13: Report and Installation ----------
WHEEL_FILE=$(ls bazel-bin/tensorflow/tools/pip_package/*.whl 2>/dev/null | head -n 1)

if [ -f "$WHEEL_FILE" ]; then
    echo "=== COMPILAÇÃO CONCLUÍDA COM SUCESSO! ==="
    echo "Pacote gerado: $WHEEL_FILE"
    echo "Instalando dependências adicionais via Conda (sem numpy — já instalado pelo conda)..."
    conda install -c conda-forge h5py grpcio libclang ml_dtypes "protobuf>=6.31.1,<8.0.0" -y
    echo "Instalando o TensorFlow gerado localmente (--no-deps evita compilar numpy do zero)..."
    pip uninstall tensorflow -y || true
    cd ~
    pip install --force-reinstall --no-deps "$TF_DIR/tensorflow/$WHEEL_FILE"
    echo "=== INSTALAÇÃO CONCLUÍDA. TensorFlow pronto para uso no ambiente $CONDA_ENV! ==="
else
    echo "=== ERROR: .whl package not found ==="
    exit 1
fi

# ---------- Step 14: Fire Test ----------
echo "=== 14/14: Running Fire Test ==="
cd ~
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
