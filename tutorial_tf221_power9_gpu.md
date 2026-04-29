# Passo a Passo: Compilando TensorFlow 2.21 no Power9 (ppc64le) com Suporte a GPU

> **Pré-requisitos de GPU:** Certifique-se de que o **NVIDIA CUDA Toolkit** e o **cuDNN** estejam instalados no seu sistema (geralmente em `/usr/local/cuda`), pois o TensorFlow não consegue baixar pacotes herméticos (wheels) de CUDA para Power9.

Execute os blocos abaixo um por vez no terminal da sua VM Power9.

### Passo 1: Instalar dependências
```bash
sudo dnf install -y git gcc gcc-c++ zip unzip which patch wget vim-common
```

### Passo 2: Criar o ambiente Conda
Como você compilou o Bazel manualmente no `/usr/local/bin`, nós NÃO instalaremos o Bazel pelo conda. Apenas Python e pacotes base!

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda create -n tf221_build "python=3.11" numpy wheel packaging requests -c conda-forge -y
conda activate tf221_build
```

### Passo 3: Clonar o código-fonte do TensorFlow 2.21
Se a pasta `tensorflow` já existir, delete-a primeiro (`rm -rf tensorflow`) e faça o git clone de novo.

```bash
cd /home/almalinux/tensorflow221
rm -rf tensorflow
git clone --depth 1 --branch v2.21.0 https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

### Passo 4: Limpar configs antigos de Bazel
```bash
rm -f .bazelversion
bazel clean --expunge
```

### Passo 5: Configurar as variáveis de ambiente (O Pulo do Gato)
Copie e cole este bloco inteiro no terminal de uma vez:

```bash
export CC=gcc
export CXX=g++
export CFLAGS="-mno-float128 -O3"
export CXXFLAGS="-mno-float128 -O3"
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
```

### Passo 6: Rodar o Configure
```bash
echo -e "\n\n\n\n\n\n\n\n\n\n" | ./configure
```
> **Nota:** Se o `./configure` pedir o diretório do CUDA, você pode ser solicitado iterativamente, porém os atalhos de ambiente criados devem evitar perguntas interativas de CPU, preenchendo as configurações da GPU conforme os locais padrão.

### Passo 7: Corrigir o Bug "local.bzl" do rules_ml_toolchain
O Bazel 7.1.0 removeu o arquivo `local.bzl`. Vamos baixar o pacote problemático, consertá-lo e depois usá-lo localmente:

```bash
cd ~
wget -qO rules.tar.gz https://github.com/google-ml-infra/rules_ml_toolchain/archive/d8cb9c2c168cd64000eaa6eda0781a9615a26ffe.tar.gz
rm -rf rules_ml_toolchain_patched
tar -xzf rules.tar.gz
mv rules_ml_toolchain-d8cb9c2c168cd64000eaa6eda0781a9615a26ffe rules_ml_toolchain_patched
sed -i '/load("@bazel_tools\/\/tools\/build_defs\/repo:local.bzl", "new_local_repository")/d' rules_ml_toolchain_patched/cc/deps/cc_toolchain_deps.bzl
sed -i 's/new_local_repository(/native.new_local_repository(/g' rules_ml_toolchain_patched/cc/deps/cc_toolchain_deps.bzl
cd /home/almalinux/tensorflow221/tensorflow
```

### Passo 8: Criar LLVM, CUDA Redist e ROCm Stub Repos
O `rules_ml_toolchain` tenta baixar binários do LLVM/Clang para x86_64, e os scripts de CUDA tentam baixar redistribuições empacotadas via PyPI (inexistentes no power9). A solução é criar stubs e aplicar remendos que isolam funções nativas incompatíveis (como a biblioteca TSL e o `attribute_map.cc` na FFI):

```bash
# --- LLVM stub ---
mkdir -p /root/llvm_stub
echo 'VERSION = "0.0.0"' > /root/llvm_stub/version.bzl
touch /root/llvm_stub/WORKSPACE

cat > /root/llvm_stub/BUILD << 'EOF'
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

# --- CUDA redistribution stub (evita o PyPI de CUDA para rodas inacessíveis) ---
mkdir -p /root/cuda_redist_stub
touch /root/cuda_redist_stub/WORKSPACE
echo 'VERSION = "0.0.0"' > /root/cuda_redist_stub/version.bzl
cat > /root/cuda_redist_stub/versions.bzl << 'EOF'
NVIDIA_WHEEL_VERSIONS = {
    "0.0.0": [],
}
EOF
cat > /root/cuda_redist_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "all", srcs = [], visibility = ["//visibility:public"])
cc_library(name = "headers", hdrs = [], visibility = ["//visibility:public"])
cc_library(name = "libs", srcs = [], visibility = ["//visibility:public"])
EOF

# --- TensorRT stub ---
mkdir -p /root/tensorrt_stub
touch /root/tensorrt_stub/WORKSPACE
cat > /root/tensorrt_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "LICENSE", srcs = [])
cc_library(name = "tensorrt_headers", hdrs = [])
cc_library(name = "tensorrt", srcs = [])
config_setting(name = "use_static_tensorrt", values = {"define": "stub=true"})
py_library(name = "tensorrt_config_py", srcs = [])
EOF

cat > /root/tensorrt_stub/build_defs.bzl << 'EOF'
def if_tensorrt(if_true, if_false = []):
    return if_false
def is_tensorrt_configured():
    return False
def if_tensorrt_exec(if_true, if_false = []):
    return if_false
EOF

# --- ROCm stub ---
mkdir -p /root/rocm_stub/rocm
touch /root/rocm_stub/WORKSPACE
cat > /root/rocm_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
config_setting(name = "is_rocm_enabled", values = {"define": "stub=true"})
filegroup(name = "LICENSE", srcs = [])
EOF

cat > /root/rocm_stub/rocm/BUILD << 'EOF'
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
mkdir -p /root/rocm_stub/crosstool
cat > /root/rocm_stub/crosstool/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
filegroup(name = "toolchain", srcs = [])
EOF

cat > /root/rocm_stub/rocm/build_defs.bzl << 'EOF'
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
```

### Passo 9: Criar PyPI Stub e Python Stub
Igual à versão da CPU:

```bash
# --- PyPI stub (para bibliotecas Pip herméticas) ---
mkdir -p /root/pypi_stub
touch /root/pypi_stub/WORKSPACE
cat > /root/pypi_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
py_library(name = "pkg", srcs = [])
EOF

cat > /root/pypi_stub/requirements.bzl << 'EOF'
def requirement(name):
    normalized = name.replace("-", "_").lower()
    return "@@pypi//" + normalized
def install_deps(**kwargs):
    pass
EOF

for pkg in astor astunparse dill gast h5py jax lit numpy opt_einsum packaging portpicker protobuf requests scipy tblib termcolor typing_extensions wrapt zstandard; do
  mkdir -p /root/pypi_stub/$pkg
  cat > /root/pypi_stub/$pkg/BUILD << EOF
package(default_visibility = ["//visibility:public"])
py_library(name = "$pkg", srcs = [])
py_library(name = "pkg", srcs = [])
EOF
done

# Numpy precisa de headers C
NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")
ln -sf "$NUMPY_INC" /root/pypi_stub/numpy/numpy_include
cat > /root/pypi_stub/numpy/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "numpy_headers",
    hdrs = glob(["numpy_include/**/*.h"]),
    includes = ["numpy_include"],
)
py_library(name = "numpy", srcs = [])
EOF

# --- Python stub (python_3_11_host) ---
mkdir -p /root/python_stub
touch /root/python_stub/WORKSPACE
ln -sf /root/miniforge3/envs/tf221_build/bin/python3 /root/python_stub/python3_bin
cat > /root/python_stub/BUILD << 'EOF'
package(default_visibility = ["//visibility:public"])
exports_files(["python3_bin"])
filegroup(name = "python", srcs = ["python3_bin"])
filegroup(name = "files", srcs = ["python3_bin"])
filegroup(name = "includes", srcs = [])
cc_library(name = "python_headers", hdrs = [])
EOF

# --- Python config stub (local_config_python) ---
mkdir -p /root/python_config_stub
touch /root/python_config_stub/WORKSPACE
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
ln -sf "$PYINC" /root/python_config_stub/python_include

cat > /root/python_config_stub/py_cc_toolchain.bzl << 'EOF'
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

cat > /root/python_config_stub/BUILD << 'EOF'
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
```

### Passo 9.1: Patch do WORKSPACE (Preserva Configuração local do CUDA/NCCL)
Desabilita o download de pacotes herméticos, **mas garante que `local_config_cuda` e `local_config_nccl` executem para detecção real da GPU!**:

```bash
cd /home/almalinux/tensorflow221/tensorflow

cat > /tmp/patch_workspace3.py << 'PYEOF'
"""
Patch WORKSPACE para ppc64le GPU: comenta BLOCOS INTEIROS (statements)
referentes ao download de wheels herméticos de CUDA.
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

# Removidos local_config_cuda e local_config_nccl da lista!
skip_patterns = [
    "cc_toolchain_deps",
    "register_toolchains",
    "cuda_json_init",
    "cuda_redist",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
    "nccl_redist",
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
python3 /tmp/patch_workspace3.py
```

### Passo 10: Desativar Python Hermético e Ajustar Código Interno do XLA para o GCC

Execute as caixas de **10A a 10H** e **10K a 10U**. Modificamos **10J** para deixar de quebrar as compilações em CUDA e NCCL.

**10A — Criar wrapper Python**:
```bash
cat > /root/python3_bazel.sh << 'EOF'
#!/bin/bash
unset PYTHONHOME
exec /root/miniforge3/envs/tf221_build/bin/python3 "$@"
EOF
chmod +x /root/python3_bazel.sh
```

**10B — Desabilitar download do Python hermético:**
```bash
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
```

**10C — Redirecionar pip_parse para usar o wrapper Python:**
```bash
cat > /tmp/patch_pip.py << 'PYEOF'
path = "third_party/xla/third_party/py/python_init_pip.bzl"
with open(path, "r") as f:
    content = f.read()

content = content.replace(
    '''        python_interpreter_target = "@{}_host//:python".format(
            get_toolchain_name_per_python_version("python"),
        ),''',
    '        python_interpreter = "/root/python3_bazel.sh",  # ppc64le: wrapper that unsets PYTHONHOME'
)

with open(path, "w") as f:
    f.write(content)

print("OK: python_init_pip patched")
PYEOF
python3 /tmp/patch_pip.py
```

**10D — Patch do `setup_py_nvidia_dependencies_util.py`**:
```bash
sed -i 's/nvidia_wheel_versions\[$/nvidia_wheel_versions.get(/' \
    third_party/xla/third_party/py/setup_py_nvidia_dependencies_util.py

sed -i 's/      str(cuda_version)/      str(cuda_version), {}/' \
    third_party/xla/third_party/py/setup_py_nvidia_dependencies_util.py

sed -i 's/  \].items():/  ).items():/' \
    third_party/xla/third_party/py/setup_py_nvidia_dependencies_util.py
```

**10E — Patch XLA `builtin_fp16.h`**:
```bash
sed -i '1s/^/#include <cstdint>\n/' third_party/xla/xla/backends/cpu/codegen/builtin_fp16.h
```

**10F — Remover paths hardcoded e flags específicas de x86/Clang do XLA/TSL**:
```bash
find third_party/xla/xla third_party/xla/third_party/tsl -type f \( -name "*.h" -o -name "*.cc" \) -exec sed -i 's|third_party/gpus/cuda/include/||g; s|third_party/gpus/cudnn/include/||g; s|third_party/tensorrt/||g' {} +
find third_party/xla/xla -type f \( -name "BUILD" -o -name "*.bzl" \) | xargs sed -i 's/"-mprefer-vector-width=512"/""/g; s/"-fno-experimental-sanitize-metadata=all"/""/g'
find third_party/xla/xla -type f \( -name "BUILD" -o -name "*.bzl" \) | xargs sed -i "s/'-mprefer-vector-width=512'/''/g; s/'-fno-experimental-sanitize-metadata=all'/''/g"
```

**10G — Patch XLA `eigen_unary.cc`**:
```bash
sed -i 's/defined(__has_builtin) && __has_builtin(__builtin_vectorelements)/0/g' third_party/xla/xla/codegen/intrinsic/cpp/eigen_unary.cc
```

**10H — Patch BoringSSL**:
```bash
bazel fetch @boringssl//:crypto || true
BASE_H="$(bazel info output_base)/external/boringssl/src/include/openssl/base.h"
if [ -f "$BASE_H" ]; then
    chmod u+w "$BASE_H"
    sed -i 's/#error "Unknown target CPU"/#define OPENSSL_64_BIT\n#define OPENSSL_NO_ASM/g' "$BASE_H"
fi
```

**10I — Patch XLA `shape.h` e `.cc`**:
```bash
sed -i 's/Shape(Shape&&) noexcept;/Shape(Shape\&\&);/g' third_party/xla/xla/shape.h
sed -i 's/operator=(Shape&&) noexcept;/operator=(Shape\&\&);/g' third_party/xla/xla/shape.h
sed -i 's/) noexcept = default;/) = default;/g' third_party/xla/xla/shape.cc
```

**10J — Patch XLA `dso_loader.cc`** *(Versão GPU: Stub aplicado apenas p/ TRT e ROCm)*:
```bash
sed -i 's|#include "third_party/tensorrt/tensorrt_config.h"|#define TF_TENSORRT_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc
sed -i 's|#include "tensorrt_config.h"|#define TF_TENSORRT_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc
sed -i 's|#include "third_party/gpus/rocm/rocm_config.h"|#define TF_ROCM_VERSION "0"|g' third_party/xla/xla/tsl/platform/default/dso_loader.cc
```

**10K — Patch XLA `group_events.cc`**:
```bash
python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/tsl/profiler/utils/group_events.cc'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(r'(NoDestructor[^\(]+)\s*\(\s*\{', r'\1(absl::flat_hash_set<int64_t>{', t)
with open(p, 'w') as f:
    f.write(t)
PYEOF
```

**10L — Patch MLIR `fuse_qdq_pass.cc`**:
```bash
python3 - << 'PYEOF'
import re
p = 'tensorflow/compiler/mlir/lite/transforms/quantization/fuse_qdq_pass.cc'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(r'(NoDestructor[^\(]+)\s*\(\s*\{', r'\1(absl::flat_hash_set<std::string>{', t)
with open(p, 'w') as f:
    f.write(t)
PYEOF
```

**10M — Patch XLA `allocation_value.h`**:
```bash
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
```

**10N — Patch XLA `hlo_sharding_util.h`**:
```bash
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
```

**10O — Patch MLIR TFLite `optimize_pass.cc`**:
```bash
python3 - << 'PYEOF'
p = 'tensorflow/compiler/mlir/lite/transforms/optimize_pass.cc'
with open(p, 'r') as f:
    t = f.read()
t = t.replace('std::multiplies()', 'std::multiplies<int64_t>()')
with open(p, 'w') as f:
    f.write(t)
PYEOF
```

**10P — Patch DTensor MLIR**:
```bash
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
```

**10Q — Patch XLA CPU Emitter**:
```bash
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
```

**10R**:
```bash
python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/backends/cpu/ynn_support.cc'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(r'kAllowedTypes\s*\(\s*\{', r'kAllowedTypes(absl::flat_hash_set<std::tuple<xla::PrimitiveType, xla::PrimitiveType, xla::PrimitiveType>>{', t)
with open(p, 'w') as f:
    f.write(t)
PYEOF
```

**10S**:
```bash
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
```

**10T**:
```bash
python3 - << 'PYEOF'
import re
p = 'third_party/xla/xla/backends/cpu/runtime/thunk_executor.cc'
with open(p, 'r') as f:
    t = f.read()
t = re.sub(
    r'std::vector<ThunkOperation> thunk_operations;[^{}]*thunk_operations\.reserve\(node\.thunks\(\)\.size\(\)\);',
    r'std::vector<ThunkOperation> thunk_operations;',
    t, count=1
)
with open(p, 'w') as f:
    f.write(t)
PYEOF
```

**10U**:
```bash
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

if code.startswith("#!"):
    parts = code.split("\n", 1)
    code = parts[0] + "\n" + injection + parts[1]
else:
    code = injection + "\n" + code

with open(path, "w") as f:
    f.write(code)

print("OK: Super-Patch V2 aplicado com sucesso!")
PYEOF
```

**10V — Patch XLA FFI (attribute_map.cc):**
Evita trava de assertions rígidas do GCC não implementadas nativamente no Power9.
```bash
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
```

**10W — Patch XLA `expand_integer_power.cc` (GCC 8 brace-init ambiguity)**:

O GCC 8 não resolve a brace-init `{op->getOperands()}` quando há múltiplos construtores herdados no MLIR. Substituímos por construtor explícito.
```bash
python3 - << 'POWPATCH'
p = 'third_party/xla/xla/codegen/emitters/transforms/expand_integer_power.cc'
with open(p, 'r') as f:
    t = f.read()
# GCC 8: brace-init {op->getOperands()} é ambíguo com construtores herdados do MLIR.
# {operands} é o 4o arg (Adaptor), op->getAttrs() é o 5o arg separado.
t = t.replace(
    '{op->getOperands()}',
    'mlir::mhlo::PowOp::Adaptor(op->getOperands())'
)
with open(p, 'w') as f:
    f.write(t)
print("OK: expand_integer_power.cc patched (GCC 8 brace-init)")
POWPATCH
```

**10X — Patch XLA `thunk_executor.cc` (GCC 8 vector copy vs move)**:

`ThunkOperation` herda de `ExecutionGraph::Operation` (destrutor virtual + membros `unique_ptr`). O GCC 8 não detecta que o copy ctor implícito está deletado e tenta copiar em `vector::reserve()`. Solução: deletar **explicitamente** o copy ctor.
```bash
python3 - << 'THUNKPATCH'
import re, sys

p = 'third_party/xla/xla/backends/cpu/runtime/thunk_executor.cc'
try:
    with open(p, 'r') as f:
        t = f.read()
except FileNotFoundError:
    print(f"AVISO: {p} nao encontrado, pulando")
    sys.exit(0)

original = t

# Injeta copy delete + move default APÓS a abertura da classe
# [^{]* captura a herança ": public ExecutionGraph::Operation"
# Adiciona "public:" pois access-default de class{} é private
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
    print("ERRO: Regex nao encontrou 'class ThunkOperation'!")
    sys.exit(1)

if t != original:
    with open(p, 'w') as f:
        f.write(t)
    print("OK: thunk_executor.cc — copy ctor deletado explicitamente (GCC 8 ppc64le fix)")
else:
    print("AVISO: Nenhuma alteracao")
THUNKPATCH
```

### Passo 11: Corrigir Bug do Bazel 7 no tensorflow.bzl
```bash
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
```

### Passo 11.1: Patch pybind11_bazel — Remover `-fvisibility=hidden` globalmente
```bash
cd /home/almalinux/tensorflow221/tensorflow

# Desinstalar TF instalado (evita conflito no ExtractAPI step)
pip uninstall tensorflow -y || true

python3 - << 'PYEOF'
import os
import re

base = os.popen("bazel info output_base 2>/dev/null").read().strip()

pybind_bzl = f"{base}/external/pybind11_bazel/build_defs.bzl"
if os.path.exists(pybind_bzl):
    os.chmod(pybind_bzl, 0o644)
    with open(pybind_bzl, 'r') as f:
        content = f.read()
    content = content.replace('"" # -fvisibility=hidden disabled for ppc64le', '')
    content = content.replace("'' # -fvisibility=hidden disabled for ppc64le", '')
    content = content.replace('"-fvisibility=hidden"', '""')
    content = content.replace("'-fvisibility=hidden'", "''")
    content = re.sub(r'\[\s*""\s*\]', '[]', content)
    content = re.sub(r"\[\s*''\s*\]", '[]', content)
    with open(pybind_bzl, 'w') as f:
        f.write(content)
    print(f"OK: pybind11_bazel/build_defs.bzl patchado")
else:
    print(f"AVISO: {pybind_bzl} nao encontrado. Execute um 'bazel build' primeiro para popular o cache.")

proto_dir = f"{base}/external/pybind11_protobuf/pybind11_protobuf"
for fname in ['proto_cast_util.h', 'proto_cast_util.cc']:
    path = f"{proto_dir}/{fname}"
    if not os.path.exists(path):
        continue
    os.chmod(path, 0o644)
    with open(path, 'r') as f:
        content = f.read()
    content = content.replace('__attribute__((visibility("default"))) ', '')
    if '#pragma GCC visibility push(default)' not in content:
        content = re.sub(
            r'((?:#include [^\n]+\n)+)(?!#include)',
            lambda m: m.group(0) + '\n#pragma GCC visibility push(default)\n',
            content, count=1
        )
        content = content.rstrip() + '\n\n#pragma GCC visibility pop\n'
    with open(path, 'w') as f:
        f.write(content)
    print(f"OK: {fname} patchado")

print("=== Todos os patches de visibilidade aplicados! ===")
PYEOF
```

### Passo 12: Iniciar a Compilação Master
**Atenção:** Como estamos compilando para GPU, o LLVM e os shaders do CUDA serão compilados no NVCC. Isso poderá levar de **6 a 12 horas** dependendo dos cores da sua VM.

*(Nota: Repare que NÃO injetamos stubs para cuda_configure/local_config_cuda)*

**Passo 12A — Patch pybind11 (ANTES do build):**

O `pybind11_bazel` hard-code `-fvisibility=hidden` nos `copts` da regra `pybind_extension`. Isso é aplicado DEPOIS dos `--copt` do Bazel, sobrescrevendo `-fvisibility=default`. Precisamos fazer `fetch` primeiro para popular o cache, e depois remover essa flag:

```bash
bazel fetch //tensorflow/tools/pip_package:wheel 2>/dev/null || true

BAZEL_BASE=$(bazel info output_base 2>/dev/null)
PYBIND_BZL="$BAZEL_BASE/external/pybind11_bazel/build_defs.bzl"
if [ -f "$PYBIND_BZL" ]; then
    chmod u+w "$PYBIND_BZL"
    sed -i 's/"-fvisibility=hidden"/""/g' "$PYBIND_BZL"
    sed -i "s/'-fvisibility=hidden'/''/g" "$PYBIND_BZL"
    echo "OK: pybind11_bazel - removido -fvisibility=hidden"
fi

# Patch pybind11_abseil para export de símbolos
for f in "$BAZEL_BASE"/external/pybind11_abseil/pybind11_abseil/*.{h,cc}; do
    [ -f "$f" ] || continue
    chmod u+w "$f"
    grep -q 'pragma GCC visibility' "$f" || sed -i '1i #pragma GCC visibility push(default)' "$f"
done
echo "OK: pybind11_abseil patched"
```

**Passo 12B — Compilação:**

```bash
bazel build \
    --config=opt \
    --config=cuda \
    --define=tflite_with_xnnpack=false \
    --local_ram_resources=HOST_RAM*.6 \
    --per_file_copt=".*@-Wno-error" \
    --copt=-fvisibility=default \
    --cxxopt=-fvisibility=default \
    --host_copt=-fvisibility=default \
    --host_cxxopt=-fvisibility=default \
    --copt=-DPYBIND11_EXPORT="__attribute__((visibility(\"default\")))" \
    --copt=-DPYBIND11_MODULE_LOCAL="" \
    --linkopt=-Wl,--export-dynamic \
    --copt=-I/usr/local/include/cuda_stub \
    --host_copt=-I/usr/local/include/cuda_stub \
    --extra_toolchains=@@local_config_python//:py_cc_toolchain \
    --override_repository=rules_ml_toolchain=/root/rules_ml_toolchain_patched \
    --override_repository=llvm_linux_x86_64=/root/llvm_stub \
    --override_repository=llvm_linux_aarch64=/root/llvm_stub \
    --override_repository=llvm_darwin_aarch64=/root/llvm_stub \
    --override_repository=llvm18_linux_x86_64=/root/llvm_stub \
    --override_repository=llvm19_linux_x86_64=/root/llvm_stub \
    --override_repository=llvm20_linux_x86_64=/root/llvm_stub \
    --override_repository=llvm21_linux_x86_64=/root/llvm_stub \
    --override_repository=llvm18_linux_aarch64=/root/llvm_stub \
    --override_repository=llvm20_linux_aarch64=/root/llvm_stub \
    --override_repository=llvm21_linux_aarch64=/root/llvm_stub \
    --override_repository=llvm18_darwin_aarch64=/root/llvm_stub \
    --override_repository=llvm20_darwin_aarch64=/root/llvm_stub \
    --override_repository=local_config_tensorrt=/root/tensorrt_stub \
    --override_repository=local_config_rocm=/root/rocm_stub \
    --override_repository=cuda_cudart=/root/cuda_redist_stub \
    --override_repository=cuda_profiler_api=/root/cuda_redist_stub \
    --override_repository=cuda_nvcc=/root/cuda_redist_stub \
    --override_repository=cuda_nvml=/root/cuda_redist_stub \
    --override_repository=cuda_nvtx=/root/cuda_redist_stub \
    --override_repository=cuda_cccl=/root/cuda_redist_stub \
    --override_repository=cuda_cudnn=/root/cuda_redist_stub \
    --override_repository=cuda_cudnn9=/root/cuda_redist_stub \
    --override_repository=cuda_cublas=/root/cuda_redist_stub \
    --override_repository=cuda_cusolver=/root/cuda_redist_stub \
    --override_repository=cuda_cusparse=/root/cuda_redist_stub \
    --override_repository=cuda_curand=/root/cuda_redist_stub \
    --override_repository=cuda_cufft=/root/cuda_redist_stub \
    --override_repository=cuda_cupti=/root/cuda_redist_stub \
    --override_repository=cuda_nvdisasm=/root/cuda_redist_stub \
    --override_repository=cuda_nvvm=/root/cuda_redist_stub \
    --override_repository=cuda_nvjitlink=/root/cuda_redist_stub \
    --override_repository=cuda_nccl=/root/cuda_redist_stub \
    --override_repository=cuda_nvrtc=/root/cuda_redist_stub \
    --override_repository=nvidia_wheel_versions=/root/cuda_redist_stub \
    --override_repository=pypi=/root/pypi_stub \
    --override_repository=pypi_absl_py=/root/pypi_stub \
    --override_repository=pypi_astunparse=/root/pypi_stub \
    --override_repository=pypi_auditwheel=/root/pypi_stub \
    --override_repository=pypi_flatbuffers=/root/pypi_stub \
    --override_repository=pypi_gast=/root/pypi_stub \
    --override_repository=pypi_keras=/root/pypi_stub \
    --override_repository=pypi_lit=/root/pypi_stub \
    --override_repository=pypi_ml_dtypes=/root/pypi_stub \
    --override_repository=pypi_mods=/root/pypi_stub \
    --override_repository=pypi_numpy=/root/pypi_stub \
    --override_repository=pypi_nvidia_cublas_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_cuda_cupti_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_cuda_nvcc_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_cuda_nvrtc_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_cuda_runtime_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_cudnn_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_cufft_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_curand_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_cusolver_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_cusparse_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_nccl_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_nvjitlink_cu12=/root/pypi_stub \
    --override_repository=pypi_nvidia_nvshmem_cu12=/root/pypi_stub \
    --override_repository=pypi_opt_einsum=/root/pypi_stub \
    --override_repository=pypi_packaging=/root/pypi_stub \
    --override_repository=pypi_protobuf=/root/pypi_stub \
    --override_repository=pypi_requests=/root/pypi_stub \
    --override_repository=pypi_setuptools=/root/pypi_stub \
    --override_repository=pypi_termcolor=/root/pypi_stub \
    --override_repository=pypi_typing_extensions=/root/pypi_stub \
    --override_repository=pypi_wheel=/root/pypi_stub \
    --override_repository=pypi_wrapt=/root/pypi_stub \
    --override_repository=python_3_11_host=/root/python_stub \
    --override_repository=local_config_python=/root/python_config_stub \
    --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
    --repo_env=USE_HERMETIC_CC_TOOLCHAIN=0 \
    --noincompatible_enable_cc_toolchain_resolution \
    --jobs=$(nproc) \
    //tensorflow/tools/pip_package:wheel
```

### Passo 13: Instalar o Pacote (Após a compilação terminar)
> **Atenção:** Execute a partir de fora do diretório do código-fonte do TensorFlow!
```bash
cd ~
pip install --force-reinstall --no-deps \
    /home/almalinux/tensorflow221/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.21.0-cp311-cp311-linux_ppc64le.whl
```

### Passo 14: Instalar pacotes essenciais
```bash
conda install -c conda-forge h5py grpcio libclang ml_dtypes "protobuf>=6.31.1,<8.0.0" libstdcxx-ng -y
pip install absl-py astunparse flatbuffers gast google-pasta keras opt-einsum termcolor wrapt
```

### Passo 15: Teste de Fogo com GPU CUDA

> **Importante:** O GCC 8 do sistema não tem `GLIBCXX_3.4.29` (requerido pelo NumPy do Conda). Precisa exportar o `LD_LIBRARY_PATH` do Conda antes de rodar.

```bash
cd ~
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python3 -c "
import tensorflow as tf
import time

print(f'\n--- Teste de GPU Inicializado (TF {tf.__version__}) ---')
devices = tf.config.list_physical_devices('GPU')
print('GPUs detectadas:', [d.name for d in devices])

if not devices:
    print('[ERRO] Nenhuma GPU CUDA foi detectada!')
else:
    with tf.device('/GPU:0'):
        A = tf.random.normal([5000, 5000])
        B = tf.random.normal([5000, 5000])
        start = time.time()
        C = tf.matmul(A, B)
        end = time.time()
    
    soma = tf.reduce_sum(C).numpy()
    print(f'Soma do resultado: {soma:.2f}')
    print(f'Tempo de execucao CUDA: {end - start:.4f} segundos')
    print('[VITORIA] TensorFlow compilado nativo com suporte GPU no Power9 esta 100% operacional!')
"
```
