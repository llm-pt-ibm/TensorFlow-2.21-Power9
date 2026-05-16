#!/bin/bash
# Forcar Bazel a recompilar todos .tramp.S via wrapper fixado.
# Depois rebuild incremental (so re-link das libs que usam, ~10-30min)

set -o pipefail
source /root/miniforge3/etc/profile.d/conda.sh
conda activate tf221_build || conda activate base || true

CACHE_BASE="/root/.cache/bazel/_bazel_root/cfc10a9bac1c788901c46f30aa7da373/execroot/org_tensorflow"

echo "=== [1] Deletar .tramp.{o,d,pic.o,pic.d} cached ==="
COUNT=$(find "$CACHE_BASE/bazel-out" -type f \( -name '*.tramp.pic.o' -o -name '*.tramp.o' -o -name '*.tramp.pic.d' -o -name '*.tramp.d' \) 2>/dev/null | wc -l)
echo ">>> $COUNT arquivos a deletar:"
find "$CACHE_BASE/bazel-out" -type f \( -name '*.tramp.pic.o' -o -name '*.tramp.o' -o -name '*.tramp.pic.d' -o -name '*.tramp.d' \) 2>/dev/null | head -20 | sed 's/^/    /'
echo ""
find "$CACHE_BASE/bazel-out" -type f \( -name '*.tramp.pic.o' -o -name '*.tramp.o' -o -name '*.tramp.pic.d' -o -name '*.tramp.d' \) -delete 2>/dev/null
echo ">>> Apos delete:"
find "$CACHE_BASE/bazel-out" -type f \( -name '*.tramp.pic.o' -o -name '*.tramp.o' \) 2>/dev/null | wc -l
echo ""

echo "=== [2] Tambem deletar .so finais que dependem dos .tramp.o (forca relink) ==="
# Bazel só re-linka se algum dep mudou. Forçando delete das .so principais.
find "$CACHE_BASE/bazel-out" -type f -name "libtensorflow_framework.so*" -delete 2>/dev/null
find "$CACHE_BASE/bazel-out" -type f -name "libtensorflow_cc.so*" -delete 2>/dev/null
find "$CACHE_BASE/bazel-out" -type f -name "_pywrap_*.so" -delete 2>/dev/null
find "$CACHE_BASE/bazel-out" -type f -name "lib_pywrap_*.so" -delete 2>/dev/null
echo ">>> .so principais deletadas, Bazel ira relinkar"
echo ""

echo "=== [3] Re-rodar build_tf221_power9_gpu_generic.sh (incremental) ==="
echo "    Esperado: Bazel recompila ~19 .tramp.S via wrapper fixado (~1-2min)"
echo "    + relink libtensorflow_framework, libtensorflow_cc, plugins (~10-30min)"
echo "    + pip install + diagnostico final"
echo ""
echo ">>> Rodando agora (com SKIP_PATCHER_V6=1)..."
cd "$HOME/tensorflow_gpu/tf221_workspace/tensorflow" 2>/dev/null || cd ~

# Achar o build script
BUILD_SCRIPT=""
for P in \
    "$HOME/tensorflow_gpu/build_tf221_power9_gpu_generic.sh" \
    "$HOME/tensorflow_gpu/gpu/build_tf221_power9_gpu_generic.sh" \
    "$(pwd)/build_tf221_power9_gpu_generic.sh"; do
    if [ -f "$P" ]; then BUILD_SCRIPT="$P"; break; fi
done

if [ -z "$BUILD_SCRIPT" ]; then
    echo "  build_tf221_power9_gpu_generic.sh nao encontrado. Achados:"
    find $HOME -name "build_tf221_power9_gpu_generic.sh" 2>/dev/null | head -5
    exit 1
fi
echo "  Usando: $BUILD_SCRIPT"
echo ""
echo "Rodar manualmente com:"
echo "  SKIP_PATCHER_V6=1 bash $BUILD_SCRIPT"
echo ""
echo "OU se preferir validacao rapida apenas, re-rode so a parte de bazel + pip install."
