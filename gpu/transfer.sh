#!/bin/bash
# Invalidar TODOS .o que dependem de cudnn + .so finais, depois rodar build + testes

set -o pipefail
source /root/miniforge3/etc/profile.d/conda.sh
conda activate tf221_build || conda activate base || true

CACHE_BASE=/root/.cache/bazel/_bazel_root/cfc10a9bac1c788901c46f30aa7da373/execroot/org_tensorflow

echo "=== [1] Invalidar TODOS .o com dep em cudnn headers ==="
COUNT=0
for D in $(find "$CACHE_BASE/bazel-out" -name "*.d" 2>/dev/null); do
    if grep -q "cudnn" "$D" 2>/dev/null; then
        O="${D%.d}.o"
        if [ -f "$O" ]; then
            rm -f "$O" "$D"
            COUNT=$((COUNT+1))
        fi
    fi
done
echo "  $COUNT pares .o/.d deletados"
echo ""

echo "=== [2] Deletar .so finais (forca relink) ==="
find "$CACHE_BASE/bazel-out" -type f \( \
    -name "libtensorflow_framework.so*" -o \
    -name "libtensorflow_cc.so*" -o \
    -name "_pywrap_*.so" -o \
    -name "lib_pywrap_*.so" \
\) -delete 2>/dev/null
echo "  .so principais deletadas"
echo ""

echo "=== [3] Build + Testes ==="
BUILD_SCRIPT="$HOME/tensorflow_gpu/tensorflow_gpu.sh"
TESTS="$HOME/tensorflow_gpu/gpu_tests.py"
LOG="/tmp/tf_build_$(date +%Y%m%d_%H%M%S).log"

bash "$BUILD_SCRIPT" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo ""
echo "Build rc=$RC | log: $LOG"
[ $RC -ne 0 ] && exit $RC

python3 "$TESTS"
