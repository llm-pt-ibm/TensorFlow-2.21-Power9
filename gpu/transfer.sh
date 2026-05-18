#!/bin/bash
# Tentar --jit=true, --mlir-print-ir-before-all p/ ver onde trava

set -o pipefail
source /root/miniforge3/etc/profile.d/conda.sh
conda activate tf221_build || conda activate base || true

CACHE_BASE=/root/.cache/bazel/_bazel_root/cfc10a9bac1c788901c46f30aa7da373/execroot/org_tensorflow
TF_BUILD=$HOME/tensorflow_gpu/tf221_workspace/tensorflow
HLO=$CACHE_BASE/bazel-out/ppc-opt/bin/tensorflow/compiler/mlir/tools/kernel_gen/hlo_to_kernel
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$HOME/cuda_unified/lib64"
export MLIR_CRASH_REPRODUCER_DIRECTORY=/tmp

TMPL="$TF_BUILD/tensorflow/core/kernels/mlir_generated/op_definitions/add_v2.mlir.tmpl"
MLIR=/tmp/add_v2_f32.mlir
sed 's/platform/gpu/g; s/elem_type/f32/g; s/output_type/f32/g' "$TMPL" > "$MLIR"

echo "=== [A] Tentar --jit=true (compila em runtime, sem PTX agora) ==="
OUT=/tmp/add_v2_jit.o
rm -f "$OUT"
LOG=/tmp/hlo_jit_$(date +%H%M%S).log
echo "  log: $LOG"
timeout 60 "$HLO" \
    --tile_sizes=256 --host-triple=powerpc64le-linux-gnu --arch=sm_70 \
    --input="$MLIR" --output="$OUT" \
    --enable_ftz=true --jit_i64_indexed_for_large_tensors=false \
    --jit=true 2>&1 | tee "$LOG" | head -40 | sed 's/^/    /'
RC=$?
echo ""
echo "  exit $RC ($([ $RC -eq 124 ] && echo TIMEOUT))"
[ -f "$OUT" ] && echo "  output: $(stat -c %s $OUT) bytes" || echo "  no output"
echo ""

echo "=== [B] Tentar --mlir-print-ir-before-all + --mlir-disable-threading 30s ==="
LOG=/tmp/hlo_verbose_$(date +%H%M%S).log
echo "  log: $LOG"
timeout 30 "$HLO" \
    --tile_sizes=256 --host-triple=powerpc64le-linux-gnu --arch=sm_70 \
    --input="$MLIR" --output=/tmp/test.o \
    --enable_ftz=true --jit_i64_indexed_for_large_tensors=false --jit=false \
    --mlir-disable-threading \
    --mlir-print-ir-before-all 2>&1 | tee "$LOG" > /dev/null
RC=$?
echo "  exit $RC"
echo ""
echo "  Ultimas 30 linhas do log (provavelmente mostra a pass onde travou):"
tail -30 "$LOG" 2>/dev/null | sed 's/^/    /'
echo ""

echo "=== [C] Reprod files gerados? ==="
ls -la /tmp/*.mlir 2>/dev/null | grep -v "add_v2_f32.mlir" | head -5 | sed 's/^/    /'
ls -la /tmp/mlir_reproducer* 2>/dev/null | head -5 | sed 's/^/    /'
