#!/bin/bash
# Diagnostic: is libppc64_savres.a actually being used by the TF build?
#
# Checks libtensorflow_framework.so for:
#   - undefined refs (U/u) to _savegpr_NN / _restgpr_NN / _savefpr_NN / _restfpr_NN / _savevr_NN / _restvr_NN
#       -> means SOME .o needs them; savres genuinely resolved the link
#   - defined symbols (T/t) for the same names
#       -> --whole-archive may have injected them; could be dead weight
#
# Interpretation:
#   undefined=0 defined=0  -> savres is completely inert; safe to drop from build
#   undefined=0 defined>0  -> --whole-archive forced inclusion, but nothing references; ~few KB bloat
#   undefined>0 defined>0  -> savres is genuinely resolving real refs; KEEP IT
#
# Usage on the container:
#   bash transfer.sh

(
set -e

SO_LIST=(
    "/root/miniforge3/envs/tf221_build/lib/python3.11/site-packages/tensorflow/libtensorflow_framework.so.2"
    "/root/.cache/bazel/_bazel_root/fc0772529e3515d3f8f16399ba3b0fa0/execroot/org_tensorflow/bazel-out/ppc-opt/bin/tensorflow/python/libtensorflow_framework.so.2"
)

for SO in "${SO_LIST[@]}"; do
    echo "=== $SO ==="
    if [ ! -f "$SO" ]; then
        echo "  (not found, skipping)"
        echo ""
        continue
    fi
    echo "  size: $(ls -la "$SO" | awk '{print $5}') bytes"

    UNDEF_COUNT=$(nm "$SO" 2>&1 | grep -cE ' [Uu] _(save|rest)(gpr|fpr|vr)_[0-9]+' || true)
    DEF_COUNT=$(nm "$SO" 2>&1   | grep -cE ' [Tt] _(save|rest)(gpr|fpr|vr)_[0-9]+' || true)

    echo "  undefined refs (U/u): $UNDEF_COUNT"
    echo "  defined symbols (T/t): $DEF_COUNT"

    if [ "$DEF_COUNT" -gt 0 ]; then
        echo "  sample defined (first 5):"
        nm "$SO" 2>&1 | grep -E ' [Tt] _(save|rest)(gpr|fpr|vr)_[0-9]+' | head -5 | sed 's/^/    /'
    fi

    if [ "$UNDEF_COUNT" -gt 0 ]; then
        echo "  sample undefined (first 5):"
        nm "$SO" 2>&1 | grep -E ' [Uu] _(save|rest)(gpr|fpr|vr)_[0-9]+' | head -5 | sed 's/^/    /'
    fi

    echo ""
    echo "  verdict:"
    if [ "$UNDEF_COUNT" = "0" ] && [ "$DEF_COUNT" = "0" ]; then
        echo "    INERT — savres is dead weight in this pipeline. Safe to drop."
    elif [ "$UNDEF_COUNT" = "0" ] && [ "$DEF_COUNT" -gt 0 ]; then
        echo "    BLOAT — savres force-linked via --whole-archive, but nothing references it."
    else
        echo "    GENUINELY USED — keep libppc64_savres.a, real refs being resolved."
    fi
    echo ""
done

# Bonus: check if any OTHER installed .so still references these symbols
# (e.g. CUDA/cuDNN libs compiled with GCC elsewhere).
echo "=== cross-check: scan other installed .so for unresolved save/restore refs ==="
echo "  (any 'U _savegpr_NN' here means GCC-compiled lib still in the loader path)"
for D in /root/cuda_unified/lib64 /root/miniforge3/envs/tf221_build/lib /root/tensorflow_gpu/llvm-install/lib ; do
    [ -d "$D" ] || continue
    find "$D" -maxdepth 1 -name '*.so*' -type f 2>/dev/null | while read F; do
        N=$(nm "$F" 2>&1 | grep -cE ' [Uu] _(save|rest)(gpr|fpr|vr)_[0-9]+' || true)
        [ "$N" -gt 0 ] && echo "    $F: $N undefined save/restore refs"
    done
done
echo "  (silence above = no other lib references them; savres truly only matters for the TF .so)"
)
