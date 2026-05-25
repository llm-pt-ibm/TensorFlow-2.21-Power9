#!/bin/bash
# Point the TF build at the newly-patched LLD.
#
# The TF build script (gpu/build_tf221_power9_gpu_generic.sh) probes:
#   $HOME/tensorflow_gpu/llvm-install/bin/ld.lld
# and falls back to the Conda LLD if missing.
#
# We just swap that path to point at the freshly-validated LLD at
# /root/lld-ppc64-install (kept around so the standalone artifact remains
# usable). The previous content of /root/tensorflow_gpu/llvm-install (if any)
# is renamed *.bak.<timestamp> rather than deleted, so you can roll back.
#
# Usage on the container:
#   bash transfer.sh                # do the swap
#   ROLLBACK=1 bash transfer.sh     # restore the most recent .bak

(
set -e

NEW_LLD_DIR="${NEW_LLD_DIR:-/root/lld-ppc64-install}"
TF_LLD_DIR="${TF_LLD_DIR:-$HOME/tensorflow_gpu/llvm-install}"

if [ "${ROLLBACK:-0}" = "1" ]; then
    LATEST_BAK="$(ls -1dt "${TF_LLD_DIR}".bak.* 2>/dev/null | head -1)"
    if [ -z "$LATEST_BAK" ]; then
        echo "ERROR: no backup found matching ${TF_LLD_DIR}.bak.*"
        exit 1
    fi
    echo ">>> Rolling back: removing current $TF_LLD_DIR, restoring $LATEST_BAK"
    rm -rf "$TF_LLD_DIR"
    mv "$LATEST_BAK" "$TF_LLD_DIR"
    ls -la "$TF_LLD_DIR/bin/ld.lld"
    exit 0
fi

echo "=== Pointing TF build at $NEW_LLD_DIR ==="

if [ ! -x "$NEW_LLD_DIR/bin/ld.lld" ]; then
    echo "ERROR: $NEW_LLD_DIR/bin/ld.lld not found or not executable."
    echo "       Build/validate it first (see lld-ppc64-fix/build_patched_lld.sh)."
    exit 1
fi

mkdir -p "$(dirname "$TF_LLD_DIR")"

# Backup whatever's currently there (file, dir, or symlink)
if [ -e "$TF_LLD_DIR" ] || [ -L "$TF_LLD_DIR" ]; then
    TS="$(date +%Y%m%d-%H%M%S)"
    BAK="${TF_LLD_DIR}.bak.${TS}"
    echo ">>> Existing $TF_LLD_DIR found — moving to $BAK"
    mv "$TF_LLD_DIR" "$BAK"
fi

# Symlink the new install into the path TF expects
ln -s "$NEW_LLD_DIR" "$TF_LLD_DIR"
echo ">>> Symlinked $TF_LLD_DIR -> $NEW_LLD_DIR"

# Sanity check: ld.lld resolves and reports the patched version
echo ""
echo "=== Sanity check ==="
ls -la "$TF_LLD_DIR/bin/ld.lld"
"$TF_LLD_DIR/bin/ld.lld" --version | head -1

# Quick smoke test: link a trivial .so using the path the TF script will use
TD="$(mktemp -d)"
cat > "$TD/x.c" <<'EOF'
int x(void) { return 42; }
EOF
gcc -c -fPIC "$TD/x.c" -o "$TD/x.o" 2>/dev/null || true
if "$TF_LLD_DIR/bin/ld.lld" --shared "$TD/x.o" -o "$TD/libx.so" 2>"$TD/err"; then
    echo ">>> Smoke link OK: $(file "$TD/libx.so" | head -1)"
else
    echo ">>> WARN: smoke link failed:"
    sed 's/^/    /' "$TD/err"
fi
rm -rf "$TD"

echo ""
echo "============================================="
echo " TF build is now wired to the patched LLD."
echo ""
echo " Next: run your TF build as usual, e.g."
echo "   cd $HOME/TensorFlow-2.21-Power9   # adjust path"
echo "   bash gpu/build_tf221_power9_gpu_generic.sh"
echo ""
echo " The TF script will pick up:"
echo "   $TF_LLD_DIR/bin/ld.lld"
echo " and set LD_LIBRARY_PATH=$TF_LLD_DIR/lib automatically."
echo ""
echo " Rollback: bash transfer.sh ROLLBACK=1"
echo "============================================="
)
