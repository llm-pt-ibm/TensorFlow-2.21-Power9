#!/bin/bash
# Diagnosticar: stOther dos cudart trampolines + ver se R2SaveStubs foram criados

set -o pipefail
source /root/miniforge3/etc/profile.d/conda.sh
conda activate tf221_build

# Restaurar do .bak-v11 antes (estado pos-LLD-patchado, pre-v11-v12)
TF=/root/miniforge3/envs/tf221_build/lib/python3.11/site-packages/tensorflow
PLUGIN="$TF/python/profiler/internal/_pywrap_profiler_plugin.so"

echo "=== [1] Restore do estado original LLD patchado (pre-patcher-binario) ==="
for SO in "$TF/libtensorflow_framework.so.2" "$TF/libtensorflow_cc.so.2" \
         "$TF/python/lib_pywrap_tensorflow_common.so" "$PLUGIN"; do
    # Procura backup mais antigo (.bak ou .bak-v11)
    for BAK in "$SO.bak" "$SO.bak-v11"; do
        if [ -f "$BAK" ]; then
            cp "$BAK" "$SO"
            echo "  restored $(basename "$SO") from $(basename "$BAK")"
            break
        fi
    done
done
echo ""

echo "=== [2] stOther dos cudart trampolines em _pywrap_profiler_plugin.so ==="
python3 << 'PYEOF'
from elftools.elf.elffile import ELFFile

PLUGIN = "/root/miniforge3/envs/tf221_build/lib/python3.11/site-packages/tensorflow/python/profiler/internal/_pywrap_profiler_plugin.so"

CUDART_NAMES = (
    "__cudaRegisterFatBinary", "__cudaRegisterFatBinaryEnd",
    "__cudaRegisterFunction", "__cudaRegisterVar",
    "cudaMalloc", "cudaFree", "cudaMemcpy",
    "cublasCreate_v2", "cudnnCreate",
)

# stOther bits PPC64 ELFv2:
#   bits 7:5 = localentry value (0..7)
#     0 = no global entry / leaf
#     1 = function clobbers r2 (caller must save)
#     2..7 = power-of-2 offset of local entry from global
def decode_stOther(so):
    le = (so >> 5) & 0x7
    if le == 0: return "no-GE/leaf"
    if le == 1: return "CLOBBERS_R2 (localentry=1)"
    return f"GE+{1 << (le-1)} bytes -> LE"

with open(PLUGIN, "rb") as f:
    e = ELFFile(f)
    ss = e.get_section_by_name(".symtab") or e.get_section_by_name(".dynsym")
    print(f"{'name':50} {'value':>14}  stOther  localentry")
    for sym in ss.iter_symbols():
        n = sym.name or ""
        if not any(n.startswith(x) for x in CUDART_NAMES): continue
        if sym["st_value"] == 0: continue  # UND
        so = sym["st_other"]["visibility"] if isinstance(sym["st_other"], dict) else sym["st_other"]
        # pyelftools converte st_other em dict {visibility, local}.
        # Vamos pegar raw bytes pra ler bits 7:5
        # Use o atributo entry
        raw_so = sym.entry["st_other"]
        if isinstance(raw_so, dict):
            # decode dict: bits 7:5 dentro de 'local' field?
            # Fallback: usar getattr ou inspecionar
            print(f"  RAW: {sym.entry['st_other']}")
            continue
        print(f"{n[:50]:50} {sym['st_value']:#14x}  {raw_so:#04x}     {decode_stOther(raw_so)}")
PYEOF
echo ""

echo "=== [3] readelf -s diretamente (mostra stOther string) ==="
readelf -sW "$PLUGIN" 2>/dev/null | grep -E "__cudaRegisterFatBinary\b|__cudaRegisterFunction\b|cudaMalloc\b|cublasCreate" | head -20
echo ""

echo "=== [4] Procurar __toc_save_ stubs (R2SaveStubs criados pelo LLD) ==="
readelf -sW "$PLUGIN" 2>/dev/null | grep "__toc_save_" | head -10
COUNT=$(readelf -sW "$PLUGIN" 2>/dev/null | grep -c "__toc_save_")
echo "Total __toc_save_ stubs em _pywrap_profiler_plugin.so: $COUNT"
echo ""

echo "=== [5] Mesmo em libtensorflow_framework.so.2 ==="
TF_FW="$TF/libtensorflow_framework.so.2"
readelf -sW "$TF_FW" 2>/dev/null | grep -E "__cudaRegisterFatBinary\b" | head -5
echo ""
COUNT2=$(readelf -sW "$TF_FW" 2>/dev/null | grep -c "__toc_save_")
echo "Total __toc_save_ stubs em libtensorflow_framework.so.2: $COUNT2"
echo ""

echo "=== [6] Verificar se o LLD pacheado esta sendo usado ==="
ls -la $HOME/tensorflow_gpu/llvm-install/bin/ld.lld 2>/dev/null
echo ""
echo ">>> Strings de version no LLD binary:"
strings $HOME/tensorflow_gpu/llvm-install/bin/ld.lld 2>/dev/null | grep -iE "LLD .*Compatible" | head -2
