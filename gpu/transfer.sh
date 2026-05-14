#!/bin/bash
# Reaplicar patcher v8 (que cobre BFD .long_branch.* stubs) - estado funcional
# conhecido onde TF importa + GPUs sao detectadas.

TF_DIR=/root/miniforge3/envs/tf221_build/lib/python3.11/site-packages/tensorflow
export LD_LIBRARY_PATH=/root/cuda_unified/lib64:/root/miniforge3/envs/tf221_build/lib:$LD_LIBRARY_PATH

PATCHER=/tmp/patch_nv_toc_save_v8.py

if [ ! -f "$PATCHER" ]; then
    echo "Patcher v8 nao existe ainda. Recriando..."
    cat > "$PATCHER" << 'PATCHER_EOF'
#!/usr/bin/env python3
import argparse, bisect, shutil, struct, sys
from pathlib import Path

PPC64_INSN_STD_R2_24_R1 = 0xF8410018
PPC64_INSN_LD_R2_24_R1 = 0xE8410018
PPC64_INSN_NOP = 0x60000000

CUDA_PREFIXES = (
    "__cuda", "__nv", "cuda", "cublas", "cudnn", "cufft",
    "curand", "cusolver", "cusparse", "cupti",
    "nccl", "nvml", "nvrtc", "nvshmem",
)

def is_cuda_target(name):
    if not name: return False
    for p in CUDA_PREFIXES:
        if name.startswith(p): return True
    if len(name) >= 3 and name[:2] == "cu" and name[2].isupper():
        return True
    return False

def encode_b(disp):
    w = disp // 4
    return (18 << 26) | ((w & ((1 << 24) - 1)) << 2)

def decode_b(insn):
    if ((insn >> 26) & 0x3F) != 18: return None
    if ((insn >> 1) & 1) != 0: return None
    if (insn & 1) != 0: return None
    li = (insn >> 2) & ((1 << 24) - 1)
    if li & (1 << 23): li -= (1 << 24)
    return li * 4

def decode_bl(insn):
    if ((insn >> 26) & 0x3F) != 18: return None
    if ((insn >> 1) & 1) != 0: return None
    if (insn & 1) != 1: return None
    li = (insn >> 2) & ((1 << 24) - 1)
    if li & (1 << 23): li -= (1 << 24)
    return li * 4

def encode_bl(disp):
    w = disp // 4
    return (18 << 26) | ((w & ((1 << 24) - 1)) << 2) | 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("so_path")
    args = ap.parse_args()
    so_path = Path(args.so_path)
    from elftools.elf.elffile import ELFFile
    with open(so_path, 'r+b') as f:
        elf = ELFFile(f)
        text = elf.get_section_by_name(".text")
        if text is None: return 1
        tv = text["sh_addr"]; to = text["sh_offset"]; ts = text["sh_size"]
        ss = elf.get_section_by_name(".symtab") or elf.get_section_by_name(".dynsym")
        if ss is None: return 1
        cuda_stubs_bfd = {}
        cuda_stubs_lld = {}
        for sym in ss.iter_symbols():
            name = sym.name or ""
            if name.startswith("__long_branch_"):
                tail = name[len("__long_branch_"):]
                if is_cuda_target(tail):
                    cuda_stubs_lld[sym["st_value"]] = name
            elif ".long_branch." in name:
                idx = name.find(".long_branch.")
                tail = name[idx + len(".long_branch."):]
                if is_cuda_target(tail):
                    cuda_stubs_bfd[sym["st_value"]] = name
        print(f">>> Stubs LLD: {len(cuda_stubs_lld)} BFD: {len(cuda_stubs_bfd)}")
        f.seek(to)
        td = bytearray(f.read(ts))
        bfd_patched = bfd_skipped = 0
        for sv in sorted(cuda_stubs_bfd):
            o = sv - tv
            if o < 0 or o + 16 > ts: bfd_skipped += 1; continue
            insn0 = struct.unpack_from("<I", td, o)[0]
            insn1 = struct.unpack_from("<I", td, o + 4)[0]
            if insn0 == PPC64_INSN_STD_R2_24_R1: continue
            disp_orig = decode_b(insn0)
            if disp_orig is None or insn1 != 0:
                bfd_skipped += 1; continue
            struct.pack_into("<II", td, o, PPC64_INSN_STD_R2_24_R1, encode_b(disp_orig - 4))
            bfd_patched += 1
        print(f">>> BFD stubs: patched={bfd_patched} skipped={bfd_skipped}")
        w0 = PPC64_INSN_STD_R2_24_R1
        w1_back20 = encode_b(-20)
        lld_patched = 0
        stub2wrap_lld = {}
        for sv in sorted(cuda_stubs_lld):
            o = sv - tv
            if o + 32 > ts: continue
            pad = o + 16
            ex = bytes(td[pad:pad+8])
            if struct.unpack("<II", ex) == (w0, w1_back20):
                stub2wrap_lld[sv] = sv + 16; continue
            if ex != b"\x00"*8: continue
            struct.pack_into("<II", td, pad, w0, w1_back20)
            stub2wrap_lld[sv] = sv + 16
            lld_patched += 1
        print(f">>> LLD wrappers: {lld_patched}")
        wrap_set = set(stub2wrap_lld.values())
        bfd_set = set(cuda_stubs_bfd.keys())
        nops = 0
        for off in range(0, len(td) - 3, 4):
            insn = struct.unpack_from("<I", td, off)[0]
            disp = decode_bl(insn)
            if disp is None: continue
            tgt = tv + off + disp
            need = False
            if tgt in stub2wrap_lld:
                struct.pack_into("<I", td, off, encode_bl(stub2wrap_lld[tgt] - (tv + off)))
                need = True
            elif tgt in wrap_set or tgt in bfd_set:
                need = True
            if need and off + 4 < len(td):
                ni = struct.unpack_from("<I", td, off + 4)[0]
                if ni == PPC64_INSN_NOP:
                    struct.pack_into("<I", td, off + 4, PPC64_INSN_LD_R2_24_R1)
                    nops += 1
        print(f">>> nops->ld r2: {nops}")
        f.seek(to); f.write(bytes(td)); f.flush()
    print(">>> Patch v8 OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
PATCHER_EOF
fi

echo "=== Aplicar v8 em todas .so com long_branch stubs ==="
for so in $(find "$TF_DIR" -name '*.so*' -type f 2>/dev/null | grep -v '\.bak'); do
    if nm "$so" 2>/dev/null | grep -qE '__long_branch_|\.long_branch\.'; then
        echo "--- $(basename $so) ---"
        python3 "$PATCHER" "$so"
    fi
done
echo ""

echo "=== Teste import + list GPUs ==="
python3 << 'PYEOF'
import tensorflow as tf
print('TF', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', len(gpus))
for g in gpus:
    print(' ', g)
PYEOF
echo "exit: $?"
