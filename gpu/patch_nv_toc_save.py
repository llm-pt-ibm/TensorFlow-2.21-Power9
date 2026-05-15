#!/usr/bin/env python3
"""
Patcher v6 para libs do TF 2.21 PPC64LE buildadas com LLD.

CAUSA RAIZ:
- Libs grandes (>64KB TOC) forçam LLD a usar multi-TOC.
- GCC ELFv2 PPC64LE nem sempre emite `std r2, 24(r1)` em prólogos de funções
  que fazem bls a símbolos same-`.so` (assume intra-TOC).
- LLD insere __long_branch_* stubs ou deixa direct bls. Os targets (trampolines
  cudart_stub.cc etc.) têm global entry prologue que troca r2.
- Sem o save, o `ld r2, 24(r1)` que LLD patcha após o bl lê stack não-init →
  r2 vira lixo → cascata de crashes em init de __sti____cudaRegisterAll,
  PlatformInitialize, ToStatusSlow, etc.

FIX (3 phases):
1. Em padding (16 bytes zeros) APÓS cada `__long_branch_<cuda lib prefix>` stub,
   instala wrapper `std r2,24(r1); b stub`. Redireciona bls do stub pro wrapper.
2. Coleta paddings disponíveis em stubs `__long_branch_*` não-cuda como reserva.
3. Para bls diretas a cuda targets em funções SEM `std r2,24(r1)` no prólogo,
   instala wrapper em padding disponível dentro de range ±32MB.

bls em funções com save no prólogo só precisam de nop→`ld r2,24(r1)` após bl.

Uso:
    python3 patch_nv_toc_save.py <path/to/lib.so> [--backup] [--dry-run]
"""
import argparse, bisect, shutil, struct, sys
from pathlib import Path

PPC64_INSN_STD_R2_24_R1 = 0xF8410018
PPC64_INSN_LD_R2_24_R1 = 0xE8410018
PPC64_INSN_NOP = 0x60000000
BL_RANGE = 1 << 25  # +/- 32MB (bl tem 24-bit signed disp em words)

CUDA_PREFIXES = (
    "__cuda", "__nv",
    "cuda", "cublas", "cudnn", "cufft",
    "curand", "cusolver", "cusparse", "cupti",
    "nccl", "nvml", "nvrtc", "nvshmem",
)


def is_cuda_target(name):
    """Heuristica: name e um trampoline de lib CUDA gerado por implib_so."""
    if not name:
        return False
    for p in CUDA_PREFIXES:
        if name.startswith(p):
            return True
    # Driver API: cu[A-Z]... (cuInit, cuMemAlloc, etc.)
    if len(name) >= 3 and name[:2] == "cu" and name[2].isupper():
        return True
    return False


def encode_b(disp):
    w = disp // 4
    return (18 << 26) | ((w & ((1 << 24) - 1)) << 2)


def decode_bl(insn):
    if ((insn >> 26) & 0x3F) != 18:
        return None
    if ((insn >> 1) & 1) != 0:  # AA must be 0
        return None
    if (insn & 1) != 1:  # LK must be 1
        return None
    li = (insn >> 2) & ((1 << 24) - 1)
    if li & (1 << 23):
        li -= (1 << 24)
    return li * 4


def encode_bl(disp):
    w = disp // 4
    return (18 << 26) | ((w & ((1 << 24) - 1)) << 2) | 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("so_path")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    so_path = Path(args.so_path)
    if not so_path.is_file():
        print(f"ERROR: {so_path} nao existe", file=sys.stderr)
        return 1

    try:
        from elftools.elf.elffile import ELFFile
    except ImportError:
        print("ERROR: pyelftools nao instalado (pip install pyelftools)", file=sys.stderr)
        return 1

    if args.backup and not args.dry_run:
        bak = so_path.with_suffix(so_path.suffix + ".bak")
        if not bak.exists():
            print(f">>> Backup: {bak}")
            shutil.copy2(so_path, bak)

    mode = "rb" if args.dry_run else "r+b"
    with open(so_path, mode) as f:
        elf = ELFFile(f)
        text = elf.get_section_by_name(".text")
        if text is None:
            print("ERROR: sem .text", file=sys.stderr)
            return 1
        tv = text["sh_addr"]
        to = text["sh_offset"]
        ts = text["sh_size"]

        ss = (elf.get_section_by_name(".symtab")
              or elf.get_section_by_name(".dynsym"))
        if ss is None:
            print("ERROR: sem symbol table", file=sys.stderr)
            return 1

        cuda_stubs = {}
        cudart_func_addrs = set()
        all_funcs = []
        all_long_branch_stubs = []

        for sym in ss.iter_symbols():
            name = sym.name or ""
            if name.startswith("__long_branch_"):
                all_long_branch_stubs.append((sym["st_value"], name))
                tail = name[len("__long_branch_"):]
                if is_cuda_target(tail):
                    cuda_stubs[sym["st_value"]] = name
                continue
            if is_cuda_target(name) and sym["st_value"] > 0:
                cudart_func_addrs.add(sym["st_value"])
            st = sym["st_info"]
            if st["type"] == "STT_FUNC" and sym["st_size"] > 0 \
                    and sym["st_value"] > 0:
                all_funcs.append(
                    (sym["st_value"], sym["st_value"] + sym["st_size"])
                )

        all_funcs.sort()
        func_starts = [a for a, _ in all_funcs]
        all_long_branch_stubs.sort()

        print(f">>> Stubs LLD: {len(cuda_stubs)} cuda / "
              f"{len(all_long_branch_stubs)} total")
        print(f">>> Targets diretos cuda libs: {len(cudart_func_addrs)} | "
              f"Funcs no .so: {len(all_funcs)}")

        f.seek(to)
        td = bytearray(f.read(ts))

        save_cache = {}

        def func_has_save(start_v, end_v):
            """Verifica se a funcao tem `std r2, 24(r1)` nos primeiros 256 bytes."""
            if start_v in save_cache:
                return save_cache[start_v]
            scan_end = min(end_v, start_v + 256)
            o0 = start_v - tv
            o1 = scan_end - tv
            if o0 < 0 or o1 > len(td):
                save_cache[start_v] = False
                return False
            has = False
            for o in range(o0, o1, 4):
                if struct.unpack_from("<I", td, o)[0] == PPC64_INSN_STD_R2_24_R1:
                    has = True
                    break
            save_cache[start_v] = has
            return has

        def find_func(v):
            """Retorna (start, end) da funcao contendo vaddr v."""
            i = bisect.bisect_right(func_starts, v) - 1
            if i < 0:
                return None
            s, e = all_funcs[i]
            return (s, e) if s <= v < e else None

        w0 = PPC64_INSN_STD_R2_24_R1
        w1_back20 = encode_b(-20)  # b -20: do offset +16 (padding) ao stub @ +0

        # Phase 1: wrappers nos stubs cuda
        installed = already = skipped = 0
        stub2wrap = {}
        for sv in sorted(cuda_stubs):
            o = sv - tv
            if o < 0 or o + 32 > ts:
                skipped += 1
                continue
            pad = o + 16
            ex = bytes(td[pad:pad + 8])
            if struct.unpack("<II", ex) == (w0, w1_back20):
                already += 1
                stub2wrap[sv] = sv + 16
                continue
            if ex != b"\x00" * 8:
                skipped += 1
                continue
            struct.pack_into("<II", td, pad, w0, w1_back20)
            stub2wrap[sv] = sv + 16
            installed += 1
        print(f">>> [P1] Cuda-stub wrappers: novos={installed} "
              f"existentes={already} pulados={skipped}")

        # Phase 2: paddings disponiveis em outros __long_branch_* stubs
        available_paddings = []
        for stub_v, _ in all_long_branch_stubs:
            if stub_v in cuda_stubs:
                continue
            pad_off = (stub_v - tv) + 16
            if pad_off < 0 or pad_off + 16 > len(td):
                continue
            if bytes(td[pad_off:pad_off + 16]) == b"\x00" * 16:
                available_paddings.append(stub_v + 16)
        available_paddings.sort()
        print(f">>> [P2] Paddings disponiveis em stubs nao-cuda: "
              f"{len(available_paddings)}")

        # Phase 3: percorrer bls
        wrap_set = set(stub2wrap.values())
        used_paddings = set()

        def find_padding_for(C, T):
            lo = max(C - BL_RANGE + 16, T + 4 - BL_RANGE)
            hi = min(C + BL_RANGE - 4, T + 4 + BL_RANGE - 4)
            i = bisect.bisect_left(available_paddings, lo)
            while i < len(available_paddings):
                p = available_paddings[i]
                if p > hi:
                    break
                if p not in used_paddings:
                    return p
                i += 1
            return None

        bls_p = bls_a = nops_p = nops_a = nops_s = 0
        direct_w_save = direct_no_save_fixed = direct_no_save_unfixable = 0
        for off in range(0, len(td) - 3, 4):
            insn = struct.unpack_from("<I", td, off)[0]
            disp = decode_bl(insn)
            if disp is None:
                continue
            src = tv + off
            tgt = src + disp
            need_nop = False
            if tgt in stub2wrap:
                struct.pack_into("<I", td, off,
                                 encode_bl(stub2wrap[tgt] - src))
                bls_p += 1
                need_nop = True
            elif tgt in wrap_set:
                bls_a += 1
                need_nop = True
            elif tgt in cudart_func_addrs:
                caller = find_func(src)
                if caller is not None and func_has_save(*caller):
                    direct_w_save += 1
                    need_nop = True
                else:
                    W = find_padding_for(src, tgt)
                    if W is not None:
                        pad_off_in_text = W - tv
                        b_disp = tgt - (W + 4)
                        if -BL_RANGE <= b_disp < BL_RANGE:
                            struct.pack_into("<II", td, pad_off_in_text,
                                             w0, encode_b(b_disp))
                            used_paddings.add(W)
                            struct.pack_into("<I", td, off,
                                             encode_bl(W - src))
                            direct_no_save_fixed += 1
                            need_nop = True
                        else:
                            direct_no_save_unfixable += 1
                    else:
                        direct_no_save_unfixable += 1
            if need_nop and off + 4 < len(td):
                ni = struct.unpack_from("<I", td, off + 4)[0]
                if ni == PPC64_INSN_NOP:
                    struct.pack_into("<I", td, off + 4,
                                     PPC64_INSN_LD_R2_24_R1)
                    nops_p += 1
                elif ni == PPC64_INSN_LD_R2_24_R1:
                    nops_a += 1
                else:
                    nops_s += 1
        print(f">>> [P3] bls a stub: redir={bls_p} ja={bls_a}")
        print(f">>> [P3] bls diretas (com save): {direct_w_save}")
        print(f">>> [P3] bls diretas (sem save): "
              f"fixed={direct_no_save_fixed} "
              f"unfixable={direct_no_save_unfixable}")
        print(f">>> [P3] nops->ld r2: novos={nops_p} ja={nops_a} "
              f"pulados={nops_s}")
        print(f">>> Paddings consumidos: {len(used_paddings)}/"
              f"{len(available_paddings)}")

        if args.dry_run:
            print(">>> dry-run: nada escrito")
            return 0

        f.seek(to)
        f.write(bytes(td))
        f.flush()

    print(">>> Patch v6 aplicado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
