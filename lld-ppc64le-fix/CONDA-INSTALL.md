# Installing `lld-ppc64le-fix` via conda

Patched `ld.lld` for PPC64LE distributed as a conda package on
[anaconda.org/ufcg-ibm/lld-ppc64le-fix](https://anaconda.org/ufcg-ibm/lld-ppc64le-fix).
One-command install, no build from source.

## TL;DR

```bash
conda install -c ufcg-ibm lld-ppc64le-fix
```

## Prerequisites

- **Architecture**: `linux-ppc64le` (little-endian POWER). The package is
  arch-restricted via `skip: true  # [not (linux and ppc64le)]` in the
  recipe; conda on other arches will refuse the install.
- **A working conda installation**: Miniforge / Miniconda / Anaconda all work.
  Tested with Miniforge 24+.
- **No additional system packages needed.** The package brings its own
  `libstdcxx-ng >= 11` as a runtime dependency.

Quick check that conda sees the right channel:

```bash
conda search -c ufcg-ibm lld-ppc64le-fix
# expected:
#   lld-ppc64le-fix  1.0.0  ppc64le__0  ufcg-ibm
```

## Install

### Into a new isolated environment (recommended)

Keeps the patched LLD separate from your base/build envs.

```bash
conda create -n lld -c ufcg-ibm lld-ppc64le-fix -y
conda activate lld
ld.lld --version
# LLD 23.0.0 (https://github.com/llvm/llvm-project.git 38a8cd7c...) (compatible with GNU linkers)
```

### Into an existing environment

```bash
conda activate <your-env>
conda install -c ufcg-ibm lld-ppc64le-fix
```

### With a specific version

```bash
conda install -c ufcg-ibm lld-ppc64le-fix=1.0.0
```

## What gets installed

| Path | Description |
|------|-------------|
| `$CONDA_PREFIX/bin/ld.lld` | The patched linker (symlink to `lld`) |
| `$CONDA_PREFIX/bin/lld` | The actual binary |
| `$CONDA_PREFIX/lib/liblldELF.so` | LLD ELF backend with the patches |
| `$CONDA_PREFIX/lib/libLLVM*.so` | LLVM support libraries |

Total: ~3 files in `bin/`, ~2 files in `lib/`. Footprint: ~135 MB on disk.

## Verify the install

### Quick check

```bash
ld.lld --version
# should print LLD 23.0.0 with commit 38a8cd7c...

# verify it's the patched binary (looks for thunk emission marker in liblldELF):
strings $CONDA_PREFIX/lib/liblldELF.so* | grep -c "__long_branch_"
# expected: >= 1
```

### Smoke link test

Confirms `ld.lld` resolves correctly and produces a runnable binary:

```bash
cat > /tmp/hello.c <<'EOF'
#include <stdio.h>
int main(void) { puts("hello from patched ld.lld"); return 0; }
EOF

# -B injects the conda bin dir into gcc's search path so it finds our ld.lld
gcc -B$CONDA_PREFIX/bin -fuse-ld=lld -o /tmp/hello /tmp/hello.c
/tmp/hello
# hello from patched ld.lld

# confirm the linker identification in the binary
readelf -p .comment /tmp/hello | grep -i linker
#   [    5b]  Linker: LLD 23.0.0 (https://github.com/llvm/llvm-project.git 38a8cd7c...)
```

## Using the patched linker

The patched LLD is a drop-in replacement for the system `ld.lld`. Pass it to
whatever compiler driver you use.

### Raw `gcc` / `clang`

GCC ≥ 9 supports absolute paths in `-fuse-ld`:

```bash
gcc -fuse-ld=$CONDA_PREFIX/bin/ld.lld -o app *.o
clang -fuse-ld=$CONDA_PREFIX/bin/ld.lld -o app *.o
```

GCC 8 (AlmaLinux 8 default) wants `-B<bindir>`:

```bash
gcc -B$CONDA_PREFIX/bin -fuse-ld=lld -o app *.o
```

### CMake

```bash
cmake -DCMAKE_LINKER_TYPE=LLD \
      -DCMAKE_EXE_LINKER_FLAGS="-B$CONDA_PREFIX/bin -fuse-ld=lld" \
      -DCMAKE_SHARED_LINKER_FLAGS="-B$CONDA_PREFIX/bin -fuse-ld=lld" \
      -S /path/to/source -B build
cmake --build build
```

For CMake 3.29+ you can also use `-DCMAKE_LINKER=$CONDA_PREFIX/bin/ld.lld`.

### Bazel

```bash
bazel build \
    --linkopt=-B$CONDA_PREFIX/bin --linkopt=-fuse-ld=lld \
    --host_linkopt=-B$CONDA_PREFIX/bin --host_linkopt=-fuse-ld=lld \
    //your:target
```

### Default linker for an entire conda env

Symlink it as `ld` so any tool that calls `cc -B$(dirname $(which gcc)) ld`
picks it up:

```bash
ln -sf $CONDA_PREFIX/bin/ld.lld $CONDA_PREFIX/bin/ld
```

Reverse with `rm $CONDA_PREFIX/bin/ld`.

## End-to-end functional test

If you have the [source repo](https://github.com/llm-pt-ibm/llvm-project/tree/ppc64-multitoc-longbranch-thunk-fix)
checked out, the `lld-ppc64le-fix/test_lld.sh` script validates the patch
end-to-end:

```bash
LLD_PATH=$CONDA_PREFIX/bin/ld.lld bash test_lld.sh
# expected:
# === SUMMARY ===
#   PASS: 8 / 8
#   FAIL: 0
```

T1+T2 validate the multi-TOC fix (std r2,24 + nop patching); T3 validates
upstream `R2SaveStub` behavior; T4 validates that leaf callees correctly
skip the nop rewrite (the refinement that makes this patch upstreamable).

## Troubleshooting

### `conda install` says "package not found"

```bash
# verify the channel/arch
conda search -c ufcg-ibm lld-ppc64le-fix
# if nothing returns, you may be on the wrong architecture
uname -m   # must print ppc64le
```

### `ld.lld --version` shows a different commit

Another `ld.lld` is shadowing ours on PATH:

```bash
which -a ld.lld
# the conda env one should be first
```

If a system or other-env `ld.lld` comes first, either reactivate the conda
env (`conda deactivate && conda activate <env>`) or use the absolute path
`$CONDA_PREFIX/bin/ld.lld` explicitly.

### Build fails with `unknown relocation` errors

Almost always means you're using the **system** LLD (which is older and
doesn't know modern PPC relocs), not the patched one. Force the conda
binary explicitly with absolute path:

```bash
$CC -fuse-ld=$CONDA_PREFIX/bin/ld.lld ...
```

### Link fails with `undefined reference to _savefpr_NN`

LLD on PPC64LE does not synthesize the single-underscore save/restore
helpers that some GCC versions emit. This is **unrelated to the multi-TOC
fix** — it's an orthogonal LLD gap. Workaround:

```bash
# Generate the missing helpers as a tiny static lib (see the TF build
# script's savres block, ~150 lines of asm + ar)
# Or: avoid the issue by using clang as the compiler (clang inlines these).
```

A separate `libppc64-savres` package may be published in the future to
cover this case.

### `lacks nop, can't restore toc` link error

This is the **upstream** LLD diagnostic firing because some caller in your
code emits `bl <r2-clobbering-callee>` without a trailing `nop` (an ABI
violation). The fix is in the caller, not the linker. If you really need
to suppress for a one-off case, build LLD from source with the diagnostic
patched out (not recommended).

## Uninstall

```bash
conda remove -n <env> lld-ppc64le-fix
# or, if you installed into a dedicated env:
conda env remove -n lld
```

## Reporting issues

- **The fix itself** (patch logic, regressions): file at the source repo
  [llm-pt-ibm/llvm-project @ ppc64-multitoc-longbranch-thunk-fix](https://github.com/llm-pt-ibm/llvm-project/tree/ppc64-multitoc-longbranch-thunk-fix).
- **The conda package** (packaging, dependencies, install errors): file at
  the package page [anaconda.org/ufcg-ibm/lld-ppc64le-fix](https://anaconda.org/ufcg-ibm/lld-ppc64le-fix).
- **Upstream LLVM integration**: track the LLVM PR (open soon at
  llvm/llvm-project).

## See also

- [README.md](./README.md) — full project context, build-from-source, patch
  upstreamability notes
- [build_patched_lld.sh](./build_patched_lld.sh) — recipe to rebuild from
  source if you prefer
- [test_lld.sh](./test_lld.sh) — the 8-test lit suite used to validate
- GitHub release with pre-built tarball:
  [v1.0.0](https://github.com/llm-pt-ibm/llvm-project/releases/tag/v1.0.0)
