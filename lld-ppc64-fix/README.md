# lld-ppc64-fix

A patched build of LLVM `ld.lld` that fixes a latent **r2 (TOC pointer)
corruption** in upstream LLD's `PPC64LongBranchThunk` path. Affects any
PPC64LE (little-endian POWER) shared object whose TOC exceeds 64 KB and
whose `.text` exceeds ±32 MB of branch range — the conditions that make
LLD generate both a multi-TOC layout *and* long-branch thunks for
intra-module calls.

The fix is a single coherent change to `lld/ELF/Thunks.cpp` (3 edits)
that follows the same pattern LLD already uses for `PPC64PltCallStub`
and `PPC64R2SaveStub`. It is intended as an upstream-quality
contribution to LLVM, not a project-specific workaround.

## TL;DR

```bash
./build_patched_lld.sh        # ~30min on x86_64, ~2-4h native on POWER9
./test_lld.sh                 # 8 lit-style validation tests (T1..T4)

# Use it
export PATH=$HOME/lld-ppc64-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/lld-ppc64-install/lib:$LD_LIBRARY_PATH
$CC -fuse-ld=$HOME/lld-ppc64-install/bin/ld.lld ...
```

The patched binary lands at `$LLD_INSTALL_DIR/bin/ld.lld`
(`$HOME/lld-ppc64-install/bin/ld.lld` by default).

## The bug

PPC64 ELFv2 ABI uses register `r2` as the TOC base pointer. The caller of
a function that may switch TOC groups (or otherwise clobber `r2`) is
responsible for:

1. Saving `r2` to `24(r1)` **before** the call.
2. Restoring `r2` from `24(r1)` **after** the call, in the trailing `nop`
   slot that the linker rewrites to `ld r2, 24(r1)`.

LLD already does this correctly for two of the three PPC64 stub kinds:

- **`PPC64PltCallStub`** — for PLT entries.
- **`PPC64R2SaveStub`** — for global-entry callees or `.localentry sym, 1`
  (the ABI v1.5 `CLOBBERS_R2` marker).

In both cases LLD's `writeTo` prefixes the stub with `std r2, 24(r1)` and
LLD sets `needsTocRestore(true)` on the stub symbol so the post-call
`nop` is rewritten to `ld r2, 24(r1)`.

The third kind — **`PPC64LongBranchThunk`** — is used for *intra-module*
calls that exceed the 26-bit `bl` displacement (±32 MB). It does
**neither**: it neither saves r2 in the thunk nor marks the symbol
`needsTocRestore`. Upstream LLD silently assumes intra-DSO branches never
cross TOC groups.

In **multi-TOC mode** (when total TOC > 64 KB and LLD splits it into
multiple groups), that assumption breaks. A `LongBranchThunk` can land in
a different TOC group from its caller, the callee's global-entry prologue
(or any other r2 clobber) corrupts `r2`, and the caller continues with a
garbage TOC pointer. The next TOC-relative access
(`ld r3, sym@toc(r2)`, etc.) returns random memory and the process
crashes shortly after.

The bug is *silent at link time*: upstream's lit test suite never
exercises a large enough binary to trigger it, and there is no
diagnostic when the linker emits the buggy thunk. It only manifests at
runtime, often deep inside C++ static initializers, where the crash
location appears unrelated to the actual call site.

### Real-world impact

Discovered while linking
[`libtensorflow_framework.so`](https://github.com/tensorflow/tensorflow)
(~67 MB `.text`, >64 KB TOC) on POWER9. Stock LLD produced segfaults in
`__sti____cudaRegisterAll`, `PlatformInitialize`, `ToStatusSlow`, and
other early-init paths. After the patch all of them link and run
correctly. Any sufficiently large C++ shared object on PPC64LE is at
risk — the bug is in LLD, not in any particular project.

## The fix

`build_patched_lld.sh` clones `llvm-project` (shallow), applies 3
coordinated source modifications to `lld/ELF/Thunks.cpp` via
`sed`/`python`, and builds only `lld` + its dependencies (~30 min on a
beefy x86_64 box, 2–4h native on POWER9).

The 3 modifications, all in `lld/ELF/Thunks.cpp`:

| # | Change | Why |
|---|--------|-----|
| 1 | `PPC64LongBranchThunk::size()` 32 → 36 bytes | Room for the new instruction in [2]. |
| 2 | `PPC64LongBranchThunk::writeTo` emits `std r2, 24(r1)` before the load-and-branch | Save caller's `r2` to its standard ABI slot before the thunk can switch TOC groups. Mirrors `PPC64PltCallStub` / `PPC64R2SaveStub` behavior. |
| 3 | `PPC64LongBranchThunk::addSymbols` sets `needsTocRestore(true)` on the thunk symbol — *only* when `(destination.stOther >> 5) != 0` | Request post-call `nop → ld r2, 24(r1)` rewrite when the callee may clobber `r2` (has a global entry point or the ABI v1.5 `CLOBBERS_R2` marker). Leaf-style callees with `st_other == 0` preserve `r2` and need no restore. |

A single patch file in this directory mirrors the source changes for
review and upstream submission:

- `0001-lld-PPC64-Add-r2-TOC-save-to-LongBranch-thunks.patch` — all three
  modifications plus a new lit test
  `lld/test/ELF/ppc64-long-branch-r2-save.s`.

The `.patch` file is illustrative (line numbers and hashes are
placeholders); the actual modifications are applied in
`build_patched_lld.sh` via `sed`/`python` so they survive LLVM HEAD
churn around the affected functions.

### Upstreamability

The change set is designed to be submittable to LLVM as a single PR:

- It mirrors the existing pattern in `PPC64PltCallStub` and
  `PPC64R2SaveStub`, so reviewers do not have to evaluate a new design.
- Modification [3] is *conditional*, not over-broad: the upstream
  `"call to X lacks nop, can't restore toc"` diagnostic remains
  semantically correct — it fires only when a caller violates ABI by
  omitting the trailing nop after a `bl` to a callee that does clobber
  `r2`. No diagnostic suppression needed.
- Modification [2] is unconditional (the `std r2, 24(r1)` always runs,
  even when the thunk happens not to cross a TOC boundary). Reviewers
  may ask whether it should be gated on multi-TOC mode being active; we
  leave it unconditional because (a) a single store to stack is cheap,
  (b) gating would re-introduce silent failures whenever LLD's
  TOC-splitting heuristics evolve.

The included lit test (`ppc64-long-branch-r2-save.s`) constructs a
minimal multi-TOC scenario and checks both the `std r2, 24(r1)` in the
thunk and the `nop → ld r2, 24(r1)` rewrite at the call site.

## Build

```bash
./build_patched_lld.sh           # build + install
./build_patched_lld.sh --test    # build + install + run check-lld-elf
```

### Requirements

- `git`, `cmake` (≥ 3.20), `ninja` (or `make`)
- A C++17-capable host compiler (`clang`/`clang++` preferred; `gcc`/`g++`
  works as long as it can self-link LLVM)
- Python 3 (for patch [3]'s regex-based edit)
- ~16 GB RAM for the link step
- ~5 GB free disk for the shallow LLVM clone + build

The script auto-detects `clang`/`clang++` on `PATH` and uses them when
present. If a Conda environment is preferable (e.g. on hosts where the
system toolchain is too old), point `CONDA_BASE` at it.

### Tunables (environment variables)

| Variable | Default | Notes |
|----------|---------|-------|
| `LLD_INSTALL_DIR` | `$HOME/lld-ppc64-install` | install prefix; `bin/ld.lld` + `lib/lib*.so` land here |
| `CC` / `CXX` | autodetect | override the host compiler |
| `CONDA_BASE` | unset | if set + valid, source it and `conda activate ${CONDA_ENV:-base}` |
| `CONDA_ENV` | `base` | which conda env to activate when `CONDA_BASE` is used |

### Output

```
$LLD_INSTALL_DIR/
├── bin/
│   ├── ld.lld           ← use this
│   └── lld
└── lib/
    ├── liblldELF.so     ← contains the patched logic (BUILD_SHARED_LIBS=ON)
    ├── libLLVM*.so
    └── ...
```

Because `build_patched_lld.sh` configures `BUILD_SHARED_LIBS=ON`, the
patched ELF logic lives in `liblldELF.so`, *not* in the `ld.lld`
front-end binary. If you move `bin/ld.lld` to another machine, take the
matching `lib/` with it and set `LD_LIBRARY_PATH` accordingly.

## Test

```bash
./test_lld.sh
```

Runs 4 categories of validation (8 sub-tests) against the installed
patched binary:

| Test | What it checks |
|------|----------------|
| **T1** | `__long_branch_*` stub begins with `std r2, 24(r1)` (patch [2]) |
| **T2** | `bl` to a `__long_branch_*` thunk *whose target has global entry / `.localentry`* has its trailing `nop` rewritten to `ld r2, 24(r1)` (patch [3] positive case) |
| **T3** | `bl` to `.localentry sym, 1` creates a `PPC64R2SaveStub` and patches the trailing nop (upstream behavior, unaffected by our patches — included as a regression guard) |
| **T4** | `bl` to a `__long_branch_*` thunk *whose target is a leaf with `st_other == 0`* — `std r2, 24(r1)` still emitted in the thunk (patch [2] is unconditional), but the trailing `nop` is **not** patched (patch [3] correctly skips `needsTocRestore`) |

T3 requires an assembler that accepts `.localentry sym, 1` (the ELFv2
ABI v1.5 `CLOBBERS_R2` flag). GAS ≤ 2.30 rejects it; the script
auto-probes and prefers Clang over GAS when both are present. T1+T2+T4
use `.localentry sym, 8` which works in both GAS and Clang.

A passing run looks like:

```
=== SUMMARY ===
  PASS: 8 / 8
  FAIL: 0
```

## Using the patched LLD

### With raw `gcc`/`clang`

```bash
export PATCHED_LLD=$HOME/lld-ppc64-install/bin/ld.lld
export LD_LIBRARY_PATH=$HOME/lld-ppc64-install/lib:$LD_LIBRARY_PATH

$CC -fuse-ld="$PATCHED_LLD" -o app obj1.o obj2.o ...
```

### With CMake

```bash
cmake -DCMAKE_LINKER_TYPE=LLD \
      -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=$PATCHED_LLD" \
      -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=$PATCHED_LLD" \
      ...
```

### With Bazel

```bash
bazel build --linkopt=-fuse-ld="$PATCHED_LLD" \
            --host_linkopt=-fuse-ld="$PATCHED_LLD" \
            //your:target
```

### PPC64 save/restore stubs note

Independent of the multi-TOC bug fixed here, `ld.lld` does **not**
auto-generate the `_savefpr_NN` / `_restfpr_NN` / `_savegpr_NN` /
`_restgpr_NN` / `_savevr_NN` / `_restvr_NN` symbols that GCC references
from out-of-line prologues/epilogues. BFD generates these inline; LLD
requires external linkage. `libgcc` ships the *double*-underscore
variants (`__savegpr0_NN` etc.) but not the single-underscore ones.

If you hit `undefined reference to _savefpr_14` (or similar) when
linking GCC-compiled object files with `ld.lld`, you need a small `.S`
file providing those symbols. This is an orthogonal LLD-on-PPC64
limitation, not specific to this patch.

## Layout

```
lld-ppc64-fix/
├── build_patched_lld.sh                          build entry point
├── test_lld.sh                                   T1..T4 validation tests
├── 0001-lld-PPC64-Add-r2-TOC-save-to-LongBranch-thunks.patch
├── LICENSE                                       Apache-2.0 with LLVM Exception
└── README.md                                     this file
```

## Caveats

- **LLVM version.** The script clones LLVM `main` (`--depth 1`). When
  the upstream `Thunks.cpp` is refactored (e.g.
  `PPC64LongBranchThunk::addSymbols` function shape changes), the
  `sed`/`python` patterns may need updating. Pin to a known-good commit
  by editing the `git clone` step if you depend on this in CI.
- **ABI compliance is enforced.** We deliberately do not suppress the
  upstream `"call to X lacks nop, can't restore toc"` diagnostic. If
  your build hits it, the caller is genuinely violating the ELFv2 ABI by
  emitting `bl <clobber-r2-callee>` without a trailing `nop`. The
  correct fix is in the caller (whichever code generator emitted the
  bare `bl`), not in the linker. If you really need to suppress it for
  an unfixable third-party object, do so in a downstream fork.
- **Multi-TOC trigger.** If your `.so` does not exceed the 64 KB TOC
  threshold (and thus does not trigger multi-TOC mode), upstream LLD
  already works for you — no need to patch. The fix is still safe in
  that case (the extra `std r2, 24(r1)` is harmless).

## License

Apache License v2.0 with LLVM Exceptions, matching upstream LLVM. See
[LICENSE](LICENSE) and <https://llvm.org/LICENSE.txt>.
