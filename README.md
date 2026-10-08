# randlapack-bindings

MATLAB, and eventually Python, bindings for [RandLAPACK](https://github.com/BallisticLA/RandLAPACK).

## Scope (v0)

Proof-of-concept binding for a single driver, `BQRRP` (randomized blocked QR with column pivoting), in single and double precision, real matrices, column-major. More drivers as their C++ APIs stabilize. Python next.

## Requirements

* CMake 3.21+, C++20 compiler
* RandLAPACK installed. The supported route is RandLAPACK's installer (`installers/install.sh` on Linux/macOS, `installers/install.ps1` on Windows; see its `INSTALL_SCRIPT.md` and `INSTALL_WINDOWS.md`), which builds BLAS++ and LAPACK++ against a BLAS backend in a tested configuration (on Windows it finds or downloads oneMKL or OpenBLAS) and prints the `RandLAPACK_DIR` to use below
* MATLAB R2018a+ on a platform matching your C++ toolchain (Linux MATLAB for `.mexa64`, Windows MATLAB for `.mexw64`)

## Build

```sh
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK \
    -DMatlab_ROOT_DIR=/path/to/MATLAB
cmake --build build -j
```

Pass `-DCMAKE_BUILD_TYPE=Release`: without a build type the MEX compiles without optimization (Ninja with MSVC even defaults to Debug).

If MATLAB is not found, MEX targets are skipped with a clear warning; the `.m` files still ship and can be used once the MEX is built. On success, `matlab/+randlapack/private/bqrrp_mex.<ext>` is in place.

### Windows

Run from the **x64 Native Tools Command Prompt for VS** (the prompt RandLAPACK's `INSTALL_WINDOWS.md` uses for the installer) and add `-G Ninja`:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DRandLAPACK_DIR="<RandLAPACK_DIR printed by install.ps1>" ^
    -DMatlab_ROOT_DIR="C:/Program Files/MATLAB/R2026a"
cmake --build build
```

`-G Ninja` is required: the default Visual Studio generator appends a `Debug\` or `Release\` subdirectory to the MEX output path, where MATLAB's package loader never looks.

The build copies the BLAS++ and LAPACK++ DLLs and every DLL in the BLAS backend's `bin` directory next to the MEX (except Intel's OpenMP runtime; see Runtime notes), so MATLAB needs no `PATH` changes. For oneMKL that is about 1 GB. The backend directory comes from the RandLAPACK install, which records it when built by `install.ps1`; for a RandLAPACK you configured yourself, pass `-DRANDLAPACK_RUNTIME_DLL_DIRS=<backend bin directory>`, or the MEX fails to load with "The specified module could not be found".

If you built BLAS++ and LAPACK++ as static libraries yourself, your RandLAPACK install must include BallisticLA/RandLAPACK#199 (reinstall from current `main`); with an older install this build fails at the DLL-copy step.

## Usage

```matlab
addpath('/path/to/randlapack-bindings/matlab')

A = randn(2000, 200);
[Q, R, J] = randlapack.bqrrp(A);
err = norm(A(:, J) - Q*R, 'fro') / norm(A, 'fro');
```

Two output modes, chosen by a trailing string (qr-style):

```matlab
[Q, R, J]       = randlapack.bqrrp(A, 'explicit')  % default; matches qr(A, 'vector')
[A_out, tau, J] = randlapack.bqrrp(A, 'implicit')  % GEQP3-format; skips Q materialization
```

Run `help randlapack.bqrrp` for the full signature, including the optional `b_sz`, `d_factor`, and RNG `state` arguments.

## Runtime notes

On **Linux MATLAB R2026a + Ubuntu 24.04** (and similar new-glibc hosts), MATLAB ships an older `libstdc++.so.6` than your compiler's. MEX files built against the system `libstdc++` fail at load with `version GLIBCXX_3.4.32 not found`. Workaround:

```sh
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab
```

Alias it in your shell profile to make it transparent. Not needed on macOS or older Ubuntu hosts.

On **Windows**, the MEX uses MATLAB's own OpenMP runtime (`libiomp5md`) instead of MSVC's, because two OpenMP runtimes in one process abort with `OMP: Error #15`. For the same reason the build never leaves a `libiomp5md.dll` next to the MEX, even when the BLAS backend uses Intel's OpenMP (threaded oneMKL does): the backend then runs on MATLAB's copy. Do not set `KMP_DUPLICATE_LIB_OK=TRUE` to get past that error: the error message itself calls it an unsafe, unsupported workaround that may cause crashes or silently produce incorrect results. If you see Error #15, close MATLAB (Windows locks a loaded MEX file) and rebuild the MEX from the current bindings.

## Conventions

* `J` is **1-based**, matching MATLAB. BQRRP stores 1-based pivots natively; no conversion in the MEX layer.
* RNG `state` is Philox4x32: `state.counter` is `uint32[4]`, `state.key` is `uint32[2]`. Omit it (defaults to seed 0) or pass a scalar `uint32` seed.
