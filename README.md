# randlapack-bindings

MATLAB, and eventually Python, bindings for [RandLAPACK](https://github.com/BallisticLA/RandLAPACK).

## Scope

Bindings for selected RandLAPACK drivers, single and double precision, real matrices, column-major:

* **`BQRRP`** (randomized blocked QR with column pivoting) — MATLAB + Python.
* **`FunNystromPP`** (matrix-function trace estimation, `tr f(A)`) — MATLAB. The
  Python binding is deferred.

More drivers as their C++ APIs stabilize.

## Requirements

* CMake 3.20+, C++20 compiler
* RandLAPACK installed (with the `pre-bindings-cleanup` work or later) — point `RandLAPACK_DIR` at its `lib/cmake/RandLAPACK` directory
* MATLAB R2018a+ on a platform matching your C++ toolchain (Linux MATLAB for `.mexa64`, Windows MATLAB for `.mexw64`)

## Build

```sh
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK \
    -DMatlab_ROOT_DIR=/path/to/MATLAB
cmake --build build -j
```

Always pass `-DCMAKE_BUILD_TYPE=Release`: with no build type the MEX compiles
without optimization (and Ninja+MSVC even defaults to Debug, measured
2026-08-19), which is functionally correct but useless for timing.

On Windows, run from an x64 developer shell and pass two extra things:
`-G Ninja` and `-DRANDLAPACK_RUNTIME_DLL_DIRS=<BLAS backend bin dir>` (the
RandLAPACK installer prints that directory). `-G Ninja` is REQUIRED, not a
preference: without it CMake selects the multi-config Visual Studio generator,
which appends a `Debug\`/`Release\` subdirectory to the MEX output path, so the
MEX lands in `matlab/+randlapack/private/Debug/` where MATLAB's package loader
never looks (verified 2026-08-19; Ninja ships with the VS "C++ CMake tools"
component). The DLL-dirs flag stages the BLAS runtime next to the MEX; without
it MATLAB fails to load the MEX with "The specified module could not be found".

If MATLAB is not found, MEX targets are skipped with a clear warning; the `.m` files still ship and can be used once the MEX is built. On success, the per-driver MEX (`matlab/+randlapack/private/{bqrrp,fun_nystrom_pp}_mex.<ext>`) is in place.

### Prebuilt release (no compiler, no CMake)

To avoid building RandLAPACK and the MEX yourself, a release archive ships the
`.m` files plus a prebuilt `matlab/+randlapack/private/*.mexa64`. Unzip,
`addpath('.../matlab')`, and call the functions directly. Prebuilt binaries are
currently **Linux x86-64 only**; on macOS/Windows, build from source as above.

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

For matrix-function trace estimation:

```matlab
n = 1000;
G = randn(n, 200); A = G * G.' / 200;   % symmetric PSD test matrix
est = randlapack.fun_nystrom_pp(A, 100, 50, 'Func', 'sqrt');   % k=100, s=50
```

Run `help randlapack.fun_nystrom_pp` for the full name-value signature (`Func`, `Q`, `LFAType`, `Depth`, `Sketch`, `Reorth`, …).

## Runtime notes

The MEX targets statically link `libstdc++`/`libgcc` (`-static-libstdc++ -static-libgcc`), so they load in any MATLAB regardless of the `GLIBCXX` version MATLAB ships — **no `LD_PRELOAD` is needed**. (Earlier versions required preloading the system `libstdc++.so.6`; that is no longer the case.)

The MEX does dynamically link BLAS/LAPACK (via blaspp/lapackpp). When running outside the build environment, point `LD_LIBRARY_PATH` at the blaspp/lapackpp install `lib`/`lib64` dirs, and set `MKL_INTERFACE_LAYER=ILP64`, `MKL_THREADING_LAYER=GNU` if using MKL.

## Conventions

* `J` is **1-based**, matching MATLAB. BQRRP stores 1-based pivots natively; no conversion in the MEX layer.
* RNG `state` is Philox4x32: `state.counter` is `uint32[4]`, `state.key` is `uint32[2]`. Omit it (defaults to seed 0) or pass a scalar `uint32` seed.
