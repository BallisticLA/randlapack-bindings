# randlapack-bindings

MATLAB, and eventually Python, bindings for [RandLAPACK](https://github.com/BallisticLA/RandLAPACK).

## Scope (v0)

Proof-of-concept binding for a single driver, `BQRRP` (randomized blocked QR with column pivoting), in single and double precision, real matrices, column-major. More drivers as their C++ APIs stabilize. Python next.

## Requirements

* CMake 3.20+, C++20 compiler
* RandLAPACK installed (with the `pre-bindings-cleanup` work or later) — point `RandLAPACK_DIR` at its `lib/cmake/RandLAPACK` directory
* MATLAB R2018a+ on a platform matching your C++ toolchain (Linux MATLAB for `.mexa64`, Windows MATLAB for `.mexw64`)

## Build

```sh
cmake -S . -B build \
    -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK \
    -DMatlab_ROOT_DIR=/path/to/MATLAB
cmake --build build -j
```

If MATLAB is not found, MEX targets are skipped with a clear warning; the `.m` files still ship and can be used once the MEX is built. On success, `matlab/+randlapack/private/bqrrp_mex.<ext>` is in place.

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

## Conventions

* `J` is **1-based**, matching MATLAB. BQRRP stores 1-based pivots natively; no conversion in the MEX layer.
* RNG `state` is Philox4x32: `state.counter` is `uint32[4]`, `state.key` is `uint32[2]`. Omit it (defaults to seed 0) or pass a scalar `uint32` seed.
