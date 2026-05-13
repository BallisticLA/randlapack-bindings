# randlapack-bindings

MATLAB, and eventually Python, bindings for [RandLAPACK](https://github.com/BallisticLA/RandLAPACK).

## Scope (v0)

This is a proof-of-concept binding for a single RandLAPACK driver, `BQRRP` (randomized blocked QR with column pivoting). Additional drivers will be added as their C++ APIs stabilize.

* MATLAB only for v0; Python next.
* `single` and `double` precision; real matrices.
* Column-major storage (MATLAB's default).

## Requirements

* CMake 3.20 or newer
* C++20 compiler
* RandLAPACK (with the `pre-bindings-cleanup` work, or later) installed
* MATLAB R2018a or newer with a matching-platform install (Linux MATLAB for a Linux MEX build, Windows MATLAB for a Windows MEX build)

## Build

```sh
cmake -S . -B build \
    -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK
cmake --build build -j
```

If MATLAB is not on `PATH`, also pass `-DMatlab_ROOT_DIR=/path/to/MATLAB`. If MATLAB is not found at configure time, MEX targets are skipped with a warning; the `.m` files in `matlab/+randlapack/` still work as long as the MEX is built later by some other means.

After a successful build, `matlab/+randlapack/private/bqrrp_mex.<ext>` is in place.

## Usage

```matlab
addpath('/path/to/randlapack-bindings/matlab')

A = randn(2000, 200);
[Q, R, J] = randlapack.bqrrp(A);

% A(:, J) == Q * R up to numerical error.
err = norm(A(:, J) - Q * R, 'fro') / norm(A, 'fro');
```

See `matlab/+randlapack/bqrrp.m` for the full signature, including optional `b_sz`, `d_factor`, and RNG `state` arguments.

## Layout

```
matlab/
├── CMakeLists.txt
├── src/bqrrp_mex.cc           MEX C++ wrapper
├── +randlapack/
│   ├── bqrrp.m                user facing wrapper (default args, validation)
│   └── private/bqrrp_mex.<ext>  MEX binary (built artifact)
├── examples/bqrrp_demo.m      short usage demo
├── tests/test_bqrrp.m         correctness checks
└── benchmarks/bqrrp_vs_qr.m   wall-clock vs MATLAB qr(A, 'vector')
```

## Notes

* The pivot vector `J` returned by `bqrrp` is **1-based**, matching MATLAB convention. (BQRRP itself stores 1-based pivots internally; no conversion happens in the MEX layer.)
* The RNG state is a Philox4x32 state serialized as a struct with `.counter` (uint32 length 4) and `.key` (uint32 length 2). You can omit it (defaults to seed 0) or pass a scalar uint32 seed.
