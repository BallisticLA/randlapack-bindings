# randlapack-bindings

MATLAB and Python bindings for [RandLAPACK](https://github.com/BallisticLA/RandLAPACK).

Currently bound: `BQRRP` (randomized blocked QR with column pivoting), single and double
precision, real matrices. More drivers land in waves as their C++ APIs stabilize.

## Quick start

From a bare clone of this repo, one command builds everything you have the
prerequisites for:

```sh
bash bootstrap.sh
```

It finds an installed RandLAPACK — or clones and installs one for you, at the pinned
commit these bindings are tested against, using RandLAPACK's own installer — then builds
the MATLAB bindings (if MATLAB is on your PATH) and the Python package (if a Python ≥ 3.10
is), and runs each language's binding tests. Nothing is installed system-wide.

Useful flags: `--no-matlab`, `--no-python`, `--no-tests`, `--blas=mkl|openblas|...`,
`--project-dir=DIR`, `-j N`, `--yes`. Run `bash bootstrap.sh --help` for all of them.

If you already keep a shared RandNLA tree (the `RandNLA-project` layout that RandLAPACK's
and RandBLAS's installers create), set `RANDNLA_PROJECT_DIR` and bootstrap reuses its
RandLAPACK instead of installing another.

**Python users**: install into a virtualenv. Debian/Ubuntu system Pythons are marked
externally managed (PEP 668) and `pip` will refuse them:

```sh
python3 -m venv .venv && . .venv/bin/activate
bash bootstrap.sh --no-matlab
```

## Requirements

* CMake 3.21+, a C++20 compiler, Git
* An installed RandLAPACK (bootstrap can provide this), or point `RandLAPACK_DIR` at an
  existing install's `lib/cmake/RandLAPACK`
* For the MATLAB bindings: MATLAB R2018a+ on a platform matching your C++ toolchain
* For the Python package: Python ≥ 3.10 with NumPy

## Manual build

If you would rather drive CMake yourself:

```sh
cmake -S . -B build \
    -DRandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK \
    -DMatlab_ROOT_DIR=/path/to/MATLAB
cmake --build build -j
```

If MATLAB is not found, MEX targets are skipped with a clear warning. On success,
`matlab/+randlapack/private/bqrrp_mex.<ext>` is in place and
`addpath('/path/to/randlapack-bindings/matlab')` makes the package usable.

The Python package builds and installs with pip (in a virtualenv):

```sh
RandLAPACK_DIR=/path/to/RandLAPACK-install/lib/cmake/RandLAPACK pip install ./python
```

## Usage

MATLAB:

```matlab
addpath('/path/to/randlapack-bindings/matlab')

A = randn(2000, 200);
[Q, R, J] = randlapack.bqrrp(A);
err = norm(A(:, J) - Q*R, 'fro') / norm(A, 'fro');

[Q, R, J]       = randlapack.bqrrp(A, 'explicit')   % default; matches qr(A, 'vector')
[A_out, tau, J] = randlapack.bqrrp(A, 'implicit')   % GEQP3-format; skips Q materialization
```

Python:

```python
import numpy as np, randlapack as rl

A = np.random.default_rng(0).standard_normal((2000, 200))
Q, R, J = rl.bqrrp(A)
err = np.linalg.norm(A[:, J] - Q @ R) / np.linalg.norm(A)
```

Run `help randlapack.bqrrp` (MATLAB) or `help(rl.bqrrp)` (Python) for the full
signatures, including block size, sketch embedding factor, and the RNG state.
Examples live in `matlab/examples/` and `python/examples/`.

## Conventions: MATLAB vs Python

The two APIs mirror each other but follow their own ecosystem's conventions where those
diverge. These are choices, not accidents:

| | MATLAB | Python |
|---|---|---|
| Pivot vector `J` | **1-based** (MATLAB indexing; BQRRP's native base) | **0-based** (`scipy.linalg.qr(..., pivoting=True)` convention; converted at the binding) |
| Input layout | column-major (MATLAB's native layout) | any; C-order auto-converted via `np.asfortranarray` (one copy) |
| RNG state | omit, a `uint32` seed, or a struct with `.counter` (`uint32[4]`) and `.key` (`uint32[2]`) | omit, an `int` seed, or a dict with `'counter'`/`'key'` |
| Reproducibility | pass the returned state back in | same, via `return_state=True` |

The RNG is Philox4x32 (counter-based), so a given state produces the same sketch on any
platform.

## Troubleshooting

**MATLAB: `version GLIBCXX_... not found` when the MEX loads.** Should not happen with
MEX files built from this repo: they statically link `libstdc++` on GNU/Linux precisely
because MATLAB ships an older `libstdc++.so.6` than modern compilers expect. If you see
it anyway, you are likely loading a MEX built before that change — rebuild. (The old
workaround, starting MATLAB with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`,
still works but is no longer needed.)

**Linux + MATLAB: BLAS errors or crashes inside the MEX (`Parameter N incorrect on entry
to DGEMM`).** The MEX runs inside MATLAB's process, and MATLAB's own runtime BLAS is an
ILP64 MKL. The BLAS++ underneath RandLAPACK must match that integer width, or MATLAB's
MKL services the MEX's calls with the wrong integer size. Build the stack with MKL ILP64
on Linux — which is what RandLAPACK's installer does by default when MKL is selected.

**macOS: why OpenBLAS rather than Accelerate.** Apple's legacy Accelerate has a broken
divide-and-conquer `gesdd`, and RandLAPACK refuses BQRRP on Accelerate outright. Use
OpenBLAS (RandLAPACK's installer default on macOS) until BLAS++ adopts Apple's new
interface — tracked in
[RandLAPACK #165](https://github.com/BallisticLA/RandLAPACK/issues/165).

**macOS, manual build: `Could NOT find OpenMP_CXX`.** BLAS++'s installed CMake config does
`find_dependency(OpenMP)`, which stock Apple Clang cannot satisfy on its own, so every
consumer of the stack needs the same libomp hints the installer used. `bootstrap.sh` adds
them for you; if you drive CMake yourself, add:

```sh
LIBOMP=$(brew --prefix libomp)
cmake -S . -B build ... \
    -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_C_LIB_NAMES=omp \
    -DOpenMP_omp_LIBRARY=$LIBOMP/lib/libomp.dylib \
    "-DOpenMP_CXX_FLAGS=-Xpreprocessor;-fopenmp" \
    "-DOpenMP_C_FLAGS=-Xpreprocessor;-fopenmp"
```

**`pip` refuses to install (`externally-managed-environment`).** Use a virtualenv; see
Quick start.

## Tested configurations

Every row is exercised by CI on each commit (`.github/workflows/ci.yml`); the RandLAPACK
commit is pinned in `bootstrap.sh` and the CI stack action, and bumped deliberately.

| OS | Bindings | BLAS | Integer width |
|---|---|---|---|
| Ubuntu (latest) | MATLAB (MEX) + Python | oneMKL | ILP64 (matches MATLAB's runtime MKL) |
| macOS (latest) | MATLAB (MEX) | OpenBLAS | LP64 |
| Ubuntu (latest) | bootstrap end-to-end | oneMKL | ILP64 |
