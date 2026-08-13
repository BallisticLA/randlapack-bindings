#!/bin/bash
# randlapack-bindings bootstrap: get from a bare clone of THIS repo to working
# MATLAB and/or Python bindings in one command.
#
#   bash bootstrap.sh
#
# What it does, in order:
#   1. Finds an installed RandLAPACK -- yours if you have one, and if you do
#      not, it installs one for you using RandLAPACK's own installer at a
#      pinned, known-good commit.
#   2. Configures and builds the MATLAB bindings (if MATLAB is found) and the
#      Python package (if a suitable Python is found).
#   3. Runs each language's binding tests as a smoke check.
#
# Nothing is installed system-wide. A RandLAPACK installed by this script goes
# into the shared RandNLA-project layout (honouring RANDNLA_PROJECT_DIR), so
# it is the same install RandLAPACK's and RandBLAS's own installers would use
# and reuse -- not a private copy.
#
# You bring the toolchain: a C++20 compiler, CMake 3.21+, Git, a BLAS/LAPACK,
# and MATLAB and/or Python >= 3.10 for whichever bindings you want. See
# README.md for the full prerequisite list and supported configurations.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash bootstrap.sh [options]

RandLAPACK:
  --randlapack-dir=DIR  Use this RandLAPACK install (the directory that
                        contains lib/cmake/RandLAPACK). Default: probe
                        $RandLAPACK_DIR, then
                        $RANDNLA_PROJECT_DIR/install/RandLAPACK-install,
                        then offer to install one.
  --project-dir=DIR     Where an installed-by-us RandLAPACK and its
                        dependencies go. Default: $RANDNLA_PROJECT_DIR if
                        set, otherwise ../RandNLA-project next to this clone.
  --blas=BACKEND        Passed through to RandLAPACK's installer when we
                        install (auto | openblas | mkl | accelerate | custom).

Bindings:
  --no-matlab           Skip the MATLAB bindings even if MATLAB is found
  --no-python           Skip the Python package even if Python is found
  --no-tests            Build only; skip the smoke tests

Build:
  -j, --jobs N          Parallel build jobs (default: number of cores)
      --fresh           Clear the bindings build directory first
  -y, --yes             Assume "yes" at every prompt. Also the behavior when
                        stdin is not a terminal (CI, pipes).
  -h, --help            Show this help and exit

Environment-variable equivalents (flags win): BINDINGS_RANDLAPACK_DIR,
BINDINGS_PROJECT_DIR, BINDINGS_BLAS, BINDINGS_MATLAB=0, BINDINGS_PYTHON=0,
BINDINGS_TESTS=0, BINDINGS_JOBS, BINDINGS_FRESH=1, BINDINGS_YES=1.

All build output goes to ./bootstrap.log; the console shows one line per step.
USAGE
}

#==============================================================================
# The RandLAPACK commit these bindings are developed and tested against.
# Immutable ref, same discipline as every other pin in the randlibs repos.
# Update deliberately, alongside a CI run against the new commit.
#==============================================================================
RANDLAPACK_URL="https://github.com/BallisticLA/RandLAPACK.git"
RANDLAPACK_REF="ab05872aec1b8695e137fb3c7f29685e5399fc49"

#==============================================================================
# Options.
#==============================================================================
RANDLAPACK_DIR_ARG="${BINDINGS_RANDLAPACK_DIR:-}"
PROJECT_DIR_ARG="${BINDINGS_PROJECT_DIR:-}"
BLAS_ARG="${BINDINGS_BLAS:-}"
WANT_MATLAB="${BINDINGS_MATLAB:-1}"
WANT_PYTHON="${BINDINGS_PYTHON:-1}"
WANT_TESTS="${BINDINGS_TESTS:-1}"
JOBS="${BINDINGS_JOBS:-}"
FRESH="${BINDINGS_FRESH:-0}"
ASSUME_YES="${BINDINGS_YES:-0}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --randlapack-dir)   RANDLAPACK_DIR_ARG="${2:?--randlapack-dir requires a path}"; shift ;;
        --randlapack-dir=*) RANDLAPACK_DIR_ARG="${1#*=}" ;;
        --project-dir)      PROJECT_DIR_ARG="${2:?--project-dir requires a path}"; shift ;;
        --project-dir=*)    PROJECT_DIR_ARG="${1#*=}" ;;
        --blas)             BLAS_ARG="${2:?--blas requires a backend}"; shift ;;
        --blas=*)           BLAS_ARG="${1#*=}" ;;
        --no-matlab)        WANT_MATLAB=0 ;;
        --no-python)        WANT_PYTHON=0 ;;
        --no-tests)         WANT_TESTS=0 ;;
        -j|--jobs)          JOBS="${2:?--jobs requires a number}"; shift ;;
        --jobs=*)           JOBS="${1#*=}" ;;
        -j*)                JOBS="${1#-j}" ;;
        --fresh)            FRESH=1 ;;
        -y|--yes)           ASSUME_YES=1 ;;
        -h|--help)          usage; exit 0 ;;
        *) printf 'Unknown option: %s (see --help)\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

if [[ -z "$JOBS" ]]; then
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
fi

INTERACTIVE=0
if [[ -t 0 && "$ASSUME_YES" != "1" ]]; then
    INTERACTIVE=1
fi
ask() {  # <question> <default y|n>
    local question="$1" default="$2" reply
    if [[ "$INTERACTIVE" != "1" ]]; then
        [[ "$default" == "y" ]]; return
    fi
    read -r -p "$question [$( [[ $default == y ]] && echo Y/n || echo y/N )]: " reply
    reply="${reply:-$default}"
    [[ "$reply" == "y" || "$reply" == "Y" || "$reply" == "yes" ]]
}

if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
    C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_ERR=""; C_BOLD=""; C_OFF=""
fi
note() { printf '%s\n' "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/bootstrap.log"
{
    printf '\n===============================================================\n'
    printf 'bindings bootstrap started %s\n' "$(date)"
    printf '===============================================================\n'
} >> "$LOG"

STEP=0; TOTAL_STEPS=0
run_step() {
    local label="$1"; shift
    STEP=$((STEP + 1))
    printf '%s[%d/%d]%s %s ... ' "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$label"
    { printf '\n===== [%d/%d] %s =====\n$ %s\n' "$STEP" "$TOTAL_STEPS" "$label" "$*"; } >> "$LOG"
    local t0 t1; t0=$(date +%s)
    if "$@" >> "$LOG" 2>&1; then
        t1=$(date +%s)
        printf '%sdone%s (%ds)\n' "$C_OK" "$C_OFF" "$((t1 - t0))"
    else
        printf '%sFAILED%s\n' "$C_ERR" "$C_OFF" >&2
        printf '\nStep "%s" failed. Full output: %s\nThe last 20 log lines:\n' "$label" "$LOG" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
}
skip_step() {
    STEP=$((STEP + 1))
    printf '%s[%d/%d]%s %s\n' "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$1"
}

#==============================================================================
# Toolchain preflight: everything missing at once.
#==============================================================================
MISSING=()
command -v cmake >/dev/null 2>&1 || MISSING+=("cmake")
command -v git   >/dev/null 2>&1 || MISSING+=("git")
if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && \
   ! command -v clang++ >/dev/null 2>&1; then
    MISSING+=("a C++ compiler")
fi
if (( ${#MISSING[@]} )); then
    die "missing prerequisites: ${MISSING[*]} -- see README.md"
fi

#==============================================================================
# Language detection. Detection only decides the default; the flags decide.
#==============================================================================
MATLAB_BIN=""
if (( WANT_MATLAB )); then
    if command -v matlab >/dev/null 2>&1; then
        MATLAB_BIN="$(command -v matlab)"
        # matlab lives at <root>/bin/matlab; resolve symlinks first, since
        # distro installs typically link /usr/local/bin/matlab elsewhere.
        MATLAB_ROOT="$(dirname "$(dirname "$(readlink -f "$MATLAB_BIN")")")"
    else
        note "MATLAB not found on PATH; skipping the MATLAB bindings (use --no-matlab to silence this)."
        WANT_MATLAB=0
    fi
fi

PYTHON_BIN=""
if (( WANT_PYTHON )); then
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && \
           "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PYTHON_BIN="$(command -v "$cand")"
            break
        fi
    done
    if [[ -z "$PYTHON_BIN" ]]; then
        note "No Python >= 3.10 found on PATH; skipping the Python package (use --no-python to silence this)."
        WANT_PYTHON=0
    elif ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' 2>/dev/null; then
        # System interpreter, not a virtualenv. Debian/Ubuntu mark these
        # externally managed (PEP 668) and pip will refuse to install into
        # them. Say so up front instead of letting pip fail mid-run.
        note "Note: $PYTHON_BIN is a system Python, not a virtualenv. On Debian/Ubuntu,"
        note "pip will refuse to install into it (PEP 668). If the Python step fails,"
        note "create and activate a venv first:  python3 -m venv .venv && . .venv/bin/activate"
    fi
fi

if (( ! WANT_MATLAB && ! WANT_PYTHON )); then
    die "nothing to build: MATLAB and Python are both unavailable or disabled"
fi

#==============================================================================
# Locate RandLAPACK. Precedence: flag, then $RandLAPACK_DIR, then the shared
# RandNLA-project layout, then install one at the pinned commit.
#==============================================================================
find_randlapack_cmake() {  # <install-root> -> prints the cmake config dir, or nothing
    local root="$1" libdir
    for libdir in lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu; do
        if [[ -f "$root/$libdir/cmake/RandLAPACK/RandLAPACKConfig.cmake" ]]; then
            printf '%s/%s/cmake/RandLAPACK' "$root" "$libdir"
            return 0
        fi
    done
    return 0
}

if [[ -n "$PROJECT_DIR_ARG" ]]; then
    PROJECT_DIR="$PROJECT_DIR_ARG"
elif [[ -n "${RANDNLA_PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="$RANDNLA_PROJECT_DIR"
else
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")/RandNLA-project"
fi

# macOS: the RandLAPACK stack is built against Homebrew's libomp, so BLAS++'s
# installed config does find_dependency(OpenMP) -- which stock Apple Clang
# cannot satisfy unaided. Every consumer configure on macOS therefore needs
# the same OpenMP hints RandLAPACK's installer used. Computed once here, used
# for both the CMake configure and the pip build below.
OPENMP_HINTS=()
if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    LIBOMP="$(brew --prefix libomp 2>/dev/null || true)"
    if [[ -n "$LIBOMP" && -f "$LIBOMP/lib/libomp.dylib" ]]; then
        export CFLAGS="${CFLAGS:-} -Xpreprocessor -fopenmp -I$LIBOMP/include"
        export CXXFLAGS="${CXXFLAGS:-} -Xpreprocessor -fopenmp -I$LIBOMP/include"
        export LDFLAGS="${LDFLAGS:-} -L$LIBOMP/lib"
        OPENMP_HINTS=(
            "-DOpenMP_C_LIB_NAMES=omp"
            "-DOpenMP_CXX_LIB_NAMES=omp"
            "-DOpenMP_omp_LIBRARY=$LIBOMP/lib/libomp.dylib"
            "-DOpenMP_C_FLAGS=-Xpreprocessor;-fopenmp"
            "-DOpenMP_CXX_FLAGS=-Xpreprocessor;-fopenmp"
        )
    fi
fi

RANDLAPACK_CMAKE_DIR=""
RANDLAPACK_SOURCE_NOTE=""

if [[ -n "$RANDLAPACK_DIR_ARG" ]]; then
    RANDLAPACK_CMAKE_DIR="$(find_randlapack_cmake "$RANDLAPACK_DIR_ARG")"
    [[ -n "$RANDLAPACK_CMAKE_DIR" ]] || die "--randlapack-dir=$RANDLAPACK_DIR_ARG holds no RandLAPACKConfig.cmake"
    RANDLAPACK_SOURCE_NOTE="from --randlapack-dir"
elif [[ -n "${RandLAPACK_DIR:-}" ]]; then
    # $RandLAPACK_DIR may point at the cmake dir itself or at the install root.
    if [[ -f "$RandLAPACK_DIR/RandLAPACKConfig.cmake" ]]; then
        RANDLAPACK_CMAKE_DIR="$RandLAPACK_DIR"
    else
        RANDLAPACK_CMAKE_DIR="$(find_randlapack_cmake "$RandLAPACK_DIR")"
    fi
    [[ -n "$RANDLAPACK_CMAKE_DIR" ]] || die "\$RandLAPACK_DIR=$RandLAPACK_DIR holds no RandLAPACKConfig.cmake"
    RANDLAPACK_SOURCE_NOTE="from \$RandLAPACK_DIR"
else
    RANDLAPACK_CMAKE_DIR="$(find_randlapack_cmake "$PROJECT_DIR/install/RandLAPACK-install")"
    if [[ -n "$RANDLAPACK_CMAKE_DIR" ]]; then
        RANDLAPACK_SOURCE_NOTE="reused from $PROJECT_DIR (the shared RandNLA-project layout)"
    fi
fi

NEED_INSTALL=0
if [[ -z "$RANDLAPACK_CMAKE_DIR" ]]; then
    NEED_INSTALL=1
    note ""
    note "No installed RandLAPACK was found."
    note "  Looked at: --randlapack-dir, \$RandLAPACK_DIR, and"
    note "  $PROJECT_DIR/install/RandLAPACK-install"
    note ""
    if ! ask "Install RandLAPACK now (pinned commit ${RANDLAPACK_REF:0:9}, into $PROJECT_DIR)?" y; then
        die "cannot proceed without RandLAPACK. Install it (see README.md) and re-run, or pass --randlapack-dir."
    fi
fi

#==============================================================================
# Step accounting.
#==============================================================================
TOTAL_STEPS=0
(( NEED_INSTALL ))  && TOTAL_STEPS=$((TOTAL_STEPS + 2)) || true   # clone + install
TOTAL_STEPS=$((TOTAL_STEPS + 1))                                  # configure bindings
(( WANT_MATLAB )) && TOTAL_STEPS=$((TOTAL_STEPS + 1)) || true     # build MEX
(( WANT_PYTHON )) && TOTAL_STEPS=$((TOTAL_STEPS + 1)) || true     # pip install
if (( WANT_TESTS )); then
    (( WANT_MATLAB )) && TOTAL_STEPS=$((TOTAL_STEPS + 1)) || true
    (( WANT_PYTHON )) && TOTAL_STEPS=$((TOTAL_STEPS + 1)) || true
fi
note ""

#==============================================================================
# Install RandLAPACK if needed, via its own installer. The installer owns the
# dependency recipe (pins, provenance, the run-not-just-link conftest); this
# script deliberately re-implements none of it.
#==============================================================================
if (( NEED_INSTALL )); then
    RL_CLONE="$PROJECT_DIR/lib/RandLAPACK-bootstrap-clone"
    clone_randlapack() {
        mkdir -p "$PROJECT_DIR/lib"
        rm -rf "$RL_CLONE"
        git clone --quiet "$RANDLAPACK_URL" "$RL_CLONE"
        git -C "$RL_CLONE" checkout --quiet "$RANDLAPACK_REF"
        git -C "$RL_CLONE" submodule update --init --recursive --quiet
    }
    run_step "Cloning RandLAPACK (${RANDLAPACK_REF:0:9})" clone_randlapack

    install_randlapack() {
        local args=(--yes --no-gpu --no-extras --no-benchmarks
                    --project-dir="$PROJECT_DIR" -j "$JOBS")
        [[ -n "$BLAS_ARG" ]] && args+=("--blas=$BLAS_ARG")
        bash "$RL_CLONE/install.sh" "${args[@]}"
    }
    run_step "Installing RandLAPACK (its installer; see $PROJECT_DIR/install.log)" install_randlapack

    RANDLAPACK_CMAKE_DIR="$(find_randlapack_cmake "$PROJECT_DIR/install/RandLAPACK-install")"
    [[ -n "$RANDLAPACK_CMAKE_DIR" ]] || die "RandLAPACK installed but RandLAPACKConfig.cmake was not found under $PROJECT_DIR/install/RandLAPACK-install"
    RANDLAPACK_SOURCE_NOTE="installed now, into $PROJECT_DIR"
else
    note "RandLAPACK: $RANDLAPACK_CMAKE_DIR"
    note "  ($RANDLAPACK_SOURCE_NOTE)"
fi

#==============================================================================
# Configure and build the bindings.
#
# find_package(RandLAPACK) resolves blaspp/lapackpp through the paths the
# RandLAPACK install recorded, so RandLAPACK_DIR is the only required hint.
#==============================================================================
BUILD_DIR="$SCRIPT_DIR/build"
if (( FRESH )); then rm -rf "$BUILD_DIR"; fi

CMAKE_ARGS=(-S "$SCRIPT_DIR" -B "$BUILD_DIR"
            -DCMAKE_BUILD_TYPE=Release
            -DRandLAPACK_DIR="$RANDLAPACK_CMAKE_DIR"
            "${OPENMP_HINTS[@]}")
if (( WANT_MATLAB )); then
    CMAKE_ARGS+=(-DMatlab_ROOT_DIR="$MATLAB_ROOT")
fi
run_step "Configuring bindings" cmake "${CMAKE_ARGS[@]}"

if (( WANT_MATLAB )); then
    run_step "Building the MATLAB MEX bindings" \
        cmake --build "$BUILD_DIR" -j "$JOBS"
fi

if (( WANT_PYTHON )); then
    pip_install() {
        # scikit-build-core forwards $CMAKE_ARGS to its CMake configure, which
        # is how the macOS OpenMP hints reach the Python build.
        RandLAPACK_DIR="$RANDLAPACK_CMAKE_DIR" \
        CMAKE_ARGS="${OPENMP_HINTS[*]:-}" \
            "$PYTHON_BIN" -m pip install "$SCRIPT_DIR/python" -v
    }
    run_step "Installing the Python package (pip install ./python)" pip_install
fi

#==============================================================================
# Smoke tests.
#==============================================================================
if (( WANT_TESTS )); then
    if (( WANT_MATLAB )); then
        run_step "Running the MATLAB binding tests" \
            "$MATLAB_BIN" -batch "addpath('$SCRIPT_DIR/matlab'); addpath('$SCRIPT_DIR/matlab/tests'); test_bqrrp; disp('matlab binding tests passed');"
    fi
    if (( WANT_PYTHON )); then
        python_tests() {
            if "$PYTHON_BIN" -m pytest --version >/dev/null 2>&1; then
                "$PYTHON_BIN" -m pytest "$SCRIPT_DIR/python/tests" -q
            else
                # pytest not installed: at minimum prove the module loads and runs.
                "$PYTHON_BIN" - <<'PYEOF'
import numpy as np, randlapack as rl
A = np.random.default_rng(0).standard_normal((200, 40))
Q, R, J = rl.bqrrp(A, 16)
err = np.linalg.norm(A[:, J] - Q @ R) / np.linalg.norm(A)
assert err < 1e-10, f"reconstruction error {err}"
print("python import + bqrrp smoke test passed")
PYEOF
            fi
        }
        run_step "Running the Python binding tests" python_tests
    fi
fi

#==============================================================================
# Summary.
#==============================================================================
printf '\n%s%sBindings ready.%s\n\n' "$C_OK" "$C_BOLD" "$C_OFF"
printf '  RandLAPACK         %s\n' "$RANDLAPACK_CMAKE_DIR"
printf '                     (%s)\n' "$RANDLAPACK_SOURCE_NOTE"
if (( WANT_MATLAB )); then
    printf '  MATLAB             addpath(%s/matlab)\n' "$SCRIPT_DIR"
    printf '                     then e.g.  [Q, R, J] = randlapack.bqrrp(A);\n'
fi
if (( WANT_PYTHON )); then
    printf '  Python             %s -c "import randlapack"\n' "$PYTHON_BIN"
fi
printf '  Full build log     %s\n\n' "$LOG"
