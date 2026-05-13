cmake_minimum_required(VERSION 3.20)
project(randlapack-bindings VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use position-independent code; MEX binaries are shared objects.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# RandLAPACK dependency.
# Primary: find_package against an installed RandLAPACK. The user is expected
# to set -DRandLAPACK_DIR=/path/to/install/lib/cmake/RandLAPACK when invoking
# cmake. CPM fallback is a future addition.
# ---------------------------------------------------------------------------
find_package(RandLAPACK REQUIRED)
message(STATUS "RandLAPACK found at ${RandLAPACK_DIR}")

# ---------------------------------------------------------------------------
# MATLAB bindings. Gated on FindMatlab. If MATLAB is not available we print
# a clear message and skip the MEX target; the .m files still ship via the
# source tree and can be added to the MATLAB path manually.
# ---------------------------------------------------------------------------
find_package(Matlab QUIET COMPONENTS MX_LIBRARY MAIN_PROGRAM)
if(Matlab_FOUND)
    message(STATUS "MATLAB found at ${Matlab_ROOT_DIR}; building MEX targets")
    add_subdirectory(matlab)
else()
    message(WARNING
        "MATLAB not found. MEX targets will be skipped. "
        "Set -DMatlab_ROOT_DIR=/path/to/MATLAB to enable. "
        "The +randlapack package can still be used by adding "
        "${CMAKE_CURRENT_SOURCE_DIR}/matlab to the MATLAB path, "
        "but calls into bqrrp_mex will fail until the MEX is built.")
endif()
