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
#
# RandLAPACK_DIR must be the directory holding RandLAPACKConfig.cmake, but a
# common mistake is passing the RandNLA-project root or the install prefix
# instead. Accept those by searching the known install layouts beneath the
# given directory; anything else gets an actionable error instead of
# find_package's generic one.
# ---------------------------------------------------------------------------
if(DEFINED RandLAPACK_DIR AND NOT EXISTS "${RandLAPACK_DIR}/RandLAPACKConfig.cmake")
    file(GLOB _rl_configs
         "${RandLAPACK_DIR}/lib/cmake/RandLAPACK/RandLAPACKConfig.cmake"
         "${RandLAPACK_DIR}/lib64/cmake/RandLAPACK/RandLAPACKConfig.cmake"
         "${RandLAPACK_DIR}/RandLAPACK-install/lib/cmake/RandLAPACK/RandLAPACKConfig.cmake"
         "${RandLAPACK_DIR}/RandLAPACK-install/lib64/cmake/RandLAPACK/RandLAPACKConfig.cmake"
         "${RandLAPACK_DIR}/install/RandLAPACK-install/lib/cmake/RandLAPACK/RandLAPACKConfig.cmake"
         "${RandLAPACK_DIR}/install/RandLAPACK-install/lib64/cmake/RandLAPACK/RandLAPACKConfig.cmake")
    if(_rl_configs)
        list(GET _rl_configs 0 _rl_config)
        get_filename_component(_rl_dir "${_rl_config}" DIRECTORY)
        message(STATUS "RandLAPACK_DIR does not itself contain RandLAPACKConfig.cmake; "
                       "using ${_rl_dir} found beneath it")
        set(RandLAPACK_DIR "${_rl_dir}" CACHE PATH "Directory containing RandLAPACKConfig.cmake" FORCE)
    else()
        message(FATAL_ERROR
            "RandLAPACK_DIR=${RandLAPACK_DIR} does not contain RandLAPACKConfig.cmake, "
            "and none was found in the usual install layouts beneath it. "
            "If RandLAPACK's install.sh has not completed successfully yet, run it first -- "
            "it prints the exact -DRandLAPACK_DIR to use. Otherwise locate the file with\n"
            "    find <project-dir>/install -name RandLAPACKConfig.cmake\n"
            "and pass its directory.")
    endif()
endif()
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

# ---------------------------------------------------------------------------
# Benchmark external baselines (off by default). When ON, FetchContent pulls
# the external reference implementations (Epperly, Persson) used by the
# funNystrom++ benchmark at configure time — NOT vendored in this repo.
# See matlab/benchmark/funnystrompp/CMakeLists.txt and THIRD_PARTY_LICENSES.md.
# ---------------------------------------------------------------------------
option(RANDLAPACK_BINDINGS_BENCHMARKS
       "Fetch external benchmark baselines (Epperly, Persson) via FetchContent" OFF)
if(RANDLAPACK_BINDINGS_BENCHMARKS)
    add_subdirectory(matlab/benchmark/funnystrompp)
endif()
