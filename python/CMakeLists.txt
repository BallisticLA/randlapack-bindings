cmake_minimum_required(VERSION 3.21)
project(randlapack_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# nanobind ships its CMake config inside the installed wheel; ask the build
# Python where that is. This is the standard scikit-build-core recipe (the
# nanobind_example template): under pip, nanobind is a build requirement from
# pyproject.toml, so the interpreter running the build always has it.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

find_package(RandLAPACK CONFIG REQUIRED)

# NB_STATIC links nanobind's support library into the extension, so the wheel
# has no separate libnanobind to locate at import time.
nanobind_add_module(_randlapack NB_STATIC
    src/randlapack_module.cc
    src/bqrrp_binding.cc
    src/cqrrpt_binding.cc
    src/rsvd_binding.cc
)
target_link_libraries(_randlapack PRIVATE RandLAPACK::RandLAPACK)
target_compile_features(_randlapack PRIVATE cxx_std_20)

# Embed RPATH pointing at wherever blaspp/lapackpp/RandBLAS were found, so the
# extension can dlopen them at runtime without LD_LIBRARY_PATH gymnastics.
set_target_properties(_randlapack PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS _randlapack LIBRARY DESTINATION randlapack)
