cmake_minimum_required(VERSION 3.20)
project(randlapack_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(RandLAPACK CONFIG REQUIRED)

pybind11_add_module(_randlapack
    src/randlapack_module.cc
    src/bqrrp_binding.cc
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
