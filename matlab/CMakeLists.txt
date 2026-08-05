# MATLAB MEX targets for the +randlapack package.
#
# Each MEX target is built and dropped into matlab/+randlapack/private/ so
# that after `make`, the package is immediately usable from MATLAB after
# addpath(/path/to/randlapack-bindings/matlab).

set(MEX_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/+randlapack/private)
file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})

# Statically link libstdc++ into each MEX. MATLAB R2026a ships its own
# (older) libstdc++.so.6 under sys/os/glnxa64/ which lacks newer GLIBCXX
# symbols required by code built with GCC 13+. Statically linking avoids
# the runtime dependency entirely -- the MEX file ships its own libstdc++
# symbols and is loadable in any MATLAB without LD_PRELOAD gymnastics.
# Applied via target_link_options on each matlab_add_mex target below.
#
# These are GCC/Linux-only flags. AppleClang on macOS rejects -static-libgcc
# ("unsupported option"): there is no libgcc to statically link (macOS uses
# libc++, not libstdc++), and the GLIBCXX-version problem above does not exist
# there. Leave the list empty off GNU so macOS/other compilers link normally.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(MEX_STATIC_LIBSTDCXX_FLAGS -static-libstdc++ -static-libgcc)
else()
    set(MEX_STATIC_LIBSTDCXX_FLAGS "")
endif()

matlab_add_mex(
    NAME bqrrp_mex
    SRC src/bqrrp_mex.cc
    LINK_TO RandLAPACK::RandLAPACK
    R2018a
)
# matlab_add_mex's LINK_TO doesn't propagate INTERFACE_INCLUDE_DIRECTORIES from
# the linked targets in all CMake versions. Expose them explicitly. We use
# target_include_directories (not a second target_link_libraries) because
# FindMatlab itself uses the plain (no-keyword) target_link_libraries form
# internally and CMake disallows mixing keyword/plain calls on one target.
target_include_directories(bqrrp_mex PRIVATE
    $<TARGET_PROPERTY:RandLAPACK::RandLAPACK,INTERFACE_INCLUDE_DIRECTORIES>
)
set_target_properties(bqrrp_mex PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${MEX_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${MEX_OUTPUT_DIR}
)
target_compile_features(bqrrp_mex PRIVATE cxx_std_20)
target_link_options(bqrrp_mex PRIVATE ${MEX_STATIC_LIBSTDCXX_FLAGS})
