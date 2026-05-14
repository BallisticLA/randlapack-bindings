# MATLAB MEX targets for the +randlapack package.
#
# Each MEX target is built and dropped into matlab/+randlapack/private/ so
# that after `make`, the package is immediately usable from MATLAB after
# addpath(/path/to/randlapack-bindings/matlab).

set(MEX_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/+randlapack/private)
file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})

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
