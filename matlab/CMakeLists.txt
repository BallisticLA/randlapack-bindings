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
    LINK_TO RandLAPACK
    R2018a
)
set_target_properties(bqrrp_mex PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${MEX_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${MEX_OUTPUT_DIR}
)
target_compile_features(bqrrp_mex PRIVATE cxx_std_20)
