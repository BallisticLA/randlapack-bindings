// MEX wrapper for RandLAPACK::BQRRP.
//
// MATLAB-side signature (low-level; users normally call randlapack.bqrrp.m):
//
//   [Q, R, J, state_out] = bqrrp_mex(A, b_sz, d_factor, state_in)
//
// where:
//   A         m-by-n matrix, single or double, column-major (MATLAB default).
//   b_sz      block size; scalar, converted to int64.
//   d_factor  sketch factor; scalar of the same scalar type as A.
//   state_in  RNG state struct with fields .counter (uint32 length 4) and
//             .key (uint32 length 2). Philox4x32 layout.
//
// returns:
//   Q          m-by-k explicit orthogonal factor, k = min(m, n).
//   R          k-by-n upper-triangular factor.
//   J          1-by-n vector of 1-based column-pivot indices (int64).
//   state_out  RNG state struct after advancing.
//
// J is 1-based because BQRRP itself stores 1-based pivot indices internally
// (see RandLAPACK/RandLAPACK/drivers/rl_bqrrp.hh, std::iota(..., 1) at line
// 330). No conversion is performed in the MEX layer.

#include "mex.h"
#include <RandLAPACK.hh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {

using RNG = RandBLAS::DefaultRNG;  // Philox4x32

constexpr mwSize CTR_LEN = 4;
constexpr mwSize KEY_LEN = 2;

RandBLAS::RNGState<RNG> read_state(const mxArray* state_struct) {
    if (!mxIsStruct(state_struct)) {
        mexErrMsgIdAndTxt("randlapack:bqrrp_mex:state",
            "state must be a struct with .counter (uint32[4]) and .key (uint32[2])");
    }
    const mxArray* counter_field = mxGetField(state_struct, 0, "counter");
    const mxArray* key_field     = mxGetField(state_struct, 0, "key");
    if (!counter_field || !mxIsUint32(counter_field)
        || mxGetNumberOfElements(counter_field) != CTR_LEN) {
        mexErrMsgIdAndTxt("randlapack:bqrrp_mex:state",
            "state.counter must be a uint32 array of length 4");
    }
    if (!key_field || !mxIsUint32(key_field)
        || mxGetNumberOfElements(key_field) != KEY_LEN) {
        mexErrMsgIdAndTxt("randlapack:bqrrp_mex:state",
            "state.key must be a uint32 array of length 2");
    }

    RandBLAS::RNGState<RNG> state;
    const uint32_t* ctr_in = (const uint32_t*) mxGetData(counter_field);
    const uint32_t* key_in = (const uint32_t*) mxGetData(key_field);
    for (mwSize i = 0; i < CTR_LEN; ++i) state.counter.v[i] = ctr_in[i];
    for (mwSize i = 0; i < KEY_LEN; ++i) state.key.v[i]     = key_in[i];
    return state;
}

mxArray* write_state(const RandBLAS::RNGState<RNG>& state) {
    const char* fields[] = {"counter", "key"};
    mxArray* out = mxCreateStructMatrix(1, 1, 2, fields);

    mxArray* counter_mx = mxCreateNumericMatrix(1, CTR_LEN, mxUINT32_CLASS, mxREAL);
    uint32_t* ctr_out = (uint32_t*) mxGetData(counter_mx);
    for (mwSize i = 0; i < CTR_LEN; ++i) ctr_out[i] = state.counter.v[i];
    mxSetField(out, 0, "counter", counter_mx);

    mxArray* key_mx = mxCreateNumericMatrix(1, KEY_LEN, mxUINT32_CLASS, mxREAL);
    uint32_t* key_out = (uint32_t*) mxGetData(key_mx);
    for (mwSize i = 0; i < KEY_LEN; ++i) key_out[i] = state.key.v[i];
    mxSetField(out, 0, "key", key_mx);

    return out;
}

template <typename T>
constexpr mxClassID mx_class_for() {
    if constexpr (std::is_same_v<T, double>) return mxDOUBLE_CLASS;
    else if constexpr (std::is_same_v<T, float>) return mxSINGLE_CLASS;
    else { static_assert(sizeof(T) == 0, "unsupported scalar type"); return mxUNKNOWN_CLASS; }
}

template <typename T>
void run_bqrrp(int nlhs, mxArray* plhs[], const mxArray* A_in,
               int64_t b_sz, T d_factor, const mxArray* state_in)
{
    const int64_t m = (int64_t) mxGetM(A_in);
    const int64_t n = (int64_t) mxGetN(A_in);
    const int64_t k = std::min(m, n);
    const T* A_data = (const T*) mxGetData(A_in);

    // BQRRP overwrites A; copy into a mutable scratch buffer.
    std::vector<T> A_work((size_t) m * (size_t) n);
    std::memcpy(A_work.data(), A_data, sizeof(T) * (size_t) m * (size_t) n);

    // Per BQRRP's docstring, tau has size n.
    std::vector<T> tau(std::max((int64_t) 1, n));
    std::vector<int64_t> J(std::max((int64_t) 1, n));

    auto state = read_state(state_in);

    RandLAPACK::BQRRP<T, RNG> alg(false, b_sz);
    int ret = alg.call(m, n, A_work.data(), m, d_factor, tau.data(), J.data(), state);
    if (ret != 0) {
        mexErrMsgIdAndTxt("randlapack:bqrrp_mex:returnCode",
            "BQRRP returned non-zero status %d", ret);
    }

    // R first: orgqr will clobber the upper triangle when materializing Q.
    plhs[1] = mxCreateNumericMatrix((mwSize) k, (mwSize) n, mx_class_for<T>(), mxREAL);
    T* R_out = (T*) mxGetData(plhs[1]);
    for (int64_t j = 0; j < n; ++j) {
        const int64_t lim = std::min(j + 1, k);
        for (int64_t i = 0; i < lim; ++i) {
            R_out[i + j * k] = A_work[i + j * m];
        }
        for (int64_t i = lim; i < k; ++i) {
            R_out[i + j * k] = (T) 0;
        }
    }

    // Form explicit Q (m-by-k) in place via LAPACK++'s ungqr (== orgqr for real T).
    int64_t info = lapack::ungqr(m, k, k, A_work.data(), m, tau.data());
    if (info != 0) {
        mexErrMsgIdAndTxt("randlapack:bqrrp_mex:ungqr",
            "lapack::ungqr returned info=%lld while forming explicit Q",
            (long long) info);
    }

    plhs[0] = mxCreateNumericMatrix((mwSize) m, (mwSize) k, mx_class_for<T>(), mxREAL);
    std::memcpy(mxGetData(plhs[0]), A_work.data(), sizeof(T) * (size_t) m * (size_t) k);

    // J is already 1-based per BQRRP convention; pass through unchanged.
    plhs[2] = mxCreateNumericMatrix(1, (mwSize) n, mxINT64_CLASS, mxREAL);
    std::memcpy(mxGetData(plhs[2]), J.data(), sizeof(int64_t) * (size_t) n);

    if (nlhs >= 4) {
        plhs[3] = write_state(state);
    }
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs != 4) {
            mexErrMsgIdAndTxt("randlapack:bqrrp_mex:nargin",
                "Expected 4 input arguments: A, b_sz, d_factor, state");
        }
        if (nlhs < 3 || nlhs > 4) {
            mexErrMsgIdAndTxt("randlapack:bqrrp_mex:nargout",
                "Expected 3 or 4 output arguments");
        }

        const mxArray* A_in = prhs[0];
        if (!mxIsNumeric(A_in) || mxIsComplex(A_in)
            || mxGetNumberOfDimensions(A_in) != 2) {
            mexErrMsgIdAndTxt("randlapack:bqrrp_mex:A",
                "A must be a real 2D numeric matrix (single or double)");
        }
        if (!mxIsScalar(prhs[1])) {
            mexErrMsgIdAndTxt("randlapack:bqrrp_mex:b_sz", "b_sz must be scalar");
        }
        const int64_t b_sz = (int64_t) mxGetScalar(prhs[1]);

        if (!mxIsScalar(prhs[2])) {
            mexErrMsgIdAndTxt("randlapack:bqrrp_mex:d_factor", "d_factor must be scalar");
        }

        const mxArray* state_in = prhs[3];

        if (mxIsDouble(A_in)) {
            const double d_factor = mxGetScalar(prhs[2]);
            run_bqrrp<double>(nlhs, plhs, A_in, b_sz, d_factor, state_in);
        } else if (mxIsSingle(A_in)) {
            const float d_factor = (float) mxGetScalar(prhs[2]);
            run_bqrrp<float>(nlhs, plhs, A_in, b_sz, d_factor, state_in);
        } else {
            mexErrMsgIdAndTxt("randlapack:bqrrp_mex:type",
                "A must be single or double");
        }
    } catch (const RandLAPACK::Error& e) {
        mexErrMsgIdAndTxt("randlapack:bqrrp_mex:RandLAPACKError", "%s", e.what());
    } catch (const RandBLAS::Error& e) {
        mexErrMsgIdAndTxt("randlapack:bqrrp_mex:RandBLASError", "%s", e.what());
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("randlapack:bqrrp_mex:StdError", "%s", e.what());
    } catch (...) {
        mexErrMsgIdAndTxt("randlapack:bqrrp_mex:Unknown",
            "Unknown exception in bqrrp_mex");
    }
}
