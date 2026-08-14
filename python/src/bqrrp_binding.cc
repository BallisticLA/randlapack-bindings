// nanobind binding for RandLAPACK::BQRRP.
//
// Low-level Python signature (users normally call randlapack.bqrrp instead,
// which handles defaults, state normalization, and layout coercion):
//
//   (out1, out2, J, state_out) = _randlapack.bqrrp(
//       A, b_sz, d_factor, state_counter, state_key, mode)
//
// where mode is 'explicit' (out1=Q, out2=R) or 'implicit' (out1=A_out
// GEQP3-format, out2=tau Householder scalars). state_out is a dict with
// 'counter' (uint32[4]) and 'key' (uint32[2]) holding the advanced
// Philox4x32 state.
//
// Python-side conventions baked in (these are choices for the Python
// ecosystem, distinct from the MATLAB binding):
//
//   * J is 0-based. BQRRP stores pivots 1-based natively (see
//     rl_bqrrp.hh:330: std::iota(..., 1)); this binding subtracts 1 before
//     returning. Reasoning: scipy.linalg.qr(A, pivoting=True) returns a
//     0-based permutation vector, and `A[:, J]` slicing works directly in
//     NumPy with no off-by-one. The MATLAB binding leaves J 1-based since
//     MATLAB indexing is 1-based.
//
//   * Input layout: this binding requires column-major (Fortran-order) A.
//     The Python-side wrapper (randlapack.bqrrp) auto-converts C-order
//     callers via np.asfortranarray(A), matching scipy.linalg.qr's
//     "accept any layout, convert as needed" idiom.
//
// nanobind specifics worth knowing when reading this file:
//
//   * nb::ndarray strides are in ELEMENTS, not bytes (unlike the buffer
//     protocol). Fortran-order (m, n) therefore has strides {1, m}.
//   * Output arrays are heap allocations wrapped in an nb::capsule whose
//     deleter frees them when the last NumPy reference dies -- nanobind's
//     idiom for "NumPy owns this now".
//   * The GIL is released around the numerical work (the BQRRP call and the
//     ungqr that forms Q), and reacquired before any Python object is
//     created. The input buffers stay valid while released because the
//     caller holds references to the arrays.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <RandLAPACK.hh>
#include <lapack.hh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

using RNG = RandBLAS::DefaultRNG;  // Philox4x32

// Wrap a heap buffer as a NumPy-owned Fortran-order array.
template <typename T>
nb::object make_owned_array_2d(T* data, int64_t rows, int64_t cols) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<2>>(
        data, {static_cast<size_t>(rows), static_cast<size_t>(cols)},
        owner, {1, rows}));
}

template <typename T>
nb::object make_owned_array_1d(T* data, int64_t len) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<1>>(
        data, {static_cast<size_t>(len)}, owner));
}

template <typename T>
nb::tuple run_bqrrp_typed(
    const nb::ndarray<>& A,
    int64_t b_sz,
    double d_factor_in,
    const uint32_t* ctr,
    const uint32_t* key,
    bool implicit)
{
    const int64_t m = static_cast<int64_t>(A.shape(0));
    const int64_t n = static_cast<int64_t>(A.shape(1));
    const int64_t k = std::min(m, n);
    const T d_factor = static_cast<T>(d_factor_in);
    const T* A_src = static_cast<const T*>(A.data());
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);

    RandBLAS::RNGState<RNG> state;
    for (size_t i = 0; i < 4; ++i) state.counter.v[i] = ctr[i];
    for (size_t i = 0; i < 2; ++i) state.key.v[i]     = key[i];

    // Outputs and scratch, allocated up front so the whole numerical section
    // can run without the GIL. BQRRP overwrites its input, so A is copied
    // into a mutable buffer; in explicit mode that buffer later becomes Q's
    // storage prefix, in implicit mode it IS out1.
    T* A_work = new T[mn];
    std::vector<T> tau(std::max<int64_t>(1, n));
    std::vector<int64_t> J(std::max<int64_t>(1, n));

    int ret = 0;
    int64_t ungqr_info = 0;
    {
        nb::gil_scoped_release release;
        std::memcpy(A_work, A_src, sizeof(T) * mn);
        RandLAPACK::BQRRP<T, RNG> alg(false, b_sz);
        ret = alg.call(m, n, A_work, m, d_factor, tau.data(), J.data(), state);
    }
    if (ret != 0) {
        delete[] A_work;
        throw std::runtime_error(
            "BQRRP returned non-zero status " + std::to_string(ret));
    }

    nb::object out1, out2;
    if (implicit) {
        // out1 = A_out (m-by-n, GEQP3 format); out2 = tau (length n).
        out1 = make_owned_array_2d<T>(A_work, m, n);   // ownership transferred
        T* tau_out = new T[n];
        std::memcpy(tau_out, tau.data(), sizeof(T) * n);
        out2 = make_owned_array_1d<T>(tau_out, n);
    } else {
        // Extract R first; ungqr clobbers the upper triangle of A_work.
        T* R_data = new T[static_cast<size_t>(k) * static_cast<size_t>(n)];
        {
            nb::gil_scoped_release release;
            for (int64_t j = 0; j < n; ++j) {
                const int64_t lim = std::min(j + 1, k);
                for (int64_t i = 0; i < lim; ++i)
                    R_data[i + j * k] = A_work[i + j * m];
                for (int64_t i = lim; i < k; ++i)
                    R_data[i + j * k] = static_cast<T>(0);
            }
            if (k > 0)
                ungqr_info = lapack::ungqr(m, k, k, A_work, m, tau.data());
        }
        if (ungqr_info != 0) {
            delete[] A_work;
            delete[] R_data;
            throw std::runtime_error(
                "lapack::ungqr returned info=" + std::to_string(ungqr_info));
        }
        T* Q_data = new T[static_cast<size_t>(m) * static_cast<size_t>(k)];
        std::memcpy(Q_data, A_work,
                    sizeof(T) * static_cast<size_t>(m) * static_cast<size_t>(k));
        delete[] A_work;
        out1 = make_owned_array_2d<T>(Q_data, m, k);
        out2 = make_owned_array_2d<T>(R_data, k, n);
    }

    // J: convert 1-based (BQRRP native) to 0-based (scipy/NumPy convention).
    int64_t* J_out = new int64_t[n];
    for (int64_t i = 0; i < n; ++i) J_out[i] = J[i] - 1;
    nb::object J_arr = make_owned_array_1d<int64_t>(J_out, n);

    // Return advanced state as a dict.
    uint32_t* ctr_out = new uint32_t[4];
    uint32_t* key_out = new uint32_t[2];
    for (size_t i = 0; i < 4; ++i) ctr_out[i] = state.counter.v[i];
    for (size_t i = 0; i < 2; ++i) key_out[i] = state.key.v[i];
    nb::dict state_dict;
    state_dict["counter"] = make_owned_array_1d<uint32_t>(ctr_out, 4);
    state_dict["key"]     = make_owned_array_1d<uint32_t>(key_out, 2);

    return nb::make_tuple(out1, out2, J_arr, state_dict);
}

bool is_f_contiguous_2d(const nb::ndarray<>& A) {
    // nanobind strides are in elements. A degenerate dimension of extent 1
    // is contiguous regardless of its stride.
    return (A.shape(0) <= 1 || A.stride(0) == 1) &&
           (A.shape(1) <= 1 ||
            A.stride(1) == static_cast<int64_t>(A.shape(0)));
}

nb::tuple bqrrp_impl(
    nb::ndarray<> A,
    int64_t b_sz,
    double d_factor,
    nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig> state_counter,
    nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig> state_key,
    const std::string& mode)
{
    if (A.ndim() != 2)
        throw std::runtime_error("A must be 2D");
    if (!is_f_contiguous_2d(A)) {
        // The Python wrapper auto-converts; if we see C-order here, the
        // caller went around the wrapper.
        throw std::runtime_error(
            "A must be Fortran-order (column-major). The Python wrapper "
            "auto-converts; see randlapack.bqrrp.");
    }
    if (A.shape(0) == 0 || A.shape(1) == 0)
        throw std::runtime_error("A must be nonempty");
    if (state_counter.shape(0) != 4 || state_key.shape(0) != 2)
        throw std::runtime_error(
            "state_counter must be length 4 and state_key must be length 2");

    bool implicit;
    if (mode == "explicit")      implicit = false;
    else if (mode == "implicit") implicit = true;
    else
        throw std::runtime_error(
            "mode must be 'explicit' or 'implicit'; got '" + mode + "'");

    const uint32_t* ctr = state_counter.data();
    const uint32_t* key = state_key.data();

    if (A.dtype() == nb::dtype<double>())
        return run_bqrrp_typed<double>(A, b_sz, d_factor, ctr, key, implicit);
    if (A.dtype() == nb::dtype<float>())
        return run_bqrrp_typed<float>(A, b_sz, d_factor, ctr, key, implicit);
    throw std::runtime_error("A must be float32 or float64");
}

}  // namespace


void register_bqrrp(nb::module_& m) {
    m.def("bqrrp", &bqrrp_impl,
        nb::arg("A"),
        nb::arg("b_sz"),
        nb::arg("d_factor"),
        nb::arg("state_counter"),
        nb::arg("state_key"),
        nb::arg("mode"),
        "Low-level BQRRP binding. Users should call randlapack.bqrrp() "
        "instead, which handles defaults, layout coercion, and state "
        "normalization.");
}
