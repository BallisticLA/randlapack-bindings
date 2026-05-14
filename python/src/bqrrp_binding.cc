// pybind11 binding for RandLAPACK::BQRRP.
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
//     "accept any layout, convert as needed" idiom. BQRRP needs column-
//     major internally, so the conversion happens somewhere; the wrapper
//     hides it from casual callers.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <RandLAPACK.hh>
#include <lapack.hh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

using RNG = RandBLAS::DefaultRNG;  // Philox4x32

template <typename T>
std::tuple<py::array, py::array, py::array_t<int64_t>, py::dict>
run_bqrrp_typed(
    py::array A,
    int64_t b_sz,
    double d_factor_in,
    py::array_t<uint32_t> state_counter,
    py::array_t<uint32_t> state_key,
    bool implicit)
{
    auto buf = A.request();
    const int64_t m = static_cast<int64_t>(buf.shape[0]);
    const int64_t n = static_cast<int64_t>(buf.shape[1]);
    const int64_t k = std::min(m, n);
    const T d_factor = static_cast<T>(d_factor_in);
    const py::ssize_t sz = static_cast<py::ssize_t>(sizeof(T));

    // BQRRP overwrites A; copy into a mutable scratch buffer. A is
    // Fortran-order (column-major) by precondition; the contiguous memcpy
    // is correct because mn elements are laid out as A_work[i + j*m].
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    std::vector<T> A_work(mn);
    std::memcpy(A_work.data(), buf.ptr, sizeof(T) * mn);

    std::vector<T> tau(std::max<int64_t>(1, n));
    std::vector<int64_t> J(std::max<int64_t>(1, n));

    auto ctr_buf = state_counter.request();
    auto key_buf = state_key.request();
    if (ctr_buf.size != 4 || key_buf.size != 2) {
        throw std::runtime_error(
            "state_counter must be length 4 and state_key must be length 2");
    }
    RandBLAS::RNGState<RNG> state;
    const uint32_t* ctr = static_cast<const uint32_t*>(ctr_buf.ptr);
    const uint32_t* key = static_cast<const uint32_t*>(key_buf.ptr);
    for (size_t i = 0; i < 4; ++i) state.counter.v[i] = ctr[i];
    for (size_t i = 0; i < 2; ++i) state.key.v[i]     = key[i];

    RandLAPACK::BQRRP<T, RNG> alg(false, b_sz);
    int ret = alg.call(m, n, A_work.data(), m, d_factor, tau.data(),
                       J.data(), state);
    if (ret != 0) {
        throw std::runtime_error(
            "BQRRP returned non-zero status " + std::to_string(ret));
    }

    py::array out1, out2;
    if (implicit) {
        // out1 = A_out (m-by-n) in Fortran-order; out2 = tau (length n).
        auto A_out = py::array_t<T>({m, n}, {sz, sz * m});
        std::memcpy(A_out.mutable_data(), A_work.data(), sizeof(T) * mn);
        out1 = A_out;

        auto tau_out = py::array_t<T>(n);
        std::memcpy(tau_out.mutable_data(), tau.data(), sizeof(T) * n);
        out2 = tau_out;
    } else {
        // Extract R first; ungqr clobbers the upper triangle of A_work.
        auto R = py::array_t<T>({k, n}, {sz, sz * k});
        T* R_data = R.mutable_data();
        for (int64_t j = 0; j < n; ++j) {
            const int64_t lim = std::min(j + 1, k);
            for (int64_t i = 0; i < lim; ++i) {
                R_data[i + j * k] = A_work[i + j * m];
            }
            for (int64_t i = lim; i < k; ++i) {
                R_data[i + j * k] = static_cast<T>(0);
            }
        }
        out2 = R;

        if (k > 0) {
            int64_t info = lapack::ungqr(m, k, k, A_work.data(), m, tau.data());
            if (info != 0) {
                throw std::runtime_error(
                    "lapack::ungqr returned info=" + std::to_string(info));
            }
        }

        auto Q = py::array_t<T>({m, k}, {sz, sz * m});
        std::memcpy(Q.mutable_data(), A_work.data(),
                    sizeof(T) * static_cast<size_t>(m) * static_cast<size_t>(k));
        out1 = Q;
    }

    // J: convert 1-based (BQRRP native) to 0-based (scipy/NumPy convention).
    auto J_out = py::array_t<int64_t>(static_cast<py::ssize_t>(n));
    int64_t* J_ptr = J_out.mutable_data();
    for (int64_t i = 0; i < n; ++i) {
        J_ptr[i] = J[i] - 1;
    }

    // Return advanced state as a dict.
    auto ctr_out = py::array_t<uint32_t>(4);
    auto key_out = py::array_t<uint32_t>(2);
    uint32_t* ctr_ptr = ctr_out.mutable_data();
    uint32_t* key_ptr = key_out.mutable_data();
    for (size_t i = 0; i < 4; ++i) ctr_ptr[i] = state.counter.v[i];
    for (size_t i = 0; i < 2; ++i) key_ptr[i] = state.key.v[i];
    py::dict state_dict;
    state_dict["counter"] = ctr_out;
    state_dict["key"]     = key_out;

    return std::make_tuple(out1, out2, J_out, state_dict);
}


std::tuple<py::array, py::array, py::array_t<int64_t>, py::dict>
bqrrp_impl(
    py::array A,
    int64_t b_sz,
    double d_factor,
    py::array_t<uint32_t> state_counter,
    py::array_t<uint32_t> state_key,
    std::string mode)
{
    auto buf = A.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("A must be 2D");
    }
    if (!(A.flags() & py::array::f_style)) {
        // The Python wrapper auto-converts; if we see C-order here, the
        // caller went around the wrapper.
        throw std::runtime_error(
            "A must be Fortran-order (column-major). The Python wrapper "
            "auto-converts; see randlapack.bqrrp.");
    }
    if (buf.shape[0] == 0 || buf.shape[1] == 0) {
        throw std::runtime_error("A must be nonempty");
    }

    bool implicit;
    if (mode == "explicit") {
        implicit = false;
    } else if (mode == "implicit") {
        implicit = true;
    } else {
        throw std::runtime_error(
            "mode must be 'explicit' or 'implicit'; got '" + mode + "'");
    }

    if (buf.format == py::format_descriptor<double>::format()) {
        return run_bqrrp_typed<double>(A, b_sz, d_factor, state_counter,
                                       state_key, implicit);
    } else if (buf.format == py::format_descriptor<float>::format()) {
        return run_bqrrp_typed<float>(A, b_sz, d_factor, state_counter,
                                      state_key, implicit);
    } else {
        throw std::runtime_error("A must be float32 or float64");
    }
}

}  // namespace


void register_bqrrp(py::module_& m) {
    m.def("bqrrp", &bqrrp_impl,
        py::arg("A"),
        py::arg("b_sz"),
        py::arg("d_factor"),
        py::arg("state_counter"),
        py::arg("state_key"),
        py::arg("mode"),
        "Low-level BQRRP binding. Users should call randlapack.bqrrp() "
        "instead, which handles defaults, layout coercion, and state "
        "normalization.");
}
