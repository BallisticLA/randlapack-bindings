// nanobind binding for RandLAPACK::RSVD.
//
// Low-level Python signature (users normally call randlapack.rsvd instead):
//
//   (U, s, V, state_out) = _randlapack.rsvd(
//       A, k, tol, block_sz, p, passes_per_iteration,
//       state_counter, state_key)
//
// U is (m, k_out), s is (k_out,), V is (n, k_out) UN-transposed, so
//     A ~= U @ np.diag(s) @ V.T
// which is numpy.linalg.svd's (u, s, vh) with V = vh.T -- the wrapper
// documents the difference rather than transposing behind the caller's back.
//
// k is in-out at the RandLAPACK level: QB stops early once the residual
// falls under tol, so k_out <= k and the true value is carried by the
// output shapes.
//
// RSVD's allocation contract dictates the shape of this file: RSVD::call
// takes U, S and V by REFERENCE-TO-POINTER and calloc()s them internally.
// They must be copied out and released with free() -- not delete[], not a
// nanobind capsule deleter. The copies are therefore unavoidable, and the
// raw pointers live in an RAII guard so an exception between the call and
// the copy cannot leak them.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <RandLAPACK.hh>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

using RNG = RandBLAS::DefaultRNG;  // Philox4x32

template <typename T>
struct CallocGuard {
    T* U = nullptr;
    T* S = nullptr;
    T* V = nullptr;
    ~CallocGuard() { std::free(U); std::free(S); std::free(V); }
};

// Wrap a heap buffer as a NumPy-owned Fortran-order array. Strides are in
// ELEMENTS in nanobind, not bytes.
template <typename T>
nb::object owned_2d(T* data, int64_t rows, int64_t cols) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<2>>(
        data, {static_cast<size_t>(rows), static_cast<size_t>(cols)},
        owner, {1, rows}));
}

template <typename T>
nb::object owned_1d(T* data, int64_t len) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<1>>(
        data, {static_cast<size_t>(len)}, owner));
}

nb::dict state_to_dict(const RandBLAS::RNGState<RNG>& state) {
    uint32_t* ctr_out = new uint32_t[4];
    uint32_t* key_out = new uint32_t[2];
    for (size_t i = 0; i < 4; ++i) ctr_out[i] = state.counter.v[i];
    for (size_t i = 0; i < 2; ++i) key_out[i] = state.key.v[i];
    nb::dict d;
    d["counter"] = owned_1d<uint32_t>(ctr_out, 4);
    d["key"]     = owned_1d<uint32_t>(key_out, 2);
    return d;
}

template <typename T>
nb::tuple run_rsvd_typed(
    const nb::ndarray<>& A,
    int64_t k_req, double tol_in, int64_t block_sz, int64_t p, int64_t ppi,
    const uint32_t* ctr, const uint32_t* key)
{
    const int64_t m = static_cast<int64_t>(A.shape(0));
    const int64_t n = static_cast<int64_t>(A.shape(1));
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    const T tol = static_cast<T>(tol_in);
    const T* A_src = static_cast<const T*>(A.data());

    RandBLAS::RNGState<RNG> state;
    for (size_t i = 0; i < 4; ++i) state.counter.v[i] = ctr[i];
    for (size_t i = 0; i < 2; ++i) state.key.v[i]     = key[i];

    T* A_work = new T[mn];   // the QB chain overwrites its input
    int64_t k = k_req;
    CallocGuard<T> g;
    int ret = 0;
    {
        // GIL released across the whole numerical section. No Python object
        // is touched inside; the input buffer stays alive because the caller
        // holds a reference to the array.
        nb::gil_scoped_release release;
        std::memcpy(A_work, A_src, sizeof(T) * mn);

        // The dependency-injection chain RSVD requires, assembled exactly as
        // RandLAPACK's own test does (test/drivers/test_rsvd.cc). The
        // verbose/cond/orth diagnostic flags are all off: they print to
        // stdout and compute extra condition numbers, neither of which
        // belongs in a library call.
        RandLAPACK::PLUL<T>      Stab(false, false);
        RandLAPACK::RS<T, RNG>   RS(Stab, p, ppi, false, false);
        RandLAPACK::CholQRQ<T>   Orth_RF(false, false);
        RandLAPACK::RF<T, RNG>   RF(RS, Orth_RF, false, false);
        RandLAPACK::CholQRQ<T>   Orth_QB(false, false);
        RandLAPACK::QB<T, RNG>   QB(RF, Orth_QB, false, false);
        RandLAPACK::RSVD<T, RNG> RSVD(QB, block_sz);

        ret = RSVD.call(m, n, A_work, k, tol, g.U, g.S, g.V, state);
    }
    delete[] A_work;

    if (ret != 0)
        throw std::runtime_error(
            "RSVD returned non-zero status " + std::to_string(ret));
    if (k <= 0 || g.U == nullptr || g.S == nullptr || g.V == nullptr)
        throw std::runtime_error(
            "RSVD produced an empty factorization (k=" + std::to_string(k)
            + "); try a larger target rank or a looser tolerance");

    const size_t ks = static_cast<size_t>(k);
    T* U_out = new T[static_cast<size_t>(m) * ks];
    T* s_out = new T[ks];
    T* V_out = new T[static_cast<size_t>(n) * ks];
    std::memcpy(U_out, g.U, sizeof(T) * static_cast<size_t>(m) * ks);
    std::memcpy(s_out, g.S, sizeof(T) * ks);
    std::memcpy(V_out, g.V, sizeof(T) * static_cast<size_t>(n) * ks);

    return nb::make_tuple(owned_2d<T>(U_out, m, k),
                          owned_1d<T>(s_out, k),
                          owned_2d<T>(V_out, n, k),
                          state_to_dict(state));
}

bool is_f_contiguous_2d(const nb::ndarray<>& A) {
    return (A.shape(0) <= 1 || A.stride(0) == 1) &&
           (A.shape(1) <= 1 || A.stride(1) == static_cast<int64_t>(A.shape(0)));
}

nb::tuple rsvd_impl(
    nb::ndarray<> A,
    int64_t k, double tol, int64_t block_sz, int64_t p, int64_t ppi,
    nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig> state_counter,
    nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig> state_key)
{
    if (A.ndim() != 2)
        throw std::runtime_error("A must be 2D");
    if (!is_f_contiguous_2d(A))
        throw std::runtime_error(
            "A must be Fortran-order (column-major). The Python wrapper "
            "auto-converts; see randlapack.rsvd.");
    if (A.shape(0) == 0 || A.shape(1) == 0)
        throw std::runtime_error("A must be nonempty");
    if (state_counter.shape(0) != 4 || state_key.shape(0) != 2)
        throw std::runtime_error(
            "state_counter must be length 4 and state_key must be length 2");
    if (k <= 0)        throw std::runtime_error("k must be positive");
    if (block_sz <= 0) throw std::runtime_error("block_sz must be positive");
    if (p < 0)         throw std::runtime_error("p must be nonnegative");
    if (ppi <= 0)      throw std::runtime_error(
                           "passes_per_iteration must be positive");
    if (tol < 0.0)     throw std::runtime_error("tol must be nonnegative");

    const uint32_t* ctr = state_counter.data();
    const uint32_t* key = state_key.data();

    if (A.dtype() == nb::dtype<double>())
        return run_rsvd_typed<double>(A, k, tol, block_sz, p, ppi, ctr, key);
    if (A.dtype() == nb::dtype<float>())
        return run_rsvd_typed<float>(A, k, tol, block_sz, p, ppi, ctr, key);
    throw std::runtime_error("A must be float32 or float64");
}

}  // namespace


void register_rsvd(nb::module_& m) {
    m.def("rsvd", &rsvd_impl,
        nb::arg("A"), nb::arg("k"), nb::arg("tol"), nb::arg("block_sz"),
        nb::arg("p"), nb::arg("passes_per_iteration"),
        nb::arg("state_counter"), nb::arg("state_key"),
        "Low-level RSVD binding. Users should call randlapack.rsvd() "
        "instead, which handles defaults, layout coercion, and state "
        "normalization.");
}
