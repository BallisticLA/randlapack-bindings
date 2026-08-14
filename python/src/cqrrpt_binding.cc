// nanobind binding for RandLAPACK::CQRRPT.
//
// Low-level Python signature (users normally call randlapack.cqrrpt instead):
//
//   (Q, R, J, state_out) = _randlapack.cqrrpt(
//       A, d_factor, eps, state_counter, state_key)
//
// Q is (m, k), R is (k, n), J is (n,) with k = the rank CQRRPT detected, so
//     A[:, J[:k]] ~= Q @ R[:, :k]
// always, and A[:, J] ~= Q @ R once k reaches the numerical rank.
//
// J is returned 0-BASED. CQRRPT's pivots are 1-based natively (they come
// from LAPACK's geqp3, and RandLAPACK's util::col_swap reads them as
// idx[i]-1); this binding subtracts 1 so `A[:, J]` slices correctly in
// NumPy, matching scipy.linalg.qr(..., pivoting=True). The MATLAB binding
// leaves them 1-based, for the mirror-image reason.
//
// Buffer shapes, since neither output can be factored in place: CQRRPT
// overwrites A with Q and needs a full (m, n) work buffer even though only
// the first k columns are returned, and it writes R into a caller-allocated
// (n, n) buffer (ldr >= n) of which only the first k rows are meaningful.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <RandLAPACK.hh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

using RNG = RandBLAS::DefaultRNG;  // Philox4x32

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
nb::tuple run_cqrrpt_typed(
    const nb::ndarray<>& A,
    double d_factor_in, double eps_in,
    const uint32_t* ctr, const uint32_t* key)
{
    const int64_t m = static_cast<int64_t>(A.shape(0));
    const int64_t n = static_cast<int64_t>(A.shape(1));
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    const T d_factor = static_cast<T>(d_factor_in);
    const T eps      = static_cast<T>(eps_in);
    const T* A_src = static_cast<const T*>(A.data());

    RandBLAS::RNGState<RNG> state;
    for (size_t i = 0; i < 4; ++i) state.counter.v[i] = ctr[i];
    for (size_t i = 0; i < 2; ++i) state.key.v[i]     = key[i];

    T* A_work = new T[mn];
    std::vector<T> R_work(static_cast<size_t>(n) * static_cast<size_t>(n));
    // J is ZERO-INITIALIZED, and must be. CQRRPT hands it straight to
    // LAPACK's geqp3, which READS jpvt on entry: a nonzero entry means
    // "this column is fixed, move it to the front". Uninitialized memory
    // here makes the pivoting -- and the detected rank -- depend on heap
    // garbage. std::vector value-initializes, which is what RandLAPACK's
    // own tests happen to pass, which is why the precondition is invisible
    // from inside the library.
    std::vector<int64_t> J(static_cast<size_t>(n), 0);

    int64_t k = 0;
    int ret = 0;
    {
        nb::gil_scoped_release release;
        std::memcpy(A_work, A_src, sizeof(T) * mn);
        RandLAPACK::CQRRPT<T, RNG> alg(false, eps);
        ret = alg.call(m, n, A_work, m, R_work.data(), n, J.data(),
                       d_factor, state);
        k = alg.rank;
    }

    if (ret != 0) {
        delete[] A_work;
        throw std::runtime_error(
            "CQRRPT returned non-zero status " + std::to_string(ret));
    }
    if (k <= 0) {
        delete[] A_work;
        throw std::runtime_error(
            "CQRRPT detected rank 0; the input may be numerically zero, or "
            "eps may be too large");
    }

    // Q = the leading k columns of the in-place factored A (column-major,
    // so they are the first m*k contiguous elements).
    T* Q_out = new T[static_cast<size_t>(m) * static_cast<size_t>(k)];
    std::memcpy(Q_out, A_work,
                sizeof(T) * static_cast<size_t>(m) * static_cast<size_t>(k));
    delete[] A_work;

    // R = the leading k rows of the (n, n) scratch, restrided from ldr=n.
    T* R_out = new T[static_cast<size_t>(k) * static_cast<size_t>(n)];
    for (int64_t j = 0; j < n; ++j)
        std::memcpy(&R_out[j * k], &R_work[j * n], sizeof(T) * k);

    // J: 1-based (CQRRPT native) to 0-based (scipy/NumPy convention).
    int64_t* J_out = new int64_t[n];
    for (int64_t i = 0; i < n; ++i) J_out[i] = J[i] - 1;

    return nb::make_tuple(owned_2d<T>(Q_out, m, k),
                          owned_2d<T>(R_out, k, n),
                          owned_1d<int64_t>(J_out, n),
                          state_to_dict(state));
}

bool is_f_contiguous_2d(const nb::ndarray<>& A) {
    return (A.shape(0) <= 1 || A.stride(0) == 1) &&
           (A.shape(1) <= 1 || A.stride(1) == static_cast<int64_t>(A.shape(0)));
}

nb::tuple cqrrpt_impl(
    nb::ndarray<> A,
    double d_factor, double eps,
    nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig> state_counter,
    nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig> state_key)
{
    if (A.ndim() != 2)
        throw std::runtime_error("A must be 2D");
    if (!is_f_contiguous_2d(A))
        throw std::runtime_error(
            "A must be Fortran-order (column-major). The Python wrapper "
            "auto-converts; see randlapack.cqrrpt.");
    if (A.shape(0) == 0 || A.shape(1) == 0)
        throw std::runtime_error("A must be nonempty");
    if (state_counter.shape(0) != 4 || state_key.shape(0) != 2)
        throw std::runtime_error(
            "state_counter must be length 4 and state_key must be length 2");
    if (d_factor < 1.0) throw std::runtime_error("d_factor must be >= 1.0");
    if (eps < 0.0)      throw std::runtime_error("eps must be nonnegative");

    // CQRRPT sketches to d = d_factor*n rows and needs m >= d >= n. Checked
    // here, with the arithmetic spelled out, rather than letting it surface
    // as a dimension complaint from inside the sketching chain.
    const int64_t m = static_cast<int64_t>(A.shape(0));
    const int64_t n = static_cast<int64_t>(A.shape(1));
    const int64_t d = static_cast<int64_t>(d_factor * static_cast<double>(n));
    if (m < d)
        throw std::runtime_error(
            "CQRRPT requires a tall matrix: m >= d_factor*n. Got m="
            + std::to_string(m) + ", n=" + std::to_string(n)
            + ", d_factor=" + std::to_string(d_factor) + " (needs m >= "
            + std::to_string(d) + "). Reduce d_factor toward 1.0, or use "
            "randlapack.bqrrp for wide matrices.");

    const uint32_t* ctr = state_counter.data();
    const uint32_t* key = state_key.data();

    if (A.dtype() == nb::dtype<double>())
        return run_cqrrpt_typed<double>(A, d_factor, eps, ctr, key);
    if (A.dtype() == nb::dtype<float>())
        return run_cqrrpt_typed<float>(A, d_factor, eps, ctr, key);
    throw std::runtime_error("A must be float32 or float64");
}

}  // namespace


void register_cqrrpt(nb::module_& m) {
    m.def("cqrrpt", &cqrrpt_impl,
        nb::arg("A"), nb::arg("d_factor"), nb::arg("eps"),
        nb::arg("state_counter"), nb::arg("state_key"),
        "Low-level CQRRPT binding. Users should call randlapack.cqrrpt() "
        "instead, which handles defaults, layout coercion, and state "
        "normalization.");
}
