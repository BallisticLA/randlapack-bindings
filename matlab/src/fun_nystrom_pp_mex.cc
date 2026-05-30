// MEX wrapper for RandLAPACK::FunNystromPP_v2 using MATLAB's C++ Data API
// (R2018a+).
//
// MATLAB-side signature (low-level; users normally call randlapack.fun_nystrom_pp.m):
//
//   [est, t1, t2] = fun_nystrom_pp_mex(A, Omega1, Omega2, func, q, poly_lambda, lfa_type, d)
//
// where:
//   A           n × n double matrix, column-major (symmetric, upper triangle used)
//   Omega1      n × k double matrix
//   Omega2      n × s double matrix (may be empty if k == n; Phase 2 then skipped)
//   func        'sqrt' | 'log' | 'poly' | 'square' | 'identity'
//   q           subspace-iter count (>= 1)
//   poly_lambda λ in f(x) = x(x + λ)  (only used when func == 'poly')
//   lfa_type    'exact' | 'scalar' | 'block'
//                 exact:  build f(A) once via syevd, every f(A)·X is a GEMV
//                 scalar: per-column scalar Lanczos-FA at depth d
//                 block:  block Lanczos-FA at depth d
//   d           Lanczos depth (only used for lfa_type ∈ {scalar, block})
//
// Outputs:
//   est         trace estimate t1 + t2 (scalar double)
//   t1          Phase 1 contribution (scalar double)
//   t2          Phase 2 contribution (scalar double; 0 when k == n)

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <RandLAPACK.hh>
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace linops = RandLAPACK::linops;

using matlab::mex::ArgumentList;
using matlab::data::Array;
using matlab::data::ArrayFactory;
using matlab::data::ArrayType;
using matlab::data::CharArray;
template <typename T> using TypedArray = matlab::data::TypedArray<T>;


class MexFunction : public matlab::mex::Function {
private:
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    ArrayFactory factory;

    [[noreturn]] void raise(const std::string& id, const std::string& msg) {
        matlabPtr->feval(u"error", 0, std::vector<Array>{
            factory.createCharArray(id),
            factory.createCharArray(msg)
        });
        std::terminate();
    }

    std::string read_string(const Array& a, const std::string& field) {
        if (a.getType() != ArrayType::CHAR) {
            raise("randlapack:fun_nystrom_pp_mex:string",
                  field + " must be a char array (single quotes), not a MATLAB string");
        }
        CharArray ca = a;
        return ca.toAscii();
    }

    int64_t read_int(const Array& a, const std::string& field) {
        if (a.getNumberOfElements() != 1) {
            raise("randlapack:fun_nystrom_pp_mex:scalar",
                  field + " must be a scalar");
        }
        TypedArray<double> v = a;
        return static_cast<int64_t>(v[0]);
    }

    double read_double(const Array& a, const std::string& field) {
        if (a.getNumberOfElements() != 1) {
            raise("randlapack:fun_nystrom_pp_mex:scalar",
                  field + " must be a scalar");
        }
        TypedArray<double> v = a;
        return v[0];
    }

    // Copy a TypedArray<double> column-major into a contiguous std::vector.
    void copy_into(const Array& in, std::vector<double>& out, size_t n_elems) {
        out.resize(n_elems);
        TypedArray<double> typed = in;
        size_t idx = 0;
        for (const auto v : typed) {
            out[idx++] = v;
        }
    }

public:
    MexFunction() : matlabPtr(getEngine()) {}

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        try {
            if (inputs.size() != 8) {
                raise("randlapack:fun_nystrom_pp_mex:nargin",
                      "Expected 8 inputs: A, Omega1, Omega2, func, q, poly_lambda, lfa_type, d");
            }
            if (outputs.size() < 1 || outputs.size() > 3) {
                raise("randlapack:fun_nystrom_pp_mex:nargout",
                      "Expected 1 to 3 outputs: [est, t1, t2]");
            }

            // --- A: n × n ---
            const Array& A_in = inputs[0];
            auto A_dims = A_in.getDimensions();
            if (A_dims.size() != 2 || A_dims[0] != A_dims[1]) {
                raise("randlapack:fun_nystrom_pp_mex:A_shape",
                      "A must be a square n × n matrix");
            }
            if (A_in.getType() != ArrayType::DOUBLE) {
                raise("randlapack:fun_nystrom_pp_mex:A_dtype",
                      "A must be double precision");
            }
            const int64_t n = static_cast<int64_t>(A_dims[0]);
            std::vector<double> A_buf;
            copy_into(A_in, A_buf, static_cast<size_t>(n) * n);

            // --- Omega1: n × k ---
            const Array& O1_in = inputs[1];
            auto O1_dims = O1_in.getDimensions();
            if (O1_dims.size() != 2 || static_cast<int64_t>(O1_dims[0]) != n) {
                raise("randlapack:fun_nystrom_pp_mex:Omega1_shape",
                      "Omega1 must be n × k");
            }
            const int64_t k = static_cast<int64_t>(O1_dims[1]);
            std::vector<double> O1_buf;
            copy_into(O1_in, O1_buf, static_cast<size_t>(n) * k);

            // --- Omega2: n × s (may be empty when k == n) ---
            const Array& O2_in = inputs[2];
            auto O2_dims = O2_in.getDimensions();
            int64_t s = 0;
            std::vector<double> O2_buf;
            const bool phase2_skipped = (k == n);
            if (!phase2_skipped) {
                if (O2_dims.size() != 2 || static_cast<int64_t>(O2_dims[0]) != n) {
                    raise("randlapack:fun_nystrom_pp_mex:Omega2_shape",
                          "Omega2 must be n × s when k < n");
                }
                s = static_cast<int64_t>(O2_dims[1]);
                copy_into(O2_in, O2_buf, static_cast<size_t>(n) * s);
            }

            // --- scalar params ---
            const std::string func        = read_string(inputs[3], "func");
            const int64_t     q           = read_int   (inputs[4], "q");
            const double      poly_lambda = read_double(inputs[5], "poly_lambda");
            const std::string lfa_type    = read_string(inputs[6], "lfa_type");
            const int64_t     d           = read_int   (inputs[7], "d");

            // --- scalar function f ---
            std::function<double(double)> fscalar;
            if      (func == "sqrt")     fscalar = [](double x) { return std::sqrt(std::max(x, 0.0)); };
            else if (func == "log")      fscalar = [](double x) { return std::log(x); };
            else if (func == "poly")     fscalar = [poly_lambda](double x) { return x * (x + poly_lambda); };
            else if (func == "square")   fscalar = [](double x) { return x * x; };
            else if (func == "identity") fscalar = [](double x) { return x; };
            else {
                raise("randlapack:fun_nystrom_pp_mex:func",
                      "unknown func '" + func + "' (use sqrt|log|poly|square|identity)");
            }

            // --- f(A)·X oracle ---
            linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A_buf.data(), n,
                                                  Layout::ColMajor);

            using FAFun = std::function<void(int64_t, int64_t, const double*, double*)>;
            FAFun fAfun;

            // Eig of A is built once when lfa_type == 'exact' (the oracle
            // captures V and f(λ) by value below).
            RandLAPACK::LanczosFA<double>      scalar_lfa;
            RandLAPACK::BlockLanczosFA<double> block_lfa;

            if (lfa_type == "exact") {
                std::vector<double> V = A_buf;          // copy; syevd is destructive
                std::vector<double> ev(n);
                lapack::syevd(lapack::Job::Vec, lapack::Uplo::Upper, n,
                              V.data(), n, ev.data());
                std::vector<double> f_lambda(n);
                for (int64_t i = 0; i < n; ++i) f_lambda[i] = fscalar(ev[i]);

                fAfun = [n, V = std::move(V), f_lambda = std::move(f_lambda)]
                        (int64_t m_, int64_t s_, const double *B, double *Y) {
                    std::vector<double> tmp(static_cast<size_t>(n) * s_);
                    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                               n, s_, m_, 1.0, V.data(), n, B, m_, 0.0, tmp.data(), n);
                    for (int64_t j = 0; j < s_; ++j)
                        for (int64_t i = 0; i < n; ++i)
                            tmp[i + j * n] *= f_lambda[i];
                    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                               m_, s_, n, 1.0, V.data(), m_, tmp.data(), n, 0.0, Y, m_);
                };
            } else if (lfa_type == "scalar") {
                fAfun = [&scalar_lfa, &A_op, &fscalar, d]
                        (int64_t m_, int64_t s_, const double *B, double *Y) {
                    scalar_lfa.call(A_op, B, m_, s_, fscalar, d, Y);
                };
            } else if (lfa_type == "block") {
                fAfun = [&block_lfa, &A_op, &fscalar, d]
                        (int64_t m_, int64_t s_, const double *B, double *Y) {
                    block_lfa.call(A_op, B, m_, s_, fscalar, d, Y);
                };
            } else {
                raise("randlapack:fun_nystrom_pp_mex:lfa_type",
                      "unknown lfa_type '" + lfa_type + "' (use exact|scalar|block)");
            }

            // --- drive funnystrompp ---
            RandLAPACK::FunNystromPP_v2<double> driver;
            double t1 = 0.0, t2 = 0.0;
            const double *Omega2_ptr = phase2_skipped ? nullptr : O2_buf.data();
            const double est = driver.call(A_op, fAfun, fscalar,
                                            k, s, q,
                                            O1_buf.data(), Omega2_ptr,
                                            t1, t2);

            outputs[0] = factory.createScalar<double>(est);
            if (outputs.size() >= 2) outputs[1] = factory.createScalar<double>(t1);
            if (outputs.size() >= 3) outputs[2] = factory.createScalar<double>(t2);
        }
        catch (const matlab::Exception& e) {
            // Re-raise MATLAB errors via the same mechanism as raise() above.
            throw;
        }
        catch (const std::exception& e) {
            raise("randlapack:fun_nystrom_pp_mex:cpp",
                  std::string("C++ exception: ") + e.what());
        }
    }
};
