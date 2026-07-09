// MEX wrapper for RandLAPACK::FunNystromPP using MATLAB's C++ Data API
// (R2018a+).
//
// MATLAB-side signature (low-level; users normally call randlapack.fun_nystrom_pp.m):
//
//   [est, t1, t2] = fun_nystrom_pp_mex(A, arg1, arg2, func, q, poly_lambda, lfa_type, d, ...
//                                      sketch_type, vec_nnz, sketch_seed)
//
// where:
//   A           n x n matrix, single or double, column-major. BOTH triangles are
//               used: the Phase-1 sketch application goes through sparse
//               right_spmm, so the MEX mirrors upper->lower unconditionally.
//   arg1        Phase-1 sketch SIZE: a SCALAR k (the rank). The sketch itself is
//               a SparseStack/SASO generated INSIDE RandLAPACK::NystromEVD from
//               (sketch_seed, vec_nnz) — matching the paper's Algorithm 1 line 1.
//               Passing an explicit matrix is an error (the dense-sketch mode was
//               removed from the kernel).
//   arg2        Phase-2 Hutchinson probes: pass a SCALAR s to sample n x s Gaussian probes
//               internally (seed = sketch_seed + 1000), OR an explicit n x s matrix.
//               Phase 2 is skipped when k == n (arg2 then unread).
//   func        'sqrt' | 'log' | 'poly' | 'effdim' | 'square' | 'identity'
//                 poly:   f(x) = x(x + poly_lambda)
//                 effdim: f(x) = x/(x + poly_lambda)   ("effective dimension"; operator monotone)
//   q           subspace-iter count (>= 1). q = 1 is single-pass Nystrom.
//   poly_lambda lambda used by func 'poly' and 'effdim'
//   lfa_type    'exact' | 'scalar' | 'block' | 'block_qfa'
//                 exact:  build f(A) once via syevd, every f(A)*X is a GEMM (validation oracle)
//                 scalar: per-column scalar Lanczos-FA at depth d (= Lanczos quadrature per probe)
//                 block:  block Lanczos-FA at depth d (BLAS-3 accelerated variant)
//                 block_qfa: block Lanczos-QFA — forms Ω₂ᵀf(A)Ω₂ (s×s) directly,
//                            skipping the f(A)·Ω₂ mapback (driver.use_qfa = true)
//   d           Lanczos depth (only used for lfa_type in {scalar, block, block_qfa})
//   sketch_type IGNORED (accepted for call-site compatibility; the Phase-1
//               sketch is always the kernel-internal SASO). A MATLAB warning is
//               issued if anything other than 'saso' is passed explicitly.
//   vec_nnz     nonzeros per column of the SASO sketch (default 8)  [optional, input 10]
//   sketch_seed RNG seed for the Phase-1 sketch (default 42)        [optional, input 11]
//
// Outputs:
//   est         trace estimate t1 + t2 (scalar double)
//   t1          Phase 1 contribution (scalar double)
//   t2          Phase 2 contribution (scalar double; 0 when k == n)
//   times       (optional 4th output) struct of wall-clock instrumentation:
//                 marshal_in_ms  copying A/Omega1/Omega2 out of MATLAB arrays
//                 phase1_ms      NystromEVD total (driver field)
//                 phase2_ms      Phase 2 total (driver field)
//                 fafun_ms       time inside the f(A)*X oracle (subset of phase2)
//                 assembly_ms    phase2_ms - fafun_ms (trace assembly)
//                 specrec_ms     spectral-recovery block inside NystromEVD
//                 nystrom_us     1x11 NystromEVD breakdown (microseconds):
//                                [alloc syrf matvec gram potrf trsm svd post_svd
//                                 err_est rest total]
//                 lfa_us         1x5 scalar-LanczosFA breakdown (microseconds):
//                                [matvec run_lanczos apply_f rest total]
//                                (zeros for lfa_type 'exact'/'block')
//
// A may be single or double; the computation runs in that precision and the
// scalar outputs are returned as double. Omega1/Omega2 must match the class
// of A (the randlapack.fun_nystrom_pp.m wrapper enforces this).

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <RandLAPACK.hh>
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace linops  = RandLAPACK::linops;
namespace testing = RandLAPACK::testing;

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

    // Raise a MATLAB error with a given identifier and message. Delegates to
    // MATLAB's `error` builtin via feval, which raises a MATLABException that
    // propagates back through the MEX boundary.
    [[noreturn]] void raise(const std::string& id, const std::string& msg) {
        matlabPtr->feval(u"error", 0, std::vector<Array>{
            factory.createCharArray(id),
            factory.createCharArray(msg)
        });
        std::terminate();  // feval(error) does not return; this is a backstop.
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
            raise("randlapack:fun_nystrom_pp_mex:scalar", field + " must be a scalar");
        }
        TypedArray<double> v = a;
        return static_cast<int64_t>(v[0]);
    }

    double read_double(const Array& a, const std::string& field) {
        if (a.getNumberOfElements() != 1) {
            raise("randlapack:fun_nystrom_pp_mex:scalar", field + " must be a scalar");
        }
        TypedArray<double> v = a;
        return v[0];
    }

    // Copy a TypedArray<T> column-major into a fresh heap buffer (raw new[];
    // caller frees with delete[], RandLAPACK style). MATLAB arrays are
    // column-major and contiguous, so we bulk-copy the whole buffer in one
    // shot; &*begin() is the contiguous storage of a full array, so a single
    // memcpy replaces a per-element loop (a measured marshal bottleneck at
    // large n). `new T[]` without () default-initializes — no wasted zeroing
    // pass before the memcpy overwrites every entry (std::vector::resize
    // value-initializes, i.e. touches the n^2 buffer twice).
    template <typename T>
    T* copy_into(const Array& in, size_t n_elems) {
        T* out = new T[n_elems];
        TypedArray<T> typed = in;
        std::memcpy(out, &*typed.begin(), n_elems * sizeof(T));
        return out;
    }

    // Templated worker: reads the T-typed matrices, builds the f(A)*X oracle,
    // and drives FunNystromPP<T>. T is double or float (dispatched in operator()).
    template <typename T>
    void run_fun_nystrom(ArgumentList& outputs, ArgumentList& inputs) {
        // --- input prep (timed). Seed-driven API: inputs 2 and 3 are EITHER a
        // scalar (the sketch SIZE, sampled internally here from sketch_seed) OR
        // an explicit dense sketch matrix (escape hatch for validation/repro).
        // The driver always receives plain buffers, so it is untouched. ---
        const auto t_marshal_start = std::chrono::steady_clock::now();
        using RNG = r123::Philox4x32;

        // A: n x n
        const Array& A_in = inputs[0];
        const int64_t n = static_cast<int64_t>(A_in.getDimensions()[0]);
        T* A_buf = copy_into<T>(A_in, static_cast<size_t>(n) * n);

        // scalar params (read before sketch handling: internal sampling needs them)
        const std::string func        = read_string(inputs[3], "func");
        const int64_t     q           = read_int   (inputs[4], "q");
        const T           poly_lambda = static_cast<T>(read_double(inputs[5], "poly_lambda"));
        const std::string lfa_type    = read_string(inputs[6], "lfa_type");
        const int64_t     d           = read_int   (inputs[7], "d");
        const std::string sketch_type = (inputs.size() >= 9)  ? read_string(inputs[8], "sketch_type") : "gaussian";
        const int64_t     vec_nnz     = (inputs.size() >= 10) ? read_int(inputs[9],  "vec_nnz")     : 8;
        const int64_t     sketch_seed = (inputs.size() >= 11) ? read_int(inputs[10], "sketch_seed") : 42;
        // Optional input 12: Lanczos reorthogonalization flag (default 1 = full).
        const int64_t     reorth_flag = (inputs.size() >= 12) ? read_int(inputs[11], "reorth") : 1;

        if (inputs.size() >= 9 && sketch_type != "saso") {
            matlabPtr->feval(u"warning", 0, std::vector<Array>{
                factory.createCharArray("randlapack:fun_nystrom_pp_mex:sketch_type"),
                factory.createCharArray(
                    "sketch_type '" + sketch_type + "' ignored: the Phase-1 sketch "
                    "is always a kernel-internal SASO now.")
            });
        }

        // --- Phase-1 sketch (input 2): scalar k. The sketch itself (SASO) is
        // generated inside NystromEVD from (sketch_seed, vec_nnz); explicit
        // matrices were rejected by the validation block in operator(). ---
        const int64_t k = read_int(inputs[1], "k");

        // --- Phase-2 Hutchinson probes (input 3): scalar s -> generated INSIDE
        // the kernel (FunNystromPP draws a Gaussian n x s block from `state` and
        // normalizes each column to ‖·‖₂ = √n; O2_buf stays nullptr). A matrix
        // arg overrides with an explicit Omega2. Skipped if k == n. ---
        const Array& O2_in = inputs[2];
        const bool phase2_skipped = (k == n);
        int64_t s = 0;
        T* O2_buf = nullptr;
        if (!phase2_skipped) {
            if (O2_in.getNumberOfElements() == 1) {
                s = read_int(O2_in, "s");
                // O2_buf = nullptr -> the driver generates the normalized
                // Gaussian probes internally (point 3 of the 2026-07-09 plan).
            } else {
                s = static_cast<int64_t>(O2_in.getDimensions()[1]);
                O2_buf = copy_into<T>(O2_in, static_cast<size_t>(n) * s);
            }
        }

        // Mirror upper triangle into lower unconditionally: the kernel's sparse
        // first A-application goes through right_spmm, which reads A as generic
        // dense (symmetry not exploited).
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = j + 1; i < n; ++i)
                A_buf[i + j * n] = A_buf[j + i * n];

        const double marshal_in_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_marshal_start).count();

        // --- scalar function f ---
        std::function<T(T)> fscalar;
        if      (func == "sqrt")     fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
        else if (func == "log")      fscalar = [](T x) { return std::log(x + (T)1); }; // log(x+1) = tr(log(A+I))
        else if (func == "poly")     fscalar = [poly_lambda](T x) { return x * (x + poly_lambda); };
        else if (func == "effdim")   fscalar = [poly_lambda](T x) { return x / (x + poly_lambda); };
        else if (func == "square")   fscalar = [](T x) { return x * x; };
        else if (func == "identity") fscalar = [](T x) { return x; };
        else {
            raise("randlapack:fun_nystrom_pp_mex:func",
                  "unknown func '" + func + "' (use sqrt|log|poly|effdim|square|identity)");
        }

        // --- f(A)*X oracle ---
        linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A_buf, n, Layout::ColMajor);

        using FAFun = std::function<void(int64_t, int64_t, const T*, T*)>;
        FAFun fAfun;
        RandLAPACK::LanczosFA<T>       scalar_lfa;
        RandLAPACK::BlockLanczosFA<T>  block_lfa;
        RandLAPACK::BlockLanczosQFA<T> block_qfa;
        // Lanczos-QFA fills the s×s quadratic form Ω₂ᵀf(A)Ω₂ directly (no f(A)·Ω₂
        // mapback); the driver takes its trace. Signalled to the driver below.
        const bool qfa_mode = (lfa_type == "block_qfa");

        if (lfa_type == "exact") {
            // V*diag(f(lambda))*V^T*B via a one-shot syevd. Shared implementation
            // with the RandLAPACK test + benchmark (single point of correctness).
            fAfun = testing::make_exact_fa_oracle<T>(n, A_buf, fscalar);
        } else if (lfa_type == "scalar") {
            fAfun = [&scalar_lfa, &A_op, &fscalar, d]
                    (int64_t m_, int64_t s_, const T *B, T *Y) {
                scalar_lfa.call(A_op, B, m_, s_, fscalar, d, Y);
            };
        } else if (lfa_type == "block") {
            fAfun = [&block_lfa, &A_op, &fscalar, d]
                    (int64_t m_, int64_t s_, const T *B, T *Y) {
                block_lfa.call(A_op, B, m_, s_, fscalar, d, Y);
            };
        } else if (lfa_type == "block_qfa") {
            // Y is the s×s matrix M = Ω₂ᵀ f(A) Ω₂ (driver sizes fAOmega to s²).
            fAfun = [&block_qfa, &A_op, &fscalar, d]
                    (int64_t m_, int64_t s_, const T *B, T *Y) {
                block_qfa.call(A_op, B, m_, s_, fscalar, d, Y);
            };
        } else {
            raise("randlapack:fun_nystrom_pp_mex:lfa_type",
                  "unknown lfa_type '" + lfa_type + "' (use exact|scalar|block|block_qfa)");
        }

        // --- drive funNystrom++ ---
        RandLAPACK::FunNystromPP<T> driver;
        // Sub-timers always on: a handful of steady_clock reads per call,
        // negligible next to any BLAS work; consumed by the 4th output.
        driver.nystrom_ws.times_enabled = true;
        scalar_lfa.timing = true;
        block_lfa.timing  = true;
        block_qfa.timing  = true;
        scalar_lfa.reorth = reorth_flag;
        block_lfa.reorth  = reorth_flag;
        block_qfa.reorth  = reorth_flag;
        driver.vec_nnz    = vec_nnz;
        driver.use_qfa    = qfa_mode;
        T t1 = (T)0, t2 = (T)0;
        const T *Omega2_ptr = phase2_skipped ? nullptr : O2_buf;
        RandBLAS::RNGState<RNG> state(static_cast<uint32_t>(sketch_seed));
        T est = driver.call(A_op, fAfun, fscalar,
                            k, s, q,
                            state, Omega2_ptr,
                            t1, t2);

        outputs[0] = factory.createScalar<double>(static_cast<double>(est));
        if (outputs.size() >= 2) outputs[1] = factory.createScalar<double>(static_cast<double>(t1));
        if (outputs.size() >= 3) outputs[2] = factory.createScalar<double>(static_cast<double>(t2));

        // --- optional 4th output: wall-clock instrumentation struct ---
        if (outputs.size() >= 4) {
            // NystromEVD 11-slot breakdown (microseconds, as doubles).
            std::vector<double> nys_us(driver.nystrom_ws.times.begin(),
                                       driver.nystrom_ws.times.end());
            if (nys_us.size() != 11) nys_us.assign(11, 0.0);
            // Scalar-LFA 5-slot breakdown; zeros for exact/block oracles.
            std::vector<double> lfa_us(5, 0.0);
            if (lfa_type == "scalar" && scalar_lfa.times.size() == 5) {
                lfa_us.assign(scalar_lfa.times.begin(), scalar_lfa.times.end());
            }
            matlab::data::StructArray ts = factory.createStructArray({1, 1},
                {"marshal_in_ms", "phase1_ms", "phase2_ms", "fafun_ms",
                 "assembly_ms", "specrec_ms", "nystrom_us", "lfa_us"});
            ts[0]["marshal_in_ms"] = factory.createScalar<double>(marshal_in_ms);
            ts[0]["phase1_ms"]     = factory.createScalar<double>(driver.t_phase1_ms);
            ts[0]["phase2_ms"]     = factory.createScalar<double>(driver.t_phase2_ms);
            ts[0]["fafun_ms"]      = factory.createScalar<double>(driver.t_fafun_ms);
            ts[0]["assembly_ms"]   = factory.createScalar<double>(driver.t_phase2_ms - driver.t_fafun_ms);
            ts[0]["specrec_ms"]    = factory.createScalar<double>(driver.t_specrec_ms);
            ts[0]["nystrom_us"]    = factory.createArray<double>({1, nys_us.size()},
                                         nys_us.data(), nys_us.data() + nys_us.size());
            ts[0]["lfa_us"]        = factory.createArray<double>({1, lfa_us.size()},
                                         lfa_us.data(), lfa_us.data() + lfa_us.size());
            outputs[3] = std::move(ts);
        }

        delete[] A_buf;
        delete[] O2_buf;
    }

public:
    MexFunction() : matlabPtr(getEngine()) {}

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        try {
            if (inputs.size() < 8 || inputs.size() > 12) {
                raise("randlapack:fun_nystrom_pp_mex:nargin",
                      "Expected 8 to 12 inputs: A, Omega1, Omega2, func, q, poly_lambda, "
                      "lfa_type, d [, sketch_type, vec_nnz, sketch_seed, reorth]");
            }
            if (outputs.size() < 1 || outputs.size() > 4) {
                raise("randlapack:fun_nystrom_pp_mex:nargout",
                      "Expected 1 to 4 outputs: [est, t1, t2, times]");
            }

            // --- A: square, real, single/double ---
            const Array& A_in = inputs[0];
            auto A_dims = A_in.getDimensions();
            if (A_dims.size() != 2 || A_dims[0] != A_dims[1]) {
                raise("randlapack:fun_nystrom_pp_mex:A_shape",
                      "A must be a square n x n matrix");
            }
            const ArrayType A_type = A_in.getType();
            if (A_type != ArrayType::DOUBLE && A_type != ArrayType::SINGLE) {
                raise("randlapack:fun_nystrom_pp_mex:A_dtype",
                      "A must be single or double precision");
            }
            const int64_t n = static_cast<int64_t>(A_dims[0]);

            // --- arg 2 (Phase-1): scalar k only. The Phase-1 sketch is a SASO
            // generated inside RandLAPACK::NystromEVD from (sketch_seed, vec_nnz);
            // the explicit-matrix escape hatch was removed with the kernel's
            // dense-sketch mode. ---
            const Array& O1_in = inputs[1];
            if (O1_in.getNumberOfElements() != 1) {
                raise("randlapack:fun_nystrom_pp_mex:Omega1_explicit",
                      "arg 2 must be a scalar k. Explicit Omega1 matrices are no "
                      "longer supported: the Phase-1 sketch is a SparseStack/SASO "
                      "generated inside RandLAPACK (control it via sketch_seed and "
                      "vec_nnz).");
            }
            const int64_t k = read_int(O1_in, "k");
            if (k < 1 || k > n) {
                raise("randlapack:fun_nystrom_pp_mex:k_range",
                      "k (arg 2 scalar) must satisfy 1 <= k <= n");
            }

            // --- arg 3 (Phase-2): scalar s OR explicit n x s matrix; only when k < n ---
            const Array& O2_in = inputs[2];
            if (k != n && O2_in.getNumberOfElements() != 1) {
                auto O2_dims = O2_in.getDimensions();
                if (O2_dims.size() != 2 || static_cast<int64_t>(O2_dims[0]) != n) {
                    raise("randlapack:fun_nystrom_pp_mex:Omega2_shape",
                          "Omega2 (arg 3 matrix) must be n x s when k < n");
                }
                if (O2_in.getType() != A_type) {
                    raise("randlapack:fun_nystrom_pp_mex:Omega2_dtype",
                          "Omega2 must be the same class (single/double) as A");
                }
            }

            // --- dispatch on precision ---
            if (A_type == ArrayType::DOUBLE) {
                run_fun_nystrom<double>(outputs, inputs);
            } else {
                run_fun_nystrom<float>(outputs, inputs);
            }
        }
        catch (const RandLAPACK::Error& e) {
            raise("randlapack:fun_nystrom_pp_mex:RandLAPACKError", e.what());
        }
        catch (const RandBLAS::Error& e) {
            raise("randlapack:fun_nystrom_pp_mex:RandBLASError", e.what());
        }
        catch (const matlab::Exception&) {
            // MATLAB exceptions (typically from feval(error)) propagate unchanged
            // so MATLAB sees the original error ID and message.
            throw;
        }
        catch (const std::exception& e) {
            raise("randlapack:fun_nystrom_pp_mex:StdError", e.what());
        }
        catch (...) {
            raise("randlapack:fun_nystrom_pp_mex:Unknown",
                  "Unknown exception in fun_nystrom_pp_mex");
        }
    }
};
