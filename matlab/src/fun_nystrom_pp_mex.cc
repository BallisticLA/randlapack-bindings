// MEX wrapper for RandLAPACK::FunNystromPP using MATLAB's C++ Data API
// (R2018a+).
//
// MATLAB-side signature (low-level; users normally call randlapack.fun_nystrom_pp.m):
//
//   [est, t1, t2] = fun_nystrom_pp_mex(A, arg1, arg2, func, q, poly_lambda, lfa_type, d, ...
//                                      sketch_type, vec_nnz, sketch_seed, reorth, ...
//                                      adaptive, adaptive_tol, adaptive_delay, ...
//                                      adaptive_min, budget, auto_eps, radau_return, ...
//                                      auto_depth_cap, auto_probe_frac, adaptive_matvec_cap)
//
// where:
//   A           n x n matrix, single or double, column-major. Upper triangle
//               defines the matrix; legacy builds mirror it for general sketch
//               products, while symmetric-sketch builds read it directly.
//   arg1        Phase-1 sketch SIZE: a SCALAR k (the rank). The sketch itself is
//               a SASO sketch generated INSIDE RandLAPACK::NystromEVD from
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
//   lfa_type    'exact' | 'scalar' | 'scalar_qfa' | 'block' | 'block_qfa' | 'auto' | 'adaptive'
//                 exact:  build f(A) once via syevd, every f(A)*X is a GEMM (validation oracle)
//                 scalar: per-column scalar Lanczos-FA at depth d (= Lanczos quadrature per probe)
//                 scalar_qfa: per-column scalar Lanczos-QFA — the per-probe
//                            quadratic forms ωⱼᵀf(A)ωⱼ directly (diagonal of
//                            Ω₂ᵀf(A)Ω₂; basis-free, O(n·s) memory). With
//                            adaptive = 1 each probe stops at its own depth via
//                            the Gauss-Radau certificate and adaptive_tol is a
//                            CERTIFIED per-probe relative error; reorth /
//                            adaptive_delay / adaptive_min are IGNORED (the
//                            certificate replaces the windowed rule; the
//                            recurrence is intrinsically no-reorth).
//                 block:  block Lanczos-FA at depth d (BLAS-3 accelerated variant)
//                 block_qfa: block Lanczos-QFA — forms Ω₂ᵀf(A)Ω₂ (s×s) directly,
//                            skipping the f(A)·Ω₂ mapback (driver.use_qfa = true).
//                            Input 13 (adaptive) selects the depth rule:
//                              0 = fixed depth d;
//                              1 = Radau-certified stop (stop_rule = Radau: the
//                                  block Gauss / Gauss-Radau bracket closes
//                                  within adaptive_tol — the same certified
//                                  meaning as scalar_qfa's adaptive mode; no
//                                  delay window);
//                              2 = legacy window rule (stop_rule = Window;
//                                  honors adaptive_delay / adaptive_min).
//                            Input 19 (radau_return) selects the value returned
//                            on a certified stop: 0 = block Gauss (default),
//                            1 = (Gauss + Radau)/2 midpoint.
//                 auto:   knob-free tier. The driver picks k, s, and the oracle
//                         depth from (budget, auto_eps) [inputs 17, 18], running
//                         the certified scalar QFA for both the depth probe and
//                         Phase 2; the positional k/s/d/reorth/adaptive inputs
//                         are IGNORED. Inputs 20/21 (auto_depth_cap,
//                         auto_probe_frac) tune the depth probe. An infeasible
//                         budget raises MATLAB error id
//                         'randlapack:fun_nystrom_pp:infeasibleBudget'.
//                 adaptive: fully knob-free, EPS-TARGETED tier (no upfront
//                         matvec budget). The driver picks k, s, and the
//                         oracle depth t itself from a certified BLOCK
//                         Gauss-Radau depth probe run at rtol = auto_eps
//                         [input 18, reused as eps]; the positional
//                         k/s/d/reorth/adaptive inputs are IGNORED. Input 20
//                         (auto_depth_cap) caps the probe depth, shared with
//                         the 'auto' tier. Input 22 (adaptive_matvec_cap, 0 =
//                         no cap) optionally bounds the total matvec spend;
//                         when it clamps the rank/probe split infeasibly
//                         small, or when the eps/n regime cannot fund the
//                         block-Krylov-coupled probe count (s*t <= n), the
//                         driver throws and the MEX raises the SAME error id
//                         as the 'auto' tier's infeasible budget:
//                         'randlapack:fun_nystrom_pp:infeasibleBudget'.
//   d           Lanczos depth (used for lfa_type in {scalar, scalar_qfa, block,
//               block_qfa}; the adaptive QFA modes treat it as a depth CAP;
//               ignored, like the positional k/s, for 'auto'/'adaptive')
//   sketch_type IGNORED (accepted for call-site compatibility; the Phase-1
//               sketch is always the kernel-internal SASO). A MATLAB warning is
//               issued if anything other than 'saso' is passed explicitly.
//   vec_nnz     nonzeros per ROW of the SASO sketch (default 8; 0 = auto,
//               resolved to ~log(k) inside NystromEVD)   [optional, input 10]
//   sketch_seed RNG seed for the Phase-1 sketch (default 42)        [optional, input 11]
//   ...         optional inputs 12-18 (reorth, adaptive, adaptive_tol,
//               adaptive_delay, adaptive_min, budget, auto_eps) are documented
//               inline where they are read in run_fun_nystrom below.
//   radau_return    block_qfa certified return value: 0 = Gauss (default),
//                   1 = midpoint                        [optional, input 19]
//   auto_depth_cap  'auto' tier: fixed cap on the probe depth (0 = no fixed
//                   cap, the default)                   [optional, input 20]
//   auto_probe_frac 'auto' tier: fraction of the matvec budget the depth
//                   probe may spend, in (0, 1) (default 0.125) [optional, input 21]
//   adaptive_matvec_cap  'adaptive' tier only: optional total matvec cap (0 =
//                   no cap, the default). Ignored by every other lfa_type.
//                                                       [optional, input 22]
//   ...         optional inputs 23-27 (adaptive_k_const, adaptive_s_const,
//               cap_rank_fraction, quadrature_fraction, first_row_ql) are
//               documented inline where they are read.
//   spend_cap   'scalar_auto'/'block_auto' only: 1 = spend the whole Budget
//               (driver.adaptive_spend_cap: rank and probe count chosen to use
//               the cap, Phase-2 probes at the probe's depth with no early
//               stop); 0 = shipped behaviour (default)  [optional, input 28]
//
// Outputs:
//   est         trace estimate t1 + t2 (scalar double)
//   t1          Phase 1 contribution (scalar double)
//   t2          Phase 2 contribution (scalar double; 0 when k == n)
//   times       (optional 4th output) struct of wall-clock instrumentation:
//                 marshal_in_ms  reading A/Omega2 out of MATLAB arrays: validation only by default (the
//                                driver reads MATLAB's buffers in place), plus the private copy when
//                                RANDLAPACK_FNPP_COPY_INPUT=1
//                 input_copied   true when that private copy was made (see copy_input_requested)
//                 phase1_ms      NystromEVD total (driver field)
//                 phase2_ms      Phase 2 total (driver field)
//                 fafun_ms       time inside the f(A)*X oracle (subset of phase2)
//                 assembly_ms    phase2_ms - fafun_ms (trace assembly)
//                 specrec_ms     spectral-recovery block inside NystromEVD
//                 nystrom_us     1x11 NystromEVD breakdown (microseconds).
//                                Only slots 0, 1, 2, 6, 10 (C++ 0-based; MATLAB
//                                indices 1, 2, 3, 7, 11) are populated:
//                                slot 0 = alloc, 1 = syrf (QR stabilization),
//                                2 = matvec, 6 = the WHOLE shifted spectral-
//                                recovery block (Alg. 2 lines 3-8, not just an
//                                svd), 10 = total. The remaining slots are 0.
//                 lfa_us         1x6 Lanczos-oracle breakdown (microseconds):
//                                [matvec run_lanczos apply rest total reorth]
//                                for lfa_type 'scalar'/'scalar_qfa'/'block'/
//                                'block_qfa'/'auto' (for the QFA types, apply =
//                                certificate checks + final quadrature evals;
//                                zeros for 'exact'). Slot 6 (reorth) is the
//                                recurrence's reorthogonalization time; it is
//                                real for 'scalar'/'block'/'block_qfa' (the
//                                block QFA reuses the FA recurrence and pays
//                                reorth whenever reorth = 1) and 0 for
//                                'scalar_qfa'/'auto' (basis-free recurrence,
//                                no reorth by design).
//                 d_used         Actual depth for fixed/QFA modes; a selected
//                                depth limit for automatic modes. For
//                                'block_qfa' + adaptive this is the online-chosen
//                                depth (<= d cap); for 'scalar_qfa' + adaptive it
//                                is the MAX per-probe certified depth; for 'auto'
//                                it is the probe-discovered depth cap t; for
//                                'adaptive' it is the depth probe's certified (or
//                                reached) depth t (driver.adaptive_t); for 'exact'
//                                it is NaN (no Lanczos recurrence runs; the fixed
//                                d input is not a meaningful cost figure there);
//                                otherwise (scalar/block/block_qfa fixed-depth) it
//                                equals the fixed d.
//                 oracle_mv      total A-matvecs the Phase-2 oracle actually
//                                spent: Σ per-probe depths for 'scalar_qfa' and
//                                'auto'; s*d_used for 'block_qfa'; the block
//                                oracle's matvecs member for 'adaptive'
//                                (driver.adaptive_oracle_matvecs); 0 for
//                                'exact'/'scalar'/'block' (their count is the
//                                analytic s*d_used). The benchmark should prefer
//                                this over d_used-based re-costing.
//                 auto_k         'auto'/'adaptive' only: chosen Nystrom rank
//                                (driver.auto_k / driver.adaptive_k; else 0)
//                 auto_s         'auto'/'adaptive' only: chosen probe count
//                                (driver.auto_s / driver.adaptive_s; else 0)
//                 probe_mv       'auto'/'adaptive' only: matvecs the depth probe
//                                actually spent (else 0). With the certified
//                                oracle the budget closes as an upper bound:
//                                probe_mv + q*auto_k + oracle_mv <= budget
//                                (Phase 1 costs q*auto_k matvecs, q=1 here; the
//                                'adaptive' tier has no upfront budget, so this
//                                is an accounting identity, not a constraint).
//
//               New fields below use the NaN convention: NaN when the path
//               that produces them did not run (instead of a fake 0, which
//               would be indistinguishable from a real measurement).
//                 tr_U           'block_qfa'/'adaptive' only: final block Gauss
//                                trace of Ω₂ᵀf(A)Ω₂ (upper side of the Radau
//                                bracket); for 'adaptive' this is
//                                driver.adaptive_tr_U; NaN otherwise.
//                 tr_L           'block_qfa'/'adaptive' only: final block
//                                Gauss-Radau trace (lower side; equals tr_U when
//                                no certificate ran); for 'adaptive' this is
//                                driver.adaptive_tr_L, NaN under spend_cap (no
//                                bracket evaluated in Phase 2); NaN otherwise.
//                 certified      'block_qfa'/'adaptive'/'scalar_qfa' (with
//                                Adaptive=1) only: 1 if the Radau bracket
//                                closed within (adaptive_tol / auto_eps), 0 if
//                                not; for 'adaptive' this is
//                                driver.adaptive_phase2_certified (NaN under
//                                spend_cap); for
//                                'scalar_qfa' this is scalar_qfa.all_certified
//                                (every probe column certified before the
//                                depth cap); NaN otherwise (including
//                                'scalar_qfa' with Adaptive=0, where no
//                                certificate was checked).
//                 probe_ms       'auto'/'adaptive' only: wall-clock of the depth
//                                probe (runs BEFORE phase1_ms's clock starts, so
//                                phase1_ms + phase2_ms excludes it; 'adaptive'
//                                reads driver.t_adaptive_probe_ms); NaN
//                                otherwise.
//                 probe_converged   'auto'/'adaptive' only: 1 if the depth probe
//                                certified before the probe cap ('adaptive'
//                                reads driver.adaptive_probe_certified); NaN
//                                otherwise.
//                 phase2_certified  'auto'/'adaptive' only: 1 if every Phase-2
//                                oracle column/block certified at its depth cap
//                                (distinct from probe_converged, which names
//                                ONLY the probe; 'adaptive' reads
//                                driver.adaptive_phase2_certified); NaN
//                                otherwise.
//                 phase2_checked 'adaptive' only: 1 if Phase 2 evaluated its
//                                bracket, 0 under spend_cap (phase2_certified,
//                                certified and tr_L are then NaN because
//                                nothing was checked); NaN otherwise.
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
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <cstdlib>
#include <vector>

namespace linops  = RandLAPACK::linops;
namespace testing = RandLAPACK::testing;

namespace {
// Carrier for MATLAB errors raised inside the MEX. raise() throws this and
// operator() converts it to feval('error', id, msg) as the LAST action. The
// former design called feval directly from raise(): the engine's
// MATLABException then re-entered operator()'s catch chain, where the typed
// catch (const matlab::Exception&) fails to match it (this MEX statically
// links libstdc++ while MATLAB's process carries its own copy, and typed
// catches of foreign classes are unreliable across the two RTTI domains), so
// the exception fell into catch (const std::exception&) and EVERY error id
// degraded to the generic StdError. A TU-local type in an anonymous
// namespace has its typeinfo wholly inside this MEX, so its catch always
// matches, and the terminal feval exception escapes operator() with no
// handler left to clobber the id.
struct MexError {
    std::string id;
    std::string msg;
};
} // namespace

using matlab::mex::ArgumentList;
using matlab::data::Array;
using matlab::data::ArrayFactory;
using matlab::data::ArrayType;
using matlab::data::CharArray;
using matlab::data::SparseArray;
template <typename T> using TypedArray = matlab::data::TypedArray<T>;


class MexFunction : public matlab::mex::Function {
private:
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    ArrayFactory factory;

    // Raise a MATLAB error with a given identifier and message. Throws the
    // TU-local MexError; operator() catches it and emits the actual MATLAB
    // error (see the MexError comment for why the feval must happen there).
    [[noreturn]] void raise(const std::string& id, const std::string& msg) {
        throw MexError{id, msg};
    }

    // Terminal error emission: feval('error', id, msg) raises a MATLAB-side
    // error whose engine exception propagates out of the MEX unhandled,
    // which is exactly what delivers the id to the MATLAB caller. Called
    // only from operator()'s catch handlers.
    [[noreturn]] void emit_matlab_error(const std::string& id, const std::string& msg) {
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

    // Any real numeric scalar class is accepted (double, single, and the
    // signed/unsigned integer classes): a direct MEX call with int32(60) or
    // single(1e-2) is legitimate, and the Data API's TypedArray<double>
    // conversion throws InvalidArrayTypeException on anything but double.
    double read_double(const Array& a, const std::string& field) {
        if (a.getNumberOfElements() != 1) {
            raise("randlapack:fun_nystrom_pp_mex:scalar", field + " must be a scalar");
        }
        switch (a.getType()) {
            case ArrayType::DOUBLE: { TypedArray<double>   v = a; return static_cast<double>(v[0]); }
            case ArrayType::SINGLE: { TypedArray<float>    v = a; return static_cast<double>(v[0]); }
            case ArrayType::INT8:   { TypedArray<int8_t>   v = a; return static_cast<double>(v[0]); }
            case ArrayType::INT16:  { TypedArray<int16_t>  v = a; return static_cast<double>(v[0]); }
            case ArrayType::INT32:  { TypedArray<int32_t>  v = a; return static_cast<double>(v[0]); }
            case ArrayType::INT64:  { TypedArray<int64_t>  v = a; return static_cast<double>(v[0]); }
            case ArrayType::UINT8:  { TypedArray<uint8_t>  v = a; return static_cast<double>(v[0]); }
            case ArrayType::UINT16: { TypedArray<uint16_t> v = a; return static_cast<double>(v[0]); }
            case ArrayType::UINT32: { TypedArray<uint32_t> v = a; return static_cast<double>(v[0]); }
            case ArrayType::UINT64: { TypedArray<uint64_t> v = a; return static_cast<double>(v[0]); }
            default:
                raise("randlapack:fun_nystrom_pp_mex:scalar",
                      field + " must be a real numeric scalar");
        }
    }

    int64_t read_int(const Array& a, const std::string& field) {
        return static_cast<int64_t>(read_double(a, field));
    }

    // Copy a TypedArray<T> column-major into a fresh heap buffer, owned by a
    // std::unique_ptr<T[]> so that every raise() (which throws through this
    // frame) and any exception out of driver.call frees it — the raw
    // new[]/delete[] form leaked n^2 + n*s elements on every error path after
    // the marshal. MATLAB arrays are
    // column-major and contiguous, so we bulk-copy the whole buffer in one
    // shot; &*begin() is the contiguous storage of a full array, so a single
    // memcpy replaces a per-element loop (a measured marshal bottleneck at
    // large n). `new T[]` without () default-initializes — no wasted zeroing
    // pass before the memcpy overwrites every entry (std::vector::resize
    // value-initializes, i.e. touches the n^2 buffer twice).
    //
    // `const TypedArray<T>` is essential: a non-const TypedArray selects the
    // mutable begin(), which forces MATLAB's copy-on-write to UNSHARE (deep-
    // copy) the whole n^2 array before the memcpy copies it again. The const
    // overload reads the shared buffer in place, so the input A is copied once,
    // not twice — this is the dominant piece of the flat marshal cost at n=3000.
    template <typename T>
    std::unique_ptr<T[]> copy_into(const Array& in, size_t n_elems) {
        std::unique_ptr<T[]> out(new T[n_elems]);
        const TypedArray<T> typed = in;
        if (n_elems == 0) return out;
        // Opt-in performance switch, off by default: RANDLAPACK_PERF_PARCOPY=1 splits the copy over
        // threads. A fresh buffer's pages land on the memory node of the thread that first writes them, so a
        // one-thread memcpy puts the whole matrix on one node and every later product streams it from there.
        const char* pc = std::getenv("RANDLAPACK_PERF_PARCOPY");
        if (pc != nullptr && pc[0] == '1' && n_elems >= ((size_t)1 << 20)) {
            // Never more threads than cores or than 2^20-element chunks, and never let a failed
            // thread creation escape: a joinable std::thread destroyed by an exception would
            // terminate MATLAB. On failure the started threads are joined and the plain copy runs.
            const char* ot = std::getenv("OMP_NUM_THREADS");
            const size_t hw = (size_t)std::max(1u, std::thread::hardware_concurrency());
            size_t nt = ot ? (size_t)std::max(1, std::atoi(ot)) : hw;
            nt = std::min({nt, hw, n_elems >> 20});
            const T* src = &*typed.cbegin(); T* dst = out.get();
            auto chunk = [=](size_t t) {
                size_t lo = n_elems * t / nt, hi = n_elems * (t + 1) / nt;
                std::memcpy(dst + lo, src + lo, (hi - lo) * sizeof(T));
            };
            std::vector<std::thread> pool;
            try {
                pool.reserve(nt);
                for (size_t t = 1; t < nt; ++t) pool.emplace_back(chunk, t);
            } catch (const std::exception&) {
                for (auto& th : pool) th.join();
                std::memcpy(dst, src, n_elems * sizeof(T));
                return out;
            }
            chunk(0);
            for (auto& th : pool) th.join();
            return out;
        }
        std::memcpy(out.get(), &*typed.cbegin(), n_elems * sizeof(T));
        return out;
    }

    // Zero-copy input (the default): the address of MATLAB's own column-major buffer, read in place. The const
    // TypedArray selects the const iterator, so MATLAB's copy-on-write never unshares the array (see copy_into).
    // The temporary TypedArray shares the buffer with `in`, which is an element of the MEX's ArgumentList and
    // keeps the data alive for the whole call, so the pointer stays valid after `typed` goes out of scope.
    // Every consumer reads A through const T*: ExplicitSymLinOp::A_buff, DiagSymLinOp::lambda,
    // make_exact_fa_oracle (which copies before its destructive syevd) and FunNystromPP::call's Omega2; there is
    // no const_cast or pointer cast anywhere on those paths (RandLAPACK fc4423e, RandBLAS 86ed633), so a write
    // into MATLAB's array would not compile. The read-only proof is in the 2026-10-07 memopt report.
    template <typename T>
    static const T* matlab_data(const Array& in) {
        const TypedArray<T> typed = in;
        if (typed.getNumberOfElements() == 0) return nullptr;
        return &*typed.cbegin();
    }

    // Input marshalling: RANDLAPACK_FNPP_COPY_INPUT=1 restores the private copy of A, Omega2 and the sparse CSC arrays
    // (the code the paper ran; RANDLAPACK_PERF_PARCOPY=1 still threads that copy). Off by default: the MEX reads
    // MATLAB's buffers in place, so a dense n x n problem holds one copy of A instead of two. Builds without
    // RANDLAPACK_SYMMETRIC_SKETCH mirror the upper triangle into the buffer and therefore always copy.
    static bool copy_input_requested() {
#ifndef RANDLAPACK_SYMMETRIC_SKETCH
        return true;
#else
        const char* v = std::getenv("RANDLAPACK_FNPP_COPY_INPUT");
        return v != nullptr && v[0] == '1';
#endif
    }

    // RANDLAPACK_PERF_PARCOPY=1 threads the private copy; without the copy it has nothing to do. Scripts from before
    // 2026-10-07 set it to get a spread copy, so say once per MATLAB session (per MEX load) that it no longer applies:
    // placement is now the caller's job (harness utils/numa_place.m, randlapack.numa_spread / load_dense).
    void warn_parcopy_unused_once() {
        static bool warned = false;
        const char* pc = std::getenv("RANDLAPACK_PERF_PARCOPY");
        if (warned || pc == nullptr || pc[0] != '1') return;
        warned = true;
        matlabPtr->feval(u"warning", 0, std::vector<Array>{
            factory.createCharArray("randlapack:fun_nystrom_pp_mex:parcopyUnused"),
            factory.createCharArray("RANDLAPACK_PERF_PARCOPY=1 has no effect: the MEX reads MATLAB's input arrays in place "
                                    "(no copy to thread). For a dense A, spread its pages with randlapack.numa_spread "
                                    "or load_dense; RANDLAPACK_FNPP_COPY_INPUT=1 restores the threaded private copy.")});
    }

    // Sparse A reaches the MEX from the wrapper as a 1x1 struct of zero-based compressed-column arrays: sparse_n (int64
    // scalar), colptr (int64, n+1), rowidx (int64, nnz), vals (double, nnz), extracted in MATLAB with find() in O(nnz).
    // A raw MATLAB sparse array is refused: the Data API hides its buffers, and walking its nonzeros with getIndex costs
    // O(n) per nonzero (the copy grew 4x per doubling of n: 14 s at n = 40,000, 55 s at 80,000; about an hour at 500k).
    static bool is_sparse_struct(const Array& a) {
        if (a.getType() != ArrayType::STRUCT || a.getNumberOfElements() != 1) return false;
        const matlab::data::StructArray S = a;
        for (const auto& f : S.getFieldNames())
            if (std::string(f) == "sparse_n") return true;
        return false;
    }

    Array sparse_field(const Array& in, const char* name, ArrayType want) {
        const matlab::data::StructArray S = in;
        bool has = false;
        for (const auto& f : S.getFieldNames())
            if (std::string(f) == name) has = true;
        if (!has)
            raise("randlapack:fun_nystrom_pp_mex:A_sparse_field",
                  std::string("sparse A struct lacks field '") + name + "' (build it with randlapack.fun_nystrom_pp)");
        Array a = S[0][name];
        if (a.getType() != want)
            raise("randlapack:fun_nystrom_pp_mex:A_sparse_field",
                  std::string("sparse A field '") + name + "' has the wrong class (int64 for sparse_n, colptr, rowidx; double for vals)");
        return a;
    }

    int64_t sparse_n_of(const Array& in) {
        const TypedArray<int64_t> nA = sparse_field(in, "sparse_n", ArrayType::INT64);
        if (nA.getNumberOfElements() != 1 || nA[0] < 1)
            raise("randlapack:fun_nystrom_pp_mex:A_sparse_n", "sparse A: sparse_n must be one positive int64");
        return nA[0];
    }

    // Validates the struct's arrays, then either copies them (copy == true) or points at MATLAB's own buffers: colptr
    // starts at 0, never decreases and ends at nnz; every row index is in [0, n) and rows strictly increase within a
    // column (the order element access and the sparse kernels rely on); every value is finite. The checks read the
    // MATLAB arrays in place, so a refused matrix costs no copy. The zero-copy pointers are const, but the CSCMatrix
    // built from them (a non-owning view, own_memory = false) stores plain T* / int64_t* members, so const SpMat& does
    // NOT protect them: the sparse path is read-only by inspection, not by type. On this path RandBLAS's left_spmm /
    // right_spmm, its CSC kernels and its sparse-times-sparse sketch only read the arrays; the MKL backend wraps them
    // in mkl_sparse_?_create_csc handles for mkl_sparse_?_mm / _spmmd, which leave them unchanged; RandBLAS never calls
    // mkl_sparse_order or reindex_inplace (RandBLAS 86ed633). Through randlapack.fun_nystrom_pp the arrays are fresh
    // find() temporaries in any case, so a caller's sparse A could not be reached.
    void marshal_csc(const Array& in, int64_t n, bool copy, std::unique_ptr<int64_t[]>& colptr_own,
                     std::unique_ptr<int64_t[]>& rowidx_own, std::unique_ptr<double[]>& vals_own,
                     const int64_t*& colptr, const int64_t*& rowidx, const double*& vals, int64_t& nnz) {
        const TypedArray<int64_t> cp = sparse_field(in, "colptr", ArrayType::INT64);
        const TypedArray<int64_t> ri = sparse_field(in, "rowidx", ArrayType::INT64);
        const TypedArray<double>  vv = sparse_field(in, "vals",   ArrayType::DOUBLE);
        nnz = static_cast<int64_t>(ri.getNumberOfElements());
        if (static_cast<int64_t>(cp.getNumberOfElements()) != n + 1)
            raise("randlapack:fun_nystrom_pp_mex:A_sparse_colptr", "sparse A: colptr must have n+1 entries");
        if (static_cast<int64_t>(vv.getNumberOfElements()) != nnz)
            raise("randlapack:fun_nystrom_pp_mex:A_sparse_nnz", "sparse A: rowidx and vals must have the same length");
        colptr = matlab_data<int64_t>(cp);
        rowidx = matlab_data<int64_t>(ri);
        vals   = matlab_data<double>(vv);
        validate_csc(n, nnz, colptr, rowidx, vals);
        if (copy) {
            colptr_own.reset(new int64_t[n + 1]);
            rowidx_own.reset(new int64_t[std::max<int64_t>(nnz, 1)]);
            vals_own.reset(new double[std::max<int64_t>(nnz, 1)]);
            std::memcpy(colptr_own.get(), colptr, static_cast<size_t>(n + 1) * sizeof(int64_t));
            if (nnz > 0) {
                std::memcpy(rowidx_own.get(), rowidx, static_cast<size_t>(nnz) * sizeof(int64_t));
                std::memcpy(vals_own.get(), vals, static_cast<size_t>(nnz) * sizeof(double));
            }
            colptr = colptr_own.get(); rowidx = rowidx_own.get(); vals = vals_own.get();
        } else if (nnz == 0) {
            // An empty MATLAB array has no buffer; the kernels never dereference these when nnz == 0, but keep them
            // non-null, as the copy path's one-element allocations are.
            rowidx_own.reset(new int64_t[1]); vals_own.reset(new double[1]);
            rowidx = rowidx_own.get(); vals = vals_own.get();
        }
    }

    void validate_csc(int64_t n, int64_t nnz, const int64_t* colptr, const int64_t* rowidx, const double* vals) {
        if (colptr[0] != 0 || colptr[n] != nnz)
            raise("randlapack:fun_nystrom_pp_mex:A_sparse_colptr", "sparse A: colptr must start at 0 and end at nnz");
        for (int64_t j = 0; j < n; ++j)   // all of colptr first: a decrease would merge columns in the row scan below
            if (colptr[j + 1] < colptr[j])
                raise("randlapack:fun_nystrom_pp_mex:A_sparse_colptr",
                      "sparse A: colptr decreases at column " + std::to_string(j + 1));
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t p = colptr[j]; p < colptr[j + 1]; ++p) {
                if (rowidx[p] < 0 || rowidx[p] >= n)
                    raise("randlapack:fun_nystrom_pp_mex:A_sparse_row",
                          "sparse A: row index out of range in column " + std::to_string(j + 1));
                if (p > colptr[j] && rowidx[p] <= rowidx[p - 1])
                    raise("randlapack:fun_nystrom_pp_mex:A_sparse_order",
                          "sparse A has unsorted or repeated row indices in column " + std::to_string(j + 1));
                if (!std::isfinite(vals[p]))
                    raise("randlapack:fun_nystrom_pp_mex:A_sparse_value",
                          "sparse A has a non-finite value in column " + std::to_string(j + 1));
            }
        }
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

        // A: n x n, or an n-vector holding diag(A) (RAII-owned; freed on every
        // exit path, including raise() and exceptions out of driver.call).
        // Vector mode copies n values, not n^2, and skips the triangle mirror
        // below; the shape was validated in operator().
        const Array& A_in = inputs[0];
        const auto A_dims = A_in.getDimensions();
        // Sparse mode (sparse double A, both triangles stored, passed as the wrapper's CSC struct): a copy of the nonzeros,
        // no n^2 buffer; A_buf stays null.
        const bool sparse_mode = is_sparse_struct(A_in);
        const bool vector_mode = !sparse_mode && (A_dims.size() == 2) &&
            ((A_dims[0] == 1 && A_dims[1] > 1) || (A_dims[1] == 1 && A_dims[0] > 1));
        const int64_t n = sparse_mode ? sparse_n_of(A_in)
            : vector_mode ? static_cast<int64_t>(A_in.getNumberOfElements())
            : static_cast<int64_t>(A_dims[0]);
        // Zero-copy by default (A_buf points into MATLAB's array); RANDLAPACK_FNPP_COPY_INPUT=1 or a legacy build
        // copies into A_own as before.
        const bool copy_input = copy_input_requested();
        if (!copy_input) warn_parcopy_unused_once();
        std::unique_ptr<T[]> A_own;
        std::unique_ptr<int64_t[]> sp_colptr_own, sp_rowidx_own;
        std::unique_ptr<double[]> sp_vals_own;
        const int64_t* sp_colptr = nullptr;
        const int64_t* sp_rowidx = nullptr;
        const double*  sp_vals   = nullptr;
        int64_t sp_nnz = 0;
        const T* A_buf = nullptr;
        if (sparse_mode) {
            marshal_csc(A_in, n, copy_input, sp_colptr_own, sp_rowidx_own, sp_vals_own,
                        sp_colptr, sp_rowidx, sp_vals, sp_nnz);
        } else if (copy_input) {
            A_own = copy_into<T>(A_in, vector_mode ? static_cast<size_t>(n) : static_cast<size_t>(n) * n);
            A_buf = A_own.get();
        } else {
            A_buf = matlab_data<T>(A_in);
        }

        // scalar params (read before sketch handling: internal sampling needs them)
        const std::string func        = read_string(inputs[3], "func");
        const int64_t     q           = read_int   (inputs[4], "q");
        const T           poly_lambda = static_cast<T>(read_double(inputs[5], "poly_lambda"));
        const std::string requested_lfa = read_string(inputs[6], "lfa_type");
        const bool target_auto = (requested_lfa == "scalar_auto" || requested_lfa == "block_auto");
        const std::string lfa_type = target_auto ? "adaptive" : requested_lfa;
        // The Gauss / Radau-at-0 bracket needs the derivatives of f from order two up to
        // alternate in sign, not f to be operator monotone. -x log x satisfies that with the
        // same pattern as log(1+x): its nth derivative is (-1)^(n-1) (n-2)! / x^(n-1), so even
        // orders are negative and odd orders positive. poly and square do not: x^2 has a second
        // derivative of +2, the opposite sign, and the bracket does not hold for them.
        if (target_auto && func != "sqrt" && func != "log" && func != "effdim"
                        && func != "identity" && func != "entropy")
            raise("randlapack:fun_nystrom_pp:unsupportedAutoFunction",
                  "scalar_auto/block_auto support sqrt, log, effdim, identity and entropy; "
                  "poly and square have no Radau certificate");
        const int64_t     d           = read_int   (inputs[7], "d");
        const std::string sketch_type = (inputs.size() >= 9)  ? read_string(inputs[8], "sketch_type") : "saso";
        const int64_t     vec_nnz_in  = (inputs.size() >= 10) ? read_int(inputs[9],  "vec_nnz")     : 8;
        // sketch_type 'gaussian' selects NystromEVD's dense Gaussian sketch (vec_nnz = -1).
        const int64_t     vec_nnz     = (sketch_type == "gaussian") ? -1 : vec_nnz_in;
        const int64_t     sketch_seed = (inputs.size() >= 11) ? read_int(inputs[10], "sketch_seed") : 42;
        // Optional input 12: Lanczos reorthogonalization flag (default 1 = full).
        const int64_t     reorth_flag = (inputs.size() >= 12) ? read_int(inputs[11], "reorth") : 1;
        // Adaptive Lanczos-QFA depth (block_qfa only): choose the Lanczos depth
        // online from the qfa certificate instead of the fixed d (which becomes
        // the cap). adaptive_tol is the relative-change tolerance on tr(M_k).
        const int64_t     adaptive_fl = (inputs.size() >= 13) ? read_int(inputs[12], "adaptive") : 0;
        const T           adaptive_tl = (inputs.size() >= 14)
                                        ? static_cast<T>(read_double(inputs[13], "adaptive_tol")) : (T)1e-2;
        // Certificate window (block_qfa adaptive only). The first convergence
        // test is at depth adaptive_min + adaptive_delay, so these set the floor
        // on d_used. 0 = keep the library default (see BlockLanczosQFA).
        const int64_t     adaptive_dl = (inputs.size() >= 15) ? read_int(inputs[14], "adaptive_delay") : 0;
        const int64_t     adaptive_mn = (inputs.size() >= 16) ? read_int(inputs[15], "adaptive_min")   : 0;
        // Knob-free tier (lfa_type == "auto"): total A-matvec budget and target
        // accuracy; the driver picks k, s, and the oracle depth itself. The
        // positional k/s/d/reorth/adaptive inputs are ignored in this mode.
        const int64_t     auto_budget = (inputs.size() >= 17) ? read_int(inputs[16], "budget") : 0;
        const double      auto_eps    = (inputs.size() >= 18) ? read_double(inputs[17], "auto_eps") : 1e-3;
        // Certified-return selector (block_qfa adaptive = 1 only): 0 = block
        // Gauss, 1 = (Gauss + Radau)/2 midpoint.
        const int64_t     radau_ret   = (inputs.size() >= 19) ? read_int(inputs[18], "radau_return") : 0;
        // Auto-tier probe knobs: a fixed cap on the probe depth (0 = no fixed
        // cap) and the fraction of the budget the probe may spend, in (0, 1).
        const int64_t     auto_dcap   = (inputs.size() >= 20) ? read_int(inputs[19], "auto_depth_cap") : 0;
        const double      auto_pfrac  = (inputs.size() >= 21) ? read_double(inputs[20], "auto_probe_frac") : 0.125;
        // Eps-targeted tier (lfa_type == "adaptive") only: optional total
        // matvec cap, 0 (default) => nullopt (no cap) at the driver call site.
        const int64_t adaptive_mvcap = target_auto ? auto_budget
            : ((inputs.size() >= 22) ? read_int(inputs[21], "adaptive_matvec_cap") : 0);
        if (target_auto && auto_budget < 1)
            raise("randlapack:fun_nystrom_pp:Budget", "scalar_auto/block_auto require a positive Budget");
        // Eps-targeted tier only: the leading constants of the paper's split
        // rule k = c_k sqrt(t)/eps, s = max(1, c_s/(sqrt(t) eps)). Both default
        // to 1, the driver's own default, so an omitted argument reproduces the
        // paper's unit-constant allocation exactly.
        const double      adaptive_kc = (inputs.size() >= 23) ? read_double(inputs[22], "adaptive_k_const") : 1.0;
        const double      adaptive_sc = (inputs.size() >= 24) ? read_double(inputs[23], "adaptive_s_const") : 1.0;
        const double cap_rank_fraction = (inputs.size() >= 25) ? read_double(inputs[24], "cap_rank_fraction") : 1.0;
        const double quadrature_fraction = (inputs.size() >= 26) ? read_double(inputs[25], "quadrature_fraction") : 1.0;
        const double first_row_ql = (inputs.size() >= 27) ? read_double(inputs[26], "first_row_ql") : 0.0;
        if (first_row_ql != 0.0 && first_row_ql != 1.0)
            raise("randlapack:fun_nystrom_pp_mex:first_row_ql", "first_row_ql must be 0 or 1");
        const double spend_cap = (inputs.size() >= 28) ? read_double(inputs[27], "spend_cap") : 0.0;
        if (spend_cap != 0.0 && spend_cap != 1.0)
            raise("randlapack:fun_nystrom_pp_mex:spend_cap", "spend_cap must be 0 or 1");
        if (spend_cap != 0.0 && !target_auto)
            raise("randlapack:fun_nystrom_pp_mex:spend_cap",
                  "spend_cap applies only to lfa_type 'scalar_auto' or 'block_auto'");

        if (!std::isfinite(adaptive_kc) || !std::isfinite(adaptive_sc) ||
            !(adaptive_kc > 0.0) || !(adaptive_sc > 0.0)) {
            raise("randlapack:fun_nystrom_pp_mex:adaptive_split_const",
                  "adaptive_k_const (input 23) and adaptive_s_const (input 24) must be > 0");
        }
        if (adaptive_fl < 0 || adaptive_fl > 2) {
            raise("randlapack:fun_nystrom_pp_mex:adaptive",
                  "adaptive (input 13) must be 0 (fixed depth), 1 (Radau-certified), "
                  "or 2 (legacy window; block_qfa only)");
        }
        if (radau_ret != 0 && radau_ret != 1) {
            raise("randlapack:fun_nystrom_pp_mex:radau_return",
                  "radau_return (input 19) must be 0 (Gauss) or 1 (midpoint)");
        }
        if (auto_dcap < 0) {
            raise("randlapack:fun_nystrom_pp_mex:auto_depth_cap",
                  "auto_depth_cap (input 20) must be >= 0 (0 = no fixed cap)");
        }
        if (!(auto_pfrac > 0.0 && auto_pfrac < 1.0)) {
            raise("randlapack:fun_nystrom_pp_mex:auto_probe_frac",
                  "auto_probe_frac (input 21) must lie in (0, 1)");
        }
        if (adaptive_mvcap < 0) {
            raise("randlapack:fun_nystrom_pp_mex:adaptive_matvec_cap",
                  "adaptive_matvec_cap (input 22) must be >= 0 (0 = no cap)");
        }
        if (vec_nnz_in < 0) {
            raise("randlapack:fun_nystrom_pp_mex:vec_nnz",
                  "vec_nnz (input 10) must be >= 0 (0 = auto, ~log(k))");
        }

        if (inputs.size() >= 9 && sketch_type != "saso" && sketch_type != "gaussian") {
            matlabPtr->feval(u"warning", 0, std::vector<Array>{
                factory.createCharArray("randlapack:fun_nystrom_pp_mex:sketch_type"),
                factory.createCharArray(
                    "sketch_type '" + sketch_type + "' ignored: the Phase-1 sketch "
                    "is always a kernel-internal SASO now.")
            });
        }

        // Ignored-knob warnings for 'auto'/'adaptive' (q hardcoded to 1
        // internally; Adaptive*/Depth/Reorth/RadauReturn all unread by
        // these two tiers) used to live here as a value-comparison against
        // the .m wrapper's own defaults -- which could not distinguish "the
        // caller never touched this" from "the caller explicitly passed
        // the default value", and fired on every call from a caller that
        // always forwards a value verbatim (e.g. this repo's own MATLAB
        // harness). Moved to randlapack.fun_nystrom_pp.m, which has access
        // to inputParser's UsingDefaults and can tell the two apart; this
        // low-level MEX entry point no longer warns on ignored knobs at all
        // (matches its "low-level" framing in the header comment above).

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
        std::unique_ptr<T[]> O2_own;   // RAII: freed on every exit path (copy path only)
        const T* O2_buf = nullptr;
        // 'auto'/'adaptive' never read Omega2/O2_buf: their driver.call
        // overloads (below) don't even take an Omega2 argument, generating
        // their own internal probes instead. Skip the marshal entirely for
        // those two tiers, including the O(n*s) heap copy branch below,
        // rather than paying a wasted copy of a caller-supplied explicit
        // Omega2 (e.g. the "same probes for every estimator" comparison
        // escape hatch) on every such call.
        const bool skip_omega2_marshal =
            (lfa_type == "auto" || lfa_type == "adaptive");
        if (!phase2_skipped && !skip_omega2_marshal) {
            if (O2_in.getNumberOfElements() == 1) {
                s = read_int(O2_in, "s");
                if (s < 1) {
                    raise("randlapack:fun_nystrom_pp_mex:s_range",
                          "s (arg 3 scalar) must be >= 1 when k < n");
                }
                // O2_buf = nullptr -> the driver generates the normalized
                // Gaussian probes internally (point 3 of the 2026-07-09 plan).
            } else {
                s = static_cast<int64_t>(O2_in.getDimensions()[1]);
                if (s < 1) {
                    // An n x 0 array would otherwise flow into t2 = .../0 = NaN
                    // with no error raised.
                    raise("randlapack:fun_nystrom_pp_mex:Omega2_empty",
                          "explicit Omega2 (arg 3 matrix) must have s >= 1 "
                          "columns when k < n; got an n x 0 array");
                }
                // The driver reads Omega2 through const T* (FunNystromPP::call's Omega2 and every oracle's B).
                if (copy_input) {
                    O2_own = copy_into<T>(O2_in, static_cast<size_t>(n) * s);
                    O2_buf = O2_own.get();
                } else {
                    O2_buf = matlab_data<T>(O2_in);
                }
            }
        }

#ifndef RANDLAPACK_SYMMETRIC_SKETCH
        // The legacy general sketch product requires both triangles. Legacy builds always copy (copy_input_requested),
        // so this writes the MEX's own buffer, never MATLAB's.
        if (!vector_mode && !sparse_mode) {
            T* A_w = A_own.get();
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = j + 1; i < n; ++i)
                    A_w[i + j * n] = A_w[j + i * n];
        }
#endif

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
        else if (func == "entropy")  fscalar = [](T x) { return x > (T)0 ? -x * std::log(x) : (T)0; };
        else {
            raise("randlapack:fun_nystrom_pp_mex:func",
                  "unknown func '" + func + "' (use sqrt|log|poly|effdim|square|identity|entropy)");
        }

        // --- f(A)*X oracle ---
        using FAFun = std::function<void(int64_t, int64_t, const T*, T*)>;
        RandLAPACK::LanczosFA<T>       scalar_lfa;
        RandLAPACK::LanczosQFA<T>      scalar_qfa;
        RandLAPACK::BlockLanczosFA<T>  block_lfa;
        RandLAPACK::BlockLanczosQFA<T> block_qfa;
        // Lanczos-QFA fills the quadratic form Ω₂ᵀf(A)Ω₂ directly (no f(A)·Ω₂
        // mapback); the driver takes its trace, reading only the diagonal.
        // Signalled to the driver below.
        const bool qfa_mode = (lfa_type == "block_qfa" || lfa_type == "scalar_qfa");

        // --- drive funNystrom++ ---
        // The driver and the four oracle objects are operator-independent and
        // are read after the tier body (the timing outputs below), so they are
        // declared here and the tier body is a generic lambda over the operator
        // type: one implementation serves the dense and the diagonal operator.
        RandLAPACK::FunNystromPP<T> driver;
        // Sub-timers always on: a handful of steady_clock reads per call,
        // negligible next to any BLAS work; consumed by the 4th output.
        driver.nystrom_ws.times_enabled = true;
        scalar_lfa.timing = true;
        scalar_qfa.timing = true;
        block_lfa.timing  = true;
        block_qfa.timing  = true;
        driver.auto_sqfa.timing = true;
        scalar_qfa.use_first_row_ql = (first_row_ql != 0.0);
        driver.auto_sqfa.use_first_row_ql = (first_row_ql != 0.0);
        driver.adaptive_bqfa.timing = true;
        scalar_lfa.reorth = reorth_flag;
        block_lfa.reorth  = reorth_flag;
        block_qfa.reorth  = reorth_flag;
        // scalar_qfa: intrinsically no-reorth (basis-free recurrence); reorth /
        // adaptive_delay / adaptive_min do not apply: the Gauss-Radau
        // certificate has no window. adaptive_tol is its CERTIFIED tolerance.
        // Any nonzero adaptive selects it (the scalar class has no window rule).
        scalar_qfa.adaptive      = (adaptive_fl != 0);
        scalar_qfa.adaptive_rtol = adaptive_tl;
        // block_qfa adaptive semantics: 0 = fixed depth; 1 = Radau-certified
        // (consistent with scalar_qfa's meaning); 2 = legacy window rule
        // (honors adaptive_delay / adaptive_min). stop_rule and return_mode
        // are assigned unconditionally: the oracle may be cached across calls
        // (persistent-handle path), so every mode must be set by name, never
        // inherited from a previous call.
        block_qfa.adaptive      = (adaptive_fl != 0);
        block_qfa.adaptive_rtol = adaptive_tl;
        block_qfa.stop_rule     = (adaptive_fl == 2)
            ? RandLAPACK::BlockQFAStop::Window : RandLAPACK::BlockQFAStop::Radau;
        block_qfa.return_mode   = (radau_ret == 1)
            ? RandLAPACK::BlockQFAReturn::Midpoint : RandLAPACK::BlockQFAReturn::Gauss;
        // Assign UNCONDITIONALLY, restoring the library default by name when the
        // caller passed 0. The former `if (x > 0)` form depended on the oracle
        // being freshly constructed every call to supply the default; once the
        // oracle is cached (persistent-handle path) that assumption is false and
        // the previous call's window leaks forward, changing d_used and hence
        // both the estimate and the matvec count. The oracle is non-copyable and
        // non-assignable, so "reset by reconstruction" is not available.
        block_qfa.adaptive_delay = (adaptive_dl > 0)
            ? adaptive_dl : RandLAPACK::BlockLanczosQFA<T>::default_adaptive_delay;
        block_qfa.adaptive_min   = (adaptive_mn > 0)
            ? adaptive_mn : RandLAPACK::BlockLanczosQFA<T>::default_adaptive_min;
        driver.vec_nnz         = vec_nnz;
        driver.use_qfa         = qfa_mode;
        driver.auto_depth_cap  = auto_dcap;
        driver.auto_probe_frac = static_cast<T>(auto_pfrac);
        // A +-1 quadratic form is exact on a diagonal matrix, which would make
        // the eps-targeted tier's error degenerate rather than sampled, so
        // vector mode draws that tier's probes from the sphere instead.
        driver.adaptive_rademacher = !target_auto && !vector_mode;
        driver.adaptive_use_scalar = (requested_lfa == "scalar_auto");
        driver.adaptive_reuse_pilot = !target_auto;
        driver.adaptive_gauss_return = target_auto;
        driver.adaptive_k_const    = static_cast<T>(adaptive_kc);
        driver.adaptive_s_const    = static_cast<T>(adaptive_sc);
        driver.adaptive_cap_rank_fraction = static_cast<T>(cap_rank_fraction);
        driver.adaptive_quadrature_fraction = static_cast<T>(quadrature_fraction);
        driver.adaptive_spend_cap = (spend_cap != 0.0);

        T t1 = (T)0, t2 = (T)0, est = (T)0;
        const T *Omega2_ptr = phase2_skipped ? nullptr : O2_buf;
        RandBLAS::RNGState<RNG> state(static_cast<uint32_t>(sketch_seed));
        // f(lambda) for the vector-mode exact oracle; unused otherwise.
        std::unique_ptr<T[]> flam_own;

        auto drive_tiers = [&](auto& A_op) {
        FAFun fAfun;
        if (lfa_type == "exact") {
            if (vector_mode) {
                // f(A)B is a row scaling by f(lambda). make_exact_fa_oracle's
                // syevd needs the dense A and is not called here.
                flam_own.reset(new T[n]);
                for (int64_t i = 0; i < n; ++i) flam_own[i] = fscalar(A_buf[i]);
                fAfun = [flam = flam_own.get()]
                        (int64_t m_, int64_t s_, const T *B, T *Y) {
                    for (int64_t jj = 0; jj < s_; ++jj)
                        for (int64_t ii = 0; ii < m_; ++ii)
                            Y[ii + jj * m_] = flam[ii] * B[ii + jj * m_];
                };
            } else if (sparse_mode) {
                raise("randlapack:fun_nystrom_pp_mex:exact_sparse",
                      "lfa_type 'exact' needs the dense matrix (one-shot syevd); it is not available for sparse A");
            } else {
                // V*diag(f(lambda))*V^T*B via a one-shot syevd. Shared implementation
                // with the RandLAPACK test + benchmark (single point of correctness).
                fAfun = testing::make_exact_fa_oracle<T>(n, A_buf, fscalar);
            }
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
        } else if (lfa_type == "scalar_qfa") {
            // Per-column quadratic forms onto the DIAGONAL of the s×s Y (the
            // only part the driver's use_qfa trace read touches; off-diagonals
            // are left unwritten — same convention as the auto tier).
            fAfun = [&scalar_qfa, &A_op, &fscalar, d]
                    (int64_t m_, int64_t s_, const T *B, T *Y) {
                // unique_ptr: an exception out of scalar_qfa.call must not
                // leak the per-call buffer.
                std::unique_ptr<T[]> qv(new T[s_]);
                scalar_qfa.call(A_op, B, m_, s_, fscalar, d, qv.get());
                for (int64_t jj = 0; jj < s_; ++jj)
                    Y[jj + jj * s_] = qv[jj];
            };
        } else if (lfa_type == "block_qfa") {
            // Y is the s×s matrix M = Ω₂ᵀ f(A) Ω₂ (driver sizes fAOmega to s²).
            fAfun = [&block_qfa, &A_op, &fscalar, d]
                    (int64_t m_, int64_t s_, const T *B, T *Y) {
                block_qfa.call(A_op, B, m_, s_, fscalar, d, Y);
            };
        } else if (lfa_type == "auto") {
            // Knob-free tier: no oracle to build here; the driver's auto call
            // owns its QFA oracle and picks k/s/depth from (budget, auto_eps).
            // Same id as randlapack.fun_nystrom_pp.m's own pre-check (unified:
            // previously this used a different, undocumented
            // 'randlapack:fun_nystrom_pp_mex:budget' id, unreachable through
            // the documented .m API since .m always validates first; a direct
            // fun_nystrom_pp_mex(...) call now raises the same id the .m
            // wrapper would have).
            if (auto_budget < 1)
                raise("randlapack:fun_nystrom_pp:Budget",
                      "lfa_type 'auto' requires a positive matvec budget (input 17)");
        } else if (lfa_type == "adaptive") {
            // Eps-targeted knob-free tier: no oracle to build here either; the
            // driver's eps-targeted call owns its BLOCK QFA oracle and picks
            // k/s/depth from a certified depth probe at rtol = auto_eps. No
            // upfront budget is required (unlike 'auto'); adaptive_matvec_cap
            // (0 = none) is passed straight through below.
        } else {
            raise("randlapack:fun_nystrom_pp_mex:lfa_type",
                  "unknown lfa_type '" + lfa_type +
                  "' (use exact|scalar|scalar_qfa|block|block_qfa|auto|adaptive)");
        }

        if (lfa_type == "auto") {
            // The auto overload throws std::invalid_argument for an infeasible
            // matvec budget. Surface that as a dedicated MATLAB error id so
            // the benchmark can catch it and write a skip row; other
            // invalid_argument reasons (eps / probe_frac range) keep the
            // generic StdError id.
            try {
                est = driver.call(A_op, fscalar, auto_budget, static_cast<T>(auto_eps),
                                  state, t1, t2);
            } catch (const std::invalid_argument& e) {
                const std::string msg = e.what();
                if (msg.find("infeasible") != std::string::npos) {
                    raise("randlapack:fun_nystrom_pp:infeasibleBudget", msg);
                }
                raise("randlapack:fun_nystrom_pp_mex:StdError", msg);
            }
        } else if (lfa_type == "adaptive") {
            // Eps-targeted overload: auto_eps doubles as eps, auto_dcap (shared
            // with the 'auto' tier's driver.auto_depth_cap member, already set
            // above) bounds the probe depth. adaptive_mvcap == 0 => nullopt (no
            // cap). The driver throws std::invalid_argument for two distinct
            // infeasibility cases: the matvec_cap case and the block-Krylov
            // s*t <= n case reachable at non-default probe block widths. Both
            // driver messages contain "infeasible", so the OR-match above on
            // "s*t <= n" is redundant-but-harmless; it is kept for robustness
            // in case the driver message text changes. Both cases route to
            // the SAME MATLAB error id as the 'auto' tier's infeasible-budget
            // case, so callers can catch one id regardless of which
            // knob-free tier they picked.
            const std::optional<int64_t> mv_cap = (adaptive_mvcap > 0)
                ? std::optional<int64_t>(adaptive_mvcap) : std::nullopt;
            try {
                est = driver.call(A_op, fscalar, static_cast<T>(auto_eps),
                                  state, t1, t2, mv_cap);
            } catch (const std::invalid_argument& e) {
                const std::string msg = e.what();
                if (msg.find("infeasible") != std::string::npos ||
                    msg.find("s*t <= n") != std::string::npos) {
                    raise("randlapack:fun_nystrom_pp:infeasibleBudget", msg);
                }
                raise("randlapack:fun_nystrom_pp_mex:StdError", msg);
            }
        } else {
            est = driver.call(A_op, fAfun, fscalar,
                              k, s, q,
                              state, Omega2_ptr,
                              t1, t2);
        }
        };   // end drive_tiers

        if (vector_mode) {
            linops::DiagSymLinOp<T> A_op(n, A_buf);
            drive_tiers(A_op);
        } else if (sparse_mode) {
            if constexpr (std::is_same_v<T, double>) {
                using CSC = RandBLAS::sparse_data::CSCMatrix<double, int64_t>;
                // CSCMatrix's non-owning constructor takes non-const pointers; the view is only read (see marshal_csc),
                // so dropping const here never lets a write reach MATLAB's arrays.
                CSC A_csc(n, n, sp_nnz, const_cast<double*>(sp_vals), const_cast<int64_t*>(sp_rowidx),
                          const_cast<int64_t*>(sp_colptr));
                linops::SparseSymLinOp<double, CSC> A_op(A_csc);
                drive_tiers(A_op);
            } else {
                raise("randlapack:fun_nystrom_pp_mex:A_sparse_single", "sparse A is double only");
            }
        } else {
            linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A_buf, n, Layout::ColMajor);
            A_op.both_triangles = true;   // A_buf is MATLAB's full symmetric matrix (or, with copy_input, a copy of it)
            drive_tiers(A_op);
        }

        outputs[0] = factory.createScalar<double>(static_cast<double>(est));
        if (outputs.size() >= 2) outputs[1] = factory.createScalar<double>(static_cast<double>(t1));
        if (outputs.size() >= 3) outputs[2] = factory.createScalar<double>(static_cast<double>(t2));

        // --- optional 4th output: wall-clock instrumentation struct ---
        if (outputs.size() >= 4) {
            // NystromEVD 11-slot breakdown (microseconds, as doubles).
            std::vector<double> nys_us(driver.nystrom_ws.times.begin(),
                                       driver.nystrom_ws.times.end());
            if (nys_us.size() != 11) nys_us.assign(11, 0.0);
            // Lanczos-oracle 6-slot breakdown (whichever oracle ran); zeros for
            // the exact oracle, which has no Lanczos phase to instrument.
            std::vector<double> lfa_us(6, 0.0);
            if (lfa_type == "scalar" && scalar_lfa.times.size() >= 5) {
                lfa_us.assign(scalar_lfa.times.begin(), scalar_lfa.times.end());
            } else if (lfa_type == "scalar_qfa" && scalar_qfa.times.size() >= 5) {
                lfa_us.assign(scalar_qfa.times.begin(), scalar_qfa.times.end());
            } else if (lfa_type == "block" && block_lfa.times.size() >= 5) {
                lfa_us.assign(block_lfa.times.begin(), block_lfa.times.end());
            } else if (lfa_type == "block_qfa" && block_qfa.times.size() >= 5) {
                lfa_us.assign(block_qfa.times.begin(), block_qfa.times.end());
            } else if (lfa_type == "auto" && driver.auto_sqfa.times.size() >= 5) {
                lfa_us.assign(driver.auto_sqfa.times.begin(), driver.auto_sqfa.times.end());
            } else if (requested_lfa == "scalar_auto" && driver.auto_sqfa.times.size() >= 5) {
                lfa_us.assign(driver.auto_sqfa.times.begin(), driver.auto_sqfa.times.end());
            } else if (lfa_type == "adaptive" && driver.adaptive_bqfa.times.size() >= 5) {
                lfa_us.assign(driver.adaptive_bqfa.times.begin(), driver.adaptive_bqfa.times.end());
            }
            // NaN convention (see the header comment): used below for both the
            // path-specific fields and 'exact''s d_used (declared here, before
            // its first use, rather than where the path-specific block below
            // used to declare it).
            const double nan_v = std::numeric_limits<double>::quiet_NaN();
            // Lanczos depth actually used by the f(A) oracle. For block_qfa with
            // adaptive stopping this is the online-chosen depth (< the d cap);
            // for 'auto' it is the probe-discovered depth t; for 'adaptive' it is
            // the depth probe's certified (or reached) depth t
            // (driver.adaptive_t); for every other lfa_type it is just the fixed
            // d, EXCEPT 'exact', which never runs a Lanczos recurrence at all (it
            // builds f(A) once via syevd) and so has no meaningful depth: NaN,
            // not the leftover default-depth input `d`, which would otherwise
            // read as a real (if constant and meaningless) cost figure in a
            // benchmark plot. Exposed so the benchmark can count matvecs
            // (matvecs of A are proportional to this depth, for oracles that
            // have one).
            double d_used;
            if      (lfa_type == "block_qfa")  d_used = static_cast<double>(block_qfa.d_used);
            else if (lfa_type == "scalar_qfa") d_used = static_cast<double>(scalar_qfa.d_used);
            else if (lfa_type == "auto")       d_used = static_cast<double>(driver.auto_t);
            else if (lfa_type == "adaptive")   d_used = static_cast<double>(driver.adaptive_t);
            else if (lfa_type == "exact")      d_used = nan_v;
            else                               d_used = static_cast<double>(d);
            // Actual Phase-2 oracle matvecs: Σ per-probe certified depths for
            // the scalar-QFA-backed types, s*d_used (the class's matvecs
            // member) for block_qfa, the block oracle's matvecs member for
            // 'adaptive'; 0 otherwise (analytic s*d_used).
            double oracle_mv = 0.0;
            if      (lfa_type == "scalar_qfa") oracle_mv = static_cast<double>(scalar_qfa.matvecs);
            else if (lfa_type == "block_qfa")  oracle_mv = static_cast<double>(block_qfa.matvecs);
            else if (lfa_type == "auto")       oracle_mv = static_cast<double>(driver.auto_oracle_matvecs);
            else if (lfa_type == "adaptive")   oracle_mv = static_cast<double>(driver.adaptive_oracle_matvecs);
            // Knob-free bookkeeping (zeros unless lfa_type == 'auto'/'adaptive'):
            // the chosen rank / probe count and the matvecs the depth probe
            // actually spent. With the certified oracle the budget closes as an
            // upper bound: probe_mv + q*auto_k + oracle_mv <= budget (q = 1 in
            // the auto tier; 'adaptive' has no upfront budget so this is an
            // accounting identity, not a constraint enforced against an input).
            const bool is_auto     = (lfa_type == "auto");
            const bool is_adaptive = (lfa_type == "adaptive");
            const double auto_k_out  = is_auto ? static_cast<double>(driver.auto_k)
                                      : is_adaptive ? static_cast<double>(driver.adaptive_k) : 0.0;
            const double auto_s_out  = is_auto ? static_cast<double>(driver.auto_s)
                                      : is_adaptive ? static_cast<double>(driver.adaptive_s) : 0.0;
            const double probe_mv    = is_auto ? static_cast<double>(driver.auto_probe_matvecs)
                                      : is_adaptive ? static_cast<double>(driver.adaptive_probe_matvecs) : 0.0;
            // Path-specific fields, NaN when the path did not run (see the
            // header comment): block_qfa's/adaptive's Gauss/Radau traces +
            // certification, and the auto/adaptive tiers' probe wall-clock +
            // certification flags. nan_v declared above, next to d_used.
            double tr_U_out = nan_v, tr_L_out = nan_v, cert_out = nan_v;
            if (lfa_type == "block_qfa") {
                tr_U_out = static_cast<double>(block_qfa.tr_U);
                tr_L_out = static_cast<double>(block_qfa.tr_L);
                cert_out = block_qfa.certified ? 1.0 : 0.0;
            } else if (is_adaptive) {
                tr_U_out = static_cast<double>(driver.adaptive_tr_U);
                tr_L_out = static_cast<double>(driver.adaptive_tr_L);
                cert_out = driver.adaptive_phase2_certified ? 1.0 : 0.0;
                // scalar_auto: the driver's block traces are NaN by construction. The
                // scalar oracle keeps per-column Gauss/Radau values for its last call,
                // which is the Phase-2 call over driver.adaptive_s columns; report
                // their sums so the bracket is available for every self-tuning tier.
                if (driver.adaptive_use_scalar && driver.adaptive_s > 0 &&
                    driver.auto_sqfa.gauss_val_sz >= driver.adaptive_s) {
                    double su = 0.0, sl = 0.0;
                    for (int64_t j = 0; j < driver.adaptive_s; ++j) {
                        su += static_cast<double>(driver.auto_sqfa.gauss_val[j]);
                        sl += static_cast<double>(driver.auto_sqfa.radau_val[j]);
                    }
                    tr_U_out = su; tr_L_out = sl;
                }
                // spend_cap: Phase 2 ran at fixed depth and evaluated no bracket, so the lower
                // side and the certificate are not measurements (the oracles leave 0 or tr_U there).
                if (!driver.adaptive_phase2_checked) { tr_L_out = nan_v; cert_out = nan_v; }
            } else if (lfa_type == "scalar_qfa" && scalar_qfa.adaptive) {
                // NaN when Adaptive=0 (fixed depth): no certificate was ever
                // checked, so all_certified's default-false would misreport
                // "uncertified" rather than "not applicable".
                cert_out = scalar_qfa.all_certified ? 1.0 : 0.0;
                // Sum of the per-column Gauss and Gauss-Radau values over the s
                // Phase-2 columns: the scalar analogue of the block tier's tr_U/tr_L.
                if (s > 0 && scalar_qfa.gauss_val_sz >= s) {
                    double su = 0.0, sl = 0.0;
                    for (int64_t j = 0; j < s; ++j) {
                        su += static_cast<double>(scalar_qfa.gauss_val[j]);
                        sl += static_cast<double>(scalar_qfa.radau_val[j]);
                    }
                    tr_U_out = su; tr_L_out = sl;
                }
            }
            double probe_ms_out = nan_v, probe_conv_out = nan_v, ph2_cert_out = nan_v, ph2_checked_out = nan_v;
            if (is_auto) {
                probe_ms_out   = driver.t_probe_ms;
                probe_conv_out = driver.auto_probe_converged ? 1.0 : 0.0;
                ph2_cert_out   = driver.auto_phase2_certified ? 1.0 : 0.0;
            } else if (is_adaptive) {
                probe_ms_out   = driver.t_adaptive_probe_ms;
                probe_conv_out = driver.adaptive_probe_certified ? 1.0 : 0.0;
                ph2_cert_out   = driver.adaptive_phase2_checked ? (driver.adaptive_phase2_certified ? 1.0 : 0.0) : nan_v;
                ph2_checked_out = driver.adaptive_phase2_checked ? 1.0 : 0.0;
            }
            matlab::data::StructArray ts = factory.createStructArray({1, 1},
                {"marshal_in_ms", "phase1_ms", "phase2_ms", "fafun_ms",
                 "assembly_ms", "specrec_ms", "nystrom_us", "lfa_us", "d_used",
                 "oracle_mv", "auto_k", "auto_s", "probe_mv",
                 "tr_U", "tr_L", "certified",
                 "probe_ms", "probe_converged", "phase2_certified",
                 "first_row_ql_requested", "first_row_ql_fallback", "symmetric_sketch",
                 "ritz_clamped", "rank_deficient_steps", "min_diag_ratio",
                 "nystrom_clamped", "bracket_evaluated", "phase2_checked", "input_copied"});
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
            ts[0]["d_used"]        = factory.createScalar<double>(d_used);
            ts[0]["oracle_mv"]     = factory.createScalar<double>(oracle_mv);
            ts[0]["auto_k"]        = factory.createScalar<double>(auto_k_out);
            ts[0]["auto_s"]        = factory.createScalar<double>(auto_s_out);
            ts[0]["probe_mv"]      = factory.createScalar<double>(probe_mv);
            ts[0]["tr_U"]          = factory.createScalar<double>(tr_U_out);
            ts[0]["tr_L"]          = factory.createScalar<double>(tr_L_out);
            ts[0]["certified"]     = factory.createScalar<double>(cert_out);
            ts[0]["probe_ms"]      = factory.createScalar<double>(probe_ms_out);
            ts[0]["probe_converged"]  = factory.createScalar<double>(probe_conv_out);
            ts[0]["phase2_certified"] = factory.createScalar<double>(ph2_cert_out);
            ts[0]["phase2_checked"]   = factory.createScalar<double>(ph2_checked_out);
            // true when A (and an explicit Omega2) were copied into MEX-owned buffers (RANDLAPACK_FNPP_COPY_INPUT=1 or a
            // legacy build); false when the driver read MATLAB's arrays in place.
            ts[0]["input_copied"]     = factory.createScalar<bool>(copy_input);
            ts[0]["first_row_ql_requested"] = factory.createScalar<bool>(first_row_ql != 0.0);
            ts[0]["first_row_ql_fallback"] = factory.createScalar<bool>(
                scalar_qfa.first_row_ql_fallback_seen || driver.auto_sqfa.first_row_ql_fallback_seen);
#ifdef RANDLAPACK_SYMMETRIC_SKETCH
            ts[0]["symmetric_sketch"] = factory.createScalar<bool>(true);
#else
            ts[0]["symmetric_sketch"] = factory.createScalar<bool>(false);
#endif
            // ---- Numerical-degradation diagnostics ----
            // Until now the kernel's guards were unobservable: a clamped Ritz value, a
            // rank-deficient block QR or a clamped Nystrom eigenvalue left no trace in any
            // exported field, so a degraded run and a healthy one produced identical telemetry.
            // These five fields are the channel out. Exactly one code path runs per call, so
            // summing across the objects reports that path's totals and leaves the rest at zero.
            const double ritz_clamped_total =
                  (double)scalar_qfa.ritz_clamped
                + (double)block_qfa.ritz_clamped + (double)block_qfa.fa.ritz_clamped
                + (double)driver.auto_sqfa.ritz_clamped
                + (double)driver.adaptive_bqfa.ritz_clamped
                + (double)driver.adaptive_bqfa.fa.ritz_clamped;
            const double rank_def_total =
                  (double)block_qfa.fa.rank_deficient_steps
                + (double)driver.adaptive_bqfa.fa.rank_deficient_steps;
            // Smallest diag ratio seen by whichever block QR ran; stays Inf when none did, so a
            // reader can tell "no block QR" from "a block QR that was comfortably full rank".
            double min_ratio = (double)block_qfa.fa.min_diag_ratio;
            if ((double)driver.adaptive_bqfa.fa.min_diag_ratio < min_ratio)
                min_ratio = (double)driver.adaptive_bqfa.fa.min_diag_ratio;
            ts[0]["ritz_clamped"]         = factory.createScalar<double>(ritz_clamped_total);
            ts[0]["rank_deficient_steps"] = factory.createScalar<double>(rank_def_total);
            ts[0]["min_diag_ratio"]       = factory.createScalar<double>(min_ratio);
            ts[0]["nystrom_clamped"]      = factory.createScalar<double>((double)driver.nystrom_ws.clamped_eigenvalues);
            ts[0]["bracket_evaluated"]    = factory.createScalar<bool>(
                block_qfa.bracket_evaluated || driver.adaptive_bqfa.bracket_evaluated);
            outputs[3] = std::move(ts);
        }
        // A_own / O2_own release their buffers here (RAII).
    }

public:
    MexFunction() : matlabPtr(getEngine()) {}

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        try {
            if (inputs.size() < 8 || inputs.size() > 28) {
                raise("randlapack:fun_nystrom_pp_mex:nargin",
                      "Expected 8 to 28 inputs: A, Omega1, Omega2, func, q, poly_lambda, "
                      "lfa_type, d [, sketch_type, vec_nnz, sketch_seed, reorth, "
                      "adaptive, adaptive_tol, adaptive_delay, adaptive_min, "
                      "budget, auto_eps, radau_return, auto_depth_cap, "
                      "auto_probe_frac, adaptive_matvec_cap, adaptive_k_const, "
                      "adaptive_s_const, cap_rank_fraction, quadrature_fraction, first_row_ql, spend_cap]");
            }
            if (outputs.size() < 1 || outputs.size() > 4) {
                raise("randlapack:fun_nystrom_pp_mex:nargout",
                      "Expected 1 to 4 outputs: [est, t1, t2, times]");
            }

            // --- A: square n x n, OR an n-vector read as diag(A), real,
            // single/double. Vector mode never forms the n x n matrix. ---
            const Array& A_in = inputs[0];
            auto A_dims = A_in.getDimensions();
            if (A_in.getType() == ArrayType::SPARSE_DOUBLE) {
                raise("randlapack:fun_nystrom_pp_mex:A_sparse_raw",
                      "pass sparse A through randlapack.fun_nystrom_pp, which hands the MEX its compressed-column arrays "
                      "(reading a MATLAB sparse array element by element here costs O(n) per nonzero)");
            }
            const bool sparse_in = is_sparse_struct(A_in);
            const bool vector_mode = !sparse_in && (A_dims.size() == 2) &&
                ((A_dims[0] == 1 && A_dims[1] > 1) || (A_dims[1] == 1 && A_dims[0] > 1));
            if (!sparse_in && (A_dims.size() != 2 || (!vector_mode && A_dims[0] != A_dims[1]))) {
                raise("randlapack:fun_nystrom_pp_mex:A_shape",
                      "A must be a square n x n matrix or an n-vector of diagonal entries");
            }
            const ArrayType A_type = sparse_in ? ArrayType::DOUBLE : A_in.getType();
            if (A_type != ArrayType::DOUBLE && A_type != ArrayType::SINGLE) {
                raise("randlapack:fun_nystrom_pp_mex:A_dtype",
                      "A must be single or double precision, or a sparse double matrix");
            }
            // A_dims[0] is 1 for a row vector, so numel is the only safe read.
            const int64_t n = sparse_in ? sparse_n_of(A_in)
                : vector_mode ? static_cast<int64_t>(A_in.getNumberOfElements())
                : static_cast<int64_t>(A_dims[0]);

            // --- arg 2 (Phase-1): scalar k only. The Phase-1 sketch is a SASO
            // generated inside RandLAPACK::NystromEVD from (sketch_seed, vec_nnz);
            // the explicit-matrix escape hatch was removed with the kernel's
            // dense-sketch mode. ---
            const Array& O1_in = inputs[1];
            if (O1_in.getNumberOfElements() != 1) {
                raise("randlapack:fun_nystrom_pp_mex:Omega1_explicit",
                      "arg 2 must be a scalar k. Explicit Omega1 matrices are no "
                      "longer supported: the Phase-1 sketch is a SASO "
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
        catch (const MexError& r) {
            emit_matlab_error(r.id, r.msg);
        }
        catch (const RandLAPACK::Error& e) {
            emit_matlab_error("randlapack:fun_nystrom_pp_mex:RandLAPACKError", e.what());
        }
        catch (const RandBLAS::Error& e) {
            emit_matlab_error("randlapack:fun_nystrom_pp_mex:RandBLASError", e.what());
        }
        catch (const matlab::Exception&) {
            // Engine exceptions (e.g. an interrupt) propagate unchanged. NB in
            // the dual-libstdc++ setup this clause may fail to match, in which
            // case the std::exception clause below re-labels the error as
            // StdError with the message preserved.
            throw;
        }
        catch (const std::exception& e) {
            emit_matlab_error("randlapack:fun_nystrom_pp_mex:StdError", e.what());
        }
        catch (...) {
            emit_matlab_error("randlapack:fun_nystrom_pp_mex:Unknown",
                              "Unknown exception in fun_nystrom_pp_mex");
        }
    }
};
