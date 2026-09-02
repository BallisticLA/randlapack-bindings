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
//                 d_used         Lanczos depth the oracle actually used. For
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
//                                driver.adaptive_tr_L; NaN otherwise.
//                 certified      'block_qfa'/'adaptive'/'scalar_qfa' (with
//                                Adaptive=1) only: 1 if the Radau bracket
//                                closed within (adaptive_tol / auto_eps), 0 if
//                                not; for 'adaptive' this is
//                                driver.adaptive_phase2_certified; for
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
        std::memcpy(out.get(), &*typed.cbegin(), n_elems * sizeof(T));
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

        // A: n x n (RAII-owned; freed on every exit path, including raise()
        // and exceptions out of driver.call)
        const Array& A_in = inputs[0];
        const int64_t n = static_cast<int64_t>(A_in.getDimensions()[0]);
        std::unique_ptr<T[]> A_own = copy_into<T>(A_in, static_cast<size_t>(n) * n);
        T* A_buf = A_own.get();

        // scalar params (read before sketch handling: internal sampling needs them)
        const std::string func        = read_string(inputs[3], "func");
        const int64_t     q           = read_int   (inputs[4], "q");
        const T           poly_lambda = static_cast<T>(read_double(inputs[5], "poly_lambda"));
        const std::string lfa_type    = read_string(inputs[6], "lfa_type");
        const int64_t     d           = read_int   (inputs[7], "d");
        const std::string sketch_type = (inputs.size() >= 9)  ? read_string(inputs[8], "sketch_type") : "saso";
        const int64_t     vec_nnz     = (inputs.size() >= 10) ? read_int(inputs[9],  "vec_nnz")     : 8;
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
        const int64_t     adaptive_mvcap = (inputs.size() >= 22) ? read_int(inputs[21], "adaptive_matvec_cap") : 0;

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
        if (vec_nnz < 0) {
            raise("randlapack:fun_nystrom_pp_mex:vec_nnz",
                  "vec_nnz (input 10) must be >= 0 (0 = auto, ~log(k))");
        }

        if (inputs.size() >= 9 && sketch_type != "saso") {
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
        std::unique_ptr<T[]> O2_own;   // RAII: freed on every exit path
        T* O2_buf = nullptr;
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
                O2_own = copy_into<T>(O2_in, static_cast<size_t>(n) * s);
                O2_buf = O2_own.get();
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
        RandLAPACK::LanczosQFA<T>      scalar_qfa;
        RandLAPACK::BlockLanczosFA<T>  block_lfa;
        RandLAPACK::BlockLanczosQFA<T> block_qfa;
        // Lanczos-QFA fills the quadratic form Ω₂ᵀf(A)Ω₂ directly (no f(A)·Ω₂
        // mapback); the driver takes its trace, reading only the diagonal.
        // Signalled to the driver below.
        const bool qfa_mode = (lfa_type == "block_qfa" || lfa_type == "scalar_qfa");

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

        // --- drive funNystrom++ ---
        RandLAPACK::FunNystromPP<T> driver;
        // Sub-timers always on: a handful of steady_clock reads per call,
        // negligible next to any BLAS work; consumed by the 4th output.
        driver.nystrom_ws.times_enabled = true;
        scalar_lfa.timing = true;
        scalar_qfa.timing = true;
        block_lfa.timing  = true;
        block_qfa.timing  = true;
        driver.auto_sqfa.timing = true;
        driver.adaptive_bqfa.timing = true;
        scalar_lfa.reorth = reorth_flag;
        block_lfa.reorth  = reorth_flag;
        block_qfa.reorth  = reorth_flag;
        // scalar_qfa: intrinsically no-reorth (basis-free recurrence); reorth /
        // adaptive_delay / adaptive_min do not apply — the Gauss-Radau
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
        T t1 = (T)0, t2 = (T)0;
        const T *Omega2_ptr = phase2_skipped ? nullptr : O2_buf;
        RandBLAS::RNGState<RNG> state(static_cast<uint32_t>(sketch_seed));
        T est;
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
            } else if (lfa_type == "scalar_qfa" && scalar_qfa.adaptive) {
                // NaN when Adaptive=0 (fixed depth): no certificate was ever
                // checked, so all_certified's default-false would misreport
                // "uncertified" rather than "not applicable".
                cert_out = scalar_qfa.all_certified ? 1.0 : 0.0;
            }
            double probe_ms_out = nan_v, probe_conv_out = nan_v, ph2_cert_out = nan_v;
            if (is_auto) {
                probe_ms_out   = driver.t_probe_ms;
                probe_conv_out = driver.auto_probe_converged ? 1.0 : 0.0;
                ph2_cert_out   = driver.auto_phase2_certified ? 1.0 : 0.0;
            } else if (is_adaptive) {
                probe_ms_out   = driver.t_adaptive_probe_ms;
                probe_conv_out = driver.adaptive_probe_certified ? 1.0 : 0.0;
                ph2_cert_out   = driver.adaptive_phase2_certified ? 1.0 : 0.0;
            }
            matlab::data::StructArray ts = factory.createStructArray({1, 1},
                {"marshal_in_ms", "phase1_ms", "phase2_ms", "fafun_ms",
                 "assembly_ms", "specrec_ms", "nystrom_us", "lfa_us", "d_used",
                 "oracle_mv", "auto_k", "auto_s", "probe_mv",
                 "tr_U", "tr_L", "certified",
                 "probe_ms", "probe_converged", "phase2_certified"});
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
            outputs[3] = std::move(ts);
        }
        // A_own / O2_own release their buffers here (RAII).
    }

public:
    MexFunction() : matlabPtr(getEngine()) {}

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        try {
            if (inputs.size() < 8 || inputs.size() > 22) {
                raise("randlapack:fun_nystrom_pp_mex:nargin",
                      "Expected 8 to 22 inputs: A, Omega1, Omega2, func, q, poly_lambda, "
                      "lfa_type, d [, sketch_type, vec_nnz, sketch_seed, reorth, "
                      "adaptive, adaptive_tol, adaptive_delay, adaptive_min, "
                      "budget, auto_eps, radau_return, auto_depth_cap, "
                      "auto_probe_frac, adaptive_matvec_cap]");
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
