// MEX wrapper for RandLAPACK::CQRRT using MATLAB's C++ Data API (R2018a+).
//
// MATLAB-side signature (low-level; users normally call randlapack.cqrrt.m):
//
//   [Q, R, state_out] = cqrrt_mex(A, d_factor, compute_Q, state)
//
// Outputs:
//   Q          m-by-n orthonormal factor; 0-by-0 when compute_Q is false
//   R          n-by-n upper-triangular factor, so A ~= Q * R
//   state_out  RNG state after advancing
//
// CQRRT is unpivoted: it sketches A, takes R_sk from a QR of the sketch,
// preconditions A := A * inv(R_sk), and finishes with Cholesky QR. It expects
// a full-rank tall input and does not detect rank: a rank-deficient input
// usually completes, with R numerically singular. The driver returns a
// nonzero status only when the sketch's R has an exact zero on its diagonal
// (e.g. A has a zero column) or the Cholesky factorization fails; that is
// reported below as an error pointing at the rank-revealing alternatives.
//
// compute_Q = false selects CQRRT's R-only mode, which skips forming Q. The
// driver still overwrites A while preconditioning, so a work copy of the
// input is needed either way; with compute_Q = true that copy is the Q output
// buffer itself, and A is factored in place into it.

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <RandLAPACK.hh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using matlab::mex::ArgumentList;
using matlab::data::Array;
using matlab::data::ArrayFactory;
using matlab::data::ArrayType;
using matlab::data::StructArray;
template <typename T> using TypedArray = matlab::data::TypedArray<T>;


class MexFunction : public matlab::mex::Function {
private:
    using RNG = RandBLAS::DefaultRNG;
    static constexpr size_t CTR_LEN = 4;
    static constexpr size_t KEY_LEN = 2;

    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    ArrayFactory factory;

    [[noreturn]] void raise(const std::string& id, const std::string& msg) {
        matlabPtr->feval(u"error", 0, std::vector<Array>{
            factory.createCharArray(id),
            factory.createCharArray(msg)
        });
        std::terminate();  // feval(error) does not return; this is a backstop.
    }

    RandBLAS::RNGState<RNG> read_state(const Array& state_arr) {
        if (state_arr.getType() != ArrayType::STRUCT) {
            raise("randlapack:cqrrt_mex:state",
                "state must be a struct with .counter (uint32[4]) and .key (uint32[2])");
        }
        StructArray s = state_arr;
        if (s.getNumberOfElements() == 0) {
            raise("randlapack:cqrrt_mex:state", "state struct must be nonempty");
        }
        Array ctr_field = s[0]["counter"];
        Array key_field = s[0]["key"];
        if (ctr_field.getType() != ArrayType::UINT32
            || ctr_field.getNumberOfElements() != CTR_LEN) {
            raise("randlapack:cqrrt_mex:state",
                "state.counter must be a uint32 array of length 4");
        }
        if (key_field.getType() != ArrayType::UINT32
            || key_field.getNumberOfElements() != KEY_LEN) {
            raise("randlapack:cqrrt_mex:state",
                "state.key must be a uint32 array of length 2");
        }
        TypedArray<uint32_t> ctr = std::move(ctr_field);
        TypedArray<uint32_t> key = std::move(key_field);
        RandBLAS::RNGState<RNG> state;
        for (size_t i = 0; i < CTR_LEN; ++i) state.counter.v[i] = ctr[i];
        for (size_t i = 0; i < KEY_LEN; ++i) state.key.v[i]     = key[i];
        return state;
    }

    Array write_state(const RandBLAS::RNGState<RNG>& state) {
        StructArray out = factory.createStructArray({1, 1}, {"counter", "key"});
        TypedArray<uint32_t> ctr = factory.createArray<uint32_t>({1, CTR_LEN});
        TypedArray<uint32_t> key = factory.createArray<uint32_t>({1, KEY_LEN});
        for (size_t i = 0; i < CTR_LEN; ++i) ctr[i] = state.counter.v[i];
        for (size_t i = 0; i < KEY_LEN; ++i) key[i] = state.key.v[i];
        out[0]["counter"] = std::move(ctr);
        out[0]["key"]     = std::move(key);
        return out;
    }

    template <typename T>
    void run_cqrrt(ArgumentList& outputs, ArgumentList& inputs,
                   double d_factor_in, bool compute_Q)
    {
        const Array& A_in = inputs[0];
        auto dims = A_in.getDimensions();
        const int64_t m = static_cast<int64_t>(dims[0]);
        const int64_t n = static_cast<int64_t>(dims[1]);
        const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
        const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
        const T d_factor = static_cast<T>(d_factor_in);

        // Work copy of A: the Q output buffer when Q is wanted (factored in
        // place, no second copy), otherwise scratch. Const view + one bulk
        // memcpy: a non-const TypedArray would unshare (deep-copy) the input
        // before any element is read.
        const TypedArray<T> A_typed = A_in;
        const T* A_src = &*A_typed.cbegin();
        auto Q_buf = factory.createBuffer<T>(compute_Q ? mn : 0);
        std::unique_ptr<T[]> A_scratch(compute_Q ? nullptr : new T[mn]);
        T* A_work = compute_Q ? Q_buf.get() : A_scratch.get();
        std::memcpy(A_work, A_src, sizeof(T) * mn);

        // R must be zero on entry. CQRRT writes only the upper triangle until
        // its final trmm, which multiplies the whole n-by-n buffer: anything
        // left below the diagonal would leak into the upper triangle of R.
        // createBuffer returns uninitialized memory, hence the fill.
        auto R_buf = factory.createBuffer<T>(nn);
        std::fill(R_buf.get(), R_buf.get() + nn, static_cast<T>(0));

        auto state = read_state(inputs[3]);
        // CQRRT keeps an eps member for interface symmetry with CQRRPT but
        // does not read it.
        RandLAPACK::CQRRT<T, RNG> alg(false, static_cast<T>(0));
        alg.compute_Q = compute_Q;

        int ret = alg.call(m, n, A_work, m, R_buf.get(), n, d_factor, state);
        if (ret != 0) {
            raise("randlapack:cqrrt_mex:breakdown",
                "CQRRT could not factor A: the R factor of its sketch is "
                "exactly singular (e.g. A has a zero column) or its Cholesky "
                "step failed, so A is rank-deficient or too ill-conditioned "
                "for an unpivoted factorization. Use randlapack.cqrrpt, which "
                "reveals the rank, or randlapack.bqrrp.");
        }

        if (compute_Q) {
            outputs[0] = factory.createArrayFromBuffer<T>(
                {static_cast<size_t>(m), static_cast<size_t>(n)}, std::move(Q_buf));
        } else {
            outputs[0] = factory.createArray<T>({0, 0});
        }
        if (outputs.size() >= 2) {
            outputs[1] = factory.createArrayFromBuffer<T>(
                {static_cast<size_t>(n), static_cast<size_t>(n)}, std::move(R_buf));
        }
        if (outputs.size() >= 3) {
            outputs[2] = write_state(state);
        }
    }

    double read_double(const Array& a, const char* name) {
        if (a.getNumberOfElements() != 1) {
            raise(std::string("randlapack:cqrrt_mex:") + name,
                  std::string(name) + " must be a scalar");
        }
        switch (a.getType()) {
            case ArrayType::DOUBLE: return TypedArray<double>(a)[0];
            case ArrayType::SINGLE: return static_cast<double>(TypedArray<float>(a)[0]);
            case ArrayType::INT64:  return static_cast<double>(TypedArray<int64_t>(a)[0]);
            default:
                raise(std::string("randlapack:cqrrt_mex:") + name,
                      std::string(name) + " must be a numeric scalar");
        }
    }

    bool read_logical(const Array& a, const char* name) {
        if (a.getNumberOfElements() != 1 || a.getType() != ArrayType::LOGICAL) {
            raise(std::string("randlapack:cqrrt_mex:") + name,
                  std::string(name) + " must be a logical scalar");
        }
        return TypedArray<bool>(a)[0];
    }

public:
    MexFunction() : matlabPtr(getEngine()) {}

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        try {
            if (inputs.size() != 4) {
                raise("randlapack:cqrrt_mex:nargin",
                    "Expected 4 input arguments: A, d_factor, compute_Q, state");
            }
            if (outputs.size() < 2 || outputs.size() > 3) {
                raise("randlapack:cqrrt_mex:nargout",
                    "Expected 2 or 3 output arguments");
            }

            const Array& A_in = inputs[0];
            if (A_in.getDimensions().size() != 2) {
                raise("randlapack:cqrrt_mex:A",
                    "A must be a 2D real numeric matrix (single or double)");
            }
            const ArrayType A_type = A_in.getType();
            if (A_type != ArrayType::DOUBLE && A_type != ArrayType::SINGLE) {
                raise("randlapack:cqrrt_mex:type", "A must be single or double");
            }

            const double d_factor = read_double(inputs[1], "d_factor");
            const bool compute_Q  = read_logical(inputs[2], "compute_Q");
            if (d_factor < 1.0) {
                raise("randlapack:cqrrt_mex:d_factor", "d_factor must be >= 1.0");
            }

            // CQRRT sketches to d = d_factor*n rows and needs m >= d >= n.
            // Checked here, with the arithmetic spelled out, rather than
            // letting the requirement surface as a dimension complaint from
            // inside the sketch-and-precondition chain.
            const int64_t m = static_cast<int64_t>(A_in.getDimensions()[0]);
            const int64_t n = static_cast<int64_t>(A_in.getDimensions()[1]);
            const int64_t d = static_cast<int64_t>(d_factor * static_cast<double>(n));
            if (m < d) {
                raise("randlapack:cqrrt_mex:notTall",
                    "CQRRT requires a tall matrix: m >= d_factor*n. Got m="
                    + std::to_string(m) + ", n=" + std::to_string(n)
                    + ", d_factor=" + std::to_string(d_factor)
                    + " (needs m >= " + std::to_string(d) + "). Reduce d_factor "
                    "toward 1.0, or use randlapack.bqrrp for wide matrices.");
            }

            if (A_type == ArrayType::DOUBLE) {
                run_cqrrt<double>(outputs, inputs, d_factor, compute_Q);
            } else {
                run_cqrrt<float>(outputs, inputs, d_factor, compute_Q);
            }
        } catch (const RandLAPACK::Error& e) {
            raise("randlapack:cqrrt_mex:RandLAPACKError", e.what());
        } catch (const RandBLAS::Error& e) {
            raise("randlapack:cqrrt_mex:RandBLASError", e.what());
        } catch (const matlab::engine::Exception&) {
            // Errors raised by this MEX (raise() calls feval("error"), which
            // throws matlab::engine::MATLABException) and other MATLAB engine
            // errors propagate unchanged, so MATLAB sees the original error ID
            // and message. In the MEX API (cppmex/mexException.hpp) these
            // derive from std::exception but not from matlab::Exception (the
            // C++ Engine API's classes of the same name differ): without this
            // handler, the std::exception one below would re-raise them as
            // ...:StdError.
            throw;
        } catch (const matlab::Exception&) {
            // MATLAB Data API exceptions propagate unchanged.
            throw;
        } catch (const std::exception& e) {
            raise("randlapack:cqrrt_mex:StdError", e.what());
        } catch (...) {
            raise("randlapack:cqrrt_mex:Unknown", "Unknown exception in cqrrt_mex");
        }
    }
};
