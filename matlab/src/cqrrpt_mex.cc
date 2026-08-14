// MEX wrapper for RandLAPACK::CQRRPT using MATLAB's C++ Data API (R2018a+).
//
// MATLAB-side signature (low-level; users normally call randlapack.cqrrpt.m):
//
//   [Q, R, J, state_out] = cqrrpt_mex(A, d_factor, eps, state)
//
// Outputs:
//   Q          m-by-k orthonormal factor, k = the rank CQRRPT detected
//   R          k-by-n upper-trapezoidal factor
//   J          1-by-n vector of 1-based column pivots (int64), so
//              A(:, J) ~= Q * R
//   state_out  RNG state after advancing
//
// k is discovered, not requested: CQRRPT is rank-revealing, and the rank it
// detects (its public `rank` member, governed by `eps`) sets the shared inner
// dimension of the returned Q and R. On a full-rank input k = n.
//
// J is 1-based, matching both MATLAB and CQRRPT's own convention -- the
// pivots originate in LAPACK's geqp3 and RandLAPACK's util::col_swap reads
// them as `idx[i] - 1`. No conversion here. (The Python binding converts, for
// the opposite reason.)
//
// Shape contract worth noting while reading the buffer sizes below: CQRRPT
// factors A IN PLACE into Q, requiring an m-by-n work buffer even though only
// the first k columns come back as output, and it writes R into a caller-
// allocated n-by-n buffer (ldr >= n) of which only the first k rows are
// meaningful. Both outputs are therefore sliced out of oversized scratch,
// which is why neither can be factored directly into its output buffer the
// way bqrrp_mex's implicit mode does.

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
            raise("randlapack:cqrrpt_mex:state",
                "state must be a struct with .counter (uint32[4]) and .key (uint32[2])");
        }
        StructArray s = state_arr;
        if (s.getNumberOfElements() == 0) {
            raise("randlapack:cqrrpt_mex:state", "state struct must be nonempty");
        }
        Array ctr_field = s[0]["counter"];
        Array key_field = s[0]["key"];
        if (ctr_field.getType() != ArrayType::UINT32
            || ctr_field.getNumberOfElements() != CTR_LEN) {
            raise("randlapack:cqrrpt_mex:state",
                "state.counter must be a uint32 array of length 4");
        }
        if (key_field.getType() != ArrayType::UINT32
            || key_field.getNumberOfElements() != KEY_LEN) {
            raise("randlapack:cqrrpt_mex:state",
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
    void run_cqrrpt(ArgumentList& outputs, ArgumentList& inputs,
                    double d_factor_in, double eps_in)
    {
        const Array& A_in = inputs[0];
        auto dims = A_in.getDimensions();
        const int64_t m = static_cast<int64_t>(dims[0]);
        const int64_t n = static_cast<int64_t>(dims[1]);
        const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
        const T d_factor = static_cast<T>(d_factor_in);
        const T eps      = static_cast<T>(eps_in);

        // Const view + one bulk memcpy: a non-const TypedArray would unshare
        // (deep-copy) the input before any element is read.
        const TypedArray<T> A_typed = A_in;
        const T* A_src = &*A_typed.cbegin();
        std::unique_ptr<T[]> A_work(new T[mn]);
        std::memcpy(A_work.get(), A_src, sizeof(T) * mn);

        // R is n-by-n scratch (ldr = n) regardless of the detected rank; J is
        // written directly into its final output buffer.
        std::vector<T> R_work(static_cast<size_t>(n) * static_cast<size_t>(n));
        auto J_buf = factory.createBuffer<int64_t>(static_cast<size_t>(n));

        // J MUST be zeroed before the call. CQRRPT hands it straight to
        // LAPACK's geqp3, which READS jpvt on entry: a nonzero entry means
        // "this column is fixed, move it to the front". createBuffer returns
        // uninitialized memory, so skipping this makes the pivoting depend on
        // whatever was in the heap -- nondeterministic results and a rank
        // estimate that collapses. (BQRRP is not affected: it iotas J itself.
        // RandLAPACK's own tests pass a std::vector, which zero-initializes,
        // so the precondition is invisible from inside the library.)
        std::fill(J_buf.get(), J_buf.get() + n, static_cast<int64_t>(0));

        auto state = read_state(inputs[3]);
        RandLAPACK::CQRRPT<T, RNG> alg(false, eps);

        int ret = alg.call(m, n, A_work.get(), m, R_work.data(), n,
                           J_buf.get(), d_factor, state);
        if (ret != 0) {
            raise("randlapack:cqrrpt_mex:returnCode",
                "CQRRPT returned non-zero status " + std::to_string(ret));
        }

        const int64_t k = alg.rank;
        if (k <= 0) {
            raise("randlapack:cqrrpt_mex:zeroRank",
                "CQRRPT detected rank 0; the input may be numerically zero, "
                "or eps may be too large");
        }

        // Q = first k columns of the in-place factored A (column-major, so
        // they are the leading m*k contiguous elements).
        auto Q_buf = factory.createBuffer<T>(
            static_cast<size_t>(m) * static_cast<size_t>(k));
        std::memcpy(Q_buf.get(), A_work.get(),
                    sizeof(T) * static_cast<size_t>(m) * static_cast<size_t>(k));
        outputs[0] = factory.createArrayFromBuffer<T>(
            {static_cast<size_t>(m), static_cast<size_t>(k)}, std::move(Q_buf));

        // R = first k rows of the n-by-n scratch, restrided from ldr=n to k.
        auto R_buf = factory.createBuffer<T>(
            static_cast<size_t>(k) * static_cast<size_t>(n));
        T* R_out = R_buf.get();
        for (int64_t j = 0; j < n; ++j) {
            std::memcpy(&R_out[j * k], &R_work[j * n], sizeof(T) * k);
        }
        outputs[1] = factory.createArrayFromBuffer<T>(
            {static_cast<size_t>(k), static_cast<size_t>(n)}, std::move(R_buf));

        // J: 1-based int64 row vector, written in place by CQRRPT above.
        outputs[2] = factory.createArrayFromBuffer<int64_t>(
            {1, static_cast<size_t>(n)}, std::move(J_buf));

        if (outputs.size() >= 4) {
            outputs[3] = write_state(state);
        }
    }

    double read_double(const Array& a, const char* name) {
        if (a.getNumberOfElements() != 1) {
            raise(std::string("randlapack:cqrrpt_mex:") + name,
                  std::string(name) + " must be a scalar");
        }
        switch (a.getType()) {
            case ArrayType::DOUBLE: return TypedArray<double>(a)[0];
            case ArrayType::SINGLE: return static_cast<double>(TypedArray<float>(a)[0]);
            case ArrayType::INT64:  return static_cast<double>(TypedArray<int64_t>(a)[0]);
            default:
                raise(std::string("randlapack:cqrrpt_mex:") + name,
                      std::string(name) + " must be a numeric scalar");
        }
    }

public:
    MexFunction() : matlabPtr(getEngine()) {}

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        try {
            if (inputs.size() != 4) {
                raise("randlapack:cqrrpt_mex:nargin",
                    "Expected 4 input arguments: A, d_factor, eps, state");
            }
            if (outputs.size() < 3 || outputs.size() > 4) {
                raise("randlapack:cqrrpt_mex:nargout",
                    "Expected 3 or 4 output arguments");
            }

            const Array& A_in = inputs[0];
            if (A_in.getDimensions().size() != 2) {
                raise("randlapack:cqrrpt_mex:A",
                    "A must be a 2D real numeric matrix (single or double)");
            }
            const ArrayType A_type = A_in.getType();
            if (A_type != ArrayType::DOUBLE && A_type != ArrayType::SINGLE) {
                raise("randlapack:cqrrpt_mex:type", "A must be single or double");
            }

            const double d_factor = read_double(inputs[1], "d_factor");
            const double eps      = read_double(inputs[2], "eps");

            if (d_factor < 1.0) {
                raise("randlapack:cqrrpt_mex:d_factor",
                      "d_factor must be >= 1.0");
            }
            if (eps < 0.0) {
                raise("randlapack:cqrrpt_mex:eps", "eps must be nonnegative");
            }

            // CQRRPT sketches to d = d_factor*n rows and needs m >= d >= n.
            // Checked here, with the arithmetic spelled out, rather than
            // letting the requirement surface as a dimension complaint from
            // three layers down inside the sketch-and-precondition chain.
            const int64_t m = static_cast<int64_t>(A_in.getDimensions()[0]);
            const int64_t n = static_cast<int64_t>(A_in.getDimensions()[1]);
            const int64_t d = static_cast<int64_t>(d_factor * static_cast<double>(n));
            if (m < d) {
                raise("randlapack:cqrrpt_mex:notTall",
                    "CQRRPT requires a tall matrix: m >= d_factor*n. Got m="
                    + std::to_string(m) + ", n=" + std::to_string(n)
                    + ", d_factor=" + std::to_string(d_factor)
                    + " (needs m >= " + std::to_string(d) + "). Reduce d_factor "
                    "toward 1.0, or use randlapack.bqrrp for wide matrices.");
            }

            if (A_type == ArrayType::DOUBLE) {
                run_cqrrpt<double>(outputs, inputs, d_factor, eps);
            } else {
                run_cqrrpt<float>(outputs, inputs, d_factor, eps);
            }
        } catch (const RandLAPACK::Error& e) {
            raise("randlapack:cqrrpt_mex:RandLAPACKError", e.what());
        } catch (const RandBLAS::Error& e) {
            raise("randlapack:cqrrpt_mex:RandBLASError", e.what());
        } catch (const matlab::Exception&) {
            // MATLAB exceptions (typically from feval(error)) propagate through
            // unchanged so MATLAB sees the original error ID and message.
            throw;
        } catch (const std::exception& e) {
            raise("randlapack:cqrrpt_mex:StdError", e.what());
        } catch (...) {
            raise("randlapack:cqrrpt_mex:Unknown", "Unknown exception in cqrrpt_mex");
        }
    }
};
