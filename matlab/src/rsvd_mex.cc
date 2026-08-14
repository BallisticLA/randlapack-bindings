// MEX wrapper for RandLAPACK::RSVD using MATLAB's C++ Data API (R2018a+).
//
// MATLAB-side signature (low-level; users normally call randlapack.rsvd.m):
//
//   [U, s, V, state_out] = rsvd_mex(A, k, tol, block_sz, p, ppi, state)
//
// Outputs:
//   U          m-by-k_out orthonormal left singular vectors
//   s          length-k_out singular values (a VECTOR here; the .m wrapper
//              turns it into a diagonal matrix when called with 3 outputs,
//              matching svd/svds)
//   V          n-by-k_out right singular vectors, UN-transposed, so
//              A ~= U * diag(s) * V'  -- MATLAB's [U, S, V] convention
//   state_out  RNG state after advancing
//
// k is in-out at the RandLAPACK level: QB may return a factorization of
// smaller rank than requested (that is the point of the tolerance), so
// k_out <= k and the true value is carried by the output shapes.
//
// RSVD's own allocation contract, and why it dictates the shape of this
// file: RSVD::call takes U, S and V by REFERENCE-TO-POINTER and calloc()s
// them internally. They must be copied out and released with free() -- not
// delete[], not MATLAB's deleters. The copies below are therefore
// unavoidable, and the raw pointers are held in an RAII guard so an
// exception between the call and the copy cannot leak them.

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <RandLAPACK.hh>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

    // Frees the three calloc'd buffers RSVD hands back, whatever path leaves
    // the scope. free() specifically: they come from calloc.
    template <typename T>
    struct CallocGuard {
        T* U = nullptr;
        T* S = nullptr;
        T* V = nullptr;
        ~CallocGuard() { std::free(U); std::free(S); std::free(V); }
    };

    [[noreturn]] void raise(const std::string& id, const std::string& msg) {
        matlabPtr->feval(u"error", 0, std::vector<Array>{
            factory.createCharArray(id),
            factory.createCharArray(msg)
        });
        std::terminate();  // feval(error) does not return; this is a backstop.
    }

    RandBLAS::RNGState<RNG> read_state(const Array& state_arr) {
        if (state_arr.getType() != ArrayType::STRUCT) {
            raise("randlapack:rsvd_mex:state",
                "state must be a struct with .counter (uint32[4]) and .key (uint32[2])");
        }
        StructArray s = state_arr;
        if (s.getNumberOfElements() == 0) {
            raise("randlapack:rsvd_mex:state", "state struct must be nonempty");
        }
        Array ctr_field = s[0]["counter"];
        Array key_field = s[0]["key"];
        if (ctr_field.getType() != ArrayType::UINT32
            || ctr_field.getNumberOfElements() != CTR_LEN) {
            raise("randlapack:rsvd_mex:state",
                "state.counter must be a uint32 array of length 4");
        }
        if (key_field.getType() != ArrayType::UINT32
            || key_field.getNumberOfElements() != KEY_LEN) {
            raise("randlapack:rsvd_mex:state",
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
    void run_rsvd(ArgumentList& outputs, ArgumentList& inputs,
                  int64_t k_req, double tol_in, int64_t block_sz,
                  int64_t p, int64_t ppi)
    {
        const Array& A_in = inputs[0];
        auto dims = A_in.getDimensions();
        const int64_t m = static_cast<int64_t>(dims[0]);
        const int64_t n = static_cast<int64_t>(dims[1]);
        const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
        const T tol = static_cast<T>(tol_in);

        // Const view, then one bulk memcpy: binding a non-const TypedArray
        // would trigger MATLAB's copy-on-write unshare and deep-copy the
        // whole input before a single element is read. A must be copied
        // regardless -- the QB chain overwrites its input.
        const TypedArray<T> A_typed = A_in;
        const T* A_src = &*A_typed.cbegin();
        std::unique_ptr<T[]> A_work(new T[mn]);
        std::memcpy(A_work.get(), A_src, sizeof(T) * mn);

        auto state = read_state(inputs[6]);

        // The dependency-injection chain RSVD requires, assembled exactly as
        // RandLAPACK's own test does (test/drivers/test_rsvd.cc). All the
        // verbose/cond/orth diagnostic flags are off: they print to stdout
        // and compute extra condition numbers, neither of which belongs in a
        // library call.
        RandLAPACK::PLUL<T>      Stab(false, false);
        RandLAPACK::RS<T, RNG>   RS(Stab, p, ppi, false, false);
        RandLAPACK::CholQRQ<T>   Orth_RF(false, false);
        RandLAPACK::RF<T, RNG>   RF(RS, Orth_RF, false, false);
        RandLAPACK::CholQRQ<T>   Orth_QB(false, false);
        RandLAPACK::QB<T, RNG>   QB(RF, Orth_QB, false, false);
        RandLAPACK::RSVD<T, RNG> RSVD(QB, block_sz);

        // k is in-out: QB may return a smaller rank than requested.
        int64_t k = k_req;
        CallocGuard<T> g;
        int ret = RSVD.call(m, n, A_work.get(), k, tol, g.U, g.S, g.V, state);
        if (ret != 0) {
            raise("randlapack:rsvd_mex:returnCode",
                "RSVD returned non-zero status " + std::to_string(ret));
        }
        if (k <= 0 || g.U == nullptr || g.S == nullptr || g.V == nullptr) {
            raise("randlapack:rsvd_mex:emptyFactorization",
                "RSVD produced an empty factorization (k=" + std::to_string(k)
                + "); try a larger target rank or a looser tolerance");
        }

        const size_t ks = static_cast<size_t>(k);
        auto U_buf = factory.createBuffer<T>(static_cast<size_t>(m) * ks);
        auto S_buf = factory.createBuffer<T>(ks);
        auto V_buf = factory.createBuffer<T>(static_cast<size_t>(n) * ks);
        std::memcpy(U_buf.get(), g.U, sizeof(T) * static_cast<size_t>(m) * ks);
        std::memcpy(S_buf.get(), g.S, sizeof(T) * ks);
        std::memcpy(V_buf.get(), g.V, sizeof(T) * static_cast<size_t>(n) * ks);

        outputs[0] = factory.createArrayFromBuffer<T>(
            {static_cast<size_t>(m), ks}, std::move(U_buf));
        outputs[1] = factory.createArrayFromBuffer<T>(
            {ks, 1}, std::move(S_buf));
        outputs[2] = factory.createArrayFromBuffer<T>(
            {static_cast<size_t>(n), ks}, std::move(V_buf));

        if (outputs.size() >= 4) {
            outputs[3] = write_state(state);
        }
    }

    // Read a scalar that MATLAB may have handed us as int64 or double.
    int64_t read_int64(const Array& a, const char* name) {
        if (a.getNumberOfElements() != 1) {
            raise(std::string("randlapack:rsvd_mex:") + name,
                  std::string(name) + " must be a scalar");
        }
        switch (a.getType()) {
            case ArrayType::INT64:  return TypedArray<int64_t>(a)[0];
            case ArrayType::DOUBLE: return static_cast<int64_t>(TypedArray<double>(a)[0]);
            case ArrayType::SINGLE: return static_cast<int64_t>(TypedArray<float>(a)[0]);
            default:
                raise(std::string("randlapack:rsvd_mex:") + name,
                      std::string(name) + " must be a numeric scalar");
        }
    }

    double read_double(const Array& a, const char* name) {
        if (a.getNumberOfElements() != 1) {
            raise(std::string("randlapack:rsvd_mex:") + name,
                  std::string(name) + " must be a scalar");
        }
        switch (a.getType()) {
            case ArrayType::DOUBLE: return TypedArray<double>(a)[0];
            case ArrayType::SINGLE: return static_cast<double>(TypedArray<float>(a)[0]);
            case ArrayType::INT64:  return static_cast<double>(TypedArray<int64_t>(a)[0]);
            default:
                raise(std::string("randlapack:rsvd_mex:") + name,
                      std::string(name) + " must be a numeric scalar");
        }
    }

public:
    MexFunction() : matlabPtr(getEngine()) {}

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        try {
            if (inputs.size() != 7) {
                raise("randlapack:rsvd_mex:nargin",
                    "Expected 7 input arguments: A, k, tol, block_sz, p, ppi, state");
            }
            if (outputs.size() < 3 || outputs.size() > 4) {
                raise("randlapack:rsvd_mex:nargout",
                    "Expected 3 or 4 output arguments");
            }

            const Array& A_in = inputs[0];
            if (A_in.getDimensions().size() != 2) {
                raise("randlapack:rsvd_mex:A",
                    "A must be a 2D real numeric matrix (single or double)");
            }
            const ArrayType A_type = A_in.getType();
            if (A_type != ArrayType::DOUBLE && A_type != ArrayType::SINGLE) {
                raise("randlapack:rsvd_mex:type", "A must be single or double");
            }

            const int64_t k        = read_int64(inputs[1], "k");
            const double  tol      = read_double(inputs[2], "tol");
            const int64_t block_sz = read_int64(inputs[3], "block_sz");
            const int64_t p        = read_int64(inputs[4], "p");
            const int64_t ppi      = read_int64(inputs[5], "passes_per_iteration");

            if (k <= 0)
                raise("randlapack:rsvd_mex:k", "k must be a positive integer");
            if (block_sz <= 0)
                raise("randlapack:rsvd_mex:block_sz",
                      "block_sz must be a positive integer");
            if (p < 0)
                raise("randlapack:rsvd_mex:p", "p must be nonnegative");
            if (ppi <= 0)
                raise("randlapack:rsvd_mex:passes_per_iteration",
                      "passes_per_iteration must be a positive integer");
            if (tol < 0.0)
                raise("randlapack:rsvd_mex:tol", "tol must be nonnegative");

            if (A_type == ArrayType::DOUBLE) {
                run_rsvd<double>(outputs, inputs, k, tol, block_sz, p, ppi);
            } else {
                run_rsvd<float>(outputs, inputs, k, tol, block_sz, p, ppi);
            }
        } catch (const RandLAPACK::Error& e) {
            raise("randlapack:rsvd_mex:RandLAPACKError", e.what());
        } catch (const RandBLAS::Error& e) {
            raise("randlapack:rsvd_mex:RandBLASError", e.what());
        } catch (const matlab::Exception&) {
            // MATLAB exceptions (typically from feval(error)) propagate through
            // unchanged so MATLAB sees the original error ID and message.
            throw;
        } catch (const std::exception& e) {
            raise("randlapack:rsvd_mex:StdError", e.what());
        } catch (...) {
            raise("randlapack:rsvd_mex:Unknown", "Unknown exception in rsvd_mex");
        }
    }
};
