// MEX wrapper for RandLAPACK::BQRRP using MATLAB's C++ Data API (R2018a+).
//
// MATLAB-side signature (low-level; users normally call randlapack.bqrrp.m):
//
//   [out1, out2, J, state_out] = bqrrp_mex(A, b_sz, d_factor, state, mode)
//
// where mode is 'explicit' or 'implicit'.
//
// 'explicit' (default in the .m wrapper):
//   out1 = Q (m-by-k explicit orthogonal factor, k = min(m, n))
//   out2 = R (k-by-n upper-triangular factor)
//
// 'implicit':
//   out1 = A_out (m-by-n, BQRRP's native GEQP3-format: Householder vectors
//                  below the diagonal, R in and above the diagonal)
//   out2 = tau   (length-n vector of Householder scalars)
//
// Other outputs:
//   J          1-by-n vector of 1-based column-pivot indices (int64)
//   state_out  RNG state struct after advancing
//
// J is 1-based because BQRRP itself stores 1-based pivots internally
// (rl_bqrrp.hh:330: std::iota(..., 1)). No conversion in the MEX layer.

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <RandLAPACK.hh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using matlab::mex::ArgumentList;
using matlab::data::Array;
using matlab::data::ArrayFactory;
using matlab::data::ArrayType;
using matlab::data::CharArray;
using matlab::data::StructArray;
template <typename T> using TypedArray = matlab::data::TypedArray<T>;


class MexFunction : public matlab::mex::Function {
private:
    using RNG = RandBLAS::DefaultRNG;
    static constexpr size_t CTR_LEN = 4;
    static constexpr size_t KEY_LEN = 2;

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

    RandBLAS::RNGState<RNG> read_state(const Array& state_arr) {
        if (state_arr.getType() != ArrayType::STRUCT) {
            raise("randlapack:bqrrp_mex:state",
                "state must be a struct with .counter (uint32[4]) and .key (uint32[2])");
        }
        StructArray s = state_arr;
        if (s.getNumberOfElements() == 0) {
            raise("randlapack:bqrrp_mex:state", "state struct must be nonempty");
        }
        Array ctr_field = s[0]["counter"];
        Array key_field = s[0]["key"];
        if (ctr_field.getType() != ArrayType::UINT32
            || ctr_field.getNumberOfElements() != CTR_LEN) {
            raise("randlapack:bqrrp_mex:state",
                "state.counter must be a uint32 array of length 4");
        }
        if (key_field.getType() != ArrayType::UINT32
            || key_field.getNumberOfElements() != KEY_LEN) {
            raise("randlapack:bqrrp_mex:state",
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
    void run_bqrrp(ArgumentList& outputs, ArgumentList& inputs,
                   int64_t b_sz, double d_factor_in, bool implicit)
    {
        const Array& A_in = inputs[0];
        auto dims = A_in.getDimensions();
        const int64_t m = static_cast<int64_t>(dims[0]);
        const int64_t n = static_cast<int64_t>(dims[1]);
        const int64_t k = std::min(m, n);
        const T d_factor = static_cast<T>(d_factor_in);

        // BQRRP overwrites A; copy into a mutable scratch buffer (column-major,
        // matching MATLAB's storage layout).
        std::vector<T> A_work(static_cast<size_t>(m) * static_cast<size_t>(n));
        TypedArray<T> A_typed = A_in;
        {
            size_t idx = 0;
            for (const auto v : A_typed) {
                A_work[idx++] = v;
            }
        }

        std::vector<T> tau(std::max<int64_t>(1, n));
        std::vector<int64_t> J(std::max<int64_t>(1, n));

        auto state = read_state(inputs[3]);

        RandLAPACK::BQRRP<T, RNG> alg(false, b_sz);
        int ret = alg.call(m, n, A_work.data(), m, d_factor, tau.data(),
                           J.data(), state);
        if (ret != 0) {
            raise("randlapack:bqrrp_mex:returnCode",
                "BQRRP returned non-zero status " + std::to_string(ret));
        }

        if (implicit) {
            // out1 = A_out (m-by-n, GEQP3-format); out2 = tau (length n).
            auto A_buf = factory.createBuffer<T>(
                static_cast<size_t>(m) * static_cast<size_t>(n));
            std::memcpy(A_buf.get(), A_work.data(),
                        sizeof(T) * static_cast<size_t>(m)
                                  * static_cast<size_t>(n));
            outputs[0] = factory.createArrayFromBuffer<T>(
                {static_cast<size_t>(m), static_cast<size_t>(n)},
                std::move(A_buf));

            auto tau_buf = factory.createBuffer<T>(static_cast<size_t>(n));
            std::memcpy(tau_buf.get(), tau.data(),
                        sizeof(T) * static_cast<size_t>(n));
            outputs[1] = factory.createArrayFromBuffer<T>(
                {1, static_cast<size_t>(n)}, std::move(tau_buf));
        } else {
            // out1 = Q (m-by-k explicit); out2 = R (k-by-n).
            // Order matters: ungqr clobbers the upper triangle of A_work, so
            // copy R out first.
            auto R_buf = factory.createBuffer<T>(
                static_cast<size_t>(k) * static_cast<size_t>(n));
            T* R_data = R_buf.get();
            for (int64_t j = 0; j < n; ++j) {
                const int64_t lim = std::min(j + 1, k);
                for (int64_t i = 0; i < lim; ++i) {
                    R_data[i + j * k] = A_work[i + j * m];
                }
                for (int64_t i = lim; i < k; ++i) {
                    R_data[i + j * k] = static_cast<T>(0);
                }
            }
            outputs[1] = factory.createArrayFromBuffer<T>(
                {static_cast<size_t>(k), static_cast<size_t>(n)},
                std::move(R_buf));

            int64_t info = lapack::ungqr(m, k, k, A_work.data(), m, tau.data());
            if (info != 0) {
                raise("randlapack:bqrrp_mex:ungqr",
                    "lapack::ungqr returned info=" + std::to_string(info));
            }

            auto Q_buf = factory.createBuffer<T>(
                static_cast<size_t>(m) * static_cast<size_t>(k));
            std::memcpy(Q_buf.get(), A_work.data(),
                        sizeof(T) * static_cast<size_t>(m)
                                  * static_cast<size_t>(k));
            outputs[0] = factory.createArrayFromBuffer<T>(
                {static_cast<size_t>(m), static_cast<size_t>(k)},
                std::move(Q_buf));
        }

        // J: 1-based int64 row vector.
        auto J_buf = factory.createBuffer<int64_t>(static_cast<size_t>(n));
        std::memcpy(J_buf.get(), J.data(), sizeof(int64_t) * static_cast<size_t>(n));
        outputs[2] = factory.createArrayFromBuffer<int64_t>(
            {1, static_cast<size_t>(n)}, std::move(J_buf));

        if (outputs.size() >= 4) {
            outputs[3] = write_state(state);
        }
    }

public:
    MexFunction() : matlabPtr(getEngine()) {}

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        try {
            if (inputs.size() != 5) {
                raise("randlapack:bqrrp_mex:nargin",
                    "Expected 5 input arguments: A, b_sz, d_factor, state, mode");
            }
            if (outputs.size() < 3 || outputs.size() > 4) {
                raise("randlapack:bqrrp_mex:nargout",
                    "Expected 3 or 4 output arguments");
            }

            const Array& A_in = inputs[0];
            if (A_in.getDimensions().size() != 2) {
                raise("randlapack:bqrrp_mex:A",
                    "A must be a 2D real numeric matrix (single or double)");
            }
            const ArrayType A_type = A_in.getType();
            if (A_type != ArrayType::DOUBLE && A_type != ArrayType::SINGLE) {
                raise("randlapack:bqrrp_mex:type", "A must be single or double");
            }

            const Array& b_sz_arr = inputs[1];
            if (b_sz_arr.getNumberOfElements() != 1
                || b_sz_arr.getType() != ArrayType::INT64) {
                raise("randlapack:bqrrp_mex:b_sz",
                    "b_sz must be an int64 scalar");
            }
            const int64_t b_sz = TypedArray<int64_t>(b_sz_arr)[0];

            const Array& d_factor_arr = inputs[2];
            if (d_factor_arr.getNumberOfElements() != 1) {
                raise("randlapack:bqrrp_mex:d_factor", "d_factor must be scalar");
            }
            double d_factor;
            if (d_factor_arr.getType() == ArrayType::DOUBLE) {
                d_factor = static_cast<double>(TypedArray<double>(d_factor_arr)[0]);
            } else if (d_factor_arr.getType() == ArrayType::SINGLE) {
                d_factor = static_cast<double>(TypedArray<float>(d_factor_arr)[0]);
            } else {
                raise("randlapack:bqrrp_mex:d_factor",
                    "d_factor must be a single or double scalar");
            }

            const Array& mode_arr = inputs[4];
            if (mode_arr.getType() != ArrayType::CHAR) {
                raise("randlapack:bqrrp_mex:mode",
                    "mode must be a character vector ('explicit' or 'implicit')");
            }
            const std::string mode = CharArray(mode_arr).toAscii();
            bool implicit;
            if (mode == "explicit") {
                implicit = false;
            } else if (mode == "implicit") {
                implicit = true;
            } else {
                raise("randlapack:bqrrp_mex:mode",
                    "mode must be 'explicit' or 'implicit'; got '" + mode + "'");
            }

            if (A_type == ArrayType::DOUBLE) {
                run_bqrrp<double>(outputs, inputs, b_sz, d_factor, implicit);
            } else {
                run_bqrrp<float>(outputs, inputs, b_sz, d_factor, implicit);
            }
        } catch (const RandLAPACK::Error& e) {
            raise("randlapack:bqrrp_mex:RandLAPACKError", e.what());
        } catch (const RandBLAS::Error& e) {
            raise("randlapack:bqrrp_mex:RandBLASError", e.what());
        } catch (const matlab::Exception&) {
            // MATLAB exceptions (typically from feval(error)) propagate through
            // unchanged so MATLAB sees the original error ID and message.
            throw;
        } catch (const std::exception& e) {
            raise("randlapack:bqrrp_mex:StdError", e.what());
        } catch (...) {
            raise("randlapack:bqrrp_mex:Unknown",
                  "Unknown exception in bqrrp_mex");
        }
    }
};
