"""Low-rank approximation with RandLAPACK RSVD.

Builds a matrix with a known decaying spectrum, then compares the randomized
rank-k approximation against the optimal (truncated SVD) one across a sweep
of k. Eckart-Young says the truncated SVD is optimal, so the interesting
quantity is the RATIO: how much the randomized method gives up for its speed.

Run after installing the package (see bootstrap.sh or README):
    python python/examples/rsvd_demo.py
"""
import time

import numpy as np

import randlapack as rl


def main():
    m, n = 2000, 500
    rng = np.random.default_rng(0)
    Q1, _ = np.linalg.qr(rng.standard_normal((m, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sv = 0.93 ** np.arange(n)               # geometric decay, known exactly
    A = np.asfortranarray((Q1 * sv) @ Q2.T)
    norm_A = np.linalg.norm(A)

    sv_true = np.linalg.svd(A, compute_uv=False)

    print(f"{'k':>4}  {'randomized':>12}  {'optimal':>12}  {'ratio':>8}  {'time (s)':>8}")
    for k in (10, 25, 50, 100, 200):
        t0 = time.perf_counter()
        U, s, V = rl.rsvd(A, k)
        dt = time.perf_counter() - t0

        k_out = s.shape[0]                  # may be < k if tol fired
        rand_err = np.linalg.norm(A - U @ np.diag(s) @ V.T) / norm_A
        # The optimal rank-k_out error is the tail of the true spectrum.
        opt_err = np.linalg.norm(sv_true[k_out:]) / norm_A

        print(f"{k_out:>4}  {rand_err:>12.4e}  {opt_err:>12.4e}  "
              f"{rand_err / opt_err:>8.3f}  {dt:>8.3f}")

    _, s, _ = rl.rsvd(A, 50)
    print("\nLeading singular values (randomized vs true):")
    for i in range(5):
        rel = abs(s[i] - sv_true[i]) / sv_true[i]
        print(f"  sigma_{i}: {s[i]:10.6f} vs {sv_true[i]:10.6f}  (rel err {rel:.2e})")


if __name__ == "__main__":
    main()
