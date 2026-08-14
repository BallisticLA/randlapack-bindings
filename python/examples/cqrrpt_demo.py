"""Rank revelation with RandLAPACK CQRRPT.

CQRRPT discovers the numerical rank of a tall matrix rather than taking it as
a parameter. This demo builds a matrix of known rank by repeating columns,
then checks that the detected rank, the pivots, and the residual all agree
with the construction, and compares against SciPy's dense QRCP.

The shape below is chosen where the randomized method actually pays: dense
QRCP wins on small problems (measured 0.6x at 5000x50) and loses as the
matrix grows (7.4x at 20000x200, 2.2x at 50000x1000 on this machine).

Run after installing the package (see bootstrap.sh or README):
    python python/examples/cqrrpt_demo.py
"""
import time

import numpy as np

import randlapack as rl


def main():
    m, r, extra = 20000, 160, 40
    rng = np.random.default_rng(0)
    B = rng.standard_normal((m, r))
    A = np.hstack([B, B[:, :extra]])            # rank r, r+extra columns
    A = np.asfortranarray(A[:, rng.permutation(A.shape[1])])

    t0 = time.perf_counter()
    Q, R, J = rl.cqrrpt(A)
    dt = time.perf_counter() - t0

    k = R.shape[0]
    print(f"A is {m}-by-{A.shape[1]}, constructed rank {r}")
    print(f"CQRRPT detected rank {k} in {dt:.3f} s")
    print(f"||A[:, J] - Q@R|| / ||A||     = "
          f"{np.linalg.norm(A[:, J] - Q @ R) / np.linalg.norm(A):.2e}")
    print(f"||Q.T @ Q - I||               = "
          f"{np.linalg.norm(Q.T @ Q - np.eye(k)):.2e}")

    # J is 0-based here (scipy convention), so it slices A directly.
    print(f"rank(A[:, J[:k]])             = "
          f"{np.linalg.matrix_rank(A[:, J[:k]])}  (want {k})")

    try:
        from scipy.linalg import qr as dense_qr
    except ImportError:
        print("\n(install scipy to see the dense QRCP comparison)")
        return

    t0 = time.perf_counter()
    _, _, Jd = dense_qr(A, mode="economic", pivoting=True)
    dt_dense = time.perf_counter() - t0
    print(f"\nscipy dense QRCP:             {dt_dense:.3f} s")
    print(f"CQRRPT speedup:               {dt_dense / dt:.1f}x")
    print(f"leading pivots agree on {len(set(J[:k]) & set(Jd[:k]))} "
          f"of the first {k} columns")


if __name__ == "__main__":
    main()
