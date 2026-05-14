"""Wall-clock comparison of randlapack.bqrrp against scipy.linalg.qr(A, pivoting=True).

No warmup, no statistics; single timing per size. The point is a quick
directional signal, not a publication-grade measurement.
"""
import time

import numpy as np
import scipy.linalg as sla

import randlapack as rl


def main():
    sizes = [(1000, 100), (5000, 500), (20000, 1000)]
    rng = np.random.default_rng(0)
    print(f"{'size':>12} | {'qr piv (s)':>12} | {'bqrrp (s)':>12} | {'speedup':>10}")
    print("-" * 56)
    for m, n in sizes:
        A = rng.standard_normal((m, n))

        t0 = time.perf_counter()
        sla.qr(A, pivoting=True)
        t_scipy = time.perf_counter() - t0

        t0 = time.perf_counter()
        rl.bqrrp(A)
        t_rl = time.perf_counter() - t0

        print(f"{m:6d}x{n:>4d} | {t_scipy:12.3f} | {t_rl:12.3f} | "
              f"{t_scipy / t_rl:9.2f}x")


if __name__ == "__main__":
    main()
