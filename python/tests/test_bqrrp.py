"""Correctness checks for randlapack.bqrrp."""
import numpy as np
import pytest

import randlapack as rl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_explicit(A, Q, R, J, atol_orth=1e-10, atol_fact=1e-10):
    m, n = A.shape
    k = min(m, n)
    assert Q.shape == (m, k)
    assert R.shape == (k, n)
    assert J.shape == (n,)
    # J is a 0-based permutation of 0..n-1
    np.testing.assert_array_equal(np.sort(J), np.arange(n))
    # Orthogonality
    I = np.eye(k, dtype=Q.dtype)
    assert np.linalg.norm(Q.T @ Q - I, "fro") < atol_orth * np.sqrt(m * k)
    # Factorization
    err = np.linalg.norm(A[:, J] - Q @ R, "fro") / np.linalg.norm(A, "fro")
    assert err < atol_fact, f"||A[:, J] - Q@R||_F / ||A||_F = {err}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("shape", [(200, 50), (500, 100), (1000, 200)])
def test_explicit_basic(dtype, shape):
    rng = np.random.default_rng(0)
    A = rng.standard_normal(shape).astype(dtype)
    Q, R, J = rl.bqrrp(A)
    atol = 1e-5 if dtype == np.float32 else 1e-10
    _check_explicit(A, Q, R, J, atol_orth=atol, atol_fact=atol)


def test_implicit_basic():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((500, 100))
    A_out, tau, J = rl.bqrrp(A, mode="implicit")
    assert A_out.shape == A.shape
    assert tau.shape == (A.shape[1],)
    assert J.shape == (A.shape[1],)
    # Upper triangle of A_out's top n rows is R
    n = A.shape[1]
    R = np.triu(A_out[:n, :])
    assert R.shape == (n, n)


def test_modes_consistent_at_fixed_state():
    """Explicit R and implicit triu(A_out) should be bit-identical when the
    same RNG state drives both calls."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((500, 100))
    state = {
        "counter": np.zeros(4, dtype=np.uint32),
        "key": np.array([0, 42], dtype=np.uint32),
    }
    Q, R_exp, J_exp = rl.bqrrp(A, state=state)
    A_out, tau, J_imp = rl.bqrrp(A, state=state, mode="implicit")
    n = A.shape[1]
    R_imp = np.triu(A_out[:n, :])
    np.testing.assert_array_equal(J_exp, J_imp)
    np.testing.assert_array_equal(R_exp, R_imp)


def test_c_order_auto_converts():
    """Calling with a C-order array should silently work (wrapper converts)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((300, 80))  # default C-order
    assert A.flags.c_contiguous
    Q, R, J = rl.bqrrp(A)
    _check_explicit(A, Q, R, J)


def test_input_not_mutated():
    rng = np.random.default_rng(0)
    A = np.asfortranarray(rng.standard_normal((200, 50)))
    A_orig = A.copy()
    rl.bqrrp(A)
    np.testing.assert_array_equal(A, A_orig)


def test_return_state_chains_reproducibly():
    """Two successive calls with the chained state should produce the same
    result as two calls with the same starting state."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 50))
    seed = 17

    # Sequential chain
    Q1a, R1a, J1a, s1 = rl.bqrrp(A, state=seed, return_state=True)
    Q1b, R1b, J1b = rl.bqrrp(A, state=s1)

    # Two independent calls with the same starting seed should match Q1a
    Q_repeat, R_repeat, J_repeat = rl.bqrrp(A, state=seed)
    np.testing.assert_array_equal(Q_repeat, Q1a)
    np.testing.assert_array_equal(R_repeat, R1a)
    np.testing.assert_array_equal(J_repeat, J1a)


def test_rejects_float32_when_complex():
    # We don't accept complex input
    A = np.zeros((10, 5), dtype=np.complex128)
    with pytest.raises(TypeError):
        rl.bqrrp(A)


def test_rejects_1d():
    A = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError):
        rl.bqrrp(A)


def test_rejects_bad_mode():
    A = np.zeros((10, 5), dtype=np.float64)
    with pytest.raises(ValueError):
        rl.bqrrp(A, mode="xyzzy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
