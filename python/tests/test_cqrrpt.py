"""Correctness checks for randlapack.cqrrpt."""
import numpy as np
import pytest

import randlapack as rl


DTYPES = [np.float64, np.float32]


def tol_for(dtype, m, n):
    return np.sqrt(float(m * n)) * np.finfo(dtype).eps * 100


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(500, 50), (1000, 100), (2000, 200)])
def test_full_rank_factorization(dtype, shape):
    m, n = shape
    A = np.asfortranarray(
        np.random.default_rng(0).standard_normal(shape).astype(dtype))

    Q, R, J = rl.cqrrpt(A)

    # Full-rank input: the detected rank is n.
    assert R.shape[0] == n, f"expected rank {n}, got {R.shape[0]}"
    assert Q.shape == (m, n)
    assert J.shape == (n,)

    err = np.linalg.norm(A[:, J] - Q @ R) / np.linalg.norm(A)
    assert err < tol_for(dtype, m, n), f"||A[:,J] - QR||/||A|| = {err:g}"

    orth = np.linalg.norm(Q.T @ Q - np.eye(n, dtype=dtype))
    assert orth < tol_for(dtype, m, n), f"||Q^T Q - I|| = {orth:g}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_pivots_are_zero_based_permutation(dtype):
    A = np.asfortranarray(
        np.random.default_rng(1).standard_normal((800, 60)).astype(dtype))
    _, _, J = rl.cqrrpt(A)
    # 0-based, matching scipy.linalg.qr(pivoting=True) -- not the 1-based
    # form CQRRPT uses natively and the MATLAB binding preserves.
    assert np.array_equal(np.sort(J), np.arange(60))


@pytest.mark.parametrize("dtype", DTYPES)
def test_rank_detection(dtype):
    """Duplicated columns give an exactly known rank.

    Also a regression test for the zero-initialization of J: CQRRPT passes it
    to LAPACK's geqp3, which reads jpvt on entry, so an uninitialized buffer
    makes the pivoting depend on heap garbage and the detected rank wander.
    """
    m, r, extra = 2000, 40, 10
    rng = np.random.default_rng(2)
    B = rng.standard_normal((m, r)).astype(dtype)
    A = np.asfortranarray(np.hstack([B, B[:, :extra]]))

    Q, R, J = rl.cqrrpt(A)

    assert R.shape[0] == r, f"expected rank {r}, got {R.shape[0]}"
    err = np.linalg.norm(A[:, J] - Q @ R) / np.linalg.norm(A)
    assert err < tol_for(dtype, m, r + extra)


@pytest.mark.parametrize("dtype", DTYPES)
def test_leading_block_identity(dtype):
    """A[:, J[:k]] == Q @ R[:, :k] holds whatever rank came back."""
    m, n = 1500, 80
    A = np.asfortranarray(
        np.random.default_rng(3).standard_normal((m, n)).astype(dtype))
    Q, R, J = rl.cqrrpt(A)
    k = R.shape[0]
    err = np.linalg.norm(A[:, J[:k]] - Q @ R[:, :k]) / np.linalg.norm(A)
    assert err < tol_for(dtype, m, k)


def test_c_order_input_matches_fortran_order():
    """The wrapper's asfortranarray coercion must not change the result."""
    A_c = np.random.default_rng(4).standard_normal((1000, 50))
    A_f = np.asfortranarray(A_c)
    assert not A_c.flags.f_contiguous

    Q1, R1, J1 = rl.cqrrpt(A_c, state=5)
    Q2, R2, J2 = rl.cqrrpt(A_f, state=5)

    assert np.array_equal(J1, J2)
    assert np.allclose(Q1, Q2, rtol=0, atol=0)
    assert np.allclose(R1, R2, rtol=0, atol=0)


def test_reproducible_from_explicit_state():
    A = np.asfortranarray(
        np.random.default_rng(6).standard_normal((1000, 50)))
    st = {"counter": np.zeros(4, dtype=np.uint32),
          "key": np.array([0, 11], dtype=np.uint32)}
    Q1, R1, J1 = rl.cqrrpt(A, state=st)
    Q2, R2, J2 = rl.cqrrpt(A, state=st)
    assert np.array_equal(Q1, Q2)
    assert np.array_equal(R1, R2)
    assert np.array_equal(J1, J2)


def test_state_advances():
    A = np.asfortranarray(
        np.random.default_rng(7).standard_normal((1000, 50)))
    st = {"counter": np.zeros(4, dtype=np.uint32),
          "key": np.array([0, 11], dtype=np.uint32)}
    _, _, _, st_out = rl.cqrrpt(A, state=st, return_state=True)
    assert not (np.array_equal(st_out["counter"], st["counter"])
                and np.array_equal(st_out["key"], st["key"]))


def test_wide_matrix_rejected_with_a_useful_message():
    A = np.random.default_rng(8).standard_normal((50, 500))
    with pytest.raises(ValueError, match=r"m >= d_factor\*n"):
        rl.cqrrpt(A)


def test_square_matrix_rejected():
    # Square is already too short once d_factor > 1.
    A = np.random.default_rng(9).standard_normal((100, 100))
    with pytest.raises(ValueError, match=r"m >= d_factor\*n"):
        rl.cqrrpt(A)


def test_error_paths():
    A = np.random.default_rng(10).standard_normal((1000, 50))
    with pytest.raises(ValueError):
        rl.cqrrpt(np.zeros((0, 5)))
    with pytest.raises(ValueError):
        rl.cqrrpt(A, d_factor=0.5)
    with pytest.raises(ValueError):
        rl.cqrrpt(A, eps=-1.0)
    with pytest.raises(TypeError):
        rl.cqrrpt(A.astype(np.int32))
    with pytest.raises(ValueError):
        rl.cqrrpt(np.random.default_rng(11).standard_normal((10, 10, 10)))
