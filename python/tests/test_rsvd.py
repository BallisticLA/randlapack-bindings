"""Correctness checks for randlapack.rsvd."""
import numpy as np
import pytest

import randlapack as rl


DTYPES = [np.float64, np.float32]


def tol_for(dtype, m, n):
    return np.sqrt(float(m * n)) * np.finfo(dtype).eps * 100


def decaying_matrix(m, n, dtype, seed=0):
    """Geometric spectrum with orthogonal factors: the true singular values
    are known, so the optimal rank-k truncation is computable exactly."""
    rng = np.random.default_rng(seed)
    Q1, _ = np.linalg.qr(rng.standard_normal((m, min(m, n))))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, min(m, n))))
    s = 0.85 ** np.arange(min(m, n))
    return np.asfortranarray((Q1 * s) @ Q2.T).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_low_rank_reconstruction(dtype):
    m, n, r = 600, 300, 40
    rng = np.random.default_rng(0)
    A = np.asfortranarray(
        (rng.standard_normal((m, r)) @ rng.standard_normal((r, n))).astype(dtype))

    U, s, V = rl.rsvd(A, r)

    assert U.shape[0] == m and V.shape[0] == n
    k = s.shape[0]
    assert k <= r
    assert U.shape[1] == k and V.shape[1] == k

    err = np.linalg.norm(A - U @ np.diag(s) @ V.T) / np.linalg.norm(A)
    assert err < tol_for(dtype, m, n), f"rel err {err:g}"


def orthogonality_error(X):
    k = X.shape[1]
    return np.linalg.norm(
        X.T.astype(np.float64) @ X.astype(np.float64) - np.eye(k))


@pytest.mark.parametrize("dtype", DTYPES)
def test_factors_are_orthonormal_well_conditioned(dtype):
    """On a well-conditioned input, both factors are orthonormal to
    machine precision."""
    m, n, r = 600, 300, 40
    rng = np.random.default_rng(1)
    A = np.asfortranarray(
        (rng.standard_normal((m, r)) @ rng.standard_normal((r, n))).astype(dtype))
    U, s, V = rl.rsvd(A, r)
    k = s.shape[0]
    assert orthogonality_error(U) < tol_for(dtype, m, k)
    assert orthogonality_error(V) < tol_for(dtype, n, k)


@pytest.mark.parametrize("dtype", DTYPES)
def test_orthogonality_follows_the_cholesky_qr_model(dtype):
    """U and V are orthonormal to DIFFERENT accuracies, by construction.

    V comes straight out of LAPACK's gesdd, so it is orthonormal to machine
    precision whatever the conditioning. U is formed as Q @ UT, and Q comes
    from the QB chain's CholeskyQR orthogonalizer, whose orthogonality loss
    scales like u * cond(A_sketch)**2 -- not u. On an ill-conditioned input
    U is therefore much looser than V, and power iterations do not fix it
    (they improve the approximation while making the sketch no better
    conditioned).

    Measured here (400x200, geometric spectrum, k=30): U ~ 3e-11 vs V ~ 7e-15
    in float64, U ~ 1.5e-2 vs V ~ 3e-6 in float32. This test asserts the
    model, so it catches a genuine regression without failing on expected
    numerical behaviour.
    """
    A = decaying_matrix(400, 200, dtype, seed=1)
    U, s, V = rl.rsvd(A, 30)
    k = s.shape[0]
    u = np.finfo(dtype).eps
    kappa = float(s[0] / s[-1])

    # V: machine precision.
    assert orthogonality_error(V) < tol_for(dtype, A.shape[1], k)

    # U: bounded by the CholeskyQR model, with a factor-100 safety margin
    # for the constants the model omits.
    bound = 100 * u * kappa ** 2
    err = orthogonality_error(U)
    assert err < bound, f"||U^T U - I|| = {err:g} exceeds u*kappa^2 bound {bound:g}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_near_optimal_versus_truncated_svd(dtype):
    """Eckart-Young: the deterministic rank-k truncation is optimal, so the
    randomized method can only match it or lose. Bounding the loss is the
    real check; a fixed error threshold would only test the test matrix."""
    A = decaying_matrix(400, 200, dtype, seed=2)
    U, s, V = rl.rsvd(A, 30, p=2)
    k = s.shape[0]

    rand_err = np.linalg.norm(
        A.astype(np.float64) - (U @ np.diag(s) @ V.T).astype(np.float64))

    Us, ss, Vh = np.linalg.svd(A.astype(np.float64), full_matrices=False)
    opt = (Us[:, :k] * ss[:k]) @ Vh[:k, :]
    opt_err = np.linalg.norm(A.astype(np.float64) - opt)

    assert rand_err <= 1.5 * opt_err + 10 * np.finfo(dtype).eps, (
        f"randomized {rand_err:g} vs optimal {opt_err:g}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_singular_values_descending_and_accurate(dtype):
    A = decaying_matrix(400, 200, dtype, seed=3)
    _, s, _ = rl.rsvd(A, 30, p=2)
    assert np.all(s > 0)
    assert np.all(np.diff(s.astype(np.float64)) <= 1e-6 * float(s[0]))

    s_true = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    assert abs(float(s[0]) - s_true[0]) / s_true[0] < 0.05


def test_v_is_untransposed():
    """rsvd returns V, not vh. Verify against numpy's convention directly."""
    A = decaying_matrix(300, 150, np.float64, seed=4)
    U, s, V = rl.rsvd(A, 20, p=2)
    assert V.shape == (A.shape[1], s.shape[0])
    # Reconstruction works with V.T, and would be a shape error with V.
    approx = U @ np.diag(s) @ V.T
    assert approx.shape == A.shape


def test_c_order_input_matches_fortran_order():
    A_c = np.random.default_rng(5).standard_normal((600, 200))
    A_f = np.asfortranarray(A_c)
    assert not A_c.flags.f_contiguous

    U1, s1, V1 = rl.rsvd(A_c, 20, state=5)
    U2, s2, V2 = rl.rsvd(A_f, 20, state=5)

    assert np.array_equal(s1, s2)
    assert np.array_equal(U1, U2)
    assert np.array_equal(V1, V2)


def test_reproducible_from_explicit_state():
    A = np.asfortranarray(
        np.random.default_rng(6).standard_normal((500, 200)))
    st = {"counter": np.zeros(4, dtype=np.uint32),
          "key": np.array([0, 7], dtype=np.uint32)}
    U1, s1, V1 = rl.rsvd(A, 20, state=st)
    U2, s2, V2 = rl.rsvd(A, 20, state=st)
    assert np.array_equal(U1, U2)
    assert np.array_equal(s1, s2)
    assert np.array_equal(V1, V2)


def test_state_advances():
    A = np.asfortranarray(
        np.random.default_rng(7).standard_normal((500, 200)))
    st = {"counter": np.zeros(4, dtype=np.uint32),
          "key": np.array([0, 7], dtype=np.uint32)}
    _, _, _, st_out = rl.rsvd(A, 20, state=st, return_state=True)
    assert not (np.array_equal(st_out["counter"], st["counter"])
                and np.array_equal(st_out["key"], st["key"]))


def test_tolerance_can_truncate_below_the_requested_rank():
    """A loose tol lets QB stop early, so k_out < k. The returned shapes are
    the only place that rank is reported."""
    A = decaying_matrix(400, 200, np.float64, seed=8)
    _, s_tight, _ = rl.rsvd(A, 60, tol=0.0, block_sz=4)
    _, s_loose, _ = rl.rsvd(A, 60, tol=1e-2, block_sz=4)
    assert s_loose.shape[0] <= s_tight.shape[0]


def test_error_paths():
    A = np.random.default_rng(9).standard_normal((300, 100))
    with pytest.raises(ValueError):
        rl.rsvd(np.zeros((0, 5)), 1)
    with pytest.raises(ValueError):
        rl.rsvd(A, 500)             # k > min(m, n)
    with pytest.raises(ValueError):
        rl.rsvd(A, 0)
    with pytest.raises(ValueError):
        rl.rsvd(A, 10, tol=-1.0)
    with pytest.raises(ValueError):
        rl.rsvd(A, 10, block_sz=0)
    with pytest.raises(ValueError):
        rl.rsvd(A, 10, p=-1)
    with pytest.raises(TypeError):
        rl.rsvd(A.astype(np.int32), 10)
