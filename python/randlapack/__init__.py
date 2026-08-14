"""Python bindings for selected RandLAPACK drivers.

Currently exposes BQRRP (randomized blocked QR with column pivoting),
CQRRPT (randomized rank-revealing QR for tall matrices), and RSVD
(randomized truncated SVD). The
Python API surface mirrors the MATLAB binding (randlapack-bindings/matlab/
+randlapack/) where reasonable, but follows Python ecosystem conventions
where they diverge:

* ``J`` (pivot vector) is **0-based**, matching ``scipy.linalg.qr(A, pivoting=True)``.
  BQRRP stores pivots 1-based natively; the binding converts on return.
* Input layout (``A``) is auto-coerced to Fortran-order (column-major) via
  ``np.asfortranarray`` if the caller passes a C-order array. This matches
  ``scipy.linalg.qr``'s convention.
"""
import warnings

import numpy as np

from . import _randlapack as _impl

__version__ = "0.0.1"

__all__ = ["bqrrp", "cqrrpt", "rsvd", "__version__"]


def bqrrp(A, b_sz=None, d_factor=None, state=None, mode="explicit",
          return_state=False):
    """Randomized blocked QR with column pivoting.

    Parameters
    ----------
    A : ndarray, shape (m, n), float32 or float64
        Input matrix. C-order arrays are auto-converted to Fortran-order
        (one extra copy) since BQRRP requires column-major internally.
    b_sz : int, optional
        Block size. Default ``min(m, n)``; pass a smaller value to actually
        exercise the blocked code path.
    d_factor : float, optional
        Sketch embedding factor (must be >= 1.0). Default 1.25.
    state : None, int, or dict, optional
        RNG state for the Philox4x32 generator.
        * ``None`` (default): start from the all-zero state.
        * ``int``: use as a 32-bit seed; placed in the second key slot.
        * ``dict``: must have keys ``'counter'`` (uint32, length 4) and
          ``'key'`` (uint32, length 2). Typically a state returned from a
          previous call with ``return_state=True``.
    mode : {'explicit', 'implicit'}
        Output form.

        * ``'explicit'`` (default): returns ``(Q, R, J)``. ``Q`` is m-by-k
          orthogonal, ``R`` is k-by-n upper-triangular, k = min(m, n).
          Matches ``scipy.linalg.qr(A, pivoting=True)``.
        * ``'implicit'``: returns ``(A_out, tau, J)``. ``A_out`` is m-by-n
          with Householder vectors below the diagonal and ``R`` in and above
          the diagonal (GEQP3-format); ``tau`` is the length-n vector of
          Householder scalars. Skips the ``lapack.ungqr`` cost of forming
          ``Q`` explicitly.

    return_state : bool, optional
        If True, returns a 4-tuple appending the advanced RNG state as a
        dict. Default False; the 3-tuple form is more ergonomic for one-shot
        calls.

    Returns
    -------
    out1, out2, J : ndarray
        See ``mode`` for the meaning of ``out1`` and ``out2``. ``J`` is a
        length-n int64 0-based permutation vector such that
        ``A[:, J] == Q @ R`` (explicit mode) up to numerical error.
    state_out : dict, only if return_state=True
        Dict with ``'counter'`` (uint32 length 4) and ``'key'`` (uint32
        length 2). Pass it back as ``state=...`` for reproducible chained
        calls.

    Notes
    -----
    Conventions follow the Python ecosystem and may differ from the MATLAB
    binding in this repo:

    * ``J`` is **0-based**; ``scipy.linalg.qr(A, pivoting=True)``-style. The
      MATLAB binding's ``J`` is 1-based.
    * Input layout: column-major preferred; C-order auto-converted via
      ``np.asfortranarray``. ``scipy.linalg.qr`` accepts either; BQRRP itself
      needs column-major.

    Examples
    --------
    >>> import numpy as np, randlapack as rl
    >>> A = np.random.default_rng(0).standard_normal((2000, 200))
    >>> Q, R, J = rl.bqrrp(A)
    >>> err = np.linalg.norm(A[:, J] - Q @ R, 'fro') / np.linalg.norm(A, 'fro')
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D; got {A.ndim}D")
    if A.size == 0:
        raise ValueError("A must be nonempty")
    if A.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"A must be float32 or float64; got dtype {A.dtype}")

    # Auto-convert to Fortran-order. scipy.linalg.qr convention: accept any
    # layout, copy to column-major internally. BQRRP needs column-major.
    if not A.flags.f_contiguous:
        A = np.asfortranarray(A)

    m, n = A.shape

    if b_sz is None:
        b_sz = min(m, n)
    b_sz = int(b_sz)
    if b_sz <= 0:
        raise ValueError(f"b_sz must be positive; got {b_sz}")
    if b_sz > min(m, n):
        warnings.warn(
            f"b_sz={b_sz} exceeds min(m, n)={min(m, n)}; "
            "effective block size will be clipped at the matrix boundary.",
            stacklevel=2,
        )

    if d_factor is None:
        d_factor = 1.25
    d_factor = float(d_factor)
    if d_factor < 1.0:
        raise ValueError(f"d_factor must be >= 1.0; got {d_factor}")

    if mode not in ("explicit", "implicit"):
        raise ValueError(
            f"mode must be 'explicit' or 'implicit'; got {mode!r}")

    counter, key = _normalize_state(state)

    out1, out2, J, state_out = _impl.bqrrp(
        A, b_sz, d_factor, counter, key, mode)

    if return_state:
        return out1, out2, J, state_out
    return out1, out2, J


def _normalize_state(state):
    """Return (counter, key) as uint32 ndarrays of length 4, 2."""
    if state is None:
        return (np.zeros(4, dtype=np.uint32), np.zeros(2, dtype=np.uint32))
    if isinstance(state, (int, np.integer)):
        seed = np.uint32(int(state) & 0xFFFFFFFF)
        return (np.zeros(4, dtype=np.uint32),
                np.array([0, seed], dtype=np.uint32))
    if isinstance(state, dict):
        if "counter" not in state or "key" not in state:
            raise ValueError(
                "state dict must have keys 'counter' and 'key'")
        counter = np.ascontiguousarray(state["counter"], dtype=np.uint32)
        key = np.ascontiguousarray(state["key"], dtype=np.uint32)
        if counter.shape != (4,):
            raise ValueError(
                f"state['counter'] must be shape (4,); got {counter.shape}")
        if key.shape != (2,):
            raise ValueError(
                f"state['key'] must be shape (2,); got {key.shape}")
        return counter, key
    raise TypeError(
        f"state must be None, int, or dict; got {type(state).__name__}")


def cqrrpt(A, d_factor=None, eps=None, state=None, return_state=False):
    """Randomized rank-revealing QR with column pivoting, for tall matrices.

    Computes ``Q``, ``R`` and a pivot vector ``J`` such that
    ``A[:, J] ~= Q @ R``, with ``Q`` (m, k) orthonormal and ``R`` (k, n)
    upper-trapezoidal.

    ``k`` is *discovered*, not requested: CQRRPT reveals the numerical rank
    of ``A``, and that rank is the inner dimension of the outputs. On a
    full-rank input ``k == n``. Read it off ``R.shape[0]``.

    What is guaranteed, and what is not: ``Q`` is orthonormal and the
    *leading* ``k`` pivoted columns always reproduce to machine precision,
    ``A[:, J[:k]] == Q @ R[:, :k]``. The full identity ``A[:, J] == Q @ R``
    additionally requires ``k`` to have reached the numerical rank -- the
    ordinary case, including on rank-deficient input, but it can fall short
    when the internal Cholesky QR breaks down and the conservative fallback
    rank estimate takes over.

    CQRRPT sketches ``A`` down to ``d = d_factor * n`` rows, so it requires
    ``m >= d_factor * n``. For wide or square matrices, use `bqrrp`.

    Parameters
    ----------
    A : ndarray, shape (m, n), float32 or float64
        Input matrix, tall. C-order arrays are auto-converted to
        Fortran-order (one extra copy).
    d_factor : float, optional
        Sketch embedding factor (>= 1.0). Default 1.25. Larger values sketch
        more rows: steadier rank detection, more work, and a taller input
        requirement.
    eps : float, optional
        Conditioning tolerance for rank detection. Default
        ``sqrt(finfo(A.dtype).eps)``.

        ``eps`` is *not* a singular-value truncation threshold. CQRRPT first
        truncates using a fixed machine-epsilon cutoff, so columns well above
        machine precision are kept whatever ``eps`` says. It takes effect
        only in the re-estimation that runs when the internal Cholesky QR
        fails, where the rank is cut once the ``R`` diagonal spans a ratio of
        ``sqrt(eps / finfo.eps)``. It therefore bounds the orthogonality
        loss, which for Cholesky QR scales like ``u * cond(R)**2``.
    state : None, int, or dict, optional
        RNG state; see `bqrrp`.
    return_state : bool, optional
        If True, append the advanced RNG state to the returned tuple.

    Returns
    -------
    Q : ndarray, shape (m, k)
    R : ndarray, shape (k, n)
    J : ndarray, shape (n,), int64
        **0-based** pivot indices, matching
        ``scipy.linalg.qr(A, pivoting=True)``.
    state_out : dict, only if ``return_state=True``

    Examples
    --------
    >>> import numpy as np, randlapack as rl
    >>> B = np.random.default_rng(0).standard_normal((5000, 40))
    >>> A = np.hstack([B, B[:, :10]])          # 50 columns, rank 40
    >>> Q, R, J = rl.cqrrpt(A)
    >>> R.shape[0]                             # detected rank
    40
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D; got shape {A.shape}")
    if A.size == 0:
        raise ValueError("A must be nonempty")
    if A.dtype not in (np.float32, np.float64):
        raise TypeError(f"A must be float32 or float64; got dtype {A.dtype}")
    if not A.flags.f_contiguous:
        A = np.asfortranarray(A)

    m, n = A.shape

    if d_factor is None:
        d_factor = 1.25
    d_factor = float(d_factor)
    if d_factor < 1.0:
        raise ValueError(f"d_factor must be >= 1.0; got {d_factor}")

    if eps is None:
        # sqrt of machine epsilon in the working precision: the usual
        # tolerance scale for a Gram-based (Cholesky QR) method, whose
        # accuracy floor is the square root of the precision.
        eps = float(np.sqrt(np.finfo(A.dtype).eps))
    eps = float(eps)
    if eps < 0.0:
        raise ValueError(f"eps must be nonnegative; got {eps}")

    # Checked here as well as in the binding, so the common mistake is caught
    # before marshalling and the message can name the Python alternative.
    d = int(d_factor * n)
    if m < d:
        raise ValueError(
            f"CQRRPT requires m >= d_factor*n. Got m={m}, n={n}, "
            f"d_factor={d_factor} (needs m >= {d}). Lower d_factor toward "
            "1.0, or use randlapack.bqrrp, which has no tall requirement.")

    counter, key = _normalize_state(state)
    Q, R, J, state_out = _impl.cqrrpt(A, d_factor, eps, counter, key)

    if return_state:
        return Q, R, J, state_out
    return Q, R, J


def rsvd(A, k, tol=None, block_sz=None, p=None, passes_per_iteration=None,
         state=None, return_state=False):
    """Randomized truncated SVD.

    Computes a rank-``k`` (or lower) approximate SVD of ``A``, so that
    ``A ~= U @ np.diag(s) @ V.T``.

    Note the ``V``, not ``vh``: this follows MATLAB's and
    ``scipy.sparse.linalg.svds``' convention of returning the right singular
    vectors un-transposed, unlike ``numpy.linalg.svd``, which returns
    ``vh == V.T``. Transpose if you need the numpy form.

    The returned rank may be *smaller* than the ``k`` you asked for: the
    underlying QB factorization stops early once the residual falls below
    ``tol``. Read the actual rank off ``s.shape[0]``.

    ``U`` and ``V`` are orthonormal to *different* accuracies, and this is
    structural rather than incidental. ``V`` comes straight out of LAPACK's
    ``gesdd`` and is orthonormal to machine precision. ``U`` is formed from
    the QB chain's basis, which is orthogonalized by CholeskyQR, whose
    orthogonality loss scales like ``u * cond(A_sketch)**2``. On an
    ill-conditioned input ``U`` can be far looser: measured on a 400x200
    matrix with a geometric spectrum at k=30, ``||U.T @ U - I||`` was ~3e-11
    in float64 and ~1.5e-2 in float32, against ~7e-15 and ~3e-6 for ``V``.
    Raising ``p`` does not fix this. If you need an orthonormal ``U`` to
    machine precision on a badly conditioned input, re-orthogonalize it
    (e.g. ``np.linalg.qr``) or work in float64.

    Parameters
    ----------
    A : ndarray, shape (m, n), float32 or float64
        Input matrix. C-order arrays are auto-converted to Fortran-order.
    k : int
        Target rank; must be positive and at most ``min(m, n)``.
    tol : float, optional
        Residual tolerance for early termination. Default 0.0 (run to the
        requested rank).
    block_sz : int, optional
        QB block size. Default ``min(k, 32)``. Smaller blocks give the
        tolerance more chances to trigger; larger blocks do more work per
        BLAS-3 call.
    p : int, optional
        Power iterations in the sketching stage. Default 1. Raise it when
        the spectrum decays slowly.
    passes_per_iteration : int, optional
        Stabilization frequency. Default 1.
    state : None, int, or dict, optional
        RNG state; see `bqrrp`.
    return_state : bool, optional
        If True, append the advanced RNG state to the returned tuple.

    Returns
    -------
    U : ndarray, shape (m, k_out)
    s : ndarray, shape (k_out,)
        Singular values, descending.
    V : ndarray, shape (n, k_out)
        Right singular vectors, **un-transposed**.
    state_out : dict, only if ``return_state=True``

    Examples
    --------
    >>> import numpy as np, randlapack as rl
    >>> rng = np.random.default_rng(0)
    >>> A = rng.standard_normal((2000, 100)) @ rng.standard_normal((100, 500))
    >>> U, s, V = rl.rsvd(A, 100)
    >>> float(np.linalg.norm(A - U @ np.diag(s) @ V.T) / np.linalg.norm(A)) < 1e-10
    True
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D; got shape {A.shape}")
    if A.size == 0:
        raise ValueError("A must be nonempty")
    if A.dtype not in (np.float32, np.float64):
        raise TypeError(f"A must be float32 or float64; got dtype {A.dtype}")
    if not A.flags.f_contiguous:
        A = np.asfortranarray(A)

    m, n = A.shape

    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    if k > min(m, n):
        raise ValueError(f"k={k} exceeds min(m, n)={min(m, n)}")

    tol = 0.0 if tol is None else float(tol)
    if tol < 0.0:
        raise ValueError(f"tol must be nonnegative; got {tol}")

    # block_sz caps at 32 so the tolerance gets a chance to trigger on large
    # k, rather than the whole factorization arriving in a single block.
    block_sz = min(k, 32) if block_sz is None else int(block_sz)
    if block_sz <= 0:
        raise ValueError(f"block_sz must be positive; got {block_sz}")

    p = 1 if p is None else int(p)
    if p < 0:
        raise ValueError(f"p must be nonnegative; got {p}")

    ppi = 1 if passes_per_iteration is None else int(passes_per_iteration)
    if ppi <= 0:
        raise ValueError(
            f"passes_per_iteration must be positive; got {ppi}")

    counter, key = _normalize_state(state)
    U, s, V, state_out = _impl.rsvd(A, k, tol, block_sz, p, ppi, counter, key)

    if return_state:
        return U, s, V, state_out
    return U, s, V
