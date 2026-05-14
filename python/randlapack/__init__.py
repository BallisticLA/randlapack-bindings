"""Python bindings for selected RandLAPACK drivers.

Currently exposes BQRRP (randomized blocked QR with column pivoting). The
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

__all__ = ["bqrrp", "__version__"]


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
