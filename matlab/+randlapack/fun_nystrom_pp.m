function [est, t1, t2, times] = fun_nystrom_pp(A, k, Omega2, varargin)
%FUN_NYSTROM_PP  Trace estimator tr(f(A)) via RandLAPACK FunNystromPP.
%
%   est = randlapack.fun_nystrom_pp(A, k, s, ...)
%   est = randlapack.fun_nystrom_pp(A, k, Omega2, ...)   % explicit Phase-2 probes
%   [est, t1, t2, times] = randlapack.fun_nystrom_pp(...)
%       the 4th output is a wall-clock instrumentation struct (fields:
%       marshal_in_ms, phase1_ms, phase2_ms, fafun_ms, assembly_ms,
%       specrec_ms, nystrom_us (1x11), lfa_us (1x5)) for performance
%       breakdowns. See fun_nystrom_pp_mex.cc for slot definitions.
%
%   A        n x n matrix, single or double (symmetric; both triangles are
%            used by the sparse sketch application — the MEX mirrors
%            upper->lower internally).
%   k        Phase-1 rank (SCALAR). The Phase-1 sketch is a SparseStack/SASO
%            generated INSIDE RandLAPACK::NystromEVD from (SketchSeed, VecNnz),
%            matching the paper's Algorithm 1 line 1. Explicit Omega1 matrices
%            are no longer accepted (the dense-sketch mode was removed from
%            the kernel).
%   arg3     Phase-2 Hutchinson probes. Pass a SCALAR s and the n x s Gaussian
%            probes are sampled internally (seed = SketchSeed + 1000); OR pass
%            an explicit n x s matrix (escape hatch for validation, e.g.
%            feeding every estimator in a comparison the SAME probes).
%            Skipped when k == n (then arg3 may be a 0/[] placeholder).
%
%   Optional Name-Value pairs:
%     'Func'        char in {'sqrt', 'log', 'poly', 'effdim', 'square', 'identity'}
%                   (default 'sqrt'). 'poly' is f(x) = x(x + PolyLambda);
%                   'effdim' is f(x) = x/(x + PolyLambda) (the "effective
%                   dimension" function; operator monotone).
%     'Q'           subspace-iter count (default 2)
%     'PolyLambda'  lambda used by Func 'poly' and 'effdim' (default 10)
%     'LFAType'     {'exact', 'scalar', 'block', 'block_qfa'} oracle for f(A)*X
%                   (default 'block' — the matrix-free Krylov oracle).
%                   'block_qfa' = block Lanczos-QFA: forms the s×s quadratic
%                   form Ω₂ᵀf(A)Ω₂ directly (no f(A)·Ω₂ mapback); cheapest.
%                   'exact' builds a full eigendecomposition of A (O(n^3)) and
%                   is intended for validation/reference use, not production.
%                   'scalar' runs one Lanczos recurrence per probe (equivalent
%                   to Lanczos quadrature on each quadratic form).
%     'Depth'       Lanczos depth for 'scalar' / 'block' LFAType
%                   (default 200 for scalar, 20 for block; ignored for 'exact')
%     'Sketch'      DEPRECATED/IGNORED (default 'saso'). The Phase-1 sketch is
%                   always the kernel-internal SASO; anything other than 'saso'
%                   triggers a warning from the MEX. Kept so existing call
%                   sites keep running.
%     'VecNnz'      nonzeros per column of the SASO sketch (default 8)
%     'SketchSeed'  RNG seed for the Phase-1 sketch (default 42)
%
%   Returns:
%     est   trace estimate t1 + t2
%     t1    Phase 1 contribution sum f(lambda_hat_i)
%     t2    Phase 2 Hutchinson correction (0 when k == n)
%
%   Algorithm: two-phase funNystrom++ (rank-k shifted Nystrom approximation
%   per arXiv:2508.21189 Alg. 2 + Hutchinson correction on the residual).
%   MEX wrapper over RandLAPACK::FunNystromPP<T>, T matching the class of A.
%   NB the Phase-1 sketch is drawn by RandBLAS inside the kernel, so runs are
%   reproducible at fixed SketchSeed but are NOT sketch-for-sketch identical
%   to the Persson MATLAB reference (which draws its own Gaussian sketch).

    % --- Required-argument validation (friendly MATLAB-side errors) ---
    validateattributes(A, {'single', 'double'}, {'2d', 'square', 'real', 'finite'}, ...
                       mfilename, 'A', 1);
    cls = class(A);
    n   = size(A, 1);

    if ~isscalar(k)
        error('randlapack:fun_nystrom_pp:Omega1Explicit', ...
              ['arg 2 must be a scalar rank k. Explicit Omega1 matrices are no ' ...
               'longer supported: the Phase-1 sketch is a SparseStack/SASO ' ...
               'generated inside RandLAPACK (control it via ''SketchSeed'' and ' ...
               '''VecNnz'').']);
    end
    validateattributes(k, {'numeric'}, {'integer', 'positive', '<=', n}, ...
                       mfilename, 'k', 2);
    k  = double(k);
    a1 = k;

    if k == n
        a2 = double(0);                 % Phase 2 skipped; MEX leaves arg 3 unread
    elseif isscalar(Omega2)
        validateattributes(Omega2, {'numeric'}, {'integer', 'positive'}, ...
                           mfilename, 's', 3);
        a2 = double(Omega2);
    elseif isempty(Omega2)
        error('randlapack:fun_nystrom_pp:Omega2Empty', ...
              ['Omega2/s may be empty only when k == n (Phase 2 skipped). ' ...
               'Here k = %d and n = %d; pass a scalar s or an n x s sketch.'], k, n);
    else
        validateattributes(Omega2, {'single', 'double'}, {'2d', 'real', 'finite'}, ...
                           mfilename, 'Omega2', 3);
        if size(Omega2, 1) ~= n
            error('randlapack:fun_nystrom_pp:Omega2Shape', ...
                  'Omega2 must have n = %d rows to match A; got %d.', n, size(Omega2, 1));
        end
        a2 = cast(Omega2, cls);
    end

    % --- Name-Value options ---
    p = inputParser;
    addParameter(p, 'Func',       'sqrt',     @(x) ischar(x) || isstring(x));
    addParameter(p, 'Q',          2,          @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'PolyLambda', 10,         @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'LFAType',    'block',    @(x) ischar(x) || isstring(x));
    addParameter(p, 'Depth',      [],         @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
    addParameter(p, 'Sketch',     'saso',     @(x) ischar(x) || isstring(x));
    addParameter(p, 'VecNnz',     8,          @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'SketchSeed', 42,         @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'Reorth',     1,          @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'Adaptive',   0,          @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'AdaptiveTol',1e-2,       @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, varargin{:});

    func     = char(p.Results.Func);
    q        = double(p.Results.Q);
    pl       = double(p.Results.PolyLambda);
    lfa_type = char(p.Results.LFAType);
    sketch   = char(p.Results.Sketch);
    vec_nnz  = double(p.Results.VecNnz);
    sk_seed  = double(p.Results.SketchSeed);
    reorth   = double(p.Results.Reorth);
    adaptive = double(p.Results.Adaptive);        % block_qfa only; Depth is the cap
    adapt_tol = double(p.Results.AdaptiveTol);
    if isempty(p.Results.Depth)
        if any(strcmp(lfa_type, {'block', 'block_qfa'})), d = 20; else, d = 200; end
    else
        d = double(p.Results.Depth);
    end

    % A is templated on by the MEX; keep it in the working precision.
    A = cast(A, cls);

    if nargout <= 1
        est = fun_nystrom_pp_mex(A, a1, a2, func, q, pl, lfa_type, d, ...
                                 sketch, vec_nnz, sk_seed, reorth, adaptive, adapt_tol);
    elseif nargout == 2
        [est, t1] = fun_nystrom_pp_mex(A, a1, a2, func, q, pl, lfa_type, d, ...
                                       sketch, vec_nnz, sk_seed, reorth, adaptive, adapt_tol);
    elseif nargout == 3
        [est, t1, t2] = fun_nystrom_pp_mex(A, a1, a2, func, q, pl, lfa_type, d, ...
                                           sketch, vec_nnz, sk_seed, reorth, adaptive, adapt_tol);
    else
        [est, t1, t2, times] = fun_nystrom_pp_mex(A, a1, a2, func, q, pl, lfa_type, d, ...
                                                  sketch, vec_nnz, sk_seed, reorth, adaptive, adapt_tol);
    end
end
