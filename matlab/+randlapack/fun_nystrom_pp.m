function [est, t1, t2, times] = fun_nystrom_pp(A, Omega1, Omega2, varargin)
%FUN_NYSTROM_PP  Trace estimator tr(f(A)) via RandLAPACK FunNystromPP.
%
%   est = randlapack.fun_nystrom_pp(A, k, s, ...)            % seed-driven
%   est = randlapack.fun_nystrom_pp(A, Omega1, Omega2, ...)  % explicit sketches
%   [est, t1, t2, times] = randlapack.fun_nystrom_pp(...)
%       the 4th output is a wall-clock instrumentation struct (fields:
%       marshal_in_ms, phase1_ms, phase2_ms, fafun_ms, assembly_ms,
%       specrec_ms, nystrom_us (1x11), lfa_us (1x5)) for performance
%       breakdowns. See fun_nystrom_pp_mex.cc for slot definitions.
%
%   A        n x n matrix, single or double (symmetric; upper triangle used).
%   arg2     Phase-1 sketch. Pass a SCALAR k (the rank) and the sketch is
%            sampled internally from SketchSeed (Gaussian, or SASO per Sketch);
%            OR pass an explicit n x k matrix to use as the sketch.
%   arg3     Phase-2 Hutchinson probes. Pass a SCALAR s and the n x s Gaussian
%            probes are sampled internally; OR pass an explicit n x s matrix.
%            Skipped when k == n (then arg3 may be a 0/[] placeholder).
%
%   The seed-driven form (sizes + SketchSeed) is the normal usage and matches
%   the RandLAPACK / BQRRP-binding convention. The explicit-matrix form is an
%   escape hatch for validation/reproducibility (e.g. matching a reference run
%   that fixed Omega1/Omega2), and lets a comparison study feed every estimator
%   the SAME randomness.
%
%   Optional Name-Value pairs:
%     'Func'        char in {'sqrt', 'log', 'poly', 'effdim', 'square', 'identity'}
%                   (default 'sqrt'). 'poly' is f(x) = x(x + PolyLambda);
%                   'effdim' is f(x) = x/(x + PolyLambda) (the "effective
%                   dimension" function; operator monotone).
%     'Q'           subspace-iter count (default 2)
%     'PolyLambda'  lambda used by Func 'poly' and 'effdim' (default 10)
%     'LFAType'     {'exact', 'scalar', 'block'} oracle for f(A)*X
%                   (default 'block' — the matrix-free Krylov oracle).
%                   'exact' builds a full eigendecomposition of A (O(n^3)) and
%                   is intended for validation/reference use, not production.
%                   'scalar' runs one Lanczos recurrence per probe (equivalent
%                   to Lanczos quadrature on each quadratic form).
%     'Depth'       Lanczos depth for 'scalar' / 'block' LFAType
%                   (default 200 for scalar, 20 for block; ignored for 'exact')
%     'Sketch'      {'gaussian', 'saso'} Phase-1 sketch type (default 'gaussian').
%                   'gaussian' uses the dense Omega1 you pass in. 'saso' draws a
%                   sparse sketching operator (RandBLAS SparseSkOp; the
%                   SparseStack-family sketch of the paper's Algorithm 1) inside
%                   the MEX — Omega1 is then read ONLY for its size n x k. Works
%                   at any Q >= 1 (Q = 1 densifies the sketch internally).
%     'VecNnz'      nonzeros per column of the SASO sketch (default 8)
%     'SketchSeed'  RNG seed for the SASO sketch (default 42)
%
%   Returns:
%     est   trace estimate t1 + t2
%     t1    Phase 1 contribution sum f(lambda_hat_i)
%     t2    Phase 2 Hutchinson correction (0 when k == n)
%
%   Algorithm: Persson-Kressner two-phase funNystrom++ (rank-k Nystrom
%   approximation + Hutchinson correction on the residual). MEX wrapper over
%   RandLAPACK::FunNystromPP<T>, T matching the class of A. Bit-identical to
%   the reference MATLAB implementation (Persson) for LFAType='exact'.

    % --- Required-argument validation (friendly MATLAB-side errors) ---
    validateattributes(A, {'single', 'double'}, {'2d', 'square', 'real', 'finite'}, ...
                       mfilename, 'A', 1);
    cls = class(A);
    n   = size(A, 1);

    % Args 2 and 3 are seed-driven: a SCALAR (the sketch size) samples the
    % sketch internally in the MEX; an explicit n x <cols> MATRIX is used as the
    % sketch (escape hatch for validation/reproducibility). a1/a2 are what we
    % forward to the MEX (scalar size as double, or matrix cast to class(A)).
    if isscalar(Omega1)
        validateattributes(Omega1, {'numeric'}, {'integer', 'positive', '<=', n}, ...
                           mfilename, 'k', 2);
        k = double(Omega1);  a1 = double(Omega1);
    else
        validateattributes(Omega1, {'single', 'double'}, {'2d', 'real', 'finite'}, ...
                           mfilename, 'Omega1', 2);
        if size(Omega1, 1) ~= n
            error('randlapack:fun_nystrom_pp:Omega1Shape', ...
                  'Omega1 must have n = %d rows to match A; got %d.', n, size(Omega1, 1));
        end
        k = size(Omega1, 2);  a1 = cast(Omega1, cls);
    end

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
    addParameter(p, 'Sketch',     'gaussian', @(x) ischar(x) || isstring(x));
    addParameter(p, 'VecNnz',     8,          @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'SketchSeed', 42,         @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'Reorth',     1,          @(x) isnumeric(x) && isscalar(x));
    parse(p, varargin{:});

    func     = char(p.Results.Func);
    q        = double(p.Results.Q);
    pl       = double(p.Results.PolyLambda);
    lfa_type = char(p.Results.LFAType);
    sketch   = char(p.Results.Sketch);
    vec_nnz  = double(p.Results.VecNnz);
    sk_seed  = double(p.Results.SketchSeed);
    reorth   = double(p.Results.Reorth);
    if isempty(p.Results.Depth)
        if strcmp(lfa_type, 'block'), d = 20; else, d = 200; end
    else
        d = double(p.Results.Depth);
    end
    % Sketch='saso' works at any Q >= 1: for Q == 1 (single-pass Nystrom) the
    % MEX densifies the sampled sparse sketch internally; for Q >= 2 it routes
    % the first matvec through the sparse SkOp path.

    % A is templated on by the MEX; keep it in the working precision. (a1/a2 were
    % already set above: scalar size as double, or explicit sketch cast to cls.)
    A = cast(A, cls);

    if nargout <= 1
        est = fun_nystrom_pp_mex(A, a1, a2, func, q, pl, lfa_type, d, ...
                                 sketch, vec_nnz, sk_seed, reorth);
    elseif nargout == 2
        [est, t1] = fun_nystrom_pp_mex(A, a1, a2, func, q, pl, lfa_type, d, ...
                                       sketch, vec_nnz, sk_seed, reorth);
    elseif nargout == 3
        [est, t1, t2] = fun_nystrom_pp_mex(A, a1, a2, func, q, pl, lfa_type, d, ...
                                           sketch, vec_nnz, sk_seed, reorth);
    else
        [est, t1, t2, times] = fun_nystrom_pp_mex(A, a1, a2, func, q, pl, lfa_type, d, ...
                                                  sketch, vec_nnz, sk_seed, reorth);
    end
end
