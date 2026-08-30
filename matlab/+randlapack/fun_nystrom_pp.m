function [est, t1, t2, times] = fun_nystrom_pp(A, k, Omega2, varargin)
%FUN_NYSTROM_PP  Trace estimator tr(f(A)) via RandLAPACK FunNystromPP.
%
%   est = randlapack.fun_nystrom_pp(A, k, s, ...)
%   est = randlapack.fun_nystrom_pp(A, k, Omega2, ...)   % explicit Phase-2 probes
%   [est, t1, t2, times] = randlapack.fun_nystrom_pp(...)
%       the 4th output is a wall-clock instrumentation struct (fields:
%       marshal_in_ms, phase1_ms, phase2_ms, fafun_ms, assembly_ms,
%       specrec_ms, nystrom_us (1x11; only slots 1,2,3,7,11 populated, slot 7
%       = the whole spectral-recovery block), lfa_us (1x6; slot 6 = reorth
%       time, real for scalar/block/block_qfa, 0 for the basis-free
%       scalar_qfa/auto), d_used, oracle_mv, auto_k, auto_s, probe_mv, plus
%       path-specific fields that are NaN when their path did not run:
%       tr_U, tr_L, certified ('block_qfa' Gauss/Radau traces + bracket flag)
%       and probe_ms, probe_converged, phase2_certified ('auto' depth-probe
%       wall-clock + certification flags). See fun_nystrom_pp_mex.cc for the
%       full slot definitions.
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
%     'LFAType'     {'exact', 'scalar', 'scalar_qfa', 'block', 'block_qfa',
%                   'auto'} oracle for f(A)*X (default 'block' — the
%                   matrix-free Krylov oracle).
%                   'block_qfa' = block Lanczos-QFA: forms the s×s quadratic
%                   form Ω₂ᵀf(A)Ω₂ directly (no f(A)·Ω₂ mapback); cheapest.
%                   'Adaptive' selects its depth rule: 0 = fixed Depth;
%                   1 = Radau-certified stop (the block Gauss/Gauss-Radau
%                   bracket closes within 'AdaptiveTol' — the same certified
%                   meaning as scalar_qfa's adaptive mode; no delay window);
%                   2 = legacy window rule (honors AdaptiveDelay/AdaptiveMin).
%                   With Adaptive = 1, 'RadauReturn' picks the returned value
%                   (0 = block Gauss, 1 = (Gauss+Radau)/2 midpoint) and the
%                   times struct reports the tr_U >= tr_L bracket plus the
%                   certified flag.
%                   'scalar_qfa' = scalar Lanczos-QFA: the per-probe quadratic
%                   forms directly (basis-free, O(n·s) memory). With
%                   'Adaptive',1 each probe stops at its own depth via the
%                   Gauss-Radau certificate and 'AdaptiveTol' is a CERTIFIED
%                   per-probe relative error (no window/floor); Depth is the
%                   cap. Reorth/AdaptiveDelay/AdaptiveMin are ignored. The
%                   times struct reports d_used (max per-probe depth) and
%                   oracle_mv (actual Σ per-probe matvecs).
%                   'exact' builds a full eigendecomposition of A (O(n^3)) and
%                   is intended for validation/reference use, not production.
%                   'scalar' runs one Lanczos recurrence per probe (equivalent
%                   to Lanczos quadrature on each quadratic form).
%                   'auto' = knob-free tier: pass 'Budget' (total A-matvec
%                   budget) and 'AutoEps'; the driver picks k, s, and the
%                   oracle depth itself (positional k/s become placeholders and
%                   Depth/Reorth/Adaptive* are ignored), running the certified
%                   scalar QFA for both the depth probe and Phase 2. The times
%                   struct reports the choices (auto_k, auto_s, d_used,
%                   probe_mv, oracle_mv), the depth-probe wall-clock
%                   (probe_ms; the probe runs before phase1_ms's clock, so
%                   phase1_ms + phase2_ms excludes it) and the certification
%                   flags (probe_converged, phase2_certified); spend closes
%                   as an upper bound
%                   probe_mv + q*auto_k + oracle_mv <= Budget (Phase 1 costs
%                   q*auto_k matvecs; q = 1 here). An infeasible Budget
%                   raises error id 'randlapack:fun_nystrom_pp:infeasibleBudget'.
%     'Depth'       Lanczos depth for 'scalar' / 'scalar_qfa' / 'block' /
%                   'block_qfa' LFAType (default 200 for scalar/scalar_qfa,
%                   20 for block AND block_qfa; a CAP in the adaptive QFA
%                   modes; ignored for 'exact')
%     'RadauReturn' block_qfa + Adaptive 1 only: value returned on a certified
%                   stop. 0 = block Gauss (default; matches the scalar
%                   oracle), 1 = (Gauss + Radau)/2 midpoint (for operator-
%                   monotone f the two quadratures err on opposite sides, so
%                   the midpoint halves the one-sided Gauss bias for free).
%     'AutoDepthCap' LFAType 'auto' only: fixed cap on the depth probe
%                   (default 0 = no fixed cap; the probe is then bounded only
%                   by n and by AutoProbeFrac).
%     'AutoProbeFrac' LFAType 'auto' only: fraction of the matvec Budget the
%                   depth probe may spend, in (0, 1) (default 0.125).
%     'Sketch'      DEPRECATED/IGNORED (default 'saso'). The Phase-1 sketch is
%                   always the kernel-internal SASO; anything other than 'saso'
%                   triggers a warning from the MEX. Kept so existing call
%                   sites keep running.
%     'VecNnz'      nonzeros per column of the SASO sketch (default 8;
%                   0 = auto, resolved to ~log(k) inside the kernel)
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
    % VecNnz 0 = auto (~log k, resolved inside the kernel).
    addParameter(p, 'VecNnz',     8,          @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'SketchSeed', 42,         @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'Reorth',     1,          @(x) isnumeric(x) && isscalar(x));
    % QFA depth rule: 0 = fixed Depth; 1 = Radau-certified; 2 = legacy window
    % (block_qfa only; scalar_qfa treats any nonzero as certified adaptive).
    addParameter(p, 'Adaptive',   0,          @(x) isnumeric(x) && isscalar(x) && any(x == [0 1 2]));
    addParameter(p, 'AdaptiveTol',1e-2,       @(x) isnumeric(x) && isscalar(x) && x > 0);
    % block_qfa certified return value: 0 = block Gauss, 1 = midpoint.
    addParameter(p, 'RadauReturn', 0,         @(x) isnumeric(x) && isscalar(x) && any(x == [0 1]));
    % Certificate window for block_qfa adaptive (0 = MEX/library default). The
    % first convergence test is at depth AdaptiveMin + AdaptiveDelay.
    addParameter(p, 'AdaptiveDelay', 0,       @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'AdaptiveMin',   0,       @(x) isnumeric(x) && isscalar(x) && x >= 0);
    % Knob-free tier (LFAType 'auto'): total A-matvec budget + target accuracy;
    % the driver picks k, s, and the oracle depth (positional k/s and
    % Depth/Reorth/Adaptive are ignored in this mode).
    addParameter(p, 'Budget',  0,             @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'AutoEps', 1e-3,          @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
    % Auto-tier probe knobs: a fixed cap on the probe depth (0 = no fixed cap)
    % and the fraction of the Budget the probe may spend, in (0, 1).
    addParameter(p, 'AutoDepthCap',  0,       @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'AutoProbeFrac', 0.125,   @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
    parse(p, varargin{:});

    func     = char(p.Results.Func);
    q        = double(p.Results.Q);
    pl       = double(p.Results.PolyLambda);
    lfa_type = char(p.Results.LFAType);
    sketch   = char(p.Results.Sketch);
    vec_nnz  = double(p.Results.VecNnz);
    sk_seed  = double(p.Results.SketchSeed);
    reorth   = double(p.Results.Reorth);
    adaptive = double(p.Results.Adaptive);        % QFA types only; Depth is the cap
    adapt_tol = double(p.Results.AdaptiveTol);
    adapt_dl  = double(p.Results.AdaptiveDelay);
    adapt_mn  = double(p.Results.AdaptiveMin);
    radau_ret = double(p.Results.RadauReturn);
    budget    = double(p.Results.Budget);
    auto_eps  = double(p.Results.AutoEps);
    auto_dcap = double(p.Results.AutoDepthCap);
    auto_pfr  = double(p.Results.AutoProbeFrac);
    if strcmp(lfa_type, 'auto') && budget < 1
        error('randlapack:fun_nystrom_pp:Budget', ...
              'LFAType ''auto'' requires a positive ''Budget'' (total A-matvec budget).');
    end
    if isempty(p.Results.Depth)
        if any(strcmp(lfa_type, {'block', 'block_qfa'})), d = 20; else, d = 200; end
    else
        d = double(p.Results.Depth);
    end

    % A is templated on by the MEX; keep it in the working precision.
    A = cast(A, cls);

    mex_args = {A, a1, a2, func, q, pl, lfa_type, d, ...
                sketch, vec_nnz, sk_seed, reorth, adaptive, adapt_tol, ...
                adapt_dl, adapt_mn, budget, auto_eps, radau_ret, ...
                auto_dcap, auto_pfr};
    if nargout <= 1
        est = fun_nystrom_pp_mex(mex_args{:});
    elseif nargout == 2
        [est, t1] = fun_nystrom_pp_mex(mex_args{:});
    elseif nargout == 3
        [est, t1, t2] = fun_nystrom_pp_mex(mex_args{:});
    else
        [est, t1, t2, times] = fun_nystrom_pp_mex(mex_args{:});
    end
end
