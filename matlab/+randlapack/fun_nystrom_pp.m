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
%       scalar_qfa/auto/adaptive), d_used (NaN for LFAType 'exact', which has
%       no Lanczos depth), oracle_mv, auto_k, auto_s,
%       probe_mv (also populated for 'adaptive', reusing these same generic
%       fields — see LFAType 'adaptive' below), plus path-specific fields
%       that are NaN when their path did not run: tr_U, tr_L, certified
%       ('block_qfa'/'adaptive' Gauss/Radau traces + bracket flag) and
%       probe_ms, probe_converged, phase2_certified ('auto'/'adaptive'
%       depth-probe wall-clock + certification flags). See
%       fun_nystrom_pp_mex.cc for the full slot definitions.
%
%   A        n x n matrix, single or double, REQUIRED symmetric. The MEX
%            reads only the upper triangle and mirrors it into the lower
%            triangle unconditionally (the lower triangle is never read, not
%            "used"). By default this wrapper checks symmetry with a
%            tolerance-gated maxabs(A-A') <= sqrt(eps(class(A)))*maxabs(A)
%            test and raises 'randlapack:fun_nystrom_pp:NotSymmetric' on
%            failure; pass 'SkipSymCheck', true to skip the O(n^2) check for
%            callers who already know A is symmetric (e.g. a large-matrix
%            benchmark loop).
%   k        Phase-1 rank (SCALAR). The Phase-1 sketch is a SparseStack/SASO
%            generated INSIDE RandLAPACK::NystromEVD from (SketchSeed, VecNnz),
%            matching the paper's Algorithm 1 line 1. Explicit Omega1 matrices
%            are no longer accepted (the dense-sketch mode was removed from
%            the kernel).
%   arg3     Phase-2 Hutchinson probes. Pass a SCALAR s and the n x s Gaussian
%            probes are sampled internally (seed = SketchSeed + 1000); OR pass
%            an explicit n x s matrix (escape hatch for validation, e.g.
%            feeding every estimator in a comparison the SAME probes).
%            Skipped when k == n (then arg3 may be a 0/[] placeholder). Also
%            never read for LFAType 'auto'/'adaptive' (those tiers derive
%            their own probes from a certified depth probe): the MEX skips
%            the marshal of an explicit Omega2 entirely in that case, so it
%            is validated here but never copied.
%
%   A knob explicitly passed but not consumed by the chosen 'LFAType' raises
%   warning 'randlapack:fun_nystrom_pp:ignored_knob' (fired HERE, using
%   inputParser's UsingDefaults so a knob left at its own default is silent
%   while one explicitly repeated at its default value still warns; see
%   each 'LFAType' entry below for what it reads). This replaces a prior
%   MEX-side value-comparison warning that could not tell "the caller left
%   this alone" from "the caller passed the default value on purpose", and
%   that fired unconditionally whenever a caller forwarded a knob's value
%   verbatim (e.g. every harness call).
%
%   Optional Name-Value pairs:
%     'Func'        char in {'sqrt', 'log', 'poly', 'effdim', 'square', 'identity'}
%                   (default 'sqrt'). 'poly' is f(x) = x(x + PolyLambda);
%                   'effdim' is f(x) = x/(x + PolyLambda) (the "effective
%                   dimension" function; operator monotone).
%     'Q'           subspace-iter count (default 2)
%     'PolyLambda'  lambda used by Func 'poly' and 'effdim' (default 10)
%     'LFAType'     {'exact', 'scalar', 'scalar_qfa', 'block', 'block_qfa',
%                   'auto', 'adaptive'} oracle for f(A)*X (default 'block' —
%                   the matrix-free Krylov oracle).
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
%                   Q/Depth/Reorth/Adaptive* are ignored: q is hardcoded to 1
%                   internally; this wrapper warns if any of these were
%                   explicitly passed), running the certified
%                   scalar QFA for both the depth probe and Phase 2. The times
%                   struct reports the choices (auto_k, auto_s, d_used,
%                   probe_mv, oracle_mv), the depth-probe wall-clock
%                   (probe_ms; the probe runs before phase1_ms's clock, so
%                   phase1_ms + phase2_ms excludes it) and the certification
%                   flags (probe_converged, phase2_certified); spend closes
%                   as an upper bound
%                   probe_mv + q*auto_k + oracle_mv <= Budget (Phase 1 costs
%                   q*auto_k matvecs; q = 1 here). A non-positive Budget (< 1)
%                   raises error id 'randlapack:fun_nystrom_pp:Budget'; an
%                   otherwise-infeasible Budget (too small once feasibility is
%                   checked against eps/n) raises
%                   'randlapack:fun_nystrom_pp:infeasibleBudget'.
%                   'adaptive' = fully knob-free, EPS-TARGETED tier: no
%                   upfront Budget; pass 'AutoEps' (reused here as the target
%                   accuracy eps) and the driver picks k, s, and the oracle
%                   depth itself from a certified BLOCK Gauss-Radau depth
%                   probe (positional k/s become placeholders and
%                   Q/Depth/Reorth/Adaptive* are ignored, same as 'auto').
%                   'AutoDepthCap' caps the probe depth, shared with the
%                   'auto' tier. 'AdaptiveMatvecCap' (default 0 = no cap)
%                   optionally bounds the total matvec spend. The times
%                   struct reports the choices in the SAME generic fields as
%                   'auto' (auto_k, auto_s, d_used, probe_mv, oracle_mv,
%                   probe_ms, probe_converged, phase2_certified) plus tr_U /
%                   tr_L / certified (the block oracle's final Gauss/
%                   Gauss-Radau trace bracket and certification flag, the
%                   same fields 'block_qfa' populates). An infeasible
%                   AdaptiveMatvecCap, or an eps/n regime the block-Krylov
%                   probe count cannot fund, raises the SAME error id as
%                   'auto': 'randlapack:fun_nystrom_pp:infeasibleBudget'.
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
%     'AdaptiveMatvecCap' LFAType 'adaptive' only: optional total matvec cap
%                   (default 0 = no cap). Ignored by every other LFAType.
%     'Sketch'      DEPRECATED/IGNORED (default 'saso'). The Phase-1 sketch is
%                   always the kernel-internal SASO; anything other than 'saso'
%                   triggers a warning from the MEX. Kept so existing call
%                   sites keep running.
%     'VecNnz'      nonzeros per ROW of the SASO sketch (default 8;
%                   0 = auto, resolved to ~log(k) inside the kernel)
%     'SketchSeed'  RNG seed for the Phase-1 sketch (default 42)
%     'SkipSymCheck' skip the default symmetry validation of A (default
%                   false). Set true only when the caller already knows A is
%                   symmetric (e.g. a large-n benchmark loop where the O(n^2)
%                   check is unwanted overhead); does not change what the MEX
%                   does with A (upper triangle mirrored into lower either way).
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
    validateattributes(A, {'single', 'double'}, ...
                       {'2d', 'square', 'real', 'finite', 'nonsparse'}, ...
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
        validateattributes(Omega2, {'single', 'double'}, ...
                           {'2d', 'real', 'finite', 'nonsparse'}, ...
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
    % C++ side treats Reorth as a plain boolean (full reorth vs none); restrict
    % to {0, 1} like the neighboring Adaptive/RadauReturn guards so e.g.
    % Reorth = -3 (which would otherwise silently behave like Reorth = 1) is
    % rejected instead.
    addParameter(p, 'Reorth',     1,          @(x) isnumeric(x) && isscalar(x) && any(x == [0 1]));
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
    % Eps-targeted tier (LFAType 'adaptive') only: optional total matvec cap
    % (0 = no cap).
    addParameter(p, 'AdaptiveMatvecCap', 0,   @(x) isnumeric(x) && isscalar(x) && x >= 0);
    % Default-on symmetry check (see the 'A' doc above); set true to skip it.
    addParameter(p, 'SkipSymCheck', false,    @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
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
    adapt_mvc = double(p.Results.AdaptiveMatvecCap);
    skip_sym_check = logical(p.Results.SkipSymCheck);

    % --- Ignored-knob warnings (moved here from the MEX; see the doc note
    % above 'Optional Name-Value pairs'). The consumption matrix below is
    % built from this docstring's own 'LFAType' entries plus
    % rl_fun_nystrom_pp.hh (q hardcoded to 1 and Adaptive*/Depth/Reorth
    % unread by the 'auto'/'adaptive' overloads; scalar_qfa's own
    % Reorth/AdaptiveDelay/AdaptiveMin no-op documented above;
    % AdaptiveMatvecCap read only by the 'adaptive' overload's matvec_cap
    % argument). p.UsingDefaults lists knobs NOT explicitly passed, so this
    % fires iff a knob was explicitly passed (any value, including its own
    % default) and the chosen LFAType does not consume it. ---
    switch lfa_type
        case 'exact'
            ignored = {'Depth', 'Reorth', 'Adaptive', 'AdaptiveTol', ...
                       'AdaptiveDelay', 'AdaptiveMin', 'RadauReturn', 'AdaptiveMatvecCap'};
        case 'scalar'
            ignored = {'Adaptive', 'AdaptiveTol', 'AdaptiveDelay', 'AdaptiveMin', ...
                       'RadauReturn', 'AdaptiveMatvecCap'};
        case 'scalar_qfa'
            ignored = {'Reorth', 'AdaptiveDelay', 'AdaptiveMin', 'RadauReturn', 'AdaptiveMatvecCap'};
        case 'block'
            ignored = {'Adaptive', 'AdaptiveTol', 'AdaptiveDelay', 'AdaptiveMin', ...
                       'RadauReturn', 'AdaptiveMatvecCap'};
        case 'block_qfa'
            ignored = {'AdaptiveMatvecCap'};
        case 'auto'
            ignored = {'Q', 'Depth', 'Reorth', 'Adaptive', 'AdaptiveTol', ...
                       'AdaptiveDelay', 'AdaptiveMin', 'RadauReturn', 'AdaptiveMatvecCap'};
        case 'adaptive'
            ignored = {'Q', 'Depth', 'Reorth', 'Adaptive', 'AdaptiveTol', ...
                       'AdaptiveDelay', 'AdaptiveMin', 'RadauReturn'};
        otherwise
            % Unknown lfa_type: leave the diagnostic to the MEX's own
            % 'randlapack:fun_nystrom_pp_mex:lfa_type' error.
            ignored = {};
    end
    explicitly_ignored = ignored(~ismember(ignored, p.UsingDefaults));
    if ~isempty(explicitly_ignored)
        warning('randlapack:fun_nystrom_pp:ignored_knob', ...
                ['LFAType ''%s'' ignores: %s (explicitly passed but not consumed ' ...
                 'by this tier -- see the ''LFAType'' doc above for what each tier reads).'], ...
                lfa_type, strjoin(explicitly_ignored, ', '));
    end
    % Explicit nonempty Omega2 MATRIX with auto/adaptive: both tiers derive
    % their own Phase-2 probes from a certified depth probe and never read a
    % caller-supplied Omega2 (the MEX skips the marshal entirely for these
    % two tiers, below). Scalar arg3 is exempted: it is the conventional
    % placeholder spelling used by every call site (including this
    % project's own harness), not a sign of a caller expecting their probes
    % to be used.
    if any(strcmp(lfa_type, {'auto', 'adaptive'})) && ~isscalar(Omega2) && ~isempty(Omega2)
        warning('randlapack:fun_nystrom_pp:ignored_knob', ...
                ['explicit Omega2 matrix ignored for LFAType ''%s'': this tier derives ' ...
                 'its own Phase-2 probes from a certified depth probe and never reads a ' ...
                 'caller-supplied Omega2.'], lfa_type);
    end

    % Same error id the MEX's own auto_budget < 1 guard raises (D moderate
    % finding: previously the two checks used different, undocumented ids;
    % unified so direct fun_nystrom_pp_mex(...) callers see the identical
    % identifier this wrapper would have raised).
    if strcmp(lfa_type, 'auto') && budget < 1
        error('randlapack:fun_nystrom_pp:Budget', ...
              'LFAType ''auto'' requires a positive ''Budget'' (total A-matvec budget).');
    end
    if isempty(p.Results.Depth)
        if any(strcmp(lfa_type, {'block', 'block_qfa'})), d = 20; else, d = 200; end
    else
        d = double(p.Results.Depth);
    end

    % --- Symmetry validation (default on; tolerance-gated). The MEX always
    % reads only the upper triangle of A and mirrors it into the lower
    % triangle; an asymmetric A therefore has its lower triangle silently
    % discarded with no other signal. Checked here, not in the MEX, so a
    % direct fun_nystrom_pp_mex(...) call remains a raw low-level entry point
    % (matches the header comment's "low-level" framing) while every call
    % through this wrapper is protected by default. ---
    if ~skip_sym_check
        A_diff = A - A.';
        asym   = max(abs(A_diff(:)));
        ascale = max(abs(A(:)));
        if ascale == 0, ascale = 1; end
        sym_tol = sqrt(eps(cls));
        if asym > sym_tol * ascale
            error('randlapack:fun_nystrom_pp:NotSymmetric', ...
                  ['A must be symmetric (checked within tolerance): ' ...
                   'max(abs(A-A'')) = %.3e exceeds tol*max(abs(A)) = %.3e ' ...
                   '(tol = sqrt(eps(''%s'')) = %.1e). The MEX reads only the ' ...
                   'upper triangle and mirrors it into the lower triangle; a ' ...
                   'non-symmetric A silently has its lower triangle discarded. ' ...
                   'If A is known to be symmetric (e.g. by construction) and ' ...
                   'this O(n^2) check is unwanted overhead at large n, pass ' ...
                   '''SkipSymCheck'', true.'], ...
                  asym, sym_tol * ascale, cls, sym_tol);
        end
    end

    % A is templated on by the MEX; keep it in the working precision.
    A = cast(A, cls);

    mex_args = {A, a1, a2, func, q, pl, lfa_type, d, ...
                sketch, vec_nnz, sk_seed, reorth, adaptive, adapt_tol, ...
                adapt_dl, adapt_mn, budget, auto_eps, radau_ret, ...
                auto_dcap, auto_pfr, adapt_mvc};
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
