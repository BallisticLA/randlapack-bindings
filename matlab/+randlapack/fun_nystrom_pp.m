function [est, t1, t2] = fun_nystrom_pp(A, Omega1, Omega2, varargin)
%FUN_NYSTROM_PP  Trace estimator tr(f(A)) via RandLAPACK FunNystromPP_v2.
%
%   est = randlapack.fun_nystrom_pp(A, Omega1, Omega2, ...)
%   [est, t1, t2] = randlapack.fun_nystrom_pp(A, Omega1, Omega2, ...)
%
%   A        n × n double matrix (symmetric; upper triangle used).
%   Omega1   n × k double matrix (Phase 1 sketch).
%   Omega2   n × s double matrix (Phase 2 Hutchinson sketch; may be empty
%            when k == n, in which case Phase 2 is skipped).
%
%   Optional Name-Value pairs:
%     'Func'        char in {'sqrt', 'log', 'poly', 'square', 'identity'}
%                   (default 'sqrt')
%     'Q'           subspace-iter count (default 2)
%     'PolyLambda'  λ in f(x) = x(x + λ) when Func='poly' (default 10)
%     'LFAType'     {'exact', 'scalar', 'block'} oracle for f(A)·X
%                   (default 'exact')
%     'Depth'       Lanczos depth for scalar / block LFAType
%                   (default 200 for scalar, 20 for block)
%
%   Returns:
%     est   trace estimate t1 + t2
%     t1    Phase 1 contribution Σ f(λ̂ᵢ)
%     t2    Phase 2 Hutchinson correction (0 when k == n)
%
%   Algorithm: Persson-Kressner two-phase funNyström++ (rank-k Nyström
%   approximation + Hutchinson correction on the residual). MEX wrapper
%   over RandLAPACK::FunNystromPP_v2<double>. Bit-identical to the
%   reference MATLAB implementation (Persson) for lfa_type='exact'.

    p = inputParser;
    addParameter(p, 'Func',       'sqrt',   @(x) ischar(x) || isstring(x));
    addParameter(p, 'Q',          2,        @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'PolyLambda', 10,       @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'LFAType',    'exact',  @(x) ischar(x) || isstring(x));
    addParameter(p, 'Depth',      [],       @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
    parse(p, varargin{:});

    func      = char(p.Results.Func);
    q         = double(p.Results.Q);
    pl        = double(p.Results.PolyLambda);
    lfa_type  = char(p.Results.LFAType);
    if isempty(p.Results.Depth)
        if strcmp(lfa_type, 'block'), d = 20; else, d = 200; end
    else
        d = double(p.Results.Depth);
    end

    % Inputs must be double, column-major (MATLAB default). The MEX layer
    % validates shape; pass-through here keeps the MATLAB-side simple.
    A      = double(A);
    Omega1 = double(Omega1);
    if isempty(Omega2)
        % Phase 2 will be skipped iff k == n. Use a 0-column placeholder.
        Omega2 = zeros(size(A, 1), 0);
    else
        Omega2 = double(Omega2);
    end

    if nargout <= 1
        est = fun_nystrom_pp_mex(A, Omega1, Omega2, ...
            func, q, pl, lfa_type, d);
    elseif nargout == 2
        [est, t1] = fun_nystrom_pp_mex(A, Omega1, Omega2, ...
            func, q, pl, lfa_type, d);
    else
        [est, t1, t2] = fun_nystrom_pp_mex(A, Omega1, Omega2, ...
            func, q, pl, lfa_type, d);
    end
end
