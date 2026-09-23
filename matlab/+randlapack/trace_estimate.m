function [estimate, info] = trace_estimate(A, varargin)
%TRACE_ESTIMATE Four funNystrom++ methods with explicit work limits.
% [estimate, info] = randlapack.trace_estimate(A, 'Method', 'scalar_auto', ...
%     'TargetRelError', 1e-3, 'MaxMatvecs', 4000)
%
% Method: scalar, block, scalar_auto, block_auto. Fixed methods require Depth.
% All methods use sphere probes and count block products by their column count.
% Auto methods discover depth and choose rank/probe count from the target.
% If that allocation exceeds the cap, at most half of the post-pilot work
% goes to rank; the rest funds residual probes. The expert fun_nystrom_pp
% interface retains the old policy and experimental splits for reproduction.
% The target guides allocation; it is NOT a total-error guarantee. Returned
% quadrature_certified reports only Lanczos stopping, not sampling accuracy.
% oracle_mean_depth is actual Phase-2 work per final probe. depth_limit is
% the selected limit; legacy d_used reports this limit for automatic methods.
% The fixed methods split the budget equally between rank and quadrature,
% subject to rank <= n/2 and the block Krylov dimension <= n.
%
% A is a dense symmetric matrix or a vector representing a diagonal operator.
% Func supports sqrt, log (= log(1+x)), effdim. Lambda sets the denominator
% shift for effdim x/(x+Lambda), default 10 (same as the low-level interface).
% VecNnz controls SASO nonzeros per row (default 8, clipped to rank).
% VecNnz=0 uses the kernel's logarithmic density rule. Sketch type is SASO.
% Entropy is available with every method. See fun_nystrom_pp.
% SpendCap (auto methods only, default false) makes the method spend the whole
% cap: rank and probe count use it, every probe runs to the pilot's depth, and
% quadrature_certified then reports the pilot's certificate, since nothing
% later is checked.
% FirstRowQL optionally accelerates scalar projected solves. Single precision
% and nonconverged QL solves retain the established evaluator. Its use and any
% fallback, including the pilot, are recorded in info.
    p = inputParser;
    addParameter(p, 'Method', 'scalar_auto');
    addParameter(p, 'Func', 'sqrt');
    addParameter(p, 'Lambda', 10);
    addParameter(p, 'VecNnz', 8);
    addParameter(p, 'TargetRelError', 1e-3);
    addParameter(p, 'MaxMatvecs', []);
    addParameter(p, 'Depth', []);
    addParameter(p, 'Seed', 42);
    addParameter(p, 'SkipSymCheck', false);
    addParameter(p, 'SpendCap', false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)) && any(x==[0 1]));
    addParameter(p, 'FirstRowQL', false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)) && any(x==[0 1]));
    parse(p, varargin{:});
    o = p.Results;
    method = validatestring(o.Method, {'scalar','block','scalar_auto','block_auto'});
    func = validatestring(o.Func, {'sqrt','log','effdim','identity','entropy','poly','square'});
    validateattributes(o.MaxMatvecs, {'numeric'}, {'scalar','finite','integer','positive'});
    validateattributes(o.TargetRelError, {'numeric'}, {'scalar','finite','>',0,'<',1});
    validateattributes(o.Lambda, {'numeric'}, {'scalar','real','finite','positive'});
    validateattributes(o.VecNnz, {'numeric'}, {'scalar','real','finite','integer','nonnegative'});
    B = o.MaxMatvecs;
    n = size(A, 1);
    if isvector(A), n = numel(A); end
    args = {'Func', func, 'VecNnz', o.VecNnz, 'SketchSeed', o.Seed, 'SkipSymCheck', o.SkipSymCheck};
    if o.FirstRowQL,args=[args,{'FirstRowQL',true}];end
    if ismember(func,{'effdim','poly'}), args=[args,{'PolyLambda',o.Lambda}]; end
    if endsWith(method, '_auto')
        % The Gauss / Radau-at-0 bracket needs the derivatives of f from order two up to
        % alternate in sign, not operator monotonicity. -x log x has the same pattern as
        % log(1+x), so entropy qualifies; x^2 has a second derivative of +2 and does not.
        validatestring(func, {'sqrt','log','effdim','identity','entropy'});
        [estimate, ~, ~, t] = randlapack.fun_nystrom_pp(A, 1, 1, args{:}, ...
            'LFAType', method, 'Budget', B, 'AutoEps', o.TargetRelError, ...
            'CapRankFraction', 0.5, 'SpendCap', logical(o.SpendCap));
        k = t.auto_k; s = t.auto_s; pilot = t.probe_mv;
        if o.SpendCap, cert = logical(t.probe_converged); else, cert = logical(t.phase2_certified); end
        pilot_ms = t.probe_ms;
    else
        if o.SpendCap
            error('randlapack:trace_estimate:spendCap', 'SpendCap applies only to scalar_auto and block_auto.');
        end
        validateattributes(o.Depth, {'numeric'}, {'scalar','finite','integer','positive'});
        d = min(n, o.Depth);
        k = max(1, min(floor(n/2), floor(B/2)));
        s = min(n-k, floor((B-k)/d));
        if strcmp(method, 'block'), s = min(s, floor(n/d)); end
        if s < 1
            error('randlapack:trace_estimate:infeasibleBudget', ...
                  'Budget %d cannot fund rank %d and one depth-%d probe.', B, k, d);
        end
        [estimate, ~, ~, t] = randlapack.fun_nystrom_pp(A, k, s, args{:}, ...
            'LFAType', [method '_qfa'], 'Q', 1, 'Depth', d, 'Adaptive', 0);
        pilot = 0; cert = false; pilot_ms = 0;
    end
    info = t;
    info.method = method;
    info.function_name = func;
    info.rank = k;
    info.samples = s;
    info.probe_ms = pilot_ms;
    info.matvecs = pilot + k + t.oracle_mv;
    info.algorithm_ms = pilot_ms + t.phase1_ms + t.phase2_ms;
    info.oracle_mean_depth = NaN;
    if s > 0, info.oracle_mean_depth = t.oracle_mv / s; end
    if endsWith(method, '_auto'), info.depth_limit = t.d_used; else, info.depth_limit = d; end
    info.quadrature_certified = cert;
    info.target_rel_error = o.TargetRelError;
    info.max_matvecs = B;
    info.total_error_certified = false;
    info.cap_rank_fraction = 0.5;
    info.spend_cap = logical(o.SpendCap);
    info.sketch = 'saso';
    info.vec_nnz_requested = o.VecNnz;
    info.vec_nnz_effective = min(k,o.VecNnz);
    if o.VecNnz==0, info.vec_nnz_effective=min(k,max(4,ceil(log(k)))); end
    assert(info.matvecs <= B, 'randlapack:trace_estimate:budgetExceeded', ...
           'Spent %d matvecs with a hard cap of %d.', info.matvecs, B);
end
