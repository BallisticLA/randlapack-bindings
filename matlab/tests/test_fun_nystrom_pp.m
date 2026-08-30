function test_fun_nystrom_pp()
%test_fun_nystrom_pp  Correctness checks for randlapack.fun_nystrom_pp.
%
% Seed-driven API; the Phase-1 sketch is always the kernel-internal SASO.
% For an SPD A with known spectrum, tr(f(A)) computed by funNystrom++ should
% match the analytic trace within Hutchinson sampling error.
% Checks, for both single and double:
%   * all six LFATypes ('exact','scalar','scalar_qfa','block','block_qfa',
%     'auto') round-trip within tolerance of the analytic trace
%   * the times struct carries the campaign's cost accounting: d_used;
%     oracle_mv (= s*d_used for block_qfa, = sum of per-column depths for
%     scalar_qfa); auto_k/auto_s/probe_mv with the budget closure
%     probe_mv + q*auto_k + oracle_mv <= Budget (q = 1) for 'auto';
%     probe_converged/phase2_certified present; path-specific fields are NaN
%     off their path
%   * block_qfa + Adaptive=1 (Radau-certified) with f = log: tr_U >= tr_L
%     bracket, certified flag set; RadauReturn = 1 (midpoint) runs
%   * block_qfa + Adaptive=2 (legacy window) runs and never certifies
%   * k == n => exact (rank-n Nystrom captures A; Phase 2 skipped) ~ machine eps
%   * LFAType 'auto' with an infeasible Budget raises
%     'randlapack:fun_nystrom_pp:infeasibleBudget'
%   * VecNnz = 0 ("auto" ~log k) is accepted
%   * explicit Omega2 works; an empty (n x 0) Omega2 with k < n raises
%
% Run: test_fun_nystrom_pp   (errors via assert on failure; for CI).

    n = 200;
    k = 60;
    s = 40;
    budget = 600;
    for T = {'double', 'single'}
        cls = T{1};
        rng(7);
        [Q, ~] = qr(randn(n), 0);
        lam = logspace(0, -3, n)';
        A = cast(Q * (lam .* Q'), cls);  A = (A + A') / 2;
        true_sqrt = sum(sqrt(lam));
        true_log  = sum(log(lam + 1));

        % --- 1. all six LFATypes round-trip against the sqrt fixture ---
        lfa_types = {'exact', 'scalar', 'scalar_qfa', 'block', 'block_qfa', 'auto'};
        depth_for = struct('exact', 0, 'scalar', 60, 'scalar_qfa', 60, ...
                           'block', 20, 'block_qfa', 20, 'auto', 0);
        for it = 1:numel(lfa_types)
            lfa = lfa_types{it};
            if strcmp(lfa, 'auto')
                extra = {'Budget', budget, 'AutoEps', 1e-2};
            elseif depth_for.(lfa) > 0
                extra = {'Depth', depth_for.(lfa)};
            else
                extra = {};
            end
            [est, ~, ~, times] = randlapack.fun_nystrom_pp(A, k, s, ...
                'Func', 'sqrt', 'Q', 1, 'LFAType', lfa, 'SketchSeed', 11, extra{:});
            rel = abs(est - true_sqrt) / true_sqrt;
            assert(isfinite(est) && rel < 5e-2, ...
                   '[%s/%s] rel_err = %.2e exceeds 5e-2', cls, lfa, rel);

            % d_used is the campaign's depth accounting; present for every type.
            assert(isfield(times, 'd_used') && times.d_used >= 1, ...
                   '[%s/%s] d_used missing or < 1', cls, lfa);

            switch lfa
                case 'scalar_qfa'
                    % Fixed depth: every column runs d steps, so the sum of
                    % per-column depths is exactly s * d_used.
                    assert(times.oracle_mv == s * times.d_used, ...
                           '[%s/%s] oracle_mv %g ~= s*d_used %g', ...
                           cls, lfa, times.oracle_mv, s * times.d_used);
                case 'block_qfa'
                    % The class reports matvecs = s * d_used always.
                    assert(times.oracle_mv == s * times.d_used, ...
                           '[%s/%s] oracle_mv %g ~= s*d_used %g', ...
                           cls, lfa, times.oracle_mv, s * times.d_used);
                    assert(~isnan(times.tr_U) && ~isnan(times.tr_L) && ...
                           ~isnan(times.certified), ...
                           '[%s/%s] tr_U/tr_L/certified must be real for block_qfa', cls, lfa);
                case 'auto'
                    assert(times.auto_k >= 1 && times.auto_s >= 1 && times.probe_mv >= 1, ...
                           '[%s/%s] auto_k/auto_s/probe_mv accounting missing', cls, lfa);
                    spend = times.probe_mv + 1 * times.auto_k + times.oracle_mv;  % q = 1
                    assert(spend <= budget, ...
                           '[%s/%s] budget closure violated: %g > %g', cls, lfa, spend, budget);
                    assert(~isnan(times.probe_converged) && ~isnan(times.phase2_certified), ...
                           '[%s/%s] probe_converged/phase2_certified must be real for auto', cls, lfa);
                    assert(~isnan(times.probe_ms) && times.probe_ms >= 0, ...
                           '[%s/%s] probe_ms missing for auto', cls, lfa);
                otherwise
                    % exact/scalar/block report no oracle matvec count (0).
                    assert(times.oracle_mv == 0, ...
                           '[%s/%s] oracle_mv should be 0, got %g', cls, lfa, times.oracle_mv);
            end
            % Path-specific fields must be NaN off their path.
            if ~strcmp(lfa, 'block_qfa')
                assert(isnan(times.tr_U) && isnan(times.tr_L) && isnan(times.certified), ...
                       '[%s/%s] block_qfa-only fields must be NaN here', cls, lfa);
            end
            if ~strcmp(lfa, 'auto')
                assert(isnan(times.probe_ms) && isnan(times.probe_converged) && ...
                       isnan(times.phase2_certified), ...
                       '[%s/%s] auto-only fields must be NaN here', cls, lfa);
            end
        end

        % --- 2. k == n: exact rank-n Nystrom, Phase 2 skipped -> near machine precision ---
        est_full = randlapack.fun_nystrom_pp(A, n, 0, 'Func', 'sqrt', 'Q', 1, ...
                       'LFAType', 'exact');
        rel_full = abs(est_full - true_sqrt) / true_sqrt;
        tol_full = strcmp(cls, 'single') * 1e-3 + strcmp(cls, 'double') * 1e-8;
        assert(rel_full < tol_full, ...
               '[%s] k==n rel_err = %.2e exceeds %.0e (Phase-2-skip broken?)', ...
               cls, rel_full, tol_full);

        % --- 3. block_qfa Radau-certified (Adaptive = 1) with f = log ---
        % log is operator monotone, so Gauss (tr_U) and Radau (tr_L) bracket
        % the true quadratic form from opposite sides: tr_U >= tr_L.
        [est_bq, ~, ~, tq] = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'log', ...
            'Q', 1, 'LFAType', 'block_qfa', 'Adaptive', 1, 'Depth', 60, ...
            'SketchSeed', 11);
        rel_bq = abs(est_bq - true_log) / true_log;
        assert(isfinite(est_bq) && rel_bq < 5e-2, ...
               '[%s] block_qfa Radau rel_err = %.2e exceeds 5e-2', cls, rel_bq);
        assert(tq.certified == 1, '[%s] Radau bracket did not certify', cls);
        assert(tq.tr_U >= tq.tr_L - 1e-6 * abs(tq.tr_U), ...
               '[%s] bracket inverted: tr_U = %.6e < tr_L = %.6e', cls, tq.tr_U, tq.tr_L);
        assert(tq.d_used <= 60 && tq.oracle_mv == s * tq.d_used, ...
               '[%s] certified block_qfa depth/matvec accounting wrong', cls);

        % RadauReturn = 1 (midpoint) on the same call: runs and stays accurate.
        est_mid = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'log', ...
            'Q', 1, 'LFAType', 'block_qfa', 'Adaptive', 1, 'Depth', 60, ...
            'RadauReturn', 1, 'SketchSeed', 11);
        rel_mid = abs(est_mid - true_log) / true_log;
        assert(isfinite(est_mid) && rel_mid < 5e-2, ...
               '[%s] block_qfa midpoint rel_err = %.2e exceeds 5e-2', cls, rel_mid);

        % --- 4. block_qfa legacy window (Adaptive = 2): runs, never certifies ---
        [est_w, ~, ~, tw] = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', ...
            'Q', 1, 'LFAType', 'block_qfa', 'Adaptive', 2, 'Depth', 60, ...
            'AdaptiveDelay', 3, 'AdaptiveMin', 4, 'SketchSeed', 11);
        rel_w = abs(est_w - true_sqrt) / true_sqrt;
        assert(isfinite(est_w) && rel_w < 5e-2, ...
               '[%s] block_qfa window rel_err = %.2e exceeds 5e-2', cls, rel_w);
        assert(tw.certified == 0, '[%s] window rule must not report certified', cls);
        assert(tw.d_used <= 60, '[%s] window d_used exceeds cap', cls);

        % --- 5. scalar_qfa adaptive: oracle_mv = sum of per-column certified
        %        depths, bounded by [d_used, s * d_used] ---
        [~, ~, ~, tsq] = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', ...
            'Q', 1, 'LFAType', 'scalar_qfa', 'Adaptive', 1, 'Depth', 80, ...
            'SketchSeed', 11);
        assert(tsq.oracle_mv >= tsq.d_used && tsq.oracle_mv <= s * tsq.d_used, ...
               '[%s] adaptive scalar_qfa oracle_mv %g outside [%g, %g]', ...
               cls, tsq.oracle_mv, tsq.d_used, s * tsq.d_used);

        % --- 6. infeasible auto Budget raises the dedicated error id ---
        got_id = '';
        try
            randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', 'Q', 1, ...
                'LFAType', 'auto', 'Budget', 8);
        catch err
            got_id = err.identifier;
        end
        assert(strcmp(got_id, 'randlapack:fun_nystrom_pp:infeasibleBudget'), ...
               '[%s] Budget=8 raised ''%s'', expected infeasibleBudget', cls, got_id);

        % --- 7. VecNnz = 0 ("auto" ~log k) is accepted ---
        est_v0 = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', 'Q', 1, ...
                     'LFAType', 'exact', 'VecNnz', 0, 'SketchSeed', 11);
        rel_v0 = abs(est_v0 - true_sqrt) / true_sqrt;
        assert(isfinite(est_v0) && rel_v0 < 5e-2, ...
               '[%s] VecNnz=0 rel_err = %.2e exceeds 5e-2', cls, rel_v0);

        % --- 8. explicit Omega2 path works; empty Omega2 with k < n raises ---
        rng(13);
        Om2 = cast(randn(n, s), cls);
        est_om = randlapack.fun_nystrom_pp(A, k, Om2, 'Func', 'sqrt', 'Q', 1, ...
                     'LFAType', 'exact', 'SketchSeed', 11);
        rel_om = abs(est_om - true_sqrt) / true_sqrt;
        assert(isfinite(est_om) && rel_om < 5e-2, ...
               '[%s] explicit-Omega2 rel_err = %.2e exceeds 5e-2', cls, rel_om);
        got_id = '';
        try
            randlapack.fun_nystrom_pp(A, k, zeros(n, 0, cls), 'Func', 'sqrt', ...
                'Q', 1, 'LFAType', 'exact');
        catch err
            got_id = err.identifier;
        end
        assert(strcmp(got_id, 'randlapack:fun_nystrom_pp:Omega2Empty'), ...
               '[%s] n x 0 Omega2 raised ''%s'', expected Omega2Empty', cls, got_id);

        fprintf('test_fun_nystrom_pp [%6s]: OK\n', cls);
    end
end
