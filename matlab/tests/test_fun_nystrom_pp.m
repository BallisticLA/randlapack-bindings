function test_fun_nystrom_pp()
%test_fun_nystrom_pp  Correctness checks for randlapack.fun_nystrom_pp.
%
% Seed-driven API; the Phase-1 sketch is always the kernel-internal SASO.
% For an SPD A with known spectrum, tr(f(A)) computed by funNystrom++ should
% match the analytic trace within Hutchinson sampling error.
% Checks, for both single and double:
%   * all six LFATypes ('exact','scalar','scalar_qfa','block','block_qfa',
%     'auto') round-trip within tolerance of the analytic trace
%   * the times struct carries the campaign's cost accounting: d_used (NaN
%     for 'exact', which never runs a Lanczos recurrence);
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
%   * LFAType 'adaptive' (eps-targeted, no upfront Budget): success path with
%     a loose AutoEps round-trips within tolerance, telemetry fields (auto_k,
%     auto_s, d_used, probe_mv, oracle_mv, tr_U, tr_L, certified,
%     probe_converged, phase2_certified) populated; a tiny AdaptiveMatvecCap
%     raises 'randlapack:fun_nystrom_pp:infeasibleBudget'
%   * VecNnz = 0 ("auto" ~log k) is accepted
%   * explicit Omega2 works; an empty (n x 0) Omega2 with k < n raises
%   * a non-symmetric A raises 'randlapack:fun_nystrom_pp:NotSymmetric' by
%     default; 'SkipSymCheck', true bypasses the check
%   * a warning ('randlapack:fun_nystrom_pp:ignored_knob') fires from the .m
%     wrapper (inputParser.UsingDefaults) when 'Q'/'Adaptive*' are
%     EXPLICITLY passed together with LFAType 'auto'/'adaptive' (which
%     ignore them) -- including when the explicit value equals that knob's
%     own default (UsingDefaults tracks "was it passed", not "does the
%     value differ") -- and stays silent when the knob is left unset
%   * LFAType 'auto'/'adaptive' still work when given an explicit Omega2
%     matrix (the marshal is skipped for these tiers, but the call must not
%     break)
%   * LFAType 'auto' with Budget < 1 raises 'randlapack:fun_nystrom_pp:Budget'
%     (the id unified between the .m pre-check and the MEX's own guard)
%   * ignored-knob warning coverage is UNIFORM across LFATypes, not just
%     'auto'/'adaptive': 'scalar_qfa' explicitly given 'Reorth' warns (it is
%     intrinsically no-reorth); 'exact' explicitly given 'Depth' warns (no
%     Lanczos recurrence runs)
%   * 'Reorth' outside {0, 1} is rejected by inputParser validation
%     ('MATLAB:InputParser:ArgumentFailedValidation')
%   * a sparse double A is accepted and agrees with the dense call at the
%     same seed to rounding; 'exact' on a sparse A is refused with the
%     specific id 'randlapack:fun_nystrom_pp_mex:exact_sparse'
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
                % 'Q' is ignored by 'auto' (D-I2); omit it here rather than
                % pass a non-default value that would now warn.
                extra = {'Budget', budget, 'AutoEps', 1e-2};
                [est, ~, ~, times] = randlapack.fun_nystrom_pp(A, k, s, ...
                    'Func', 'sqrt', 'LFAType', lfa, 'SketchSeed', 11, extra{:});
            else
                if depth_for.(lfa) > 0
                    extra = {'Depth', depth_for.(lfa)};
                else
                    extra = {};
                end
                [est, ~, ~, times] = randlapack.fun_nystrom_pp(A, k, s, ...
                    'Func', 'sqrt', 'Q', 1, 'LFAType', lfa, 'SketchSeed', 11, extra{:});
            end
            rel = abs(est - true_sqrt) / true_sqrt;
            assert(isfinite(est) && rel < 5e-2, ...
                   '[%s/%s] rel_err = %.2e exceeds 5e-2', cls, lfa, rel);

            % d_used is the campaign's depth accounting; present for every
            % type, but NaN for 'exact' (no Lanczos recurrence runs, so the
            % leftover default-depth input is not a meaningful cost figure).
            if strcmp(lfa, 'exact')
                assert(isfield(times, 'd_used') && isnan(times.d_used), ...
                       '[%s/%s] d_used should be NaN for exact, got %g', ...
                       cls, lfa, times.d_used);
            else
                assert(isfield(times, 'd_used') && times.d_used >= 1, ...
                       '[%s/%s] d_used missing or < 1', cls, lfa);
            end

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
        % (I1 fix) certified must now be a real 0/1 for scalar_qfa+Adaptive=1
        % (scalar_qfa.all_certified marshalled through), and stay NaN for the
        % fixed-depth (Adaptive=0) path since no certificate was ever checked.
        assert(~isnan(tsq.certified) && any(tsq.certified == [0 1]), ...
               '[%s] adaptive scalar_qfa certified must be logical (0/1), got %g', ...
               cls, tsq.certified);
        [~, ~, ~, tsq_fixed] = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', ...
            'Q', 1, 'LFAType', 'scalar_qfa', 'Adaptive', 0, 'Depth', 60, ...
            'SketchSeed', 11);
        assert(isnan(tsq_fixed.certified), ...
               '[%s] fixed-depth scalar_qfa certified should stay NaN, got %g', ...
               cls, tsq_fixed.certified);

        % --- 6. infeasible auto Budget raises the dedicated error id ---
        got_id = '';
        try
            randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', ...
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

        % --- 9. LFAType 'adaptive': eps-targeted tier, no upfront Budget.
        %        Positional k/s are placeholders (ignored, like 'auto'); the
        %        driver picks them itself from a certified depth probe. ---
        [est_ad, ~, ~, tad] = randlapack.fun_nystrom_pp(A, 1, 1, 'Func', 'sqrt', ...
            'LFAType', 'adaptive', 'AutoEps', 1e-1, 'SketchSeed', 11);
        rel_ad = abs(est_ad - true_sqrt) / true_sqrt;
        assert(isfinite(est_ad) && rel_ad < 5e-2, ...
               '[%s] adaptive rel_err = %.2e exceeds 5e-2', cls, rel_ad);
        assert(isfield(tad, 'd_used') && tad.d_used >= 1, ...
               '[%s] adaptive d_used missing or < 1', cls);
        assert(tad.oracle_mv > 0, ...
               '[%s] adaptive oracle_mv should be > 0, got %g', cls, tad.oracle_mv);
        assert(tad.auto_k > 0 && tad.auto_s > 0, ...
               '[%s] adaptive auto_k/auto_s accounting missing (k=%g, s=%g)', ...
               cls, tad.auto_k, tad.auto_s);
        assert(~isnan(tad.probe_mv) && tad.probe_mv > 0, ...
               '[%s] adaptive probe_mv missing or non-positive', cls);
        assert(~isnan(tad.tr_U) && isfinite(tad.tr_U) && ...
               ~isnan(tad.tr_L) && isfinite(tad.tr_L), ...
               '[%s] adaptive tr_U/tr_L must be finite, got tr_U=%g tr_L=%g', ...
               cls, tad.tr_U, tad.tr_L);
        assert(~isnan(tad.certified) && any(tad.certified == [0 1]), ...
               '[%s] adaptive certified must be logical (0/1), got %g', cls, tad.certified);
        assert(~isnan(tad.probe_converged) && any(tad.probe_converged == [0 1]), ...
               '[%s] adaptive probe_converged must be logical (0/1)', cls);
        assert(~isnan(tad.phase2_certified) && any(tad.phase2_certified == [0 1]), ...
               '[%s] adaptive phase2_certified must be logical (0/1)', cls);

        % --- 10. LFAType 'adaptive' with a tiny AdaptiveMatvecCap raises the
        %         SAME error id as the 'auto' tier's infeasible Budget ---
        got_id = '';
        try
            randlapack.fun_nystrom_pp(A, 1, 1, 'Func', 'sqrt', ...
                'LFAType', 'adaptive', 'AutoEps', 1e-1, 'AdaptiveMatvecCap', 5);
        catch err
            got_id = err.identifier;
        end
        assert(strcmp(got_id, 'randlapack:fun_nystrom_pp:infeasibleBudget'), ...
               '[%s] AdaptiveMatvecCap=5 raised ''%s'', expected infeasibleBudget', ...
               cls, got_id);

        % --- 11. Symmetry validation (D-I1): default-on, tolerance-gated,
        %         skippable. The MEX only reads the upper triangle and
        %         mirrors it into the lower; breaking A's upper/lower
        %         agreement by a modest absolute amount (well past the
        %         sqrt(eps) tolerance, but small next to A's own O(1) scale
        %         so the perturbed matrix is still well-conditioned enough
        %         for 'exact' to run cleanly under SkipSymCheck) must raise
        %         before ever reaching the MEX; 'SkipSymCheck', true bypasses
        %         it. SketchSeed pinned to 11 throughout, matching every
        %         other randomized call in this file (a couple of other
        %         seeds are known to hit an unrelated rare SASO
        %         rank-deficiency failure in NystromEVD at this k/vec_nnz). ---
        A_asym = A;
        A_asym(1, end) = A_asym(1, end) + 1e-2 * max(abs(A(:)));
        got_id = '';
        try
            randlapack.fun_nystrom_pp(A_asym, k, s, 'Func', 'sqrt', 'Q', 1, ...
                'LFAType', 'exact', 'SketchSeed', 11);
        catch err
            got_id = err.identifier;
        end
        assert(strcmp(got_id, 'randlapack:fun_nystrom_pp:NotSymmetric'), ...
               '[%s] asymmetric A raised ''%s'', expected NotSymmetric', cls, got_id);

        est_skip = randlapack.fun_nystrom_pp(A_asym, k, s, 'Func', 'sqrt', ...
            'Q', 1, 'LFAType', 'exact', 'SketchSeed', 11, 'SkipSymCheck', true);
        assert(isfinite(est_skip), ...
               '[%s] SkipSymCheck=true run did not produce a finite estimate', cls);

        % The fixture A used throughout this test is explicitly symmetrized
        % ((A+A')/2 above) and must NOT trip the default check.
        est_sym_ok = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', ...
            'Q', 1, 'LFAType', 'exact', 'SketchSeed', 11);
        assert(isfinite(est_sym_ok), ...
               '[%s] symmetric fixture A incorrectly failed the symmetry check', cls);

        % --- 12. Ignored-knob warning (D-I2, fix batch 2-M): moved to the .m
        %         wrapper, driven by inputParser.UsingDefaults. 'Q'/
        %         'Adaptive*' EXPLICITLY passed together with LFAType
        %         'auto'/'adaptive' warns; the knob left UNSET stays silent. ---
        lastwarn('');
        [~] = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', 'SketchSeed', 11, ...
            'LFAType', 'auto', 'Budget', budget, 'AutoEps', 1e-2, 'Q', 3);
        [~, warn_id] = lastwarn();
        assert(strcmp(warn_id, 'randlapack:fun_nystrom_pp:ignored_knob'), ...
               '[%s] Q=3 with LFAType=auto did not warn ignored_knob (got ''%s'')', ...
               cls, warn_id);

        lastwarn('');
        [~] = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', 'SketchSeed', 11, ...
            'LFAType', 'auto', 'Budget', budget, 'AutoEps', 1e-2);
        [~, warn_id] = lastwarn();
        assert(isempty(warn_id), ...
               '[%s] Q left unset on an auto call unexpectedly warned ''%s''', cls, warn_id);

        % UsingDefaults tracks "was it passed", not "does the value differ":
        % explicitly repeating 'Q''s own default (2) must still warn here,
        % unlike the old MEX-side value-comparison check it replaced (which
        % could not tell this apart from "never touched").
        lastwarn('');
        [~] = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', 'SketchSeed', 11, ...
            'LFAType', 'auto', 'Budget', budget, 'AutoEps', 1e-2, 'Q', 2);
        [~, warn_id] = lastwarn();
        assert(strcmp(warn_id, 'randlapack:fun_nystrom_pp:ignored_knob'), ...
               '[%s] explicit Q=2 (its own default) on an auto call did not warn (got ''%s'')', ...
               cls, warn_id);

        lastwarn('');
        [~] = randlapack.fun_nystrom_pp(A, 1, 1, 'Func', 'sqrt', 'SketchSeed', 11, ...
            'LFAType', 'adaptive', 'AutoEps', 1e-1, 'AdaptiveDelay', 5);
        [~, warn_id] = lastwarn();
        assert(strcmp(warn_id, 'randlapack:fun_nystrom_pp:ignored_knob'), ...
               '[%s] AdaptiveDelay=5 with LFAType=adaptive did not warn ignored_knob (got ''%s'')', ...
               cls, warn_id);

        lastwarn('');
        [~] = randlapack.fun_nystrom_pp(A, 1, 1, 'Func', 'sqrt', 'SketchSeed', 11, ...
            'LFAType', 'adaptive', 'AutoEps', 1e-1);
        [~, warn_id] = lastwarn();
        assert(isempty(warn_id), ...
               '[%s] default-knob adaptive call unexpectedly warned ''%s''', cls, warn_id);

        % --- 12b. Uniform warning coverage (M5): the SAME ignored-knob
        %          mechanism now covers every LFAType, not just auto/adaptive.
        %          'scalar_qfa' is intrinsically no-reorth; an explicit
        %          'Reorth' warns. 'exact' never runs a Lanczos recurrence;
        %          an explicit 'Depth' warns. ---
        lastwarn('');
        [~] = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', 'Q', 1, ...
            'SketchSeed', 11, 'LFAType', 'scalar_qfa', 'Depth', 60, 'Reorth', 0);
        [~, warn_id] = lastwarn();
        assert(strcmp(warn_id, 'randlapack:fun_nystrom_pp:ignored_knob'), ...
               '[%s] explicit Reorth with LFAType=scalar_qfa did not warn (got ''%s'')', ...
               cls, warn_id);

        lastwarn('');
        [~] = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', 'Q', 1, ...
            'SketchSeed', 11, 'LFAType', 'exact', 'Depth', 60);
        [~, warn_id] = lastwarn();
        assert(strcmp(warn_id, 'randlapack:fun_nystrom_pp:ignored_knob'), ...
               '[%s] explicit Depth with LFAType=exact did not warn (got ''%s'')', ...
               cls, warn_id);

        % --- 13. LFAType 'auto'/'adaptive' with an explicit Omega2 matrix
        %         (D-I3: the MEX skips marshalling it for these two tiers,
        %         since neither driver.call overload reads it: confirm the
        %         call still succeeds and returns a correct, fully-populated
        %         estimate rather than silently breaking). ---
        rng(29);
        Om2_probe = cast(randn(n, s), cls);
        [est_auto_om, ~, ~, times_auto_om] = randlapack.fun_nystrom_pp(A, k, Om2_probe, ...
            'Func', 'sqrt', 'SketchSeed', 11, 'LFAType', 'auto', 'Budget', budget, 'AutoEps', 1e-2);
        rel_auto_om = abs(est_auto_om - true_sqrt) / true_sqrt;
        assert(isfinite(est_auto_om) && rel_auto_om < 5e-2, ...
               '[%s] auto with explicit Omega2 (marshal-skip path) rel_err = %.2e exceeds 5e-2', ...
               cls, rel_auto_om);
        assert(times_auto_om.auto_k >= 1 && times_auto_om.auto_s >= 1, ...
               '[%s] auto with explicit Omega2 telemetry missing (marshal-skip path)', cls);

        [est_ad_om, ~, ~, times_ad_om] = randlapack.fun_nystrom_pp(A, k, Om2_probe, ...
            'Func', 'sqrt', 'SketchSeed', 11, 'LFAType', 'adaptive', 'AutoEps', 1e-1);
        rel_ad_om = abs(est_ad_om - true_sqrt) / true_sqrt;
        assert(isfinite(est_ad_om) && rel_ad_om < 5e-2, ...
               '[%s] adaptive with explicit Omega2 (marshal-skip path) rel_err = %.2e exceeds 5e-2', ...
               cls, rel_ad_om);
        assert(times_ad_om.auto_k > 0 && times_ad_om.auto_s > 0, ...
               '[%s] adaptive with explicit Omega2 telemetry missing (marshal-skip path)', cls);

        % --- 14. Error-id unification (M1): LFAType 'auto' with Budget < 1
        %         raises 'randlapack:fun_nystrom_pp:Budget' (previously the
        %         .m pre-check and the MEX's own guard used two different,
        %         undocumented ids; a direct fun_nystrom_pp_mex(...) call
        %         now raises the identical id this wrapper does). ---
        got_id = '';
        try
            randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', 'Q', 1, ...
                'LFAType', 'auto', 'Budget', 0, 'AutoEps', 1e-2);
        catch err
            got_id = err.identifier;
        end
        assert(strcmp(got_id, 'randlapack:fun_nystrom_pp:Budget'), ...
               '[%s] Budget=0 raised ''%s'', expected the unified Budget id', cls, got_id);

        % --- 15. Sparse input: a sparse double A (both triangles stored)
        %         is accepted and must agree with the dense call at the same
        %         seed to rounding; 'exact' needs a dense eigensolve and is
        %         refused with its specific id. MATLAB's sparse() only
        %         supports double storage, so this only runs for
        %         cls == 'double'. ---
        if strcmp(cls, 'double')
            got_id = '';
            try
                randlapack.fun_nystrom_pp(sparse(A), k, s, 'Func', 'sqrt', ...
                    'LFAType', 'exact');
            catch err
                got_id = err.identifier;
            end
            assert(strcmp(got_id, 'randlapack:fun_nystrom_pp_mex:exact_sparse'), ...
                   '[%s] exact on sparse A raised ''%s'', expected randlapack:fun_nystrom_pp_mex:exact_sparse', ...
                   cls, got_id);
            e_sparse = randlapack.fun_nystrom_pp(sparse(A), k, s, 'Func', 'sqrt', ...
                'LFAType', 'block_qfa', 'Depth', 20, 'Q', 1, 'SketchSeed', 7);
            e_dense  = randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', ...
                'LFAType', 'block_qfa', 'Depth', 20, 'Q', 1, 'SketchSeed', 7);
            assert(abs(e_sparse - e_dense) <= 1e-10 * abs(e_dense), ...
                   '[%s] sparse and dense A disagree: %.17g vs %.17g', cls, e_sparse, e_dense);
        end

        % --- 16. Reorth-range rejection: 'Reorth' outside {0, 1} is rejected
        %         by inputParser's validation function before reaching the
        %         MEX (e.g. Reorth = -3, which would otherwise silently
        %         behave like Reorth = 1). ---
        got_id = '';
        try
            randlapack.fun_nystrom_pp(A, k, s, 'Func', 'sqrt', 'Q', 1, ...
                'LFAType', 'scalar', 'Reorth', -3);
        catch err
            got_id = err.identifier;
        end
        assert(strcmp(got_id, 'MATLAB:InputParser:ArgumentFailedValidation'), ...
               '[%s] Reorth=-3 raised ''%s'', expected ArgumentFailedValidation', cls, got_id);

        % --- 17. Vector-as-diagonal mode: passing the n diagonal entries as a
        %         vector is the same computation as passing the dense
        %         diag(lambda), at the same seed, for every LFAType. The point
        %         is large n, where the dense matrix does not fit; correctness
        %         is checked here at the fixture size. ---
        lam_d = cast(lam, cls);
        A_diag = diag(lam_d);
        vec_tol = strcmp(cls, 'single') * 1e-4 + strcmp(cls, 'double') * 1e-10;
        for it = 1:numel(lfa_types)
            lfa = lfa_types{it};
            if strcmp(lfa, 'auto')
                extra = {'Budget', budget, 'AutoEps', 1e-2};
            elseif depth_for.(lfa) > 0
                extra = {'Q', 1, 'Depth', depth_for.(lfa)};
            else
                extra = {'Q', 1};
            end
            est_dense = randlapack.fun_nystrom_pp(A_diag, k, s, 'Func', 'sqrt', ...
                'LFAType', lfa, 'SketchSeed', 11, 'SkipSymCheck', true, extra{:});
            est_vec = randlapack.fun_nystrom_pp(lam_d, k, s, 'Func', 'sqrt', ...
                'LFAType', lfa, 'SketchSeed', 11, extra{:});
            assert(isfinite(est_vec), '[%s/%s] vector-mode estimate is not finite', cls, lfa);
            assert(abs(est_vec - est_dense) <= vec_tol * abs(est_dense), ...
                   '[%s/%s] vector mode %.12e differs from dense diag %.12e', ...
                   cls, lfa, est_vec, est_dense);
        end

        % Row-vector orientation is accepted and gives the same answer.
        est_row = randlapack.fun_nystrom_pp(lam_d.', k, s, 'Func', 'sqrt', ...
            'Q', 1, 'LFAType', 'scalar_qfa', 'Depth', 60, 'SketchSeed', 11);
        est_col = randlapack.fun_nystrom_pp(lam_d, k, s, 'Func', 'sqrt', ...
            'Q', 1, 'LFAType', 'scalar_qfa', 'Depth', 60, 'SketchSeed', 11);
        assert(abs(est_row - est_col) <= vec_tol * abs(est_col), ...
               '[%s] row-vector A gave %.12e, column vector gave %.12e', cls, est_row, est_col);

        % The eps-targeted tier runs in vector mode (its Rademacher probes are
        % switched to sphere there, since a +-1 quadratic form is exact on a
        % diagonal matrix) and produces a genuinely sampled, nonzero error.
        est_ad = randlapack.fun_nystrom_pp(lam_d, k, s, 'Func', 'sqrt', ...
            'LFAType', 'adaptive', 'AutoEps', 1e-2, 'SketchSeed', 11);
        rel_ad = abs(est_ad - true_sqrt) / true_sqrt;
        assert(isfinite(est_ad) && rel_ad < 5e-2, ...
               '[%s] adaptive vector-mode rel_err = %.2e exceeds 5e-2', cls, rel_ad);
        assert(rel_ad > 0, '[%s] adaptive vector-mode error is exactly zero', cls);

        % Marshal cost is O(n), not O(n^2): a 50,000-entry diagonal marshals in
        % milliseconds, where the dense matrix would be 20 GB.
        lam_big = cast(logspace(0, -3, 50000)', cls);
        [~, ~, ~, times_big] = randlapack.fun_nystrom_pp(lam_big, 8, 4, ...
            'Func', 'sqrt', 'Q', 1, 'LFAType', 'scalar_qfa', 'Depth', 4, 'SketchSeed', 11);
        assert(times_big.marshal_in_ms < 100, ...
               '[%s] n = 50000 vector marshal took %.1f ms, expected under 100 ms', ...
               cls, times_big.marshal_in_ms);

        % Bad shapes are rejected with the documented id.
        got_id = '';
        try
            randlapack.fun_nystrom_pp(cast(ones(3, 4), cls), 2, 2, 'Func', 'sqrt', ...
                'Q', 1, 'LFAType', 'exact');
        catch err
            got_id = err.identifier;
        end
        assert(strcmp(got_id, 'randlapack:fun_nystrom_pp:A_shape'), ...
               '[%s] 3 x 4 A raised ''%s'', expected A_shape', cls, got_id);

        % SpendCap (2026-09-23): the self-tuning tiers spend the cap up to one probe, run every
        % probe to the pilot's depth (phase2_checked = 0) and report the pilot's certificate; the
        % default call is the shipped tier, with Phase 2 checked.
        if strcmp(cls, 'double')
            lam_s = logspace(0, -3, 3000)';
            for m = {'scalar_auto', 'block_auto'}
                [~, i0] = randlapack.trace_estimate(lam_s, 'Method', m{1}, 'Func', 'log', ...
                    'TargetRelError', 0.3, 'MaxMatvecs', 1024, 'Seed', 5);
                [~, i1] = randlapack.trace_estimate(lam_s, 'Method', m{1}, 'Func', 'log', ...
                    'TargetRelError', 0.3, 'MaxMatvecs', 1024, 'Seed', 5, 'SpendCap', true);
                assert(i0.phase2_checked == 1 && i1.phase2_checked == 0, ...
                       '%s: phase2_checked %g / %g, expected 1 / 0', m{1}, i0.phase2_checked, i1.phase2_checked);
                assert(i1.matvecs <= 1024 && 1024 - i1.matvecs < max(i1.d_used, 1), ...
                       '%s: SpendCap spent %d of 1024 at depth %d', m{1}, i1.matvecs, i1.d_used);
                assert(i1.matvecs > i0.matvecs, '%s: SpendCap spent no more than shipped (%d vs %d)', ...
                       m{1}, i1.matvecs, i0.matvecs);
                assert(i1.quadrature_certified == logical(i1.probe_converged), ...
                       '%s: SpendCap must report the pilot certificate', m{1});
                % the allocation is detail::spend_cap_split's (ported below), not merely some split
                % that spends the cap: at n = 3000, cap 4096 and pilot depth 93 it is (1500, 23), where
                % the rejected spend-maximizing rule gave (1492, 24)
                for nb = 3000
                    lam_b = logspace(0, 4, nb)';
                    for cap = [1024, 4096]
                        [~, ib] = randlapack.trace_estimate(lam_b, 'Method', m{1}, 'Func', 'log', ...
                            'TargetRelError', 1e-3, 'MaxMatvecs', cap, 'Seed', 5, 'SpendCap', true);
                        [ke, se] = spend_cap_split(cap - ib.probe_mv, ib.d_used, nb, 1, 4, strcmp(m{1}, 'block_auto'));
                        assert(ib.auto_k == ke && ib.auto_s == se, ...
                               '%s n=%d cap=%d t=%d: (k, s) = (%d, %d), spend_cap_split gives (%d, %d)', ...
                               m{1}, nb, cap, ib.d_used, ib.auto_k, ib.auto_s, ke, se);
                    end
                end
                % nothing was checked in Phase 2, so its lower side and certificate are NaN
                assert(isnan(i1.tr_L) && isnan(i1.phase2_certified) && isfinite(i1.tr_U) && ...
                       isfinite(i0.tr_L) && ~isnan(i0.phase2_certified), ...
                       '%s: SpendCap tr_L %g, phase2_certified %g (expected NaN); shipped tr_L %g', ...
                       m{1}, i1.tr_L, i1.phase2_certified, i0.tr_L);
            end
            got_id = '';
            try
                randlapack.trace_estimate(lam_s, 'Method', 'block', 'Func', 'log', ...
                    'MaxMatvecs', 256, 'Depth', 8, 'SpendCap', true);
            catch err
                got_id = err.identifier;
            end
            assert(strcmp(got_id, 'randlapack:trace_estimate:spendCap'), ...
                   'SpendCap on a fixed method raised ''%s''', got_id);
        end

        fprintf('test_fun_nystrom_pp [%6s]: OK\n', cls);
    end
end

function [k, s] = spend_cap_split(avail, t, n, q, s_min, block_krylov)
% MATLAB copy of RandLAPACK detail::spend_cap_split (rl_fun_nystrom_pp.hh): half of the post-probe
% budget to the rank (at most n - s_min), at least s_min probes, the rank taking what dimension
% limits leave unspent.
    t = max(1, t); q = max(1, q); kcap = max(1, fix(n / 2));
    k0 = max(1, min([kcap, n - s_min, fix(avail / (2 * q)), fix((avail - s_min * t) / q)]));
    s = min(fix((avail - q * k0) / t), n - k0);
    if block_krylov, s = min(s, fix(n / t)); end
    s = max(0, s);
    k = max(k0, min([kcap, n - s, fix((avail - s * t) / q)]));
end
