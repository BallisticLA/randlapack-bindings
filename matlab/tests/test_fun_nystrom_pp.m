function test_fun_nystrom_pp()
%test_fun_nystrom_pp  Correctness checks for randlapack.fun_nystrom_pp.
%
% Seed-driven API; the Phase-1 sketch is always the kernel-internal SASO.
% For an SPD A with known spectrum, tr(sqrt(A)) computed by funNystrom++
% should match the analytic trace within Hutchinson sampling error.
% Checks, for both single and double:
%   * exact-oracle estimate within tolerance of sum(sqrt(eig(A)))
%   * k == n => exact (rank-n Nystrom captures A; Phase 2 skipped) ~ machine eps
%   * scalar Lanczos-FA oracle path runs and is sane
%
% Run: test_fun_nystrom_pp   (errors via assert on failure; for CI).

    n = 200;
    for T = {'double', 'single'}
        cls = T{1};
        rng(7);
        [Q, ~] = qr(randn(n), 0);
        lam = logspace(0, -3, n)';
        A = cast(Q * (lam .* Q'), cls);  A = (A + A') / 2;
        true_tr = sum(sqrt(lam));

        % seed-driven, exact oracle (k = 60, s = 40)
        est = randlapack.fun_nystrom_pp(A, 60, 40, 'Func', 'sqrt', 'Q', 1, ...
                  'LFAType', 'exact', 'SketchSeed', 11);
        rel = abs(est - true_tr) / true_tr;
        assert(isfinite(est) && rel < 5e-2, ...
               '[%s] tr(sqrt(A)) rel_err = %.2e exceeds 5e-2', cls, rel);

        % k == n: exact rank-n Nystrom, Phase 2 skipped -> near machine precision
        est_full = randlapack.fun_nystrom_pp(A, n, 0, 'Func', 'sqrt', 'Q', 1, ...
                       'LFAType', 'exact');
        rel_full = abs(est_full - true_tr) / true_tr;
        tol_full = strcmp(cls, 'single') * 1e-3 + strcmp(cls, 'double') * 1e-8;
        assert(rel_full < tol_full, ...
               '[%s] k==n rel_err = %.2e exceeds %.0e (Phase-2-skip broken?)', ...
               cls, rel_full, tol_full);

        % scalar Lanczos-FA oracle: runs and is sane
        est_saso = randlapack.fun_nystrom_pp(A, 60, 40, 'Func', 'sqrt', 'Q', 1, ...
                       'LFAType', 'scalar', 'Depth', 60, ...
                       'SketchSeed', 11, 'Reorth', 0);
        rel_saso = abs(est_saso - true_tr) / true_tr;
        assert(isfinite(est_saso) && rel_saso < 5e-2, ...
               '[%s] SASO rel_err = %.2e exceeds 5e-2', cls, rel_saso);

        fprintf('test_fun_nystrom_pp [%6s]: OK  (exact rel %.2e, k==n %.2e, saso %.2e)\n', ...
                cls, rel, rel_full, rel_saso);
    end
end
