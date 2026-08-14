% rsvd_demo  Low-rank approximation with RandLAPACK RSVD from MATLAB.
%
% Builds a matrix with a known decaying spectrum, then compares the
% randomized rank-k approximation against the optimal (truncated SVD) one
% across a sweep of k. Eckart-Young says the truncated SVD is optimal, so
% the interesting quantity is the RATIO -- how much the randomized method
% gives up for its speed.
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')

m = 2000; n = 500;
[Q1, ~] = qr(randn(m, n), 0);
[Q2, ~] = qr(randn(n, n), 0);
sv = 0.93 .^ (0:n-1);              % geometric decay, known exactly
A  = Q1 * diag(sv) * Q2';
normA = norm(A, 'fro');

[~, S_full, ~] = svd(A, 'econ');
sv_true = diag(S_full);

fprintf('%4s  %12s  %12s  %8s  %8s\n', ...
        'k', 'randomized', 'optimal', 'ratio', 'time (s)');
for k = [10, 25, 50, 100, 200]
    t = tic;
    [U, S, V] = randlapack.rsvd(A, k);
    dt = toc(t);

    k_out    = size(S, 1);                       % may be < k if tol fired
    rand_err = norm(A - U * S * V', 'fro') / normA;
    % Optimal rank-k_out error is the tail of the true spectrum.
    opt_err  = norm(sv_true(k_out+1:end)) / normA;

    fprintf('%4d  %12.4e  %12.4e  %8.3f  %8.3f\n', ...
            k_out, rand_err, opt_err, rand_err / opt_err, dt);
end

% Singular value accuracy at a single k.
[~, S, ~] = randlapack.rsvd(A, 50);
s_est = diag(S);
fprintf('\nLeading singular values (randomized vs true):\n');
for i = 1:5
    fprintf('  sigma_%d: %10.6f vs %10.6f  (rel err %.2e)\n', ...
            i, s_est(i), sv_true(i), abs(s_est(i) - sv_true(i)) / sv_true(i));
end
