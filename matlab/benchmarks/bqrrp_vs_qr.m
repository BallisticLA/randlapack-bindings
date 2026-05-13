% bqrrp_vs_qr  Wall-clock comparison of randlapack.bqrrp against MATLAB's
% built-in column-pivoted QR (qr(A, 'vector'), which uses LAPACK xGEQP3).
%
% No warmup, no statistics; this is a sanity-check timing run intended to
% surface a rough speedup on tall matrices. For rigorous benchmarks see
% the dedicated benchmarking project (TBD).
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')
%   bqrrp_vs_qr

sizes = {[1000,  100], ...
         [5000,  500], ...
         [20000, 1000]};

fprintf('%12s | %12s | %12s | %10s\n', ...
        'size', 'qr piv (s)', 'bqrrp (s)', 'speedup');
fprintf('%s\n', repmat('-', 1, 56));

for k = 1:numel(sizes)
    sz = sizes{k};
    A  = randn(sz(1), sz(2));

    t1 = tic;
    [~, ~, ~] = qr(A, 'vector');
    t_qr = toc(t1);

    t2 = tic;
    [~, ~, ~] = randlapack.bqrrp(A);
    t_bqrrp = toc(t2);

    fprintf('%6dx%4d | %12.3f | %12.3f | %9.2fx\n', ...
            sz(1), sz(2), t_qr, t_bqrrp, t_qr / t_bqrrp);
end
