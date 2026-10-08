% cqrrt_demo  Tall unpivoted QR with RandLAPACK CQRRT from MATLAB.
%
% CQRRT factors a tall, full-rank matrix A = Q*R by sketching A, using the
% sketch's R to precondition A, and finishing with Cholesky QR. This demo
% compares it with MATLAB's economy-size qr(A, 0), then shows the R-only
% call, which skips forming Q.
%
% CQRRT expects full rank. For rank-deficient or very ill-conditioned
% inputs use randlapack.cqrrpt, which reveals the rank.
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')

m = 50000; n = 500;
A = randn(m, n);
randlapack.cqrrt(randn(500, 50));      % warm-up: loads the MEX file

t = tic;
[Q, R] = randlapack.cqrrt(A);
dt = toc(t);
fprintf('A is %d-by-%d\n', m, n);
fprintf('CQRRT [Q, R]:          %.3f s\n', dt);
fprintf('||A - Q*R||_F / ||A||_F = %.2e\n', norm(A - Q * R, 'fro') / norm(A, 'fro'));
fprintf('||Q^T Q - I||_F         = %.2e\n', norm(Q' * Q - eye(n), 'fro'));

t = tic; [Qd, Rd] = qr(A, 0); dt_dense = toc(t);
fprintf('\ndense qr(A, 0):        %.3f s\n', dt_dense);
fprintf('CQRRT speedup:         %.1fx\n', dt_dense / dt);

% R is unique up to the signs of its rows, so compare |R| entries.
fprintf('max | |R| - |R_qr| | / max|R| = %.2e\n', ...
        max(abs(abs(R(:)) - abs(Rd(:)))) / max(abs(R(:))));

% R-only: the cheaper call when Q is not needed (least squares, preconditioning).
t = tic;
R_only = randlapack.cqrrt(A);
dt_r = toc(t);
fprintf('\nCQRRT R only:          %.3f s\n', dt_r);
