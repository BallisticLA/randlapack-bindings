% cqrrpt_demo  Rank revelation with RandLAPACK CQRRPT from MATLAB.
%
% CQRRPT discovers the numerical rank of a tall matrix rather than taking
% it as a parameter. This demo builds a matrix of known rank by repeating
% columns, then checks that the detected rank, the pivots, and the residual
% all agree with the construction.
%
% The timings compare against MATLAB's economy-size column-pivoted QR,
% qr(A, 'econ', 'vector'), after a warm-up call that loads the MEX file. The
% randomized method pays off as the matrix grows.
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')

m = 20000; r = 160; extra = 40;
B = randn(m, r);
A = [B, B(:, 1:extra)];                 % rank r, r+extra columns
perm = randperm(size(A, 2));
A = A(:, perm);                         % scatter the duplicates

randlapack.cqrrpt(randn(500, 50));     % warm-up: loads the MEX file

t = tic;
[Q, R, J] = randlapack.cqrrpt(A);
dt = toc(t);

k = size(R, 1);
fprintf('A is %d-by-%d, constructed rank %d\n', m, size(A, 2), r);
fprintf('CQRRPT detected rank %d in %.3f s\n', k, dt);
fprintf('||A(:, J) - Q*R||_F / ||A||_F = %.2e\n', ...
        norm(A(:, J) - Q * R, 'fro') / norm(A, 'fro'));
fprintf('||Q^T Q - I||_F               = %.2e\n', ...
        norm(Q' * Q - eye(k), 'fro'));

% The first k pivots should select an independent set. Checking against
% MATLAB's own rank() is the honest comparison.
fprintf('rank(A(:, J(1:k)))            = %d  (want %d)\n', ...
        rank(A(:, J(1:k))), k);

% Compare against MATLAB's dense QRCP on the same matrix (economy size, so
% that, like CQRRPT, it forms no m-by-m Q).
t = tic; [~, ~, Jd] = qr(A, 'econ', 'vector'); dt_dense = toc(t);
fprintf('\ndense qr(A, ''econ'', ''vector''): %.3f s\n', dt_dense);
fprintf('CQRRPT speedup:               %.1fx\n', dt_dense / dt);
fprintf('leading pivots agree on %d of the first %d columns\n', ...
        numel(intersect(J(1:k), Jd(1:k))), k);
