% cqrrpt_demo  Rank revelation with RandLAPACK CQRRPT from MATLAB.
%
% CQRRPT discovers the numerical rank of a tall matrix rather than taking
% it as a parameter. This demo builds a matrix of known rank by repeating
% columns, then checks that the detected rank, the pivots, and the residual
% all agree with the construction.
%
% The shape below is chosen where the randomized method actually pays: dense
% QRCP wins on small problems (measured 0.6x at 5000-by-50) and loses as the
% matrix grows (7.4x at 20000-by-200 on the machine this was written on).
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')

m = 20000; r = 160; extra = 40;
B = randn(m, r);
A = [B, B(:, 1:extra)];                 % rank r, r+extra columns
perm = randperm(size(A, 2));
A = A(:, perm);                         % scatter the duplicates

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

% Compare against MATLAB's dense QRCP on the same matrix.
t = tic; [~, ~, Jd] = qr(A, 'vector'); dt_dense = toc(t);
fprintf('\ndense qr(A, ''vector''):        %.3f s\n', dt_dense);
fprintf('CQRRPT speedup:               %.1fx\n', dt_dense / dt);
fprintf('leading pivots agree on %d of the first %d columns\n', ...
        numel(intersect(J(1:k), Jd(1:k))), k);
