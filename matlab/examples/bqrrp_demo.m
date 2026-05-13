% bqrrp_demo  Five line demo of RandLAPACK BQRRP from MATLAB.
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')

A = randn(2000, 200);
[Q, R, J] = randlapack.bqrrp(A);
err = norm(A(:, J) - Q * R, 'fro') / norm(A, 'fro');
fprintf('||A(:, J) - Q*R||_F / ||A||_F = %.2e\n', err);
fprintf('||Q^T Q - I||_F             = %.2e\n', ...
        norm(Q' * Q - eye(size(Q, 2)), 'fro'));
