function test_bqrrp()
%test_bqrrp  Correctness checks for randlapack.bqrrp.
%
% Runs BQRRP on several matrix sizes and scalar types, then checks:
%   1. Orthogonality of Q (||Q'Q - I||_F is small).
%   2. Factorization fidelity (||A(:, J) - Q*R||_F / ||A||_F is small).
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')
%   test_bqrrp

    types = {'double', 'single'};
    sizes = {[200, 50], [500, 100], [1000, 200]};

    for ti = 1:numel(types)
        T = types{ti};
        for si = 1:numel(sizes)
            sz = sizes{si};
            A = cast(randn(sz(1), sz(2)), T);

            [Q, R, J] = randlapack.bqrrp(A);

            check_orthogonality(Q, T);
            check_factorization(A, Q, R, J, T);
            check_pivot_validity(J, sz(2));

            fprintf('PASS: %s %dx%d\n', T, sz(1), sz(2));
        end
    end
end


function check_orthogonality(Q, T)
    [m, k] = size(Q);
    I   = cast(eye(k), T);
    err = norm(Q' * Q - I, 'fro');
    tol = sqrt(double(m * k)) * eps(T) * 50;
    assert(err < tol, ...
           'orthogonality check failed: ||Q^T Q - I||_F = %g > tol = %g', ...
           err, tol);
end


function check_factorization(A, Q, R, J, T)
    [m, n] = size(A);
    err = norm(A(:, J) - Q * R, 'fro') / max(norm(A, 'fro'), eps(T));
    tol = sqrt(double(m * n)) * eps(T) * 50;
    assert(err < tol, ...
           'factorization check failed: ||A(:, J) - Q*R||_F/||A||_F = %g > tol = %g', ...
           err, tol);
end


function check_pivot_validity(J, n)
    % J should be a permutation of 1:n (1-based, all entries unique).
    assert(numel(J) == n, 'J has length %d, expected %d', numel(J), n);
    assert(isequal(sort(double(J(:))), (1:n)'), ...
           'J is not a permutation of 1:n');
end
