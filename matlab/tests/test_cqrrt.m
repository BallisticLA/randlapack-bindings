function test_cqrrt()
%test_cqrrt  Correctness checks for randlapack.cqrrt.
%
% Runs CQRRT on tall full-rank matrices and checks:
%   1. Orthogonality of Q.
%   2. Factorization fidelity: A ~= Q*R.
%   3. R is upper-triangular.
%   4. The R-only call returns the same R as the [Q, R] call.
%   5. An ill-conditioned (but full-rank) input still factors accurately.
%   6. An exactly singular sketch (zero column) is reported as an error;
%      a merely rank-deficient input completes with a singular R.
%   7. Reproducibility from an explicit RNG state.
%   8. Error paths, including the tall-matrix requirement.
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')
%   addpath('/path/to/randlapack-bindings/matlab/tests')
%   test_cqrrt

    types = {'double', 'single'};
    sizes = {[500, 50], [1000, 100], [2000, 200]};

    for ti = 1:numel(types)
        T = types{ti};
        for si = 1:numel(sizes)
            sz = sizes{si};
            A = cast(randn(sz(1), sz(2)), T);

            [Q, R] = randlapack.cqrrt(A);

            assert(isequal(size(Q), sz) && isequal(size(R), [sz(2), sz(2)]), ...
                   '%s %dx%d: Q is %dx%d and R is %dx%d', T, sz(1), sz(2), ...
                   size(Q, 1), size(Q, 2), size(R, 1), size(R, 2));
            check_orthogonality(Q, T);
            check_factorization(A, Q, R, T);
            check_upper_triangular(R, T);

            fprintf('PASS: %s %dx%d\n', T, sz(1), sz(2));
        end

        % --- R-only mode returns the same R ---
        % R does not depend on whether Q is formed: compute_Q only switches
        % the triangular solve that forms Q, which reads the Cholesky factor
        % but never writes it. From the same RNG state the two calls agree
        % up to rounding. Not bit for bit: the two calls run
        % on different work buffers, and an optimized BLAS may pick a
        % different code path for a different memory alignment.
        A  = cast(randn(1000, 100), T);
        st = struct('counter', uint32([0 0 0 0]), 'key', uint32([0 7]));
        R_only = randlapack.cqrrt(A, 'state', st);
        [~, R_full] = randlapack.cqrrt(A, 'state', st);
        diff_R = norm(double(R_only) - double(R_full), 'fro') ...
                 / norm(double(R_full), 'fro');
        assert(diff_R <= 100 * eps(T), ...
               '%s: R-only and [Q, R] calls differ by %g relative', T, diff_R);
        assert(isa(R_only, T), 'R must have the same class as A');
        fprintf('PASS: %s R-only mode matches [Q, R]\n', T);

        % --- An exactly singular sketch is reported as an error ---
        % A zero column makes the sketch's R exactly singular, so CQRRT
        % cannot precondition and returns a failure status.
        A = cast(randn(1000, 100), T);
        A(:, 37) = 0;
        check_error(@() randlapack.cqrrt(A), 'randlapack:cqrrt_mex:breakdown');
        fprintf('PASS: %s error on an exactly singular sketch (zero column)\n', T);
    end

    % --- A rank-deficient input is NOT reported (documented behaviour) ---
    % Duplicated columns: the sketch's R diagonal is tiny but not zero, so
    % CQRRT completes. Q*R still reproduces A, but R is numerically
    % singular, which is what the help tells users to check for.
    B = randn(2000, 40);
    A = [B, B(:, 1:10)];
    A = A(:, randperm(50));
    [Q, R] = randlapack.cqrrt(A);
    d = abs(diag(R));
    assert(min(d) / max(d) < 1e-10, ...
           'rank-deficient input: expected a numerically singular R, got min/max diag %g', ...
           min(d) / max(d));
    check_factorization(A, Q, R, 'double');
    fprintf('PASS: double rank-deficient input completes with a singular R\n');

    % --- Ill-conditioned but full-rank input (double) ---
    % Sketch-based preconditioning is what lets Cholesky QR survive here:
    % plain Cholesky QR on A'A would square cond(A) = 1e10 past 1/eps.
    m = 4000; n = 100;
    [U, ~] = qr(randn(m, n), 0);
    [V, ~] = qr(randn(n, n));
    A = U * diag(logspace(0, -10, n)) * V';
    [Q, R] = randlapack.cqrrt(A);
    check_orthogonality(Q, 'double');
    check_factorization(A, Q, R, 'double');
    fprintf('PASS: double, cond(A) = 1e10\n');

    % --- Reproducibility from an explicit state ---
    A  = randn(1000, 100);
    st = struct('counter', uint32([0 0 0 0]), 'key', uint32([0 11]));
    [Q1, R1] = randlapack.cqrrt(A, 'state', st);
    [Q2, R2] = randlapack.cqrrt(A, 'state', st);
    assert(isequal(Q1, Q2) && isequal(R1, R2), ...
           'same explicit state must give bit-identical results');
    fprintf('PASS: reproducible from an explicit RNG state\n');

    [~, ~, st_out] = randlapack.cqrrt(A, 'state', st);
    assert(~isequal(st_out.counter, st.counter) || ~isequal(st_out.key, st.key), ...
           'returned state must differ from the input state');
    fprintf('PASS: RNG state advances\n');

    % --- d_factor is honoured ---
    [Q_a, R_a] = randlapack.cqrrt(A, 'd_factor', 1.0);
    [Q_b, R_b] = randlapack.cqrrt(A, 'd_factor', 2.0);
    check_factorization(A, Q_a, R_a, 'double');
    check_factorization(A, Q_b, R_b, 'double');
    fprintf('PASS: d_factor accepted across its range\n');

    % --- Error paths ---
    check_error(@() randlapack.cqrrt(zeros(0, 5)), 'randlapack:cqrrt:emptyInput');
    % Wide matrix: the tall requirement must be reported in MATLAB terms,
    % not as a dimension complaint from inside the sketching chain.
    check_error(@() randlapack.cqrrt(randn(50, 500)), 'randlapack:cqrrt:notTall');
    % Square is also too short once d_factor > 1.
    check_error(@() randlapack.cqrrt(randn(100, 100)), 'randlapack:cqrrt:notTall');
    check_error(@() randlapack.cqrrt(A, 'd_factor', 0.5), ...
                'MATLAB:cqrrt:notGreaterEqual');
    check_error(@() randlapack.cqrrt(A, 'nosuchopt', 1), ...
                'randlapack:cqrrt:unknownOption');
    % CQRRT has no eps option (the driver does not use one).
    check_error(@() randlapack.cqrrt(A, 'eps', 1e-8), ...
                'randlapack:cqrrt:unknownOption');
    check_error(@() randlapack.cqrrt(int32(A)), 'MATLAB:cqrrt:invalidType');
    fprintf('PASS: error paths\n');

    fprintf('test_cqrrt: all checks passed\n');
end


function check_orthogonality(Q, T)
    [m, n] = size(Q);
    I   = cast(eye(n), T);
    err = norm(double(Q)' * double(Q) - double(I), 'fro');
    tol = sqrt(double(m * n)) * eps(T) * 100;
    assert(err < tol, ...
           'orthogonality check failed: ||Q^T Q - I||_F = %g > tol = %g', ...
           err, tol);
end


function check_factorization(A, Q, R, T)
    [m, n] = size(A);
    err = norm(double(A) - double(Q) * double(R), 'fro') ...
          / norm(double(A), 'fro');
    tol = sqrt(double(m * n)) * eps(T) * 100;
    assert(err < tol, ...
           'factorization check failed: ||A - QR||_F/||A||_F = %g > tol = %g', ...
           err, tol);
end


function check_upper_triangular(R, T)
    below = tril(double(R), -1);
    assert(norm(below, 'fro') == 0, ...
           'R must be upper-triangular; found %g below the diagonal', ...
           norm(below, 'fro'));
    assert(isa(R, T), 'R must have the same class as A');
end


function check_error(fn, expected_id)
    try
        fn();
    catch err
        assert(strcmp(err.identifier, expected_id), ...
               'expected error %s, got %s', expected_id, err.identifier);
        return;
    end
    error('expected error %s, but no error was raised', expected_id);
end
