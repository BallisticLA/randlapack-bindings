function test_cqrrpt()
%test_cqrrpt  Correctness checks for randlapack.cqrrpt.
%
% Runs CQRRPT on tall matrices and checks:
%   1. Orthogonality of Q.
%   2. Factorization fidelity: A(:, J) ~= Q*R.
%   3. R is upper-trapezoidal.
%   4. J is a valid 1-based permutation.
%   5. Rank detection on a deliberately rank-deficient input.
%   6. Reproducibility from an explicit RNG state.
%   7. Error paths, including the tall-matrix requirement.
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')
%   test_cqrrpt

    types = {'double', 'single'};
    sizes = {[500, 50], [1000, 100], [2000, 200]};

    for ti = 1:numel(types)
        T = types{ti};
        for si = 1:numel(sizes)
            sz = sizes{si};
            A = cast(randn(sz(1), sz(2)), T);

            [Q, R, J] = randlapack.cqrrpt(A);

            % Full-rank input: the detected rank should be n.
            assert(size(R, 1) == sz(2), ...
                   '%s %dx%d: full-rank input gave rank %d, expected %d', ...
                   T, sz(1), sz(2), size(R, 1), sz(2));

            check_orthogonality(Q, T);
            check_factorization(A, Q, R, J, T);
            check_upper_trapezoidal(R, T);
            check_pivot_validity(J, sz(2));

            fprintf('PASS: %s %dx%d full rank\n', T, sz(1), sz(2));
        end

        % --- Rank-deficient input: the rank must be revealed exactly ---
        % Duplicated columns make the true rank exact and known, so this
        % assertion is tight on purpose. It is also a regression test for a
        % specific defect: CQRRPT passes J straight to LAPACK's geqp3, which
        % READS jpvt on entry, so a J buffer that is not zeroed makes the
        % pivoting depend on uninitialized memory. The visible symptom was
        % exactly this check -- the detected rank wandered between 1 and 40
        % across runs while the full-rank cases stayed green.
        m = 2000; r = 40; extra = 10;
        B = cast(randn(m, r), T);
        A = [B, B(:, 1:extra)];                 % rank r, r+extra columns
        A = A(:, randperm(size(A, 2)));         % scatter the duplicates

        [Q, R, J] = randlapack.cqrrpt(A);
        detected = size(R, 1);
        assert(detected == r, ...
               '%s: expected rank %d on a rank-deficient input, got %d', ...
               T, r, detected);
        check_orthogonality(Q, T);
        check_leading_factorization(A, Q, R, J, T);
        % With the rank fully revealed, the whole permuted matrix
        % reconstructs, not just the leading block.
        check_factorization(A, Q, R, J, T);
        fprintf('PASS: %s rank detection (%d of %d columns)\n', ...
                T, detected, size(A, 2));
    end

    % --- Reproducibility from an explicit state ---
    A  = randn(1000, 100);
    st = struct('counter', uint32([0 0 0 0]), 'key', uint32([0 11]));
    [Q1, R1, J1] = randlapack.cqrrpt(A, 'state', st);
    [Q2, R2, J2] = randlapack.cqrrpt(A, 'state', st);
    assert(isequal(Q1, Q2) && isequal(R1, R2) && isequal(J1, J2), ...
           'same explicit state must give bit-identical results');
    fprintf('PASS: reproducible from an explicit RNG state\n');

    [~, ~, ~, st_out] = randlapack.cqrrpt(A, 'state', st);
    assert(~isequal(st_out.counter, st.counter) || ~isequal(st_out.key, st.key), ...
           'returned state must differ from the input state');
    fprintf('PASS: RNG state advances\n');

    % --- d_factor is honoured ---
    [~, R_a, ~] = randlapack.cqrrpt(A, 'd_factor', 1.0);
    [~, R_b, ~] = randlapack.cqrrpt(A, 'd_factor', 2.0);
    assert(size(R_a, 1) == 100 && size(R_b, 1) == 100, ...
           'full-rank input must give rank 100 at either d_factor');
    fprintf('PASS: d_factor accepted across its range\n');

    % --- Error paths ---
    check_error(@() randlapack.cqrrpt(zeros(0, 5)), 'randlapack:cqrrpt:emptyInput');
    % Wide matrix: the tall requirement must be reported in MATLAB terms,
    % not as a dimension complaint from inside the sketching chain.
    check_error(@() randlapack.cqrrpt(randn(50, 500)), 'randlapack:cqrrpt:notTall');
    % Square is also too short once d_factor > 1.
    check_error(@() randlapack.cqrrpt(randn(100, 100)), 'randlapack:cqrrpt:notTall');
    check_error(@() randlapack.cqrrpt(A, 'd_factor', 0.5), ...
                'MATLAB:cqrrpt:notGreaterEqual');
    check_error(@() randlapack.cqrrpt(A, 'nosuchopt', 1), ...
                'randlapack:cqrrpt:unknownOption');
    check_error(@() randlapack.cqrrpt(int32(A)), 'MATLAB:cqrrpt:invalidType');
    fprintf('PASS: error paths\n');

    fprintf('test_cqrrpt: all checks passed\n');
end


function check_orthogonality(Q, T)
    [m, k] = size(Q);
    I   = cast(eye(k), T);
    err = norm(double(Q)' * double(Q) - double(I), 'fro');
    tol = sqrt(double(m * k)) * eps(T) * 100;
    assert(err < tol, ...
           'orthogonality check failed: ||Q^T Q - I||_F = %g > tol = %g', ...
           err, tol);
end


function check_factorization(A, Q, R, J, T)
    [m, n] = size(A);
    err = norm(double(A(:, J)) - double(Q) * double(R), 'fro') ...
          / norm(double(A), 'fro');
    tol = sqrt(double(m * n)) * eps(T) * 100;
    assert(err < tol, ...
           'factorization check failed: ||A(:,J) - QR||_F/||A||_F = %g > tol = %g', ...
           err, tol);
end


function check_leading_factorization(A, Q, R, J, T)
% The leading k pivoted columns are reproduced exactly whatever rank came
% back, because R(:, 1:k) is precisely their triangular factor. This is the
% invariant that survives a conservative rank estimate.
    [m, ~] = size(A);
    k = size(R, 1);
    lead = double(A(:, J(1:k)));
    err  = norm(lead - double(Q) * double(R(:, 1:k)), 'fro') ...
           / norm(double(A), 'fro');
    tol  = sqrt(double(m * k)) * eps(T) * 100;
    assert(err < tol, ...
           ['leading-block factorization failed: ' ...
            '||A(:,J(1:k)) - Q*R(:,1:k)||_F/||A||_F = %g > tol = %g'], ...
           err, tol);
end


function check_upper_trapezoidal(R, T)
    below = tril(double(R), -1);
    assert(norm(below, 'fro') == 0, ...
           'R must be upper-trapezoidal; found %g below the diagonal', ...
           norm(below, 'fro'));
    assert(isa(R, T), 'R must have the same class as A');
end


function check_pivot_validity(J, n)
    assert(numel(J) == n, 'J must have n entries');
    assert(isequal(sort(double(J(:)))', 1:n), ...
           'J must be a permutation of 1:n (1-based)');
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
