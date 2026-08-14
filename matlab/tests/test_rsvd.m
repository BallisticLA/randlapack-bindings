function test_rsvd()
%test_rsvd  Correctness checks for randlapack.rsvd.
%
% Runs RSVD on matrices with a known rank structure and checks:
%   1. Orthogonality of U and V.
%   2. Reconstruction: A ~= U*S*V' when A is exactly low rank.
%   3. Near-optimality against the deterministic rank-k truncation
%      (Eckart-Young lower bound: no rank-k approximation beats it).
%   4. Output shapes and svd-compatible calling conventions.
%   5. Reproducibility from an explicit RNG state.
%   6. Error paths.
%
% Run from MATLAB after:
%   addpath('/path/to/randlapack-bindings/matlab')
%   test_rsvd

    types = {'double', 'single'};

    for ti = 1:numel(types)
        T = types{ti};

        % --- Exactly rank-r input: RSVD should reconstruct it ---
        m = 600; n = 300; r = 40;
        A = cast(randn(m, r) * randn(r, n), T);

        [U, S, V] = randlapack.rsvd(A, r);

        check_orthogonality(U, T, 'U');
        check_orthogonality(V, T, 'V');
        assert(size(U, 1) == m && size(V, 1) == n, 'wrong output row counts');
        assert(size(S, 1) == size(S, 2), 'S must be square');
        k_out = size(S, 1);
        assert(k_out <= r, 'returned rank %d exceeds requested %d', k_out, r);

        err = norm(double(A) - double(U) * double(S) * double(V)', 'fro') ...
              / norm(double(A), 'fro');
        tol = sqrt(double(m * n)) * eps(T) * 100;
        assert(err < tol, ...
               '%s: rank-%d reconstruction failed: rel err %g > tol %g', ...
               T, r, err, tol);

        fprintf('PASS: %s exact rank-%d reconstruction (rel err %.2e)\n', ...
                T, r, err);

        % --- Decaying spectrum: compare against the optimal truncation ---
        % Eckart-Young says the deterministic rank-k truncation is optimal,
        % so a randomized method can only match or lose. Requiring the loss
        % to stay within a modest factor is the real quality check; asking
        % for a fixed small error would only test the test matrix.
        k = 30;
        Ad = decaying_matrix(400, 200, T);
        [U, S, V] = randlapack.rsvd(Ad, k, 'p', 2);
        rand_err = norm(double(Ad) - double(U) * double(S) * double(V)', 'fro');

        [Us, Ss, Vs] = svd(double(Ad), 'econ');
        kk  = size(S, 1);
        opt = Us(:, 1:kk) * Ss(1:kk, 1:kk) * Vs(:, 1:kk)';
        opt_err = norm(double(Ad) - opt, 'fro');

        assert(rand_err <= 1.5 * opt_err + 10 * eps(T), ...
               ['%s: randomized rank-%d error %g is more than 1.5x the ' ...
                'optimal truncation error %g'], T, kk, rand_err, opt_err);
        fprintf('PASS: %s decaying spectrum, rank %d: %.4e vs optimal %.4e\n', ...
                T, kk, rand_err, opt_err);

        % --- Singular values: descending, positive, close to the true ones ---
        s = diag(S);
        assert(all(s > 0), 'singular values must be positive');
        assert(all(diff(double(s)) <= 1e-6 * double(s(1))), ...
               'singular values must be non-increasing');
        s_true = diag(Ss);
        rel_s  = abs(double(s(1)) - s_true(1)) / s_true(1);
        assert(rel_s < 0.05, ...
               '%s: leading singular value off by %.2f%%', T, 100 * rel_s);
        fprintf('PASS: %s leading singular value within %.2f%%\n', T, 100 * rel_s);
    end

    % --- svd-compatible calling conventions ---
    A = randn(300, 100);
    s_only = randlapack.rsvd(A, 10);
    assert(isvector(s_only) && numel(s_only) <= 10, ...
           'one-output form must return a vector of singular values');
    [U, S, ~] = randlapack.rsvd(A, 10);
    assert(ismatrix(S) && size(S, 1) == size(S, 2), ...
           'three-output form must return S as a square matrix');
    assert(norm(sort(double(s_only), 'descend') - sort(diag(double(S)), 'descend')) ...
           < 1e-8 * double(s_only(1)), ...
           'the two calling forms must agree on the singular values');
    fprintf('PASS: svd-compatible output conventions\n');

    % --- Reproducibility from an explicit state ---
    st = struct('counter', uint32([0 0 0 0]), 'key', uint32([0 7]));
    [U1, S1, ~] = randlapack.rsvd(A, 10, 'state', st);
    [U2, S2, ~] = randlapack.rsvd(A, 10, 'state', st);
    assert(isequal(U1, U2) && isequal(S1, S2), ...
           'same explicit state must give bit-identical results');
    fprintf('PASS: reproducible from an explicit RNG state\n');

    % --- The advanced state actually advances ---
    [~, ~, ~, st_out] = randlapack.rsvd(A, 10, 'state', st);
    assert(~isequal(st_out.counter, st.counter) || ~isequal(st_out.key, st.key), ...
           'returned state must differ from the input state');
    fprintf('PASS: RNG state advances\n');

    % --- Error paths ---
    check_error(@() randlapack.rsvd(zeros(0, 5), 1), 'randlapack:rsvd:emptyInput');
    check_error(@() randlapack.rsvd(A, 500), 'randlapack:rsvd:rankTooLarge');
    check_error(@() randlapack.rsvd(A, 0), 'MATLAB:rsvd:expectedPositive');
    check_error(@() randlapack.rsvd(A, 10, 'nosuchopt', 1), ...
                'randlapack:rsvd:unknownOption');
    check_error(@() randlapack.rsvd(A, 10, 'tol'), 'randlapack:rsvd:nargin');
    check_error(@() randlapack.rsvd(int32(A), 10), 'MATLAB:rsvd:invalidType');
    fprintf('PASS: error paths\n');

    fprintf('test_rsvd: all checks passed\n');
end


function A = decaying_matrix(m, n, T)
% Geometrically decaying spectrum with orthogonal factors, so the true
% singular values are known and the optimal truncation is computable.
    [Q1, ~] = qr(randn(m, min(m, n)), 0);
    [Q2, ~] = qr(randn(n, min(m, n)), 0);
    s = 0.85 .^ (0:min(m, n) - 1);
    A = cast(Q1 * diag(s) * Q2', T);
end


function check_orthogonality(X, T, name)
    [rows, k] = size(X);
    I   = cast(eye(k), T);
    err = norm(double(X)' * double(X) - double(I), 'fro');
    tol = sqrt(double(rows * k)) * eps(T) * 100;
    assert(err < tol, ...
           '%s orthogonality failed: ||%s^T %s - I||_F = %g > tol = %g', ...
           name, name, name, err, tol);
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
