function [Q, R, J, state_out] = cqrrpt(A, varargin)
%RANDLAPACK.CQRRPT  Randomized rank-revealing QR with column pivoting.
%
%   [Q, R, J] = randlapack.cqrrpt(A) computes a rank-revealing QR
%   factorization of a TALL matrix A, so that
%       A(:, J) ~= Q * R
%   with Q m-by-k orthonormal, R k-by-n upper-trapezoidal, and J a length-n
%   vector of 1-based column pivots. Output layout matches qr(A, 'vector').
%
%   k is DISCOVERED, not requested: CQRRPT reveals the numerical rank of A
%   and that rank is the inner dimension of Q and R. On a full-rank input
%   k = n. Read it off size(R, 1).
%
%   What the factorization guarantees, and what it does not: Q is
%   orthonormal and the LEADING k pivoted columns always reproduce exactly,
%       A(:, J(1:k)) == Q * R(:, 1:k)
%   to machine precision. The full identity A(:, J) == Q*R additionally
%   needs k to have reached the numerical rank of A -- the ordinary case,
%   including on rank-deficient input. It can fall short when CQRRPT's
%   internal Cholesky QR breaks down and the conservative fallback estimate
%   takes over, so check size(R, 1) and the residual on inputs whose
%   conditioning you do not know.
%
%   CQRRPT sketches A down to d = d_factor*n rows, so it requires
%   m >= d_factor*n. For wide or square matrices use randlapack.bqrrp.
%
%   Name-value options:
%       'eps'        conditioning tolerance for rank detection. Default:
%                    sqrt(eps(class(A))).
%
%                    eps is NOT a singular-value truncation threshold.
%                    CQRRPT first truncates using a fixed machine-epsilon
%                    cutoff, so columns well above machine precision are
%                    kept whatever eps says. eps takes effect only in the
%                    re-estimation that runs when the internal Cholesky QR
%                    fails (i.e. on a rank-deficient input), where the rank
%                    is cut once the R diagonal spans a ratio of
%                    sqrt(eps/eps(class(A))). It therefore bounds the
%                    orthogonality loss, which for Cholesky QR scales like
%                    u*cond(R)^2. Smaller eps means a stricter bound and a
%                    smaller returned rank.
%       'd_factor'   sketch embedding factor (>= 1.0). Default: 1.25.
%                    Larger values sketch more rows: steadier rank
%                    detection, more work, and a taller input requirement.
%       'state'      RNG state: a uint32 scalar seed, or a struct with
%                    .counter (uint32[4]) and .key (uint32[2]).
%
%   [..., state] = randlapack.cqrrpt(...) returns the advanced RNG state.
%
%   A must be single or double precision, real, and 2D.
%
%   Example:
%       B = randn(5000, 40);
%       A = [B, B(:, 1:10)];              % 50 columns, rank 40
%       [Q, R, J] = randlapack.cqrrpt(A);
%       detected_rank = size(R, 1)        % ~40
%       err = norm(A(:, J) - Q*R, 'fro') / norm(A, 'fro');
%
%   See also QR, RANDLAPACK.BQRRP, RANDLAPACK.RSVD.

    % --- Required argument check ---
    validateattributes(A, {'single', 'double'}, {'2d', 'real', 'finite'}, ...
                       mfilename, 'A', 1);
    [m, n] = size(A);
    if m == 0 || n == 0
        error('randlapack:cqrrpt:emptyInput', 'A must be nonempty');
    end

    % --- Defaults ---
    % eps default: sqrt of machine epsilon in the working precision, the
    % usual rank-tolerance scale for a Gram-based (CholeskyQR) method --
    % CQRRPT forms A'A, so its accuracy floor is the square root of the
    % precision, not the precision itself.
    opts = struct('eps', sqrt(eps(class(A))), ...
                  'd_factor', 1.25, ...
                  'state', rl_default_state());

    % --- Name-value parsing ---
    if mod(numel(varargin), 2) ~= 0
        error('randlapack:cqrrpt:nargin', ...
              'Options must be given as name-value pairs.');
    end
    known = fieldnames(opts);
    for i = 1:2:numel(varargin)
        name = varargin{i};
        if ~(ischar(name) || (isstring(name) && isscalar(name)))
            error('randlapack:cqrrpt:optionName', ...
                  'Option names must be character vectors or strings.');
        end
        name = char(name);
        idx = find(strcmpi(known, name), 1);
        if isempty(idx)
            error('randlapack:cqrrpt:unknownOption', ...
                  'Unknown option ''%s''. Valid options: %s.', ...
                  name, strjoin(known', ', '));
        end
        opts.(known{idx}) = varargin{i + 1};
    end

    % --- Validation ---
    validateattributes(opts.eps, {'numeric'}, {'scalar', 'real', 'nonnegative'}, ...
                       mfilename, 'eps');
    validateattributes(opts.d_factor, {'numeric'}, {'scalar', 'real', '>=', 1}, ...
                       mfilename, 'd_factor');

    % The tall-matrix requirement is checked here as well as in the MEX, so
    % the common mistake is caught before any marshalling happens and the
    % message can name the MATLAB-level alternative.
    d = floor(opts.d_factor * n);
    if m < d
        error('randlapack:cqrrpt:notTall', ...
              ['CQRRPT requires m >= d_factor*n. Got m=%d, n=%d, ' ...
               'd_factor=%g (needs m >= %d). Lower d_factor toward 1.0, ' ...
               'or use randlapack.bqrrp, which has no tall requirement.'], ...
              m, n, opts.d_factor, d);
    end

    state = rl_normalize_state(opts.state, mfilename);

    % --- Dispatch to MEX ---
    [Q, R, J, state_out] = cqrrpt_mex(A, cast(opts.d_factor, class(A)), ...
                                      cast(opts.eps, class(A)), state);
end
