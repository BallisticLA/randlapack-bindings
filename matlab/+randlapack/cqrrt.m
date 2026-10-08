function [out1, R, state_out] = cqrrt(A, varargin)
%RANDLAPACK.CQRRT  Randomized Cholesky QR of a tall full-rank matrix.
%
%   [Q, R] = randlapack.cqrrt(A) computes an unpivoted QR factorization of a
%   TALL, FULL-RANK matrix A, so that
%       A ~= Q * R
%   with Q m-by-n orthonormal and R n-by-n upper-triangular. Output layout
%   matches qr(A, 0) (the economy-size QR).
%
%   R = randlapack.cqrrt(A), with a single output, returns R only and skips
%   forming Q (CQRRT's R-only mode), like R = qr(A, 0). This is the cheaper
%   call when only R is needed, e.g. to solve least-squares problems or to
%   precondition with R.
%
%   How it works: CQRRT sketches A down to d = d_factor*n rows, takes a QR of
%   the small sketch, uses its R factor to precondition A, and finishes with
%   Cholesky QR on the well-conditioned result. It does not pivot and does
%   NOT detect rank: on a rank-deficient A it usually returns without error,
%   with Q*R still reproducing A but R numerically singular (tiny diagonal
%   entries). It stops with an error only when the sketch's R is exactly
%   singular (for example, A has a zero column) or the Cholesky step fails.
%   If the rank of A is uncertain, check abs(diag(R)) or use
%   randlapack.cqrrpt, which reveals the rank (or randlapack.bqrrp).
%
%   CQRRT requires m >= d_factor*n. For wide or square matrices use
%   randlapack.bqrrp.
%
%   Name-value options:
%       'd_factor'   sketch embedding factor (>= 1.0). Default: 1.25.
%                    Larger values sketch more rows: a better
%                    preconditioner, more work, and a taller input
%                    requirement.
%       'state'      RNG state: a uint32 scalar seed, or a struct with
%                    .counter (uint32[4]) and .key (uint32[2]).
%
%   [Q, R, state] = randlapack.cqrrt(...) returns the advanced RNG state.
%   (The single-output R-only form does not return it.)
%
%   A must be single or double precision, real, and 2D.
%
%   Example:
%       A = randn(20000, 200);
%       [Q, R] = randlapack.cqrrt(A);
%       err = norm(A - Q*R, 'fro') / norm(A, 'fro');
%       R_only = randlapack.cqrrt(A);     % no Q formed
%
%   See also QR, RANDLAPACK.CQRRPT, RANDLAPACK.BQRRP.

    % --- Required argument check ---
    validateattributes(A, {'single', 'double'}, {'2d', 'real', 'finite'}, ...
                       mfilename, 'A', 1);
    [m, n] = size(A);
    if m == 0 || n == 0
        error('randlapack:cqrrt:emptyInput', 'A must be nonempty');
    end

    % --- Defaults ---
    opts = struct('d_factor', 1.25, ...
                  'state', rl_default_state());

    % --- Name-value parsing ---
    if mod(numel(varargin), 2) ~= 0
        error('randlapack:cqrrt:nargin', ...
              'Options must be given as name-value pairs.');
    end
    known = fieldnames(opts);
    for i = 1:2:numel(varargin)
        name = varargin{i};
        if ~(ischar(name) || (isstring(name) && isscalar(name)))
            error('randlapack:cqrrt:optionName', ...
                  'Option names must be character vectors or strings.');
        end
        name = char(name);
        idx = find(strcmpi(known, name), 1);
        if isempty(idx)
            error('randlapack:cqrrt:unknownOption', ...
                  'Unknown option ''%s''. Valid options: %s.', ...
                  name, strjoin(known', ', '));
        end
        opts.(known{idx}) = varargin{i + 1};
    end

    % --- Validation ---
    validateattributes(opts.d_factor, {'numeric'}, {'scalar', 'real', '>=', 1}, ...
                       mfilename, 'd_factor');

    % The tall-matrix requirement is checked here as well as in the MEX, so
    % the common mistake is caught before any marshalling happens and the
    % message can name the MATLAB-level alternative.
    d = floor(opts.d_factor * n);
    if m < d
        error('randlapack:cqrrt:notTall', ...
              ['CQRRT requires m >= d_factor*n. Got m=%d, n=%d, ' ...
               'd_factor=%g (needs m >= %d). Lower d_factor toward 1.0, ' ...
               'or use randlapack.bqrrp, which has no tall requirement.'], ...
              m, n, opts.d_factor, d);
    end

    state = rl_normalize_state(opts.state, mfilename);

    % --- Dispatch to MEX ---
    d_factor = cast(opts.d_factor, class(A));
    if nargout <= 1
        % R-only mode: Q is not formed.
        [~, out1] = cqrrt_mex(A, d_factor, false, state);
    else
        [out1, R, state_out] = cqrrt_mex(A, d_factor, true, state);
    end
end
