function [U, S, V, state_out] = rsvd(A, k, varargin)
%RANDLAPACK.RSVD  Randomized truncated SVD.
%
%   [U, S, V] = randlapack.rsvd(A, k) computes a rank-k (or lower)
%   approximate SVD of A, so that
%       A ~= U * S * V'
%   with U m-by-k orthonormal, S k-by-k diagonal, and V n-by-k orthonormal.
%   Output conventions match MATLAB's svd/svds.
%
%   s = randlapack.rsvd(A, k) returns the singular values as a vector,
%   again matching svd/svds called with one output.
%
%   The returned rank may be SMALLER than the k you asked for: the
%   underlying QB factorization stops early once the residual falls below
%   'tol'. Read the actual rank off the output shapes, e.g. size(S, 1).
%
%   U and V are orthonormal to DIFFERENT accuracies, structurally rather
%   than incidentally. V comes straight out of LAPACK's gesdd and is
%   orthonormal to machine precision. U is formed from the QB chain's
%   basis, orthogonalized by CholeskyQR, whose orthogonality loss scales
%   like u*cond(A_sketch)^2. On an ill-conditioned input U can be far
%   looser: measured on a 400-by-200 matrix with a geometric spectrum at
%   k=30, ||U'*U - I|| was ~3e-11 in double and ~1.5e-2 in single, against
%   ~7e-15 and ~3e-6 for V. Raising 'p' does not fix it. If you need U
%   orthonormal to machine precision on a badly conditioned input,
%   re-orthogonalize it (qr) or work in double.
%
%   Name-value options:
%       'tol'        residual tolerance for early termination.
%                    Default: 0 (run to the requested rank).
%       'block_sz'   QB block size. Default: min(k, 32). Smaller blocks
%                    give the tolerance more chances to trigger; larger
%                    blocks do more work per BLAS-3 call.
%       'p'          power iterations in the sketching stage. Default: 1.
%                    Raise it when the spectrum decays slowly.
%       'passes_per_iteration'  stabilization frequency. Default: 1.
%       'state'      RNG state: a uint32 scalar seed, or a struct with
%                    .counter (uint32[4]) and .key (uint32[2]).
%
%   [..., state] = randlapack.rsvd(...) returns the advanced RNG state;
%   passing it back in continues the stream rather than repeating it.
%
%   A must be single or double precision, real, and 2D.
%
%   Example:
%       A = randn(2000, 100) * randn(100, 500);   % rank 100
%       [U, S, V] = randlapack.rsvd(A, 100);
%       err = norm(A - U*S*V', 'fro') / norm(A, 'fro');
%
%   See also SVD, SVDS, RANDLAPACK.BQRRP.

    % --- Required arguments ---
    validateattributes(A, {'single', 'double'}, {'2d', 'real', 'finite'}, ...
                       mfilename, 'A', 1);
    [m, n] = size(A);
    if m == 0 || n == 0
        error('randlapack:rsvd:emptyInput', 'A must be nonempty');
    end
    validateattributes(k, {'numeric'}, {'scalar', 'positive', 'integer'}, ...
                       mfilename, 'k', 2);
    if k > min(m, n)
        error('randlapack:rsvd:rankTooLarge', ...
              'k=%d exceeds min(m, n)=%d.', k, min(m, n));
    end

    % --- Defaults ---
    % block_sz caps at 32 so the tolerance gets a chance to trigger on large
    % k, rather than the whole factorization arriving in one block.
    opts = struct('tol', 0, ...
                  'block_sz', min(double(k), 32), ...
                  'p', 1, ...
                  'passes_per_iteration', 1, ...
                  'state', rl_default_state());

    % --- Name-value parsing ---
    if mod(numel(varargin), 2) ~= 0
        error('randlapack:rsvd:nargin', ...
              'Options must be given as name-value pairs.');
    end
    known = fieldnames(opts);
    for i = 1:2:numel(varargin)
        name = varargin{i};
        if ~(ischar(name) || (isstring(name) && isscalar(name)))
            error('randlapack:rsvd:optionName', ...
                  'Option names must be character vectors or strings.');
        end
        name = char(name);
        idx = find(strcmpi(known, name), 1);
        if isempty(idx)
            error('randlapack:rsvd:unknownOption', ...
                  'Unknown option ''%s''. Valid options: %s.', ...
                  name, strjoin(known', ', '));
        end
        opts.(known{idx}) = varargin{i + 1};
    end

    % --- Validation ---
    validateattributes(opts.tol, {'numeric'}, {'scalar', 'real', 'nonnegative'}, ...
                       mfilename, 'tol');
    validateattributes(opts.block_sz, {'numeric'}, ...
                       {'scalar', 'positive', 'integer'}, mfilename, 'block_sz');
    validateattributes(opts.p, {'numeric'}, {'scalar', 'nonnegative', 'integer'}, ...
                       mfilename, 'p');
    validateattributes(opts.passes_per_iteration, {'numeric'}, ...
                       {'scalar', 'positive', 'integer'}, mfilename, ...
                       'passes_per_iteration');
    state = rl_normalize_state(opts.state, mfilename);

    % --- Dispatch to MEX ---
    % The MEX always returns s as a vector; the diagonal-matrix form is a
    % MATLAB-side convention (svd returns S as a matrix when called with
    % three outputs, and as a vector with one).
    [U, s, V, state_out] = rsvd_mex(A, int64(k), cast(opts.tol, class(A)), ...
                                    int64(opts.block_sz), int64(opts.p), ...
                                    int64(opts.passes_per_iteration), state);

    if nargout <= 1
        U = s;   % single output: the singular values, as svd(A) does
        return;
    end
    S = diag(s);
end
