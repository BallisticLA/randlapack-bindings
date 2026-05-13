function [Q, R, J, state_out] = bqrrp(A, varargin)
%RANDLAPACK.BQRRP  Randomized blocked QR with column pivoting.
%
%   [Q, R, J] = randlapack.bqrrp(A) computes a column pivoted QR
%   factorization of A such that
%
%       A(:, J) == Q * R   (up to numerical error),
%
%   where Q is m-by-k orthogonal, R is k-by-n upper-triangular,
%   k = min(m, n), and J is a length-n vector of 1-based pivot
%   indices. Output shape mirrors qr(A, 'vector') in economy form.
%
%   [Q, R, J] = randlapack.bqrrp(A, b_sz) sets the block size.
%   Default: 64.
%
%   [Q, R, J] = randlapack.bqrrp(A, b_sz, d_factor) sets the sketch
%   embedding factor (must satisfy d_factor >= 1.0). Default: 1.25.
%
%   [Q, R, J, state] = randlapack.bqrrp(..., state) advances and
%   returns the given RNG state. The state argument may be:
%       * omitted: a default Philox4x32 state is used (seed 0).
%       * a uint32 scalar: used as a seed.
%       * a struct with fields .counter (uint32 length 4) and
%         .key (uint32 length 2), as returned by a previous call.
%
%   A must be single or double precision, real, and 2D.
%
%   Example:
%       A = randn(2000, 200);
%       [Q, R, J] = randlapack.bqrrp(A);
%       err = norm(A(:, J) - Q * R, 'fro') / norm(A, 'fro');
%
%   See also QR.

    % --- Required argument check ---
    validateattributes(A, {'single', 'double'}, {'2d', 'real', 'finite'}, ...
                       mfilename, 'A', 1);
    [m, n] = size(A);
    if m == 0 || n == 0
        error('randlapack:bqrrp:emptyInput', 'A must be nonempty');
    end

    % --- Optional argument defaults ---
    b_sz     = int64(64);
    d_factor = cast(1.25, class(A));
    state    = make_default_state();

    if numel(varargin) >= 1 && ~isempty(varargin{1})
        b_sz = int64(varargin{1});
    end
    if numel(varargin) >= 2 && ~isempty(varargin{2})
        d_factor = cast(varargin{2}, class(A));
    end
    if numel(varargin) >= 3 && ~isempty(varargin{3})
        state = normalize_state(varargin{3});
    end
    if numel(varargin) >= 4
        error('randlapack:bqrrp:nargin', ...
              'Too many input arguments. Expected up to 4 (A, b_sz, d_factor, state).');
    end

    % --- Cross-argument validation ---
    validateattributes(b_sz, {'int64'}, {'scalar', 'positive'}, ...
                       mfilename, 'b_sz', 2);
    validateattributes(d_factor, {class(A)}, {'scalar', 'real', '>=', 1}, ...
                       mfilename, 'd_factor', 3);

    % b_sz larger than min(m, n) is wasteful but not wrong; BQRRP truncates
    % the block at the matrix boundary. Don't error, but warn loudly.
    k = min(m, n);
    if b_sz > k
        warning('randlapack:bqrrp:largeBlock', ...
                'b_sz=%d exceeds min(m, n)=%d; effective block size will be clipped.', ...
                b_sz, k);
    end

    % --- Dispatch to MEX ---
    [Q, R, J, state_out] = bqrrp_mex(A, b_sz, d_factor, state);
end


function state = make_default_state()
    state.counter = uint32([0, 0, 0, 0]);
    state.key     = uint32([0, 0]);
end


function state = normalize_state(in)
    if isstruct(in)
        if ~isfield(in, 'counter') || ~isfield(in, 'key')
            error('randlapack:bqrrp:state', ...
                  'state struct must have fields .counter and .key');
        end
        validateattributes(in.counter, {'uint32'}, {'vector', 'numel', 4}, ...
                           mfilename, 'state.counter');
        validateattributes(in.key,     {'uint32'}, {'vector', 'numel', 2}, ...
                           mfilename, 'state.key');
        state.counter = uint32(in.counter(:)).';
        state.key     = uint32(in.key(:)).';
    elseif isnumeric(in) && isscalar(in)
        % Treat scalar as a seed; place in key[1]. Counter starts at zero.
        seed = uint32(in);
        state.counter = uint32([0, 0, 0, 0]);
        state.key     = uint32([0, seed]);
    else
        error('randlapack:bqrrp:state', ...
              'state must be a struct or a scalar seed');
    end
end
