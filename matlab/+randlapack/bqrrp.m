function [out1, out2, J, state_out] = bqrrp(A, varargin)
%RANDLAPACK.BQRRP  Randomized blocked QR with column pivoting.
%
%   Two output modes, selected via a trailing string argument:
%
%   'explicit' (default):  matches MATLAB qr(A, 'vector')
%       [Q, R, J] = randlapack.bqrrp(A)
%       [Q, R, J] = randlapack.bqrrp(A, 'explicit')
%   Q is m-by-k orthogonal, R is k-by-n upper-triangular, k = min(m, n),
%   J is a length-n vector of 1-based pivot indices, and
%       A(:, J) == Q * R   (up to numerical error).
%
%   'implicit':  matches RandLAPACK's native BQRRP output (GEQP3-style)
%       [A_out, tau, J] = randlapack.bqrrp(A, 'implicit')
%   A_out is m-by-n with Householder vectors below the diagonal and the
%   upper-triangular R factor in and above the diagonal; tau is the
%   length-n vector of Householder scalars. Q is left implicit; recover
%   it via lapack::ungqr or apply it via lapack::ormqr. Skips the
%   ungqr cost the 'explicit' mode pays.
%
%   Additional positional arguments (any order before the mode string):
%       b_sz       block size (positive int). Default: min(m, n).
%       d_factor   sketch embedding factor (>= 1.0). Default: 1.25.
%       state      RNG state. May be omitted, a uint32 scalar seed, or a
%                  struct with .counter (uint32[4]) and .key (uint32[2]).
%
%   [..., state] = randlapack.bqrrp(...) returns the advanced RNG state.
%
%   A must be single or double precision, real, and 2D.
%
%   Example:
%       A = randn(2000, 200);
%       [Q, R, J]       = randlapack.bqrrp(A);              % explicit (default)
%       err             = norm(A(:, J) - Q*R, 'fro') / norm(A, 'fro');
%       [A_out, tau, J] = randlapack.bqrrp(A, 'implicit');  % GEQP3-format
%       R_from_impl     = triu(A_out(1:size(A, 2), :));     % R lives in upper triangle
%
%   See also QR.

    % --- Required argument check ---
    validateattributes(A, {'single', 'double'}, {'2d', 'real', 'finite'}, ...
                       mfilename, 'A', 1);
    [m, n] = size(A);
    if m == 0 || n == 0
        error('randlapack:bqrrp:emptyInput', 'A must be nonempty');
    end

    % --- Defaults ---
    % b_sz default is min(m, n) so it is always in range; the user can
    % pass a smaller value to actually exercise the blocked structure.
    b_sz     = int64(min(m, n));
    d_factor = cast(1.25, class(A));
    state    = rl_default_state();
    mode     = 'explicit';

    % --- Argument parsing ---
    % Mode string ('explicit' or 'implicit') may appear anywhere in
    % varargin, mirroring qr(A, 'vector') / qr(A, 'matrix'). Remaining
    % positional args are b_sz, d_factor, state in that order.
    positional = {};
    for k = 1:numel(varargin)
        arg = varargin{k};
        if (ischar(arg) || (isstring(arg) && isscalar(arg))) ...
                && ismember(char(arg), {'explicit', 'implicit'})
            mode = char(arg);
        else
            positional{end+1} = arg; %#ok<AGROW>
        end
    end

    if numel(positional) >= 1 && ~isempty(positional{1})
        b_sz = int64(positional{1});
    end
    if numel(positional) >= 2 && ~isempty(positional{2})
        d_factor = cast(positional{2}, class(A));
    end
    if numel(positional) >= 3 && ~isempty(positional{3})
        state = rl_normalize_state(positional{3}, mfilename);
    end
    if numel(positional) >= 4
        error('randlapack:bqrrp:nargin', ...
              ['Too many positional arguments. Expected up to 4 ' ...
               '(A, b_sz, d_factor, state) plus an optional mode string.']);
    end

    % --- Cross-argument validation ---
    validateattributes(b_sz, {'int64'}, {'scalar', 'positive'}, ...
                       mfilename, 'b_sz', 2);
    validateattributes(d_factor, {class(A)}, {'scalar', 'real', '>=', 1}, ...
                       mfilename, 'd_factor', 3);

    % b_sz larger than min(m, n) is wasteful but not wrong; BQRRP clips
    % the block at the matrix boundary. Don't error, but warn loudly.
    if b_sz > min(m, n)
        warning('randlapack:bqrrp:largeBlock', ...
                'b_sz=%d exceeds min(m, n)=%d; effective block size will be clipped.', ...
                b_sz, min(m, n));
    end

    % --- Dispatch to MEX ---
    % The MEX returns the pair (out1, out2) whose meaning depends on mode:
    %   mode='explicit': out1 = Q (m-by-k), out2 = R (k-by-n)
    %   mode='implicit': out1 = A_out (m-by-n, GEQP3-format),
    %                    out2 = tau   (length-n Householder scalars)
    [out1, out2, J, state_out] = bqrrp_mex(A, b_sz, d_factor, state, mode);
end

