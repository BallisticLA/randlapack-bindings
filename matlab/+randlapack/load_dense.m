function [A, stats] = load_dense(file, n_rows, n_cols, varargin)
%LOAD_DENSE Read a float64 matrix from a raw file with many threads, so its pages spread over the NUMA nodes (Linux).
%   [A, stats] = randlapack.load_dense(file, n_rows, n_cols, 'Offset', bytes, 'Divisor', d, 'Threads', t, ...
%                                      'Placement', 'interleave' | 'first_touch')
%
%   Reads n_rows*n_cols float64 values stored contiguously at byte Offset (default 0) of file, in MATLAB's column-major
%   order (as fread(fid, [n_rows n_cols], 'double') would), into a new array, and divides every value by Divisor
%   (default 1: no division) with the same correctly rounded IEEE division as MATLAB's A/d. Each of Threads threads
%   (default 0: OMP_NUM_THREADS, else all cores) reads its own contiguous span with pread into storage nobody has
%   touched yet, and only ONE copy of the matrix exists at peak (fread into zeros(n) followed by A = A + 0 holds two).
%   Placement 'interleave' (default) gives the empty array the MPOL_INTERLEAVE policy first, so its pages fault in
%   round robin over the NUMA nodes (what numactl --interleave=all does for the whole process); 'first_touch'
%   leaves each page on the node of the thread that reads it. A .npy file of a C-ordered symmetric matrix
%   can be read this way too: its row-major data is the transpose, which equals the matrix when it is symmetric.
%
%   stats: threads, ms, rss_before_mib, rss_after_alloc_mib (about rss_before: the pages are untouched until read),
%   rss_after_fill_mib, interleave_set (false on a one-node machine, where there is nothing to interleave),
%   interleave_errno, nodes (allowed node ids).
%
%   See also randlapack.numa_spread, randlapack.numa_query.
    p = inputParser;
    addParameter(p, 'Offset', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x == floor(x));
    addParameter(p, 'Divisor', 1, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x ~= 0);
    addParameter(p, 'Threads', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x == floor(x));
    addParameter(p, 'Placement', 'interleave', @(x) any(strcmp(x, {'interleave', 'first_touch'})));
    parse(p, varargin{:});
    o = p.Results;
    if exist('numa_place_mex', 'file') ~= 3
        error('randlapack:load_dense:unavailable', 'numa_place_mex is not built for this platform.');
    end
    [A, stats] = numa_place_mex('load', char(file), double(o.Offset), double(n_rows), double(n_cols), ...
                                double(o.Divisor), double(o.Threads), double(strcmp(o.Placement, 'interleave')));
end
