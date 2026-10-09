function h = numa_query(A, max_pages)
%NUMA_QUERY Memory node of the pages of a dense array, sampled evenly (Linux; a query, nothing moves).
%   h = randlapack.numa_query(A)              up to 4,096 sampled pages
%   h = randlapack.numa_query(A, max_pages)
%   h.counts(k+1) = sampled pages on node k; h.absent = pages not faulted in; h.errors = other failures;
%   h.sampled = pages sampled (one per stratum); h.page_offset, h.align64 = A's first byte modulo 4096 and 64.
%   See also randlapack.numa_spread.
    if nargin < 2, max_pages = 4096; end
    if exist('numa_place_mex', 'file') ~= 3
        h = struct('counts', [], 'absent', NaN, 'sampled', 0, 'errors', NaN, 'page_offset', NaN, 'align64', NaN); return;
    end
    h = numa_place_mex('query', A, double(max_pages));
end
