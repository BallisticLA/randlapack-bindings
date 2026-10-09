function stats = numa_spread(A)
%NUMA_SPREAD Interleave the memory pages of a large dense array over the NUMA nodes, in place (Linux).
%   stats = randlapack.numa_spread(A)
%
%   MATLAB fills a matrix read from disk (fread, load) or built by assignment with one thread, so all of its pages sit on
%   one memory node and every multithreaded product with it is limited by that node's bandwidth (2.4x slower products
%   at n = 50,000 on a 4-node Xeon 6248R bigmem host, 2026-09-24). numa_spread moves A's pages in place so they interleave over
%   the nodes this process may use, round robin by 2 MiB granule (move_pages with an explicit destination per page;
%   mbind's MPOL_INTERLEAVE + MPOL_MF_MOVE would leave pages that already sit on an allowed node unmoved). A's values
%   are not touched and no copy of A is made, unlike re-touching it with A = A + 0. A and every variable sharing its
%   data keep the new placement.
%
%   stats fields: supported, nodes (allowed node ids), bytes, threads (that ran), ms, pages (pages of A's span submitted
%   for movement; 0 when nothing needs to move, i.e. one allowed node), retry_calls (move_pages retries), on_target (pages
%   on their destination node at the end), not_moved (pages still elsewhere or refused, after one retry), unresolved
%   (pages the kernel never reported on: a failed call or an early stop), failed_batches, absent (pages never
%   touched), errno (the first error seen), and before/after page histograms (counts per node over up to 4,096 pages
%   sampled one per stratum; absent = never touched, errors = other failures). Placement is complete when
%   on_target + absent == pages. On a one-node machine nothing moves. Where the helper MEX is absent (not Linux, or
%   not built) stats.supported is false and A is unchanged.
%
%   See also randlapack.numa_query, randlapack.load_dense.
    if ~(isnumeric(A) && ~issparse(A) && isreal(A) && (isa(A,'double') || isa(A,'single')))
        error('randlapack:numa_spread:class', 'A must be a full real single or double array.');
    end
    if exist('numa_place_mex', 'file') ~= 3
        stats = struct('supported', false, 'nodes', [], 'bytes', 0, 'threads', 0, 'ms', 0, 'pages', 0, 'retry_calls', 0, 'on_target', 0, ...
                       'failed_batches', 0, 'not_moved', 0, 'unresolved', 0, 'absent', 0, 'errno', 0, 'before', [], 'after', []);
        return;
    end
    stats = numa_place_mex('spread', A);
end
