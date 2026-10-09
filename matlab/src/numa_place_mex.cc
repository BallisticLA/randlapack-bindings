// MEX helper for NUMA page placement of large dense MATLAB arrays (Linux only), using MATLAB's C++ Data API.
//
// Why: Linux places an anonymous page on the memory node of the thread that first writes it. MATLAB fills a matrix it
// reads from disk (fread, load) with one thread, so every page of A lands on one node, and every multithreaded product
// with A then streams it through that node's memory controller. On ISAAC's 4-node Xeon 6248R hosts that made the
// funNystrom++ products 2.4x slower at n = 50,000 (2026-09-24, on a 4-node bigmem host, clrm1217). The fixes below spread A over the nodes while holding
// ONE copy of it: no transient second matrix, unlike re-touching it with A = A + 0 or copying it in the MEX.
//
//   stats = numa_place_mex('spread', A)
//       Interleaves the pages of A over the memory nodes this process may use (get_mempolicy MPOL_F_MEMS_ALLOWED), in
//       place: move_pages(MPOL_MF_MOVE) with an explicit destination for every page of the page-aligned span of A's
//       buffer, round robin over the nodes by 2 MiB granule (the transparent-huge-page size, so a huge page gets one
//       destination), split over threads. mbind(MPOL_INTERLEAVE, MPOL_MF_MOVE) is NOT used: it leaves every page that
//       already sits on one of the allowed nodes where it is (mbind(2): "pages already residing on the specified nodes
//       will not be moved such that they are interleaved"), which is exactly the one-node matrix this must fix. The
//       values of A are untouched (the kernel moves pages, it does not rewrite them) and nothing is copied in MATLAB.
//       The span may include parts of the first and last page that belong to neighbouring allocations; moving those
//       is harmless.
//   hist = numa_place_mex('query', A [, max_pages])
//       Memory node of up to max_pages (default 4096) pages sampled evenly across A, via move_pages with nodes = NULL
//       (a query; nothing moves). hist.counts(k+1) = pages on node k; hist.absent = pages not yet faulted in.
//   [A, stats] = numa_place_mex('load', file, offset_bytes, n_rows, n_cols, divisor, n_threads, interleave)
//       Reads n_rows*n_cols float64 values stored contiguously (column-major as MATLAB reads them) at offset_bytes of
//       file into a NEW uninitialized MATLAB array, with n_threads threads (0 = OMP_NUM_THREADS, else all cores), each
//       pread-ing its own contiguous span. With interleave = 1 (the wrapper's default) the empty buffer first gets the
//       MPOL_INTERLEAVE policy (mbind, nothing to move yet), so the reads fault its pages in round robin over the
//       allowed nodes, the placement numactl --interleave=all gives; with interleave = 0 the pages land on the nodes
//       of the threads that first touch them (best effort: the threads are not pinned, so the scheduler decides). (On ISAAC's 6248R nodes interleave measured faster than threaded first
//       touch for products with A: gemv 100 vs 77 GB/s, stream 3, 2026-10-07.) Every
//       value is then divided by divisor unless divisor == 1: a correctly rounded IEEE division, the same operation as
//       MATLAB's blk/divisor (this file must not be built with -ffast-math or -freciprocal-math). One copy of the
//       matrix exists at peak; stats reports the resident-set growth after allocation (~0: pages untouched) and fill.
//
// The MATLAB-side wrappers are randlapack.numa_spread, randlapack.numa_query and randlapack.load_dense.

#include "mex.hpp"
#include "mexAdapter.hpp"

#if defined(__FAST_MATH__)
#error "numa_place_mex must not be built with -ffast-math: load_dense's division has to be the IEEE one (bit-identical to MATLAB's A/d)"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

struct MexError {
    std::string id;
    std::string msg;
};

#ifdef __linux__
// Constants of <linux/mempolicy.h>, spelled out so the MEX builds without libnuma's headers.
constexpr int kMpolInterleave = 3;
constexpr unsigned kMpolMfMove = 1u << 1;
constexpr unsigned long kMpolFMemsAllowed = 1ul << 2;
constexpr unsigned long kMaxNodes = 1024;   // bits in the node masks below
constexpr size_t kMaskWords = kMaxNodes / (8 * sizeof(unsigned long));

long sys_mbind(void* addr, unsigned long len, int mode, const unsigned long* mask, unsigned long maxnode,
               unsigned flags) {
    return syscall(SYS_mbind, addr, len, mode, mask, maxnode, flags);
}
long sys_get_mempolicy(int* mode, unsigned long* mask, unsigned long maxnode, void* addr, unsigned long flags) {
    return syscall(SYS_get_mempolicy, mode, mask, maxnode, addr, flags);
}
long sys_move_pages(int pid, unsigned long count, void** pages, const int* nodes, int* status, int flags) {
    return syscall(SYS_move_pages, pid, count, pages, nodes, status, flags);
}

size_t page_size() { return static_cast<size_t>(sysconf(_SC_PAGESIZE)); }

double rss_mib() {
    long pages = 0, rss = 0;
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (f) { if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0; std::fclose(f); }
    return static_cast<double>(rss) * static_cast<double>(page_size()) / 1048576.0;
}
#endif

// CPUs this process may run on (a SLURM job's cpuset), not the whole host.
size_t usable_cpus() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int c = CPU_COUNT(&set);
        if (c > 0) return static_cast<size_t>(c);
    }
#endif
    return std::max<size_t>(1, std::thread::hardware_concurrency());
}

size_t thread_count(size_t requested, size_t work_units) {
    size_t hw = usable_cpus();
    size_t nt = requested;
    if (nt == 0) {
        const char* ot = std::getenv("OMP_NUM_THREADS");
        const int v = ot ? std::atoi(ot) : 0;   // unset, empty, zero or negative: all usable CPUs
        nt = v > 0 ? static_cast<size_t>(v) : hw;
    }
    return std::max<size_t>(1, std::min({nt, hw, std::max<size_t>(1, work_units)}));
}

// Runs body(t) for t = 0..nt-1, t = 0 on the calling thread. A failed thread creation joins the started threads and
// runs the remaining indices on the calling thread, so no joinable std::thread is ever destroyed (that would
// terminate MATLAB). An exception out of body (e.g. bad_alloc) is caught in the thread that raised it, since one
// escaping a std::thread would terminate MATLAB too; the return value is false if any body threw. *ran, if given,
// receives the number of threads that actually ran bodies (the caller plus those created).
template <typename F>
bool run_threads(size_t nt, F&& body, size_t* ran = nullptr) {
    std::atomic<bool> ok{true};
    auto safe = [&](size_t t) { try { body(t); } catch (...) { ok.store(false); } };
    std::vector<std::thread> pool;
    size_t started = 1;
    try {
        pool.reserve(nt);
        for (size_t t = 1; t < nt; ++t) { pool.emplace_back(safe, t); ++started; }
    } catch (...) {
    }
    safe(0);
    for (auto& th : pool) th.join();
    for (size_t t = started; t < nt; ++t) safe(t);
    if (ran) *ran = started;
    return ok.load();
}

}  // namespace

using matlab::mex::ArgumentList;
using matlab::data::Array;
using matlab::data::ArrayFactory;
using matlab::data::ArrayType;

class MexFunction : public matlab::mex::Function {
    std::shared_ptr<matlab::engine::MATLABEngine> engine = getEngine();
    ArrayFactory factory;

    [[noreturn]] static void raise(const std::string& id, const std::string& msg) { throw MexError{id, msg}; }

    // Address and byte length of a full numeric array's buffer, read through the const iterator (no unsharing).
    static void span_of(const Array& a, const void*& ptr, size_t& bytes) {
        ptr = nullptr; bytes = 0;
        const size_t ne = a.getNumberOfElements();
        if (ne == 0) return;
        switch (a.getType()) {
            case ArrayType::DOUBLE: { const matlab::data::TypedArray<double> t = a; ptr = &*t.cbegin(); bytes = ne * 8; break; }
            case ArrayType::SINGLE: { const matlab::data::TypedArray<float> t = a; ptr = &*t.cbegin(); bytes = ne * 4; break; }
            default: raise("randlapack:numa_place:class", "A must be a full single or double array");
        }
    }

    double scalar(const Array& a, const char* what) {
        if (a.getNumberOfElements() != 1 || a.getType() != ArrayType::DOUBLE)
            raise("randlapack:numa_place:arg", std::string(what) + " must be a double scalar");
        const matlab::data::TypedArray<double> t = a;
        return t[0];
    }

    std::string text(const Array& a, const char* what) {
        if (a.getType() != ArrayType::CHAR) raise("randlapack:numa_place:arg", std::string(what) + " must be a char array");
        const matlab::data::CharArray c = a;
        return c.toAscii();
    }

#ifdef __linux__
    // Allowed memory nodes of this process (the cpuset of a SLURM job, all online nodes otherwise).
    // Placement is only a performance helper: where the NUMA syscalls are unavailable (no CONFIG_NUMA, or a seccomp
    // profile denying them) this returns no nodes and sets err, and the callers skip placement instead of failing.
    std::vector<int> allowed_nodes(std::vector<unsigned long>& mask, int& err) {
        mask.assign(kMaskWords, 0ul);
        err = 0;
        int mode = 0;
        if (sys_get_mempolicy(&mode, mask.data(), kMaxNodes, nullptr, kMpolFMemsAllowed) != 0) {
            err = errno;
            return {};
        }
        std::vector<int> nodes;
        for (unsigned long b = 0; b < kMaxNodes; ++b)
            if (mask[b / (8 * sizeof(unsigned long))] & (1ul << (b % (8 * sizeof(unsigned long))))) nodes.push_back(static_cast<int>(b));
        return nodes;
    }

    // Node histogram of up to max_pages pages sampled evenly over [base, base + bytes).
    // One page from each of m equal strata of the span, at a deterministic pseudo-random offset inside its stratum:
    // evenly spaced samples could alias with spread's round robin (a stride of a whole number of 2 MiB granules times
    // the node count would sample one node only). absent counts -ENOENT (never touched); errors, any other failure.
    void histogram(const void* base, size_t bytes, size_t max_pages, std::vector<double>& counts, double& absent,
                   double& sampled, double& errors) {
        counts.clear(); absent = 0; sampled = 0; errors = 0;
        if (bytes == 0) return;
        const size_t ps = page_size();
        const uintptr_t lo = reinterpret_cast<uintptr_t>(base) & ~(uintptr_t)(ps - 1);
        const uintptr_t hi = reinterpret_cast<uintptr_t>(base) + bytes;
        const size_t npages = (hi - lo + ps - 1) / ps;
        const size_t m = std::max<size_t>(1, std::min(max_pages, npages));
        std::vector<void*> pages(m);
        std::vector<int> status(m, -1);
        for (size_t i = 0; i < m; ++i) {
            const size_t s0 = npages * i / m, s1 = npages * (i + 1) / m;
            const size_t width = std::max<size_t>(1, s1 - s0);
            const size_t jitter = static_cast<size_t>((static_cast<uint64_t>(i) * 2654435761ull + 12345ull) % width);
            pages[i] = reinterpret_cast<void*>(lo + std::min(npages - 1, s0 + jitter) * ps);
        }
        if (sys_move_pages(0, m, pages.data(), nullptr, status.data(), 0) != 0) {
            errors = static_cast<double>(m); sampled = static_cast<double>(m);   // query unavailable: all unknown
            return;
        }
        for (size_t i = 0; i < m; ++i) {
            if (status[i] >= 0) {
                if (static_cast<size_t>(status[i]) >= counts.size()) counts.resize(status[i] + 1, 0.0);
                counts[status[i]] += 1.0;
            } else if (status[i] == -ENOENT) {
                absent += 1.0;   // not faulted in yet
            } else {
                errors += 1.0;
            }
        }
        sampled = static_cast<double>(m);
    }

    matlab::data::StructArray hist_struct(const std::vector<double>& counts, double absent, double sampled,
                                          double errors) {
        auto s = factory.createStructArray({1, 1}, {"counts", "absent", "sampled", "errors"});
        s[0]["errors"] = factory.createScalar<double>(errors);
        s[0]["counts"] = factory.createArray<double>({1, counts.size()}, counts.data(), counts.data() + counts.size());
        s[0]["absent"] = factory.createScalar<double>(absent);
        s[0]["sampled"] = factory.createScalar<double>(sampled);
        return s;
    }

    void spread(ArgumentList& outputs, ArgumentList& inputs) {
        if (inputs.size() != 2) raise("randlapack:numa_place:nargin", "spread takes one array");
        const void* ptr; size_t bytes;
        span_of(inputs[1], ptr, bytes);
        std::vector<unsigned long> mask;
        int nodes_errno = 0;
        const std::vector<int> nodes = allowed_nodes(mask, nodes_errno);
        std::vector<double> before, after; double ab = 0, aa = 0, sb = 0, sa = 0, eb = 0, ea = 0;
        histogram(ptr, bytes, 4096, before, ab, sb, eb);
        const auto t0 = std::chrono::steady_clock::now();
        std::atomic<long> failed_batches{0}, not_moved{0}, unresolved{0}, absent{0}, pages_total{0}, moved_ok{0}, retry_calls{0};
        std::atomic<int> first_errno{0};
        size_t nt = 1, nt_ran = 0;   // 0 threads when the spread is skipped (one node)
        // RANDLAPACK_NUMA_SPREAD_FORCE=1 runs the move_pages calls on a one-node machine too (every destination is node
        // 0 there, so nothing moves): a test hook for the syscall path, which a laptop cannot otherwise reach.
        const char* force = std::getenv("RANDLAPACK_NUMA_SPREAD_FORCE");
        const bool forced = force != nullptr && force[0] == '1';
        if (bytes > 0 && !nodes.empty() && (nodes.size() > 1 || forced)) {
            const size_t ps = page_size();
            const uintptr_t lo = reinterpret_cast<uintptr_t>(ptr) & ~(uintptr_t)(ps - 1);
            const uintptr_t hi = (reinterpret_cast<uintptr_t>(ptr) + bytes + ps - 1) & ~(uintptr_t)(ps - 1);
            const uintptr_t granule = (uintptr_t)2 << 20;
            const uintptr_t g0 = lo / granule, g1 = (hi + granule - 1) / granule;
            const size_t ngran = static_cast<size_t>(g1 - g0);
            const size_t nn = nodes.size();
            // Threads own whole 2 MiB granules (a transparent huge page is never split between two concurrent
            // move_pages calls), at least 32 granules (64 MiB) each; batches of 65,536 pages per call.
            nt = thread_count(0, ngran / 32);
            // pages = the whole span, counted up front: a worker that fails before or while submitting leaves its
            // remainder in no category, so on_target + absent < pages and the placement reads as incomplete.
            pages_total.store(static_cast<long>((hi - lo) / ps));
            constexpr int kUnset = std::numeric_limits<int>::min();
            const bool all_ran = run_threads(nt, [&](size_t t) {
                const uintptr_t a0 = std::max(lo, (g0 + ngran * t / nt) * granule);
                const uintptr_t a1 = std::min(hi, (g0 + ngran * (t + 1) / nt) * granule);
                if (a1 <= a0) return;
                const size_t tp = static_cast<size_t>((a1 - a0) / ps);
                constexpr size_t kBatch = 65536;
                std::vector<void*> pages, rpages;
                std::vector<int> dst, status, rdst, rstatus;
                try { pages.reserve(kBatch); dst.reserve(kBatch); status.resize(kBatch); }
                catch (...) { failed_batches.fetch_add(1); unresolved.fetch_add(static_cast<long>(tp)); return; }
                auto note_errno = [&](int e) { int z = 0; first_errno.compare_exchange_strong(z, e); };
                for (size_t b0 = 0; b0 < tp; b0 += kBatch) {
                    const size_t cnt = std::min(kBatch, tp - b0);
                    pages.clear(); dst.clear();
                    for (size_t i = 0; i < cnt; ++i) {
                        const uintptr_t a = a0 + (b0 + i) * ps;
                        pages.push_back(reinterpret_cast<void*>(a));
                        dst.push_back(nodes[(a / granule) % nn]);
                    }
                    // The kernel writes a status only for the pages it got to: a positive return (pages not migrated,
                    // Linux >= 4.17) can come back with the rest of the list unprocessed. A sentinel marks those.
                    std::fill(status.begin(), status.begin() + cnt, kUnset);
                    long r = sys_move_pages(0, cnt, pages.data(), dst.data(), status.data(), kMpolMfMove);
                    if (r < 0) {
                        note_errno(errno); failed_batches.fetch_add(1); unresolved.fetch_add(static_cast<long>(cnt));
                        continue;
                    }
                    // One retry for every page not yet on its destination (unprocessed, or failed other than -ENOENT),
                    // one 2 MiB granule per call: the kernel stops a call at the first same-destination run (granule) it
                    // cannot migrate, so one stuck granule must not hold back the rest of the batch.
                    auto off_target = [&](size_t i) {
                        return status[i] == kUnset || (status[i] < 0 && status[i] != -ENOENT) || (status[i] >= 0 && status[i] != dst[i]);
                    };
                    std::vector<size_t> ridx;
                    for (size_t i = 0; i < cnt; ++i) if (off_target(i)) ridx.push_back(i);
                    // First ask where those pages are (nodes = NULL: a query, no LRU drain). With transparent huge pages
                    // a tail page typically reports -EBUSY while its head moves the whole folio, so most "off target"
                    // entries are already on target; only the rest are retried.
                    if (!ridx.empty()) {
                        rpages.clear();
                        for (size_t i : ridx) rpages.push_back(pages[i]);
                        rstatus.assign(rpages.size(), kUnset);
                        if (sys_move_pages(0, rpages.size(), rpages.data(), nullptr, rstatus.data(), 0) == 0) {
                            std::vector<size_t> still;
                            for (size_t m = 0; m < ridx.size(); ++m) {
                                if (rstatus[m] >= 0 || rstatus[m] == -ENOENT) status[ridx[m]] = rstatus[m];
                                if (off_target(ridx[m])) still.push_back(ridx[m]);
                            }
                            ridx.swap(still);
                        }
                    }
                    for (size_t j = 0; j < ridx.size();) {
                        const uintptr_t gr = reinterpret_cast<uintptr_t>(pages[ridx[j]]) / granule;
                        size_t k = j;
                        rpages.clear(); rdst.clear();
                        while (k < ridx.size() && reinterpret_cast<uintptr_t>(pages[ridx[k]]) / granule == gr) {
                            rpages.push_back(pages[ridx[k]]); rdst.push_back(dst[ridx[k]]); ++k;
                        }
                        rstatus.assign(rpages.size(), kUnset);
                        retry_calls.fetch_add(1);
                        r = sys_move_pages(0, rpages.size(), rpages.data(), rdst.data(), rstatus.data(), kMpolMfMove);
                        if (r < 0) {   // the retried pages' outcome is unknown: unresolved, whatever the first call said
                            note_errno(errno); failed_batches.fetch_add(1);
                            for (size_t m = j; m < k; ++m) status[ridx[m]] = kUnset;
                        } else {
                            for (size_t m = j; m < k; ++m) {
                                status[ridx[m]] = rstatus[m - j];
                                if (rstatus[m - j] < 0 && rstatus[m - j] != -ENOENT && rstatus[m - j] != kUnset) note_errno(-rstatus[m - j]);
                            }
                        }
                        j = k;
                    }
                    // The kernel's last word: where is every page still not on target? move_pages with nodes = NULL moves
                    // nothing and reports each page's node. This also makes the counts exact on 4.17-5.5 kernels that
                    // drop the status of a final run of pages already on their target.
                    ridx.clear();
                    for (size_t i = 0; i < cnt; ++i) if (off_target(i)) ridx.push_back(i);
                    if (!ridx.empty()) {
                        rpages.clear();
                        for (size_t i : ridx) rpages.push_back(pages[i]);
                        rstatus.assign(rpages.size(), kUnset);
                        if (sys_move_pages(0, rpages.size(), rpages.data(), nullptr, rstatus.data(), 0) == 0)
                            for (size_t m = 0; m < ridx.size(); ++m)
                                if (rstatus[m] >= 0 || rstatus[m] == -ENOENT || status[ridx[m]] == kUnset) status[ridx[m]] = rstatus[m];
                    }
                    for (size_t i = 0; i < cnt; ++i) {
                        if (status[i] == kUnset) unresolved.fetch_add(1);
                        else if (status[i] == -ENOENT) absent.fetch_add(1);         // never touched: no page to move
                        else if (status[i] < 0) { not_moved.fetch_add(1); note_errno(-status[i]); }
                        else if (status[i] != dst[i]) not_moved.fetch_add(1);       // still on another node
                        else moved_ok.fetch_add(1);
                    }
                }
            }, &nt_ran);
            if (!all_ran) failed_batches.fetch_add(1);   // a worker threw (out of memory): its span may be unmoved
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        histogram(ptr, bytes, 4096, after, aa, sa, ea);
        auto s = factory.createStructArray({1, 1}, {"supported", "nodes", "bytes", "threads", "ms", "pages", "on_target",
                                                    "failed_batches", "not_moved", "unresolved", "absent", "errno",
                                                    "retry_calls", "before", "after"});
        s[0]["retry_calls"] = factory.createScalar<double>(static_cast<double>(retry_calls.load()));
        std::vector<double> nd(nodes.begin(), nodes.end());
        s[0]["supported"] = factory.createScalar<bool>(true);
        s[0]["nodes"] = factory.createArray<double>({1, nd.size()}, nd.data(), nd.data() + nd.size());
        s[0]["bytes"] = factory.createScalar<double>(static_cast<double>(bytes));
        s[0]["threads"] = factory.createScalar<double>(static_cast<double>(nt_ran));   // threads that ran, not asked for
        s[0]["ms"] = factory.createScalar<double>(ms);
        s[0]["pages"] = factory.createScalar<double>(static_cast<double>(pages_total.load()));
        s[0]["on_target"] = factory.createScalar<double>(static_cast<double>(moved_ok.load()));
        s[0]["failed_batches"] = factory.createScalar<double>(static_cast<double>(failed_batches.load()));
        s[0]["unresolved"] = factory.createScalar<double>(static_cast<double>(unresolved.load()));
        s[0]["not_moved"] = factory.createScalar<double>(static_cast<double>(not_moved.load()));
        s[0]["absent"] = factory.createScalar<double>(static_cast<double>(absent.load()));
        s[0]["errno"] = factory.createScalar<double>(static_cast<double>(first_errno.load() ? first_errno.load() : nodes_errno));
        s[0]["before"] = hist_struct(before, ab, sb, eb);
        s[0]["after"] = hist_struct(after, aa, sa, ea);
        outputs[0] = std::move(s);
    }

    void query(ArgumentList& outputs, ArgumentList& inputs) {
        if (inputs.size() < 2 || inputs.size() > 3) raise("randlapack:numa_place:nargin", "query takes an array and an optional page count");
        const void* ptr; size_t bytes;
        span_of(inputs[1], ptr, bytes);
        size_t mp = 4096;
        if (inputs.size() == 3) {
            const double v = scalar(inputs[2], "max_pages");
            if (!std::isfinite(v) || v < 1 || v > 1.0e9 || v != std::floor(v))
                raise("randlapack:numa_place:arg", "max_pages must be a positive integer <= 1e9");
            mp = static_cast<size_t>(v);
        }
        std::vector<double> c; double ab = 0, sm = 0, er = 0;
        histogram(ptr, bytes, mp, c, ab, sm, er);
        auto h = hist_struct(c, ab, sm, er);
        // Offset of A's first byte within its page and modulo 64 bytes: MKL's rounding can depend on alignment, so a
        // bit-level difference between two placements of the same matrix is read against this.
        auto s = factory.createStructArray({1, 1}, {"counts", "absent", "sampled", "errors", "page_offset", "align64"});
        const uintptr_t a = reinterpret_cast<uintptr_t>(ptr);
        s[0]["counts"] = h[0]["counts"]; s[0]["absent"] = h[0]["absent"]; s[0]["sampled"] = h[0]["sampled"];
        s[0]["errors"] = h[0]["errors"];
        s[0]["page_offset"] = factory.createScalar<double>(static_cast<double>(a % page_size()));
        s[0]["align64"] = factory.createScalar<double>(static_cast<double>(a % 64));
        outputs[0] = std::move(s);
    }

    void load(ArgumentList& outputs, ArgumentList& inputs) {
        if (inputs.size() != 8) raise("randlapack:numa_place:nargin", "load takes file, offset_bytes, n_rows, n_cols, divisor, n_threads, interleave");
        const std::string file = text(inputs[1], "file");
        const double off_d = scalar(inputs[2], "offset_bytes"), nr_d = scalar(inputs[3], "n_rows"),
                     nc_d = scalar(inputs[4], "n_cols"), divisor = scalar(inputs[5], "divisor"),
                     nt_d = scalar(inputs[6], "n_threads"), il_d = scalar(inputs[7], "interleave");
        if (il_d != 0 && il_d != 1) raise("randlapack:numa_place:arg", "interleave must be 0 or 1");
        for (double v : {off_d, nr_d, nc_d, nt_d})
            if (!(v >= 0) || v != std::floor(v) || v > 9.0e15) raise("randlapack:numa_place:arg", "offset, sizes and thread count must be nonnegative integers");
        if (!std::isfinite(divisor) || divisor == 0) raise("randlapack:numa_place:arg", "divisor must be finite and nonzero");
        const size_t nr = static_cast<size_t>(nr_d), nc = static_cast<size_t>(nc_d);
        const off_t off = static_cast<off_t>(off_d);
        const size_t ne = nr * nc;
        if (nr != 0 && ne / nr != nc) raise("randlapack:numa_place:arg", "n_rows * n_cols overflows");
        const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) raise("randlapack:numa_place:open", "cannot open " + file + ": " + std::strerror(errno));
        struct FdGuard { int fd; ~FdGuard() { ::close(fd); } } guard{fd};
        const off_t end = ::lseek(fd, 0, SEEK_END);
        if (end < 0 || static_cast<double>(end) < off_d + 8.0 * static_cast<double>(ne))
            raise("randlapack:numa_place:short", file + " is shorter than offset + 8*n_rows*n_cols bytes");
        const double rss0 = rss_mib();
        // createBuffer returns uninitialized storage: its pages are first touched by the pread calls below.
        matlab::data::buffer_ptr_t<double> buf = factory.createBuffer<double>(ne);
        const double rss_alloc = rss_mib();
        double* dst = buf.get();
        // Interleave policy on the untouched buffer (rounded out to whole pages; a boundary page shared with a
        // neighbouring allocation is already present and simply stays where it is).
        int il_errno = 0; bool il_set = false;
        std::vector<unsigned long> mask;
        const std::vector<int> nodes = allowed_nodes(mask, il_errno);   // reported in stats for either placement
        if (il_d == 1 && ne > 0) {
            const char* force = std::getenv("RANDLAPACK_NUMA_SPREAD_FORCE");
            if (nodes.size() > 1 || (force != nullptr && force[0] == '1')) {
                const size_t ps = page_size();
                const uintptr_t lo = reinterpret_cast<uintptr_t>(dst) & ~(uintptr_t)(ps - 1);
                const uintptr_t hi = (reinterpret_cast<uintptr_t>(dst + ne) + ps - 1) & ~(uintptr_t)(ps - 1);
                if (sys_mbind(reinterpret_cast<void*>(lo), hi - lo, kMpolInterleave, mask.data(), kMaxNodes, 0) != 0)
                    il_errno = errno;
                else
                    il_set = true;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        // Spans of at least 32 MiB per thread.
        const size_t nt = thread_count(static_cast<size_t>(nt_d), (ne * 8) / (32u << 20));
        std::atomic<int> err{0};
        std::atomic<bool> short_read{false};
        size_t nt_ran = 1;
        const bool all_ran = run_threads(nt, [&](size_t t) {
            const size_t lo = ne * t / nt, hi = ne * (t + 1) / nt;
            size_t done = 0, want = (hi - lo) * 8;
            char* p = reinterpret_cast<char*>(dst + lo);
            while (done < want && err.load() == 0 && !short_read.load()) {
                const size_t step = std::min<size_t>(want - done, 64u << 20);
                const ssize_t got = ::pread(fd, p + done, step, off + static_cast<off_t>(lo * 8 + done));
                if (got < 0) { if (errno == EINTR) continue; int z = 0; err.compare_exchange_strong(z, errno); break; }
                if (got == 0) { short_read = true; break; }
                done += static_cast<size_t>(got);
            }
            if (divisor != 1.0 && err.load() == 0 && !short_read.load())
                for (size_t i = lo; i < hi; ++i) dst[i] = dst[i] / divisor;
        }, &nt_ran);
        if (!all_ran) raise("randlapack:numa_place:read", "a reader thread failed while loading " + file);
        if (err.load() != 0) raise("randlapack:numa_place:read", "read of " + file + " failed: " + std::strerror(err.load()));
        if (short_read.load()) raise("randlapack:numa_place:short", "unexpected end of " + file);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double rss_fill = rss_mib();
        outputs[0] = factory.createArrayFromBuffer<double>({nr, nc}, std::move(buf));
        if (outputs.size() >= 2) {
            auto s = factory.createStructArray({1, 1}, {"threads", "ms", "rss_before_mib", "rss_after_alloc_mib", "rss_after_fill_mib",
                                                        "interleave_set", "interleave_errno", "nodes"});
            std::vector<double> nd(nodes.begin(), nodes.end());
            s[0]["interleave_set"] = factory.createScalar<bool>(il_set);
            s[0]["interleave_errno"] = factory.createScalar<double>(static_cast<double>(il_errno));
            s[0]["nodes"] = factory.createArray<double>({1, nd.size()}, nd.data(), nd.data() + nd.size());
            s[0]["threads"] = factory.createScalar<double>(static_cast<double>(nt_ran));   // threads that ran
            s[0]["ms"] = factory.createScalar<double>(ms);
            s[0]["rss_before_mib"] = factory.createScalar<double>(rss0);
            s[0]["rss_after_alloc_mib"] = factory.createScalar<double>(rss_alloc);
            s[0]["rss_after_fill_mib"] = factory.createScalar<double>(rss_fill);
            outputs[1] = std::move(s);
        }
    }
#endif

public:
    void operator()(ArgumentList outputs, ArgumentList inputs) {
        try {
            if (inputs.size() < 1) raise("randlapack:numa_place:nargin", "usage: numa_place_mex('spread'|'query'|'load', ...)");
            const std::string cmd = text(inputs[0], "command");
#ifdef __linux__
            if (cmd != "spread" && cmd != "query" && cmd != "load")
                raise("randlapack:numa_place:command", "unknown command '" + cmd + "'");
            const size_t max_out = (cmd == "load") ? 2 : 1;
            if (outputs.size() < 1 || outputs.size() > max_out)
                raise("randlapack:numa_place:nargout", "'" + cmd + "' returns " + (max_out == 1 ? "one output" : "one or two outputs"));
            if (cmd == "spread") spread(outputs, inputs);
            else if (cmd == "query") query(outputs, inputs);
            else if (cmd == "load") load(outputs, inputs);
            else raise("randlapack:numa_place:command", "unknown command '" + cmd + "'");
#else
            raise("randlapack:numa_place:unsupported", "numa_place_mex is Linux only");
#endif
        } catch (const MexError& e) {
            engine->feval(u"error", 0, std::vector<Array>{factory.createCharArray(e.id), factory.createCharArray(e.msg)});
        } catch (const std::exception& e) {
            engine->feval(u"error", 0, std::vector<Array>{factory.createCharArray("randlapack:numa_place:StdError"),
                                                          factory.createCharArray(e.what())});
        }
    }
};
