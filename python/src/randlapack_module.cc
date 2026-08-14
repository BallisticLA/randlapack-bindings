// nanobind module entry point for the Python `randlapack` package.
//
// Per-driver bindings (bqrrp, ...) register their functions via free
// `register_*` functions called from NB_MODULE.

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void register_bqrrp(nb::module_& m);
void register_cqrrpt(nb::module_& m);
void register_rsvd(nb::module_& m);

NB_MODULE(_randlapack, m) {
    m.doc() = "Python bindings for selected RandLAPACK drivers.";

    register_bqrrp(m);
    register_cqrrpt(m);
    register_rsvd(m);
}
