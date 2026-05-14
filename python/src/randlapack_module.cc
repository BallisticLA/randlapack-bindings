// pybind11 module entry point for the Python `randlapack` package.
//
// Per-driver bindings (bqrrp, ...) register their functions via free
// `register_*` functions called from PYBIND11_MODULE.

#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_bqrrp(py::module_& m);

PYBIND11_MODULE(_randlapack, m) {
    m.doc() = "Python bindings for selected RandLAPACK drivers.";

    register_bqrrp(m);
}
