#pragma once

#include <pybind11/pybind11.h>

namespace deepinfer::engine {

void pymodule_init(pybind11::module &engine);

// Sub-init functions (each in its own compilation unit)
void pybind_engine_sequence(pybind11::module &engine);
void pybind_engine_scheduler(pybind11::module &engine);
void pybind_engine_output(pybind11::module &engine);

} // namespace deepinfer::engine
