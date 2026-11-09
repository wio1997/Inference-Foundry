#include "pybind.hpp"

namespace deepinfer::engine {

void pymodule_init(pybind11::module &engine) {
    pybind_engine_sequence(engine);
    pybind_engine_scheduler(engine);
    pybind_engine_output(engine);
}

} // namespace deepinfer::engine
