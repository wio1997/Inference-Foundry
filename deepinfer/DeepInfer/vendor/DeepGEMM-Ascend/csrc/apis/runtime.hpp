#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../jit/jit.hpp"

namespace deep_gemm::config {

namespace py = pybind11;

static void register_apis(py::module_& m) {
    m.def("get_num_sms", []() -> int {
        return runtime->get_num_sms();
    });

    m.def("set_num_sms", [](const int& n) -> void {
        runtime->set_num_sms(n);
    });

    m.def("set_npu_arch", [](const int& arch) -> void {
        runtime->set_npu_arch(arch);
    });

    m.def("get_npu_arch", []() -> int {
        return runtime->get_npu_arch();
    });

    m.def("set_mk_alignment_for_contiguous_layout", [](const int& value) -> void {
        Runtime::set_mk_alignment_for_contiguous_layout(value);
    });

    m.def("get_mk_alignment_for_contiguous_layout", []() -> int {
        return Runtime::get_mk_alignment_for_contiguous_layout();
    });

    m.def("get_theoretical_mk_alignment_for_contiguous_layout", []() -> int {
        return Runtime::get_theoretical_mk_alignment_for_contiguous_layout();
    });

    m.def("init", [](const std::string& version, const std::string& library_root_path) -> void {
        // Keep imports free of NPU runtime initialization so callers can fork.
        init_jit(version, library_root_path);
    });

    m.def("set_dry_run", [](const bool value) -> void {
        if (value)
            jit.get();
        runtime->set_dry_run(value);
    });

    m.def("get_dry_run", []() -> bool {
        return runtime->get_dry_run();
    });

    m.def("use_deterministic_algorithms", [](const bool enabled) -> void {
        runtime->use_deterministic_algorithms(enabled);
    }, py::arg("enabled"));

    m.def("get_deterministic_algorithms", []() -> bool {
        return runtime->get_deterministic_algorithms();
    });

    m.def("npu_sleep", [](const int64_t& sleep_cycles, const uint32_t& marker) -> void {
        const auto name = marker == 0 ? "sleep" : std::format("sleep_marker_{}", marker);
        const auto code = std::format(R"(
#include <deep_gemm/sleep.hpp>

static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&sleep_impl<{}>);
}}
)", marker);
        if (runtime->get_dry_run()) {
            jit->compile_without_load(name, code);
            return;
        }
        const auto kernel = jit->compile(name, code);
        jit->launch(kernel, {.num_blocks = runtime->get_num_sms()}, sleep_cycles);
    }, py::arg("sleep_cycles"), py::kw_only(), py::arg("marker") = 0);
}

} // namespace deep_gemm::config
