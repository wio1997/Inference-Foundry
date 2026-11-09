#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <pybind11/pybind11.h>

#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/python_api.hpp>

namespace deep_ep {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::Ascend>> jit(nullptr);

class Runtime {
    int num_vec_cores = 0;
    int num_cube_cores = 0;
    int arch = 0;
    bool barrier_in_prologue = false;

public:
    int get_npu_arch() {
        return arch == 0 ? jit->device.get_npu_arch() : arch;
    }

    void set_npu_arch(const int& new_arch) {
        arch = new_arch;
        jit->default_compiler_options.arch = get_npu_arch();
    }

    int get_num_aicore_cores() {
        return jit->device.get_num_aicore_cores();
    }

    int get_num_vec_cores() {
        return num_vec_cores == 0 ? jit->device.get_num_vec_cores() : num_vec_cores;
    }

    int get_num_vec_cores_per_ai_core() {
        return get_num_vec_cores() / get_num_aicore_cores();
    }

    void set_num_vec_cores(const int& new_num_vec_cores) {
        num_vec_cores = new_num_vec_cores;
    }

    int get_num_cube_cores() {
        return num_cube_cores == 0 ? jit->device.get_num_cube_cores() : num_cube_cores;
    }

    void set_num_cube_cores(const int& new_num_cube_cores) {
        num_cube_cores = new_num_cube_cores;
    }

    int64_t get_num_ubuf_bytes_per_vec_core() {
        return jit->device.get_num_ubuf_bytes_per_vec_core();
    }

    bool get_barrier_in_prologue() const {
        return barrier_in_prologue;
    }

    bool set_barrier_in_prologue(const bool& enable) {
        const auto previous = barrier_in_prologue;
        barrier_in_prologue = enable;
        return previous;
    }
};

inline auto runtime = deep_jit::LazyInit<Runtime>([]() { return std::make_shared<Runtime>(); });

inline void init_jit(const std::string& library_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path).lexically_normal();
    const auto include_dir = library_root / "include";
    const deep_jit::Config config(library_root, "EP", {}, {include_dir}, {"deep_ep/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::Ascend>>([config] {
        return std::make_shared<deep_jit::Runtime<deep_jit::Ascend>>(config);
    });
}

namespace config {

inline void register_apis(pybind11::module_& m) {
    deep_jit::register_python_api(m, jit);

    m.def("get_num_ai_cores", []() -> int {
        return runtime->get_num_aicore_cores();
    });
    m.def("get_num_vec_cores", []() -> int {
        return runtime->get_num_vec_cores();
    });
    m.def("get_num_cube_cores", []() -> int {
        return runtime->get_num_cube_cores();
    });
    m.def("set_num_vec_cores", [](const int& n) -> void {
        runtime->set_num_vec_cores(n);
    });
    m.def("set_num_cube_cores", [](const int& n) -> void {
        runtime->set_num_cube_cores(n);
    });
    m.def("set_npu_arch", [](const int& arch) -> void {
        runtime->set_npu_arch(arch);
    });
    m.def("get_npu_arch", []() -> int {
        return runtime->get_npu_arch();
    });
    m.def("set_barrier_in_prologue", [](const bool& enable) -> bool {
        return runtime->set_barrier_in_prologue(enable);
    }, pybind11::arg("enable"));
    m.def("init_jit", [](const std::string& library_root_path) -> void {
        deep_ep::init_jit(library_root_path);
    });
}

}  // namespace config

}  // namespace deep_ep
