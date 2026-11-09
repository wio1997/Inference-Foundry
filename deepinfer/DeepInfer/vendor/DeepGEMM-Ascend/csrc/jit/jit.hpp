#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/utils/exception.hpp>

#include "../utils/format.hpp"

namespace deep_gemm {

using AscendJIT = deep_jit::Runtime<deep_jit::Ascend>;

inline deep_jit::LazyInit<AscendJIT> jit(nullptr);

class Runtime {
    int num_sms = 0;
    int arch = 0;
    bool dry_run = false;
    bool deterministic_algorithms = false;

public:
    int get_npu_arch() const {
        return arch == 0 ? jit->device.get_npu_arch() : arch;
    }

    void set_npu_arch(const int& new_arch) {
        arch = new_arch;
        jit->default_compiler_options.arch = get_npu_arch();
    }

    int get_num_sms() const {
        return num_sms == 0 ? jit->device.get_num_sms() : num_sms;
    }

    void set_num_sms(const int& n) {
        DJ_HOST_ASSERT(n > 0 and n <= jit->device.get_num_sms());
        num_sms = n;
    }

    void set_dry_run(const bool value) {
        dry_run = value;
    }

    bool get_dry_run() const {
        return dry_run;
    }

    void use_deterministic_algorithms(const bool enabled) {
        deterministic_algorithms = enabled;
    }

    bool get_deterministic_algorithms() const {
        return deterministic_algorithms;
    }

    static constexpr int kMkAlignment = 256;

    static int get_mk_alignment_for_contiguous_layout() {
        return kMkAlignment;
    }

    static void set_mk_alignment_for_contiguous_layout(const int& mk_alignment) {
        if (mk_alignment != kMkAlignment)
            DJ_PANIC("mk_alignment is fixed at {} currently, but attempted to set it to {}", kMkAlignment, mk_alignment);
        DJ_HOST_ASSERT(mk_alignment % 64 == 0);
    }

    static int get_theoretical_mk_alignment_for_contiguous_layout() {
        return kMkAlignment;
    }
};

inline auto runtime = deep_jit::LazyInit<Runtime>([] { return std::make_shared<Runtime>(); });

inline void init_jit(const std::string& version, const std::string& library_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path).lexically_normal();
    const auto include_dir = library_root / "include";
    const deep_jit::Config config(library_root, "DG", version, {include_dir}, {"deep_gemm/"});

    jit = deep_jit::LazyInit<AscendJIT>([config] {
        auto jit_runtime = std::make_shared<AscendJIT>(config);
        jit_runtime->default_compiler_options.extra_bisheng_flags.emplace_back(
            "--cce-disable-vf-stack-reserved-ubuf");
        jit_runtime->default_compiler_options.extra_bisheng_flags.emplace_back("-std=c++20");
        if (deep_jit::get_env<bool>("DG_JIT_WITH_LINEINFO"))
            jit_runtime->default_compiler_options.debug_info = true;
        return jit_runtime;
    });
}

aclrtStream get_current_npu_stream();

} // namespace deep_gemm
