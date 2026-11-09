#pragma once

#include <algorithm>
#include <cmath>
#include <tuple>

#include "../jit/jit.hpp"
#include "gemm_config.hpp"
#include "heuristics.hpp"

namespace deep_gemm {

// Dispatch a BF16 GEMM (Normal / m-grouped / k-grouped — distinguished by desc.gemm_type,
// built via get_gemm_desc()). The descriptor carries pointers, majors, outer strides,
// and grouping. D = A @ B^T (NT); D += A@B^T when desc.acc.
static void launch_bf16_gemm(const GemmDesc& desc) {
    auto config = select_gemm_config(desc);
    const auto code = std::format(R"(
#include <deep_gemm/bf16_gemm.hpp>

namespace deep_gemm {{
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&bf16_gemm_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {}>);
}}
}} // namespace deep_gemm
)",
        desc.major_a,
        desc.major_b,
        get_compiled_dim(desc.m, 'm', desc.compiled_dims),
        get_compiled_dim(desc.n, 'n', desc.compiled_dims),
        get_compiled_dim(desc.k, 'k', desc.compiled_dims),
        config.block_m,
        config.block_n,
        config.block_k,
        config.mad_m,
        config.mad_n,
        config.mad_k,
        config.num_l1_stages,
        config.num_l0_stages,
        config.num_epilogue_stages,
        config.launch_options.num_blocks.value(),
        config.l2_ctrl_a,
        config.l2_ctrl_b,
        config.l2_ctrl_store_cd,
        desc.acc ? "true" : "false",
        desc.gemm_type,
        GemmDesc::dtype_to_cd_kernel_type(desc.cd_dtype),
        desc.epilogue_class->get_epilogue_operator_type(),
        Runtime::get_mk_alignment_for_contiguous_layout(),
        config.direct_store, config.resident_a, config.resident_b);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("bf16_gemm", code);
        return;
    }
    const auto kernel = jit->compile("bf16_gemm", code);

    jit->launch(kernel, config.launch_options,
                gm_ptr<uint8_t>{desc.outer_stride_a, reinterpret_cast<uintptr_t>(desc.a_ptr), desc.stride_batch_a},
                gm_ptr<uint8_t>{desc.outer_stride_b, reinterpret_cast<uintptr_t>(desc.b_ptr), desc.stride_batch_b},
                gm_ptr<uint8_t>{desc.outer_stride_d, reinterpret_cast<uintptr_t>(desc.d_ptr), desc.stride_batch_d},
                desc.m, desc.n, desc.k, desc.grouped_layout, desc.num_groups, desc.epilogue_class->make_epilogue_operator_args(desc.n));
}

} // namespace deep_gemm
