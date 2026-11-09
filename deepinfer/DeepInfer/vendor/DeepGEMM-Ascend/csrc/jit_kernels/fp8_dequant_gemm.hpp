#pragma once

#include <deep_jit/utils/exception.hpp>

#include "../jit/jit.hpp"
#include "../utils/dtype.hpp"
#include "gemm_config.hpp"
#include "heuristics.hpp"

namespace deep_gemm {

// FP8 × FP4-dequant GEMM: A is hardware fp8, B is packed fp4 (K-major, int8) that gets
// dequantised to fp8 on the AIV vector cores. The cube datapath is identical to fp8_gemm:
// ab_dtype_t = float8_e4m3_t, e8m0 scales for both operands are applied via mad_mx.
//
// This is a separate kernel (fp8_dequant_gemm_impl) because the AIV pipeline includes the
// dequant path (build LUT → vf_deq_nd2nz → copy_ub_to_l1). The JIT wrapper is the same
// shape as FP8GemmRuntime.

// Dispatch a single FP8 × FP4-dequant GEMM. A is fp8, B is packed fp4, and both operands
// support K-major and MN-major layouts. SF uses the standard MX int16 MN-major layout.
//
// One or two AIVs decode B; DirectStore lets the AIC write D.
static void launch_fp8_dequant_gemm(const GemmDesc& desc) {
    auto config = select_gemm_config(desc);
    DJ_HOST_ASSERT(config.block_k <= 512);
    GemmArgs args{.desc = desc, .config = config};
    const auto& c = args.config;
    const auto& d = args.desc;
    const auto code = std::format(R"(
#include <deep_gemm/fp8_dequant_gemm.hpp>

using namespace deep_gemm;
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&fp8_dequant_gemm_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {},
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
        {}, {},
        {}>);
}}
)",
        d.major_a,
        d.major_b,
        get_compiled_dim(d.m, 'm', d.compiled_dims),
        get_compiled_dim(d.n, 'n', d.compiled_dims),
        get_compiled_dim(d.k, 'k', d.compiled_dims),
        c.block_m,
        c.block_n,
        c.block_k,
        c.mad_m,
        c.mad_n,
        c.mad_k,
        c.num_l1_stages,
        c.num_l0_stages,
        c.sf_k_blocks,
        c.num_l1_sf_stages,
        c.num_epilogue_stages,
        c.launch_options.num_blocks.value(),
        c.l2_ctrl_a,
        c.l2_ctrl_b,
        c.l2_ctrl_store_cd,
        d.acc ? "true" : "false",
        d.gemm_type,
        GemmDesc::dtype_to_cd_kernel_type(d.cd_dtype),
        d.epilogue_class->get_epilogue_operator_type(),
        Runtime::get_mk_alignment_for_contiguous_layout(),
        c.direct_store, c.dual_aiv_dequant,
        d.gemm_type == GemmType::Normal && d.m <= c.block_m && c.dual_aiv_dequant
    );
    if (runtime->get_dry_run()) {
        jit->compile_without_load("fp8_dequant_gemm", code);
        return;
    }
    const auto kernel = jit->compile("fp8_dequant_gemm", code);

    jit->launch(kernel, args.config.launch_options,
                gm_ptr<uint8_t>{d.outer_stride_a, reinterpret_cast<uintptr_t>(d.a_ptr), d.stride_batch_a},
                gm_ptr<uint8_t>{d.outer_stride_b, reinterpret_cast<uintptr_t>(d.b_ptr), d.stride_batch_b}, d.sfa_ptr,
                d.sfb_ptr, gm_ptr<uint8_t>{d.outer_stride_d, reinterpret_cast<uintptr_t>(d.d_ptr), d.stride_batch_d},
                d.m, d.n, d.k, d.grouped_layout, d.num_groups, d.epilogue_class->make_epilogue_operator_args(d.n));
}

} // namespace deep_gemm
