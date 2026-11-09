#pragma once

#include <cstdint>
#include <string>
#include <ATen/core/Tensor.h>

#include <deep_jit/backend/ascend/options.hpp>

#include "../utils/dtype.hpp"
#include "../utils/format.hpp"
#include "../../deep_gemm/include/deep_gemm/common.hpp"  // Major, host L2 cache types, capacities
#include "epilogue_class.hpp"

namespace deep_gemm {

// Return dim value if name is in compiled_dims, else 0 (meaning "use runtime value").
static uint32_t get_compiled_dim(uint32_t dim, char name, const std::string& compiled_dims) {
    for (const char& c : compiled_dims)
        if (c == name) return dim;
    return 0;
}

// Physical dim0 (non-contiguous axis) row stride in ELEMENTS, for the kernel's gm_ptr.
// K-major: inner axis is K (stride(-1) must be 1), outer stride is along MN = stride(-2).
// MN-major: inner axis is MN (stride(-2) must be 1), outer stride is along K = stride(-1).
// Strictly assert the axis we treat as contiguous really has unit stride.
static uint64_t get_outer_stride(const at::Tensor& t, Major major) {
    if (major == Major::K) {
        DJ_HOST_ASSERT(t.stride(-1) == 1, "K-major operand requires unit stride on the inner (K) axis");
        return static_cast<uint64_t>(t.stride(-2));
    }
    DJ_HOST_ASSERT(t.stride(-2) == 1, "MN-major operand requires unit stride on the inner (MN) axis");
    return static_cast<uint64_t>(t.stride(-1));
}

// Problem descriptor: what to compute (dimensions, dtypes, pointers, layouts).
// major_a/major_b are the GM major axis of each operand (Major::K / Major::MN); D is
// always N-major and SF is always MN-major.
struct GemmDesc {
    void* a_ptr;
    void* b_ptr;
    void* d_ptr;
    at::ScalarType a_dtype, b_dtype, d_dtype;
    void* sfa_ptr;              // FP8 only; nullptr for BF16
    void* sfb_ptr;              // FP8 only; nullptr for BF16
    uint32_t m, n, k;
    uint32_t num_cores;
    at::ScalarType cd_dtype;
    Major major_a, major_b;
    bool acc;                   // accumulate into D (D += A@B)
    std::shared_ptr<EpilogueClass> epilogue_class;
    std::string compiled_dims;  // e.g. "nk" = compile N,K as constants; M is runtime

    // m-grouped (MoE) extension. For Normal these stay at their defaults below.
    GemmType gemm_type = GemmType::Normal;
    void* grouped_layout = nullptr;  // psum prefix-sum-of-rows, int32[num_groups];
                                     // k-grouped: cumulative end K per group
    uint32_t num_groups = 0;
    uint32_t expected_m = 0;         // m-grouped: host config hint for dynamic grouped M
    uint32_t expected_k = 0;         // k-grouped: host config hint for per-group K estimate

    // Batched GEMM: per-batch strides (in ELEMENTS) of A/B/D's leading batch axis. After a
    // .permute() the batch dim's stride is not M*K / N*K / M*N, so it is carried at runtime.
    // num_groups holds the batch count. SFA/SFB are host-normalized to a dense [B, mn, k/64]
    // layout, so their per-batch block is mn*(k/64) (no extra stride needed). 0 if unbatched.
    uint64_t stride_batch_a = 0, stride_batch_b = 0, stride_batch_d = 0;

    // Physical dim0 (non-contiguous axis) row strides, in ELEMENTS, for the kernel's
    // gm_ptr. Tight inputs make these equal to k / k / n (A/B/D), reproducing the old
    // hard-coded behavior; strided or permuted operands carry their real strides. The
    // inner axis is always contiguous.
    //   outer_stride_a: A's stride along MN (K-major) or along K (MN-major)
    //   outer_stride_b: B's stride along MN (K-major) or along K (MN-major)
    //   outer_stride_d: D's stride along M (D is always N-major)
    uint64_t outer_stride_a = 0, outer_stride_b = 0, outer_stride_d = 0;

    // used to hold tensor to prevent it from being freed before the kernel is launched
    std::optional<at::Tensor> sfa, sfb;
    static const char* dtype_to_kernel_type(at::ScalarType dtype) {
        switch (dtype) {
            case at::kFloat8_e4m3fn: return "float8_e4m3_t";
            case kPackedFP4: return "float4_e2m1x2_t";
            default: return dtype_to_cd_kernel_type(dtype);
        }
    }

    static const char* dtype_to_cd_kernel_type(at::ScalarType dtype) {
        switch (dtype) {
            case at::kBFloat16: return "bfloat16_t";
            case at::kFloat: return "float";
            case at::kFloat8_e4m3fn: return "float8_e4m3_t";
            default: DJ_HOST_ASSERT(false, "Unsupported cd_dtype"); return "";
        }
    }
};

// Execution configuration: how to compute (tiling, pipeline, launch).
struct GemmConfig {
    // Tile sizes
    uint32_t block_m, block_n, block_k;
    uint32_t mad_m, mad_n, mad_k;

    // Pipeline depths
    uint32_t num_l1_stages;        // L1 data buffer stages (MTE2 <-> MTE1)
    uint32_t num_l0_stages;        // L0 AB buffer stages (MTE1 <-> Cube)
    uint32_t num_epilogue_stages;  // UB epilogue stages per AIV (AIC -> AIV)
    uint32_t num_l1_sf_stages;     // L1 SF buffer stages (0 for BF16)
    uint32_t sf_k_blocks;          // K-blocks per SF chunk (0 for BF16)

    bool direct_store = false;
    bool resident_a = false, resident_b = false, resident_sf = false;
    bool dual_aiv_dequant = false;

    // L2 cache control
    asc_load_l2_cache_mode l2_ctrl_a;
    asc_load_l2_cache_mode l2_ctrl_b;
    asc_store_l2_cache_mode l2_ctrl_store_cd;

    // Launch
    deep_jit::ascend::LaunchOptions launch_options;
};

// Kernel launch arguments: desc + config
struct GemmArgs {
    GemmDesc desc;
    GemmConfig config;
};

} // namespace deep_gemm

namespace std {

template <> struct formatter<deep_jit::ascend::LaunchOptions> : formatter<string_view> {
    static auto format(const deep_jit::ascend::LaunchOptions& value, format_context& context)  {
        auto out = std::format_to(context.out(), "LaunchOptions(num_blocks={}",
                                  value.num_blocks ? std::to_string(*value.num_blocks) : "unset");
        if (value.num_ubuf_bytes)
            out = std::format_to(out, ", num_ubuf_bytes={}", *value.num_ubuf_bytes);
        if (value.num_launch_timeout_secs)
            out = std::format_to(out, ", num_launch_timeout_secs={}", *value.num_launch_timeout_secs);
        return std::format_to(out, ")");
    }
};

template <> struct formatter<deep_gemm::GemmDesc> : formatter<string_view> {
    static auto format(deep_gemm::GemmDesc v, format_context& context)  {
        std::format_to(context.out(),
                       "GemmDesc(m={}, n={}, k={}, cd_dtype={}, major_a={}, major_b={}, acc={}, "
                       "epilogue_type={}, compiled_dims='{}', gemm_type={}",
                       v.m, v.n, v.k, deep_gemm::GemmDesc::dtype_to_kernel_type(v.cd_dtype), v.major_a, v.major_b,
                       v.acc, v.epilogue_class->get_epilogue_operator_type(), v.compiled_dims, v.gemm_type);
        if (is_m_grouped(v.gemm_type))
            std::format_to(context.out(), ", num_groups={}, expected_m={}", v.num_groups, v.expected_m);
        if (is_k_grouped(v.gemm_type))
            std::format_to(context.out(), ", num_groups={}, expected_k={}", v.num_groups, v.expected_k);
        if (is_batched(v.gemm_type))
            std::format_to(context.out(), ", batch={}", v.num_groups);
        std::format_to(context.out(), ")");
        return context.out();
    }
};

template <> struct formatter<deep_gemm::GemmConfig> : formatter<string_view> {
    static auto format(deep_gemm::GemmConfig v, format_context& context)  {
        return std::format_to(context.out(),
                              "GemmConfig(block={}x{}x{}, mad={}x{}x{}, l1_stages={}, l0_stages={}, "
                              "epi_stages={}, l1_sf_stages={}, sf_k_blocks={}, l2_ctrl={}/{}/{}, direct_store={}, resident={}/{}/{}, dual_aiv_dequant={}, launch_options={})",
                              v.block_m, v.block_n, v.block_k, v.mad_m, v.mad_n, v.mad_k, v.num_l1_stages,
                              v.num_l0_stages, v.num_epilogue_stages, v.num_l1_sf_stages, v.sf_k_blocks, v.l2_ctrl_a,
                              v.l2_ctrl_b, v.l2_ctrl_store_cd, v.direct_store, v.resident_a, v.resident_b, v.resident_sf, v.dual_aiv_dequant, v.launch_options);
    }
};

} // namespace std
