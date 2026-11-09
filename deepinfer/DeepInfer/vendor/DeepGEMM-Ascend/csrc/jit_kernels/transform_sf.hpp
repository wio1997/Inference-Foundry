#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <torch/torch.h>
#include <deep_gemm/common.hpp>
#include <deep_jit/utils/exception.hpp>

#include "../jit/jit.hpp"
#include "../utils/math.hpp"

namespace deep_gemm {

static torch::Tensor launch_transform_sf(
    torch::Tensor sf,
    uint32_t shape_m, uint32_t gran_mn,
    GemmType gemm_type,
    const std::optional<torch::Tensor>& grouped_layout = std::nullopt
) {
    const uint32_t sf_dims = sf.dim();
    DJ_HOST_ASSERT(sf_dims == 2 or sf_dims == 3, "Transform SF input must have 2 or 3 dimensions, but got {}", sf_dims);
    if (sf_dims == 2)
        sf = sf.unsqueeze(0);
    DJ_HOST_ASSERT(gran_mn >= 1 and gran_mn <= 128 and (gran_mn & (gran_mn - 1)) == 0,
                   "gran_mn only support a power of two in [1, 128]");

    const bool is_float_input = sf.scalar_type() == at::kFloat;
    const uint32_t num_batches = sf.size(0);
    const Major major = sf.stride(-1) == 1 ? Major::K : Major::MN;

    uint32_t num_groups = 0;
    if (is_grouped(gemm_type)) {
        DJ_HOST_ASSERT(num_batches == 1 and grouped_layout.has_value(),
                       "Grouped transform SF requires one logical batch and grouped_layout");
        DJ_HOST_ASSERT(grouped_layout->device() == sf.device(),
                       "grouped_layout must be on the same device as the SF input");
        DJ_HOST_ASSERT(grouped_layout->scalar_type() == at::kInt and grouped_layout->dim() == 1 and
                       grouped_layout->is_contiguous(),
                       "grouped_layout must be a contiguous 1D int32 tensor");
        num_groups = grouped_layout->numel();
        DJ_HOST_ASSERT(num_groups > 0, "grouped_layout must contain at least one group");
    }

    const uint32_t src_shape_m = sf.size(-2);
    const uint32_t src_shape_k = sf.size(-1);
    const uint32_t dst_shape_m = shape_m;
    const uint32_t dst_shape_k = ceil_div(src_shape_k, is_float_input ? 2 : 1);

    auto physical_sizes = sf.sizes().vec();
    physical_sizes[physical_sizes.size() - 2] = dst_shape_k;
    physical_sizes[physical_sizes.size() - 1] = dst_shape_m;
    auto output = torch::empty(physical_sizes, sf.options().dtype(at::kShort)).transpose(-2, -1);
    if (output.numel() == 0)
        return sf_dims == 2 ? output.squeeze(0) : output;

    uint32_t src_block_m, src_block_k;

    if (major == Major::MN) {
        src_block_m = std::min(512u, 8192u / gran_mn);
        src_block_k = gran_mn == 1 ? 32 : std::max(32u / gran_mn, 4u) * (is_float_input ? 2 : 1);
    } else {
        src_block_m = (gran_mn == 1 ? 128 : 256) / gran_mn;
        src_block_k = 128;
    }

    const uint32_t num_vector_cores = 2 * static_cast<uint32_t>(runtime->get_num_sms());
    const uint64_t input_stride = major == Major::K ? static_cast<uint64_t>(sf.stride(-2)) : static_cast<uint64_t>(sf.stride(-1));
    const uint64_t input_stride_bytes = input_stride * (is_float_input ? sizeof(float) : sizeof(int16_t));
    const auto l2_ctrl_load = deep_jit::get_env("DG_TRANSFORM_SF_LD_CACHE_HINT", input_stride_bytes > 32 ? asc_load_l2_cache_mode::NOTALLOC_KEEP : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
    const auto l2_ctrl_store = deep_jit::get_env("DG_TRANSFORM_SF_ST_CACHE_HINT", asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM);
    const auto code = std::format(R"(
#include <deep_gemm/transform_sf.hpp>

namespace deep_gemm {{
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&transform_sf<
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
}} // namespace deep_gemm
)",
        is_float_input ? "true" : "false",
        major,
        src_block_m,
        src_block_k,
        gemm_type,
        num_vector_cores,
        Runtime::get_mk_alignment_for_contiguous_layout(),
        gran_mn,
        l2_ctrl_load,
        l2_ctrl_store);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("transform_sf", code);
        return sf_dims == 2 ? output.squeeze(0) : output;
    }
    const auto kernel = jit->compile("transform_sf", code);

    jit->launch(kernel, {.num_blocks = static_cast<int>(num_vector_cores)},
                gm_ptr<uint8_t, Major::K>{input_stride, reinterpret_cast<uintptr_t>(sf.data_ptr()),
                                          static_cast<uint64_t>(sf.stride(-3))},
                output.data_ptr(), grouped_layout.has_value() ? grouped_layout->data_ptr() : nullptr, src_shape_m,
                src_shape_k, dst_shape_m, dst_shape_k, num_groups, num_batches);

    return sf_dims == 2 ? output.squeeze(0) : output;
}

} // namespace deep_gemm
