#pragma once

#include <optional>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include <deep_jit/utils/exception.hpp>

#include "../utils/math.hpp"
#include "../jit_kernels/transform_sf.hpp"
#include "c10/core/ScalarType.h"

namespace deep_gemm::layout {

namespace py = pybind11;

// Transform a scale-factor (SF) tensor into the layout required by the GEMM kernels.
//
// The input SF may be **float32** or **short**(packed ue8m0x2)
// Output is always **short** (packed ue8m0x2), MN-Major, and repeated for gran_mn if needed.
//
// Args:
//     sf: [..., ceil(MN / gran_mn), ceil(K / gran_k)], fp32 or short (packed ue8m0x2)
//     mn: MN dimension of the corresponding operand.
//     k: the K dimension of the corresponding operand.
//     recipe: quantization granularity, either (gran_m, gran_n, gran_k) or (gran_mn, gran_k) otherwise
//         if recipe is 3-element, is_sfa must be set to indicate whether the SF belongs to A or B.
//     num_groups: optional number of groups; when set, the sf.size(-3) must match it.
//     is_sfa: required when recipe is 3-element; indicates whether the SF belongs to A or B
//     disable_ue8m0_cast: when true, forbid casting a float32 SF into ue8m0.
//     psum_layout: optional partial-sum layout tensor.
//
// Returns:
//     sf: [..., MN, ceil(K / gran_k / 2)], short, MN-Major, used in internal kernels
static bool check_sf_layout(
    const torch::Tensor& sf,
    const int& mn, const int& k,
    const int& gran_mn, const int& gran_k,
    const std::optional<int>& num_groups
) {
    DJ_HOST_ASSERT(sf.scalar_type() == torch::kFloat or sf.scalar_type() == torch::kShort);
    DJ_HOST_ASSERT(sf.dim() == static_cast<int>(num_groups.has_value()) + 2);
    DJ_HOST_ASSERT(sf.stride(-1) == 1 or sf.stride(-2) == 1,
                   "SF input must be contiguous in either the last or second-to-last dimension");

    if (num_groups.has_value())
        DJ_HOST_ASSERT(sf.size(-3) == num_groups.value());

    constexpr uint32_t sf_pack_k = 2;
    const auto shape_div_k = sf.scalar_type() == at::kFloat ? gran_k : (gran_k * sf_pack_k);
    DJ_HOST_ASSERT(sf.size(-2) == ceil_div(mn, gran_mn));
    DJ_HOST_ASSERT(sf.size(-1) == ceil_div(k, shape_div_k));
    DJ_HOST_ASSERT(gran_k == 32, "gran_k must be 32");

    if (sf.scalar_type() == at::kShort and gran_mn == 1 and sf.stride(-2) == 1 and sf.stride(-1) == sf.size(-2)) {
        if (num_groups.has_value())
            return sf.stride(-3) == sf.size(-2) * sf.size(-1);
        else
            return true;
    }

    return false;
}


static torch::Tensor transform_sf_into_required_layout(
    torch::Tensor sf,
    int mn, int k,
    const std::tuple<int, int, int>& recipe,
    const std::optional<int>& num_groups,
    const std::optional<bool>& is_sfa,
    bool disable_ue8m0_cast,
    const std::optional<torch::Tensor>& psum_layout
) {
    DJ_HOST_ASSERT(is_sfa.has_value());

    int gran_mn, gran_k;
    gran_mn = is_sfa.value() ? std::get<0>(recipe) : std::get<1>(recipe);
    gran_k = std::get<2>(recipe);

    if (check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups))
        return sf;

    if (sf.scalar_type() == at::kFloat)
        DJ_HOST_ASSERT(!disable_ue8m0_cast, "Float32 SF must be converted into ue8m0, but disable_ue8m0_cast is set true");

    const auto gemm_type = psum_layout.has_value() ? GemmType::MGroupedContiguousWithPsumLayout
                                                   : num_groups.has_value() ? GemmType::Batched : GemmType::Normal;
    return launch_transform_sf(sf, mn, gran_mn, gemm_type, psum_layout);
}

static torch::Tensor transform_k_grouped_sf_into_required_layout(
    torch::Tensor sf,
    int mn, int k,
    const std::tuple<int, int, int>& recipe,
    const std::optional<bool>& is_sfa,
    const torch::Tensor& psum_layout
) {
    DJ_HOST_ASSERT(is_sfa.has_value());

    int gran_mn, gran_k;
    gran_mn = is_sfa.value() ? std::get<0>(recipe) : std::get<1>(recipe);
    gran_k = std::get<2>(recipe);

    if (check_sf_layout(sf, mn, k, gran_mn, gran_k, std::nullopt))
        return sf;

    return launch_transform_sf(sf, mn, gran_mn, GemmType::KGroupedContiguousWithPsumLayout, psum_layout);
}

static std::tuple<int, int, int> get_recipe(
    const std::optional<std::tuple<int, int, int>>& recipe,
    const std::optional<std::tuple<int, int>>& recipe_a,
    const std::optional<std::tuple<int, int>>& recipe_b
) {
    DJ_HOST_ASSERT(not (recipe.has_value() and (recipe_a.has_value() or recipe_b.has_value())));
    if (recipe.has_value()) {
        return recipe.value();
    }
    const auto [gran_m, gran_k_a] = recipe_a.value_or(std::make_tuple(1, 32));
    const auto [gran_n, gran_k_b] = recipe_b.value_or(std::make_tuple(1, 32));

    DJ_HOST_ASSERT(gran_k_a == gran_k_b, "gran_k of recipe_a and recipe_b must be the same");

    return std::make_tuple(gran_m, gran_n, gran_k_a);
}

static void register_apis(pybind11::module_& m) {
    m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout,
          py::arg("sf"), py::arg("mn"), py::arg("k"), py::arg("recipe"),
          py::arg("num_groups") = std::nullopt,
          py::arg("is_sfa") = std::nullopt,
          py::arg("disable_ue8m0_cast") = false,
          py::arg("psum_layout") = std::nullopt
    );
    m.def("transform_k_grouped_sf_into_required_layout", &transform_k_grouped_sf_into_required_layout,
          py::arg("sf"), py::arg("mn"), py::arg("k"), py::arg("recipe"), py::arg("is_sfa"), py::arg("psum_layout"));
}

} // namespace deep_gemm::layout
