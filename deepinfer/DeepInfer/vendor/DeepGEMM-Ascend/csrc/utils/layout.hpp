#pragma once

#include <torch/torch.h>
#include <tuple>
#include <string_view>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <deep_gemm/common.hpp>

#include <deep_jit/utils/exception.hpp>

#include "dtype.hpp"
#include "format.hpp"

namespace deep_gemm {

// Extract a tensor's shape as an N-tuple of ints.
template <int N>
static auto get_shape(const at::Tensor& t) {
    DJ_HOST_ASSERT(t.dim() == N);
    return [&t] <std::size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<N>());
}

static std::string format_strides(const at::Tensor& t) {
    std::string result = "[";
    for (const auto stride : t.strides()) {
        if (result.size() > 1)
            result += ", ";
        result += std::format("{}", stride);
    }
    return result + ']';
}

static void major_check(const at::Tensor& t) {
    if (t.stride(-2) == 1 and t.stride(-1) == 1) {
        DJ_PANIC("Ambiguous major axis, the tensor has both last two dimensions contiguous, strides={}", format_strides(t));
    }
    DJ_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}

static Major get_major_type_ab(const at::Tensor& t) {
    major_check(t);
    return t.stride(-1) == 1 ? Major::K : Major::MN;
}

static void check_major_type_cd(const at::Tensor& d) {
    DJ_HOST_ASSERT(d.dim() >= 2);
    DJ_HOST_ASSERT(d.stride(-1) == 1);
}

static bool gemm_early_return(const int& m, const int& n, const int& k,
                              const torch::Tensor& d, const std::optional<torch::Tensor>& c,
                              const std::optional<torch::Tensor>& sfd = std::nullopt, bool with_bias = false) {
    if (m == 0 or n == 0)
        return true;

    DJ_HOST_ASSERT(not sfd.has_value() or not c.has_value());
    const bool is_cd_same = c.has_value() and not d.is_meta() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DJ_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());

    DJ_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat or
                   d.scalar_type() == torch::kFloat8_e4m3fn);
    DJ_HOST_ASSERT((d.scalar_type() == torch::kFloat8_e4m3fn) == sfd.has_value());

    if (c.has_value()) {
        check_major_type_cd(c.value());
        DJ_HOST_ASSERT(d.scalar_type() == c.value().scalar_type());
    }

    // No accumulation
    if (k == 0) {
        if (not is_cd_same)
            c.has_value() ? d.copy_(c.value()) : d.zero_();
        if (sfd.has_value()) sfd->zero_();
        return true;
    }

    // Legacy accumulation paths initialize D before launch.
    if (c.has_value() and not is_cd_same and not with_bias)
        d.copy_(c.value());
    return false;
}

template <bool kEnable, typename T, std::integral... Dims>
static auto get_transposed(T tensor, Dims ... dims) {
    if constexpr (!kEnable) {
        return tensor;
    } else if constexpr (std::is_same_v<T, torch::Tensor> || std::is_same_v<T, at::Tensor>) {
        return tensor.transpose(dims...);
    } else if constexpr (std::is_same_v<T, std::pair<torch::Tensor, torch::Tensor>>) {
        return std::make_pair(tensor.first.transpose(dims...), tensor.second.transpose(dims...));
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for get_transposed");
    }
}

template <size_t N>
struct str {
    char value[N];

    constexpr str(const char (&s)[N]) {
        for (size_t i = 0; i < N; ++i) {
            value[i] = s[i];
        }
    }

    constexpr size_t size() const {
        return N - 1;
    }

    constexpr char operator[](size_t i) const {
        return value[i];
    }
};

struct ShapeSpec {
    const at::Tensor& tensor;
    std::string_view dims;
};

template <str dims>
struct ShapeSpecBuilder {
    ShapeSpec operator=(const at::Tensor& tensor) const {
        return ShapeSpec{
            .tensor=tensor,
            .dims=std::string_view{dims.value, dims.size()},
        };
    }
};

template <str dims>
consteval auto operator""_sp() {
    return ShapeSpecBuilder<dims>{};
}

template <str out_dims, typename shape_dtype = uint32_t, shape_dtype default_value = 0, typename... Specs>
static auto get_shape_by_spec(const Specs&... specs) {
    auto main_axis = [](const ShapeSpec& spec) -> int64_t {
        const at::Tensor& tensor = spec.tensor;
        for (int64_t i = 0; i < tensor.dim(); ++i) {
            if (tensor.stride(i) == 1) return i;
        }
        return -1;
    };

    std::array<size_t, sizeof...(specs)> major_axes;
    auto check_spec = [&](size_t tensor_idx, const ShapeSpec& spec) {
        const at::Tensor& tensor = spec.tensor;
        if (tensor.dim() != static_cast<int64_t>(spec.dims.size())) {
            DJ_PANIC("Shape mismatch: tensor #{}, annotation=\"{}\"", tensor_idx, spec.dims);
        }
        auto major_axis = main_axis(spec);
        if (major_axis < 0) {
            DJ_PANIC("Tensor #{} annotation=\"{}\" doesn't have major axis, strides={}", tensor_idx, spec.dims, format_strides(tensor));
        }
        major_axes[tensor_idx] = static_cast<size_t>(major_axis);
    };

    {
        size_t tensor_idx = 0;
        (check_spec(tensor_idx++, specs), ...);
    }


    auto get_one_dim = [&](char dim_name) -> shape_dtype {
        bool found = false;
        shape_dtype result = default_value;

        size_t first_tensor_idx = 0;
        size_t first_axis = 0;
        std::string_view first_annot;

        auto scan_spec = [&](size_t tensor_idx, const ShapeSpec& spec) {
            const at::Tensor& tensor = spec.tensor;

            for (size_t axis = 0; axis < spec.dims.size(); ++axis) {
                if (spec.dims[axis] != dim_name) continue;

                auto value = tensor.size(static_cast<int64_t>(axis));

                if (tensor.scalar_type() == kPackedFP4 && axis == major_axes[tensor_idx]) {
                    value *= 2;
                }

                if (!found) {
                    found = true;
                    result = value;
                    first_tensor_idx = tensor_idx;
                    first_axis = axis;
                    first_annot = spec.dims;
                } else if (result != value) {
                    DJ_PANIC(
                        "Conflicting shape dimension '{}': "
                        "tensor #{} annotation=\"{}\" axis={} gives {}, "
                        "but tensor #{} annotation=\"{}\" axis={} gives {}",
                        dim_name,
                        tensor_idx, spec.dims, axis, value,
                        first_tensor_idx, first_annot, first_axis, result
                    );
                }
            }
        };

        size_t tensor_idx = 0;
        (scan_spec(tensor_idx++, specs), ...);

        return result;
    };

    return [&]<size_t... I>(std::index_sequence<I...>) {
        return std::make_tuple(
            get_one_dim(out_dims[I])...
        );
    }(std::make_index_sequence<out_dims.size()>{});
}

}
