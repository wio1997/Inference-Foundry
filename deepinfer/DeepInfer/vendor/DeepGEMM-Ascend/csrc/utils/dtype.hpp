#pragma once

#include <torch/torch.h>

#include "format.hpp"

namespace deep_gemm {

inline constexpr auto kPackedFP4 = torch::kInt8;  // Canonical storage dtype for two packed FP4 values

} // namespace deep_gemm
