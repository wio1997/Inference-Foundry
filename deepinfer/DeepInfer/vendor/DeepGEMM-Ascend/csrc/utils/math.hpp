#pragma once

namespace deep_gemm {

constexpr auto ceil_div(auto a, auto b) {
    return (a + b - 1) / b;
}

constexpr auto align(auto a, auto b) {
    return ceil_div(a, b) * b;
}

} // namespace deep_gemm
