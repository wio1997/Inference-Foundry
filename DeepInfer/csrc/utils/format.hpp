#pragma once

// Keep the standard implementation where it is available. Older Ascend build
// images can use the fmt headers already bundled with PyTorch, header-only.
#if __has_include(<format>)
#include <format>
#endif

#if defined(__cpp_lib_format) && __cpp_lib_format >= 201907L
namespace deepinfer::formatting {
using std::format;
template <typename... Args> using format_string = std::format_string<Args...>;
} // namespace deepinfer::formatting
#else
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>
namespace deepinfer::formatting {
using fmt::format;
template <typename... Args> using format_string = fmt::format_string<Args...>;
} // namespace deepinfer::formatting
#endif
