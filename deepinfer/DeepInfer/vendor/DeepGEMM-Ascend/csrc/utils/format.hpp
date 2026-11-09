#pragma once

#include <format>
#include <string_view>
#include <type_traits>

#include <magic_enum/magic_enum.hpp>

namespace std {

template <typename E>
    requires is_enum_v<decay_t<E>>
struct formatter<E, char> : formatter<string_view, char> {
    template <typename FormatContext>
    auto format(E value, FormatContext& context) const {
        using D = decay_t<E>;
        return std::format_to(
            context.out(), "{}::{}", magic_enum::enum_type_name<D>(), magic_enum::enum_name<D>(value));
    }
};

} // namespace std
