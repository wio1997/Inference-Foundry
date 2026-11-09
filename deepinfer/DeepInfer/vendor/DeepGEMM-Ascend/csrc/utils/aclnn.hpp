#pragma once

#include <torch/torch.h>
#include <aclnn/acl_meta.h>

// Declared manually — including the official headers causes linker issues
// C++ symbol from libtorch_npu.so: wraps at::Tensor into aclTensor descriptor
aclTensor* ConvertType(const at::Tensor&);

namespace deep_gemm {

struct AclTensor {
    aclTensor* tensor = nullptr;
    operator aclTensor*() const { return tensor; }
    AclTensor() = default;
    AclTensor(const torch::Tensor& t): tensor(t.defined() ? ConvertType(t) : nullptr) {}
    AclTensor(aclTensor* t): tensor(t) {}
    ~AclTensor() { if (tensor) aclDestroyTensor(tensor); }
    AclTensor(AclTensor&& rhs) noexcept : tensor(rhs.tensor) {
        rhs.tensor = nullptr;
    }
    AclTensor(AclTensor& rhs) = delete;
    AclTensor(const AclTensor& rhs) = delete;
    AclTensor& operator=(const torch::Tensor& t) {
        if (tensor) aclDestroyTensor(tensor);
        tensor = t.defined() ? ConvertType(t) : nullptr;
        return *this;
    }
    AclTensor& operator=(AclTensor&& rhs) noexcept {
        if (tensor) aclDestroyTensor(tensor);
        tensor = rhs.tensor;
        rhs.tensor = nullptr;
        return *this;
    }
};

template <typename T> struct to_acl_dtype { };
template <> struct to_acl_dtype<   float> { static constexpr aclDataType value =  ACL_FLOAT; };
template <> struct to_acl_dtype< uint8_t> { static constexpr aclDataType value =  ACL_UINT8; };
template <> struct to_acl_dtype<  int8_t> { static constexpr aclDataType value =   ACL_INT8; };
template <> struct to_acl_dtype<uint16_t> { static constexpr aclDataType value = ACL_UINT16; };
template <> struct to_acl_dtype< int16_t> { static constexpr aclDataType value =  ACL_INT16; };
template <> struct to_acl_dtype<uint32_t> { static constexpr aclDataType value = ACL_UINT32; };
template <> struct to_acl_dtype< int32_t> { static constexpr aclDataType value =  ACL_INT32; };
template <> struct to_acl_dtype<    bool> { static constexpr aclDataType value =   ACL_BOOL; };
template <typename T>
constexpr aclDataType to_acl_dtype_v = to_acl_dtype<T>::value;

template <typename T>
struct AclScalar {
    T value;
    aclScalar* scalar = nullptr;
    operator aclScalar*() const { return scalar; }
    AclScalar(T value): value(value), scalar(aclCreateScalar(&this->value, to_acl_dtype_v<T>)) {}
    ~AclScalar() { if (scalar) aclDestroyScalar(scalar); }
    AclScalar(AclScalar&& rhs) noexcept : value(rhs.value), scalar(rhs.scalar) {
        rhs.scalar = nullptr;
    }
    AclScalar(AclScalar&) = delete;
    AclScalar(const AclScalar&) = delete;
    AclScalar& operator=(T value) {
        this->value = value;
        if (scalar) aclDestroyScalar(scalar);
        scalar = aclCreateScalar(&this->value, to_acl_dtype_v<T>);
        return *this;
    }
    AclScalar& operator=(AclScalar&& rhs) noexcept {
        this->value = rhs.value;
        if (scalar) aclDestroyScalar(scalar);
        scalar = aclCreateScalar(&this->value, to_acl_dtype_v<T>);
        rhs.scalar = nullptr;
        return *this;
    }
};

}
