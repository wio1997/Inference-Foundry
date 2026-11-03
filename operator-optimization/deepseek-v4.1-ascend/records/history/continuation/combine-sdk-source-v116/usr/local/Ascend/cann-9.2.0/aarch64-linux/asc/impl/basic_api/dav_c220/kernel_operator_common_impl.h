/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator_common_impl.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/basic_api/dav_c220/kernel_operator_common_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"basic_api/kernel_operator_intf.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_COMMON_IMPL_H__
#endif
#ifndef ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#define ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#include "../../../include/basic_api/kernel_common.h"
#include "../kernel_utils.h"
#include "../../../include/basic_api/kernel_tensor.h"
#include "../../../include/basic_api/kernel_struct_mm.h"
#include "../../../include/basic_api/kernel_base_types.h"
namespace AscendC {

[[deprecated("NOTICE: SetSysWorkSpace has been deprecated and will be removed in the next version.")]]
__aicore__ inline void SetSysWorkspace(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((workspace != nullptr), { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
#else
    if (g_sysWorkspaceReserved == nullptr) {
        g_sysWorkspaceReserved = workspace;
    }
#endif
}

__aicore__ inline void SetSysWorkspaceForce(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((workspace != nullptr), { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
#else
#if defined(__NPU_DEVICE__)
    __set_kfc_workspace_addr(workspace);
#else
    g_sysWorkspaceReserved = workspace;
#endif
#endif
}

__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((workspace != nullptr), { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
    return workspace;
#else
    (void)(workspace);
#if defined(__NPU_DEVICE__)
    return __get_kfc_workspace_addr() + RESERVED_WORKSPACE;
#else
    return g_sysWorkspaceReserved + RESERVED_WORKSPACE;
#endif
#endif
}

template <atomic_type_t type, atomic_op_t op>
__aicore__ inline void SetStoreAtomicConfigImpl()
{
    set_st_atomic_cfg(type, op);
}

__aicore__ inline int64_t GetStoreAtomicConfigImpl() { return get_st_atomic_cfg(); }

__aicore__ inline void GetStoreAtomicConfigImpl(uint16_t& atomicType, uint16_t& atomicOp)
{
    int64_t stAtomic = get_st_atomic_cfg();
    constexpr uint64_t typeMask = 0x7;
    constexpr uint64_t opBit = 3;
    constexpr uint64_t opMask = 0x3;
    atomicType = (static_cast<uint64_t>(stAtomic) & typeMask);
    atomicOp = ((static_cast<uint64_t>(stAtomic) >> opBit) & opMask);
}

__aicore__ inline void CheckLocalMemoryIAImpl(const CheckLocalMemoryIAParam& checkParams)
{
    uint64_t config = 0;
    config = config | (static_cast<uint64_t>(checkParams.startAddr) << 48);     // start address, DEC[63:48]
    config = config | (static_cast<uint64_t>(checkParams.endAddr) << 32);       // end address, DEC[47:32]
    config = config | (static_cast<uint64_t>(checkParams.isScalarRead) << 31);  // scalar read access, DEC[31]
    config = config | (static_cast<uint64_t>(checkParams.isScalarWrite) << 30); // scalar write access, DEC[30]
    config = config | (static_cast<uint64_t>(checkParams.isVectorRead) << 29);  // vector read access, DEC[29]
    config = config | (static_cast<uint64_t>(checkParams.isVectorWrite) << 28); // vector write access, DEC[28]
    config = config | (static_cast<uint64_t>(checkParams.isMteRead) << 27);     // vector mte read access, DEC[27]
    config = config | (static_cast<uint64_t>(checkParams.isMteWrite) << 26);    // vector mte write access, DEC[26]
    config = config | (checkParams.reserved << 1);                              // reserved, DEC[25:1]
    config = config | (static_cast<uint8_t>(checkParams.isEnable));             // enable bit, DEC[0]
    if (checkParams.enableBit == SET_DATA_EXP_ZERO) {
        set_data_exp_0(config);
    } else if (checkParams.enableBit == SET_DATA_EXP_ONE) {
        set_data_exp_1(config);
    } else if (checkParams.enableBit == SET_DATA_EXP_TWO) {
        set_data_exp_2(config);
    } else if (checkParams.enableBit == SET_DATA_EXP_THREE) {
        set_data_exp_3(config);
    } else {
        ASCENDC_REPORT_NOT_SUPPORT(false, "enableBit");
    }
}

template <int8_t startBit, int8_t endBit>
__aicore__ inline void CheckCtrlSprBitRange()
{
    static_assert((startBit == endBit && startBit == 48), "startBit and endBit must be both 48 on current platform!");
}

template <int8_t startBit, int8_t endBit>
__aicore__ static inline void SetCtrlSprImpl(int64_t value)
{
    CheckCtrlSprBitRange<startBit, endBit>();
    if (endBit - startBit == 63) {
        set_ctrl(value);
        return;
    }
    uint64_t mask = ((uint64_t(1) << (endBit - startBit + 1)) - 1) << startBit;
    mask = ~mask;
    int64_t setValue = get_ctrl() & mask;
    setValue |= (value << startBit);
    set_ctrl(setValue);
}

template <int8_t startBit, int8_t endBit>
__aicore__ static inline int64_t GetCtrlSprImpl()
{
    CheckCtrlSprBitRange<startBit, endBit>();
    int64_t value = get_ctrl();
    if (endBit - startBit == 63) {
        return value;
    }
    value = value >> startBit;
    value &= ((uint64_t(1) << (endBit - startBit + 1)) - 1);
    return value;
}

namespace Internal {
template <SaturationMode mode>
__aicore__ inline constexpr bool IsSupportedSaturationMode()
{
    return mode == SaturationMode::FLOAT || mode == SaturationMode::INT || mode == SaturationMode::CAST;
}
} // namespace Internal

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_COMMON_IMPL_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_COMMON_IMPL_H__
#endif
