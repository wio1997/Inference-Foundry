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
 * \file kernel_operator_vec_vconv_intf.h
 * \brief
 */

#if defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
#error "kernel_operator_vec_vconv_intf.h cannot be used with compile flag --enable-simt enabled."
#endif

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_VCONV_INTF_H__
#endif

#ifndef ASCENDC_MODULE_OPERATOR_VEC_VCONV_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_VCONV_INTERFACE_H

#include "../../impl/basic_api/kernel_macros.h"
#include "kernel_tensor.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"
#include "kernel_struct_vdeq.h"
#include "../../impl/basic_api/utils/kernel_utils_mode.h"

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <cstdint>
#include "stub_def.h"
#endif

namespace AscendC {
#pragma begin_pipe(V)
/* **************************************************************************************************
 * Cast                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Cast Level 0
 * @brief Precision conversion
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] roundMode round mode
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
// Cast::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__ASC_USE_RESERVED_UBUF__(3510, "Cast is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void Cast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

// Cast::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__ASC_USE_RESERVED_UBUF__(3510, "Cast is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void Cast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Cast Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] roundMode round mode
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void Cast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, const uint32_t count);

/*
 * @ingroup CastDequant Level 0
 * @brief Dequant from int16_t to uint8_t/int8_t
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.srcRepStride src repeat stride
 */
template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
__ASC_USE_RESERVED_UBUF__(
    3510, "CastDequant is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void CastDequant(
    const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint64_t mask[], uint8_t repeatTime,
    const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
__ASC_USE_RESERVED_UBUF__(
    3510, "CastDequant is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void CastDequant(
    const LocalTensor<T>& dst, const LocalTensor<U>& src, const int32_t mask, uint8_t repeatTime,
    const UnaryRepeatParams& repeatParams);

// CastDeq has been updated, please use CastDequant instead.
template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
__ASC_USE_RESERVED_UBUF__(3510, "CastDeq is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void CastDeq(
    const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint64_t mask[], uint8_t repeatTime,
    const UnaryRepeatParams& repeatParams);

// CastDeq has been updated, please use CastDequant instead.
template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
__ASC_USE_RESERVED_UBUF__(3510, "CastDeq is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void CastDeq(
    const LocalTensor<T>& dst, const LocalTensor<U>& src, const int32_t mask, uint8_t repeatTime,
    const UnaryRepeatParams& repeatParams);

/*
 * @ingroup CastDequant Level 2
 * @brief Dequant from int16_t to uint8_t/int8_t
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, typename U, bool isVecDeq = true, bool halfBlock = true>
__aicore__ inline void CastDequant(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint32_t count);

// CastDeq has been updated, please use CastDequant instead.
template <typename T, typename U, bool isVecDeq = true, bool halfBlock = true>
__aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint32_t count);

/* **************************************************************************************************
 * AddReluCast                                             *
 * ************************************************************************************************* */
/*
 * @ingroup AddReluCast Level 0
 * @brief Precision conversion
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src block stride
 * @param [in] repeatParams.src1BlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 * @param [in] repeatParams.src1RepStride src repeat stride
 */
// AddReluCast::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__ASC_USE_RESERVED_UBUF__(
    3510, "AddReluCast is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void AddReluCast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams);

// AddReluCast::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__ASC_USE_RESERVED_UBUF__(
    3510, "AddReluCast is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void AddReluCast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams);

/*
 * @ingroup AddReluCast Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void AddReluCast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, const uint32_t count);

/* **************************************************************************************************
 * SubReluCast                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SubReluCast Level 0
 * @brief Precision conversion
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src block stride
 * @param [in] repeatParams.src1BlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 * @param [in] repeatParams.src1RepStride src repeat stride
 */
// SubReluCast::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__ASC_USE_RESERVED_UBUF__(
    3510, "SubReluCast is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void SubReluCast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams);

// SubReluCast::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__ASC_USE_RESERVED_UBUF__(
    3510, "SubReluCast is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")
__aicore__ inline void SubReluCast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams);

/*
 * @ingroup SubReluCast Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void SubReluCast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, const uint32_t count);

#pragma end_pipe
__aicore__ inline void SetDeqScale(half scale);

__aicore__ inline void SetDeqScale(float scale, int16_t offset, bool signMode);

template <typename T>
__aicore__ inline void SetDeqScale(const LocalTensor<T>& vdeq, const VdeqInfo& vdeqInfo);

/*
 * @ingroup Truncate Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, RoundMode roundMode>
__aicore__ inline void Truncate(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count);
} // namespace AscendC

#include "../../impl/basic_api/kernel_operator_vec_vconv_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_INTERFACE_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_VCONV_INTF_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_VCONV_INTF_H__
#endif
