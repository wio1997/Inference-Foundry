#include <cstdint>
#include <array>
#include <vector>
#include <iostream>
#include <cstdlib>
namespace SMLA {
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_mla_kernel_metadata.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_KERNEL_METADATA_H
#define SPARSE_FLASH_MLA_KERNEL_METADATA_H


namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t SMLA_METADATA_TOTAL_SIZE = 1024;
using SMLA_METADATA_T = int32_t;

constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_S2_MAX_NUM = 8;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx, bool isAIV = false)
{
    if (isAIV) {
        return FA_METADATA_SIZE * AIC_CORE_MAX_NUM + FD_METADATA_SIZE * coreIdx + metaIdx;
    } else {
        return FA_METADATA_SIZE * coreIdx + metaIdx;
    }
}
#endif

namespace detail {
struct SmlaMetadata {
    uint32_t faMetadata[AIC_CORE_MAX_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_MAX_NUM][FD_METADATA_SIZE];
};
}; // namespace detail

static_assert(SMLA_METADATA_TOTAL_SIZE * sizeof(SMLA_METADATA_T) >= sizeof(detail::SmlaMetadata));
}; // namespace optiling

#endif // SPARSE_FLASH_MLA_KERNEL_METADATA_H

using namespace optiling;

struct FakeTensor {uint32_t* p; void* GetData(){return p;}};
struct FD {uint32_t fdUsedVecNum=0;std::vector<uint32_t> fdIdx,fdBN2Idx,fdMIdx,fdWorkspaceIdx,fdS2SplitNum,fdMStart,fdMNum; FD():fdIdx(72),fdBN2Idx(72),fdMIdx(72),fdWorkspaceIdx(72),fdS2SplitNum(72),fdMStart(72),fdMNum(72){}};
struct SplitResult {uint32_t usedCoreNum=0,maxS2LoopNum=0;std::vector<uint32_t>bN2End,gS1End,s2End,firstFdDataWorkspaceIdx;FD fdRes;SplitResult():bN2End(36),gS1End(36),s2End(36),firstFdDataWorkspaceIdx(36){}};
class SparseFlashMlaMetadataCpuKernel {public:FakeTensor* metadata_;uint32_t aicCoreNum_=24,aivCoreNum_=48;bool isSplitG_=false;bool GenMetadata(SplitResult&);};
bool SparseFlashMlaMetadataCpuKernel::GenMetadata(SplitResult &splitRes)
{
    optiling::detail::SmlaMetadata *metadataPtr = static_cast<optiling::detail::SmlaMetadata *>(metadata_->GetData());
    *metadataPtr = {};
    // FA Metadata Generate
    if (isSplitG_) {
        for (size_t i = 0; i < aicCoreNum_; i++) {
            // 单核s2计算轮次最大数量
            metadataPtr->faMetadata[2 * i][FA_S2_MAX_NUM] = splitRes.maxS2LoopNum;
            metadataPtr->faMetadata[2 * i + 1][FA_S2_MAX_NUM] = splitRes.maxS2LoopNum;

            if (i >= splitRes.usedCoreNum) {
                metadataPtr->faMetadata[2 * i][FA_CORE_ENABLE_INDEX] = 0;     // AIC disenable
                metadataPtr->faMetadata[2 * i + 1][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
                continue;
            }
            metadataPtr->faMetadata[2 * i][FA_CORE_ENABLE_INDEX] = 1;     // AIC enable
            metadataPtr->faMetadata[2 * i + 1][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
            // FA START
            metadataPtr->faMetadata[2 * i][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metadataPtr->faMetadata[2 * i][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metadataPtr->faMetadata[2 * i][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];

            metadataPtr->faMetadata[2 * i + 1][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metadataPtr->faMetadata[2 * i + 1][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metadataPtr->faMetadata[2 * i + 1][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
            // FA END
            metadataPtr->faMetadata[2 * i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metadataPtr->faMetadata[2 * i][FA_M_END_INDEX] = splitRes.gS1End[i];
            metadataPtr->faMetadata[2 * i][FA_S2_END_INDEX] = splitRes.s2End[i];

            metadataPtr->faMetadata[2 * i + 1][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metadataPtr->faMetadata[2 * i + 1][FA_M_END_INDEX] = splitRes.gS1End[i];
            metadataPtr->faMetadata[2 * i + 1][FA_S2_END_INDEX] = splitRes.s2End[i];
            // firstFdDataWorkspace
            metadataPtr->faMetadata[2 * i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
            metadataPtr->faMetadata[2 * i + 1][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] =
                splitRes.firstFdDataWorkspaceIdx[i];
        }
    } else {
        for (size_t i = 0; i < aicCoreNum_; ++i) {
            if (i >= splitRes.usedCoreNum) {
                metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
                continue;
            }
            metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
            // FA START
            metadataPtr->faMetadata[i][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metadataPtr->faMetadata[i][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metadataPtr->faMetadata[i][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
            // FA END
            metadataPtr->faMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metadataPtr->faMetadata[i][FA_M_END_INDEX] = splitRes.gS1End[i];
            metadataPtr->faMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
            // firstFdDataWorkspace
            metadataPtr->faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
        }
    }

    // FD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metadataPtr->fdMetadata[i][FD_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metadataPtr->fdMetadata[i][FD_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metadataPtr->fdMetadata[i][FD_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}
void test(){
 for(auto cores: {24u,32u,36u})for(auto used:{0u,1u}){
  std::array<uint32_t,1024>a,b;a.fill(0x11111111u);b.fill(0x22222222u);FakeTensor ta{a.data()},tb{b.data()};
  SplitResult s;s.usedCoreNum=used;s.fdRes.fdUsedVecNum=used;
  SparseFlashMlaMetadataCpuKernel k;k.aicCoreNum_=cores;k.aivCoreNum_=2*cores;k.metadata_=&ta;k.GenMetadata(s);k.metadata_=&tb;k.GenMetadata(s);
  constexpr size_t n=sizeof(optiling::detail::SmlaMetadata)/sizeof(uint32_t);bool eq=true,tail=true;
  for(size_t i=0;i<n;i++)eq &= a[i]==b[i];
  for(size_t i=n;i<1024;i++)tail &= a[i]==0x11111111u && b[i]==0x22222222u;
  if(!eq||!tail||n>=1024)std::abort();
  std::cout << "{\"operator\":\"SMLA\",\"aic\":"<<cores<<",\"used\":"<<used<<",\"struct_words\":"<<n<<",\"tail_words\":"<<1024-n<<",\"struct_equal\":true,\"tail_sentinels_unchanged\":true,\"full_packet_equal\":false}\n";
 }
}
}
namespace QLI {
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_v2_metadata.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_METADATA_H
#define QUANT_LIGHTNING_INDEXER_V2_METADATA_H


namespace optiling {

// Constants
inline constexpr uint32_t AIC_CORE_MAX_NUM = 36;
inline constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t QLI_V2_METADATA_TOTAL_SIZE = 1024;
using QLI_V2_METADATA_T = int32_t;

inline constexpr uint32_t QLI_V2_METADATA_SIZE = 8;
inline constexpr uint32_t QLD_V2_METADATA_SIZE = 8;

// LI Metadata Index Definitions
inline constexpr uint32_t QLI_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t QLI_V2_BN2_START_INDEX = 1;
inline constexpr uint32_t QLI_V2_M_START_INDEX = 2;
inline constexpr uint32_t QLI_V2_S2_START_INDEX = 3;
inline constexpr uint32_t QLI_V2_BN2_END_INDEX = 4;
inline constexpr uint32_t QLI_V2_M_END_INDEX = 5;
inline constexpr uint32_t QLI_V2_S2_END_INDEX = 6;
inline constexpr uint32_t QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX = 7;

// LD Metadata Index Definitions
inline constexpr uint32_t QLD_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t QLD_V2_BN2_IDX_INDEX = 1;
inline constexpr uint32_t QLD_V2_M_IDX_INDEX = 2;
inline constexpr uint32_t QLD_V2_WORKSPACE_IDX_INDEX = 3;
inline constexpr uint32_t QLD_V2_WORKSPACE_NUM_INDEX = 4;
inline constexpr uint32_t QLD_V2_M_START_INDEX = 5;
inline constexpr uint32_t QLD_V2_M_NUM_INDEX = 6;

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx, bool isAIV = false)
{
    if (isAIV) {
        return QLI_V2_METADATA_SIZE * AIC_CORE_MAX_NUM + QLD_V2_METADATA_SIZE * coreIdx + metaIdx;
    } else {
        return QLI_V2_METADATA_SIZE * coreIdx + metaIdx;
    }
}
#endif

namespace detail {
struct QliV2Metadata {
    uint32_t qliV2Metadata[AIC_CORE_MAX_NUM][QLI_V2_METADATA_SIZE];
    uint32_t qldV2Metadata[AIV_CORE_MAX_NUM][QLD_V2_METADATA_SIZE];
};
}; // namespace detail

static_assert(QLI_V2_METADATA_TOTAL_SIZE * sizeof(QLI_V2_METADATA_T) >= sizeof(detail::QliV2Metadata));
}; // namespace optiling

#endif // QUANT_LIGHTNING_INDEXER_V2_METADATA_H

using namespace optiling;

struct FakeTensor {uint32_t* p; void* GetData(){return p;}};
struct FD {uint32_t fdUsedVecNum=0;std::vector<uint32_t> fdIdx,fdBN2Idx,fdMIdx,fdWorkspaceIdx,fdS2SplitNum,fdMStart,fdMNum; FD():fdIdx(72),fdBN2Idx(72),fdMIdx(72),fdWorkspaceIdx(72),fdS2SplitNum(72),fdMStart(72),fdMNum(72){}};
struct SplitResult {uint32_t usedCoreNum=0,maxS2LoopNum=0;std::vector<uint32_t>bN2End,gS1End,s2End,firstFdDataWorkspaceIdx;FD fdRes;SplitResult():bN2End(36),gS1End(36),s2End(36),firstFdDataWorkspaceIdx(36){}};
class QuantLightningIndexerV2MetadataCpuKernel {public:FakeTensor* metadata_;uint32_t aicCoreNum_=24,aivCoreNum_=48;bool isSplitG_=false;bool GenMetadata(SplitResult&);};
bool QuantLightningIndexerV2MetadataCpuKernel::GenMetadata(SplitResult &splitRes)
{
    optiling::detail::QliV2Metadata *metadataPtr = static_cast<optiling::detail::QliV2Metadata *>(metadata_->GetData());
    *metadataPtr = {};
    // LI Metadata Generate
    for (size_t i = 0; i < aicCoreNum_; ++i) {
        if (i >= splitRes.usedCoreNum) {
            metadataPtr->qliV2Metadata[i][QLI_V2_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        metadataPtr->qliV2Metadata[i][QLI_V2_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        metadataPtr->qliV2Metadata[i][QLI_V2_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
        metadataPtr->qliV2Metadata[i][QLI_V2_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
        metadataPtr->qliV2Metadata[i][QLI_V2_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
        // FA END
        metadataPtr->qliV2Metadata[i][QLI_V2_BN2_END_INDEX] = splitRes.bN2End[i];
        metadataPtr->qliV2Metadata[i][QLI_V2_M_END_INDEX] = splitRes.gS1End[i];
        metadataPtr->qliV2Metadata[i][QLI_V2_S2_END_INDEX] = splitRes.s2End[i];
        metadataPtr->qliV2Metadata[i][QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX] =
            splitRes.firstFdDataWorkspaceIdx[i];
    }

    // LD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metadataPtr->qldV2Metadata[i][QLD_V2_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metadataPtr->qldV2Metadata[i][QLD_V2_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metadataPtr->qldV2Metadata[i][QLD_V2_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metadataPtr->qldV2Metadata[i][QLD_V2_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metadataPtr->qldV2Metadata[i][QLD_V2_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metadataPtr->qldV2Metadata[i][QLD_V2_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metadataPtr->qldV2Metadata[i][QLD_V2_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metadataPtr->qldV2Metadata[i][QLD_V2_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}
void test(){
 for(auto cores: {24u,32u,36u})for(auto used:{0u,1u}){
  std::array<uint32_t,1024>a,b;a.fill(0x11111111u);b.fill(0x22222222u);FakeTensor ta{a.data()},tb{b.data()};
  SplitResult s;s.usedCoreNum=used;s.fdRes.fdUsedVecNum=used;
  QuantLightningIndexerV2MetadataCpuKernel k;k.aicCoreNum_=cores;k.aivCoreNum_=2*cores;k.metadata_=&ta;k.GenMetadata(s);k.metadata_=&tb;k.GenMetadata(s);
  constexpr size_t n=sizeof(optiling::detail::QliV2Metadata)/sizeof(uint32_t);bool eq=true,tail=true;
  for(size_t i=0;i<n;i++)eq &= a[i]==b[i];
  for(size_t i=n;i<1024;i++)tail &= a[i]==0x11111111u && b[i]==0x22222222u;
  if(!eq||!tail||n>=1024)std::abort();
  std::cout << "{\"operator\":\"QLI\",\"aic\":"<<cores<<",\"used\":"<<used<<",\"struct_words\":"<<n<<",\"tail_words\":"<<1024-n<<",\"struct_equal\":true,\"tail_sentinels_unchanged\":true,\"full_packet_equal\":false}\n";
 }
}
}
int main(){SMLA::test();QLI::test();}
