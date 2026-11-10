
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#define KERNEL_LOG_ERROR(...) do {} while(0)
namespace aicpu {
struct TensorShape { std::vector<int64_t> dims; int64_t GetDimSize(size_t i) {return dims.at(i);} };
struct Tensor { std::vector<int32_t> data; TensorShape shape; void* GetData(){return data.empty()?nullptr:data.data();} int64_t NumElements(){return data.size();} TensorShape* GetTensorShape(){return &shape;} };
struct CpuKernelContext{};
struct CpuKernel {virtual uint32_t Compute(CpuKernelContext&)=0;virtual ~CpuKernel()=default;};
}
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

#include <cstdint>

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
 * \file sparse_flash_mla_metadata_aicpu.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_METADATA_AICPU_H
#define SPARSE_FLASH_MLA_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>

namespace aicpu {
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t COST_WEIGHT_M = 6U;
constexpr uint32_t COST_WEIGHT_S2 = 10U;
constexpr uint32_t BATCH_CONSISTENCY_MAX_REDUCTION_PARTS = 32U;
constexpr bool ORI_KV = false;
constexpr bool CMP_KV = true;
constexpr uint32_t NO_MASK = 0;
constexpr uint32_t HAS_MASK = 1;

enum BlockType : uint32_t {
    ORI_NORMAL_BLOCK = 0,
    ORI_TAIL_BLOCK,
    CMP_NORMAL_BLOCK,
    CMP_TAIL_BLOCK,
    BLOCK_MAX_TYPE
};

enum class SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

enum class ValidSocVersion {
    ASCEND910 = 0,
    ASCEND950
};

template <class T>
using Range = std::pair<T, T>;

template <class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template <typename T>
T Clip(T value, T minValue, T maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

template <typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

// 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
struct FlashDecodeResult {
    uint32_t fdUsedVecNum{0U}; // 归约过程使用的vector数量
    // 1、归约任务的索引信息
    std::vector<uint32_t> fdBN2Idx{}; // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> fdMIdx{};         // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> fdWorkspaceIdx{}; // 每个归约任务在workspace中的存放位置
    std::vector<uint32_t> fdS2SplitNum{}; // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    std::vector<uint32_t> fdMSize{};      // 每个归约任务m轴大小，脚标为归约任务的序号
    // 2、FD负载均衡阶段，归约任务的分核（vec）信息
    std::vector<uint32_t> fdIdx{};    // FD负载均衡阶段，每个vector处理的归约任务对应ID
    std::vector<uint32_t> fdMStart{}; // FD负载均衡阶段，每个vector处理的归约任务的m轴起点
    std::vector<uint32_t> fdMNum{};   // FD负载均衡阶段，每个vector处理的归约任务的m轴行数

    FlashDecodeResult(uint32_t aicNum, uint32_t aivNum)
        : fdBN2Idx(aicNum),
          fdMIdx(aicNum),
          fdWorkspaceIdx(aicNum),
          fdS2SplitNum(aicNum),
          fdMSize(aicNum),
          fdIdx(aivNum),
          fdMStart(aivNum),
          fdMNum(aivNum)
    {}
};

// 分核功能模块输出：FA阶段的核间分核信息
struct SplitResult {
    uint32_t usedCoreNum{0U};                        // 使用的核数量
    std::vector<uint32_t> bN2End{};                  // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End{};                  // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End{};                   // 每个核处理数据的S2结束点
    std::vector<uint32_t> firstFdDataWorkspaceIdx{}; // 每个核第一份归约任务的存放位置
    int64_t maxCost{0};                              // 慢核开销
    uint32_t numOfFdHead{0U};                        // 归约任务数量
    uint32_t maxS2SplitNum{0U};                      // 单个归约任务最大分核数量
    uint32_t maxS2LoopNum{0U};                       // 单个核最大s2计算轮次数量
    FlashDecodeResult fdRes{0U, 0U};                 // FD信息

    SplitResult(uint32_t aicNum, uint32_t aivNum)
        : bN2End(aicNum),
          gS1End(aicNum),
          s2End(aicNum),
          firstFdDataWorkspaceIdx(aicNum),
          fdRes(aicNum, aivNum) {};
};

// 分核功能模块内部使用：记录切分信息
struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum{};    // S1G方向，切了多少个基本块
    std::vector<uint32_t> oriS2BaseNum{};  // oriS2方向，切了多少个基本块
    std::vector<uint32_t> cmpS2BaseNum{};  // cmpS2方向，切了多少个基本块
    std::vector<uint32_t> s1GTailSize{};   // S1G方向，尾块size
    std::vector<uint32_t> oriS2TailSize{}; // oriS2方向，尾块size
    std::vector<uint32_t> cmpS2TailSize{}; // cmpS2方向，尾块size
    bool isKvSeqAllZero{true};

    explicit SplitInfo(uint32_t batchSize)
        : s1GBaseNum(batchSize),
          oriS2BaseNum(batchSize),
          cmpS2BaseNum(batchSize),
          s1GTailSize(batchSize),
          oriS2TailSize(batchSize),
          cmpS2TailSize(batchSize)
    {}
};

// 分核功能模块内部使用：记录batch的开销信息
struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch{};          // 整个batch的开销
    std::vector<uint32_t> bN2BlockOfEachBatch{};        // 整个batch的开销
    std::vector<uint32_t> bN2S2LoopOfEachBatch{};       // 整个batch的s2计算轮次数量
    std::vector<int64_t> bN2LastBlockCostOfEachBatch{}; // batch最后一块的开销
    uint32_t totalBlockNum{0U};
    int64_t totalCost{0};
    int64_t maxS1GCost{0}; // 记录所有S1G行中的最大开销

    explicit CostInfo(uint32_t batchSize)
        : bN2CostOfEachBatch(batchSize),
          bN2BlockOfEachBatch(batchSize),
          bN2S2LoopOfEachBatch(batchSize),
          bN2LastBlockCostOfEachBatch(batchSize)
    {}
};

// 分核功能模块内部使用：分核过程中，case基本信息的上下文信息，组合以减少接口传参数量
struct SplitContext {
    SplitInfo splitInfo{0U};
    CostInfo costInfo{0U};

    explicit SplitContext(uint32_t batchSize)
        : splitInfo(batchSize),
          costInfo(batchSize)
    {}
};

// 分核功能模块内部使用：记录batch相关的临时信息
struct BatchCache {
    uint32_t bIdx{0U};
    uint32_t s1Size{0U};
    uint32_t oriS2Size{0U};
    uint64_t cmpRevertS2Size{0U};
    int64_t oriPreTokenLeftUp{0};
    int64_t oriNextTokenLeftUp{0};
    int64_t cmpPreTokenLeftUp{0};
    int64_t cmpNextTokenLeftUp{0};
    BlockCost<int64_t> typeCost{};
};

// 分核功能模块内部使用：记录当前行（S1G）的临时信息
struct S1GCache {
    uint32_t bIdx{0U};
    uint32_t s1GIdx{0U};
    uint32_t s2Start{0U};
    uint32_t s2End{0U};
    uint32_t oriS2Start{0U};
    uint32_t oriS2End{0U};
    uint32_t cmpS2Start{0U}; // win部分与cmp部分的切分点
    uint32_t cmpS2End{0U};
    int64_t s1GCost{0};
    int64_t s1GLastBlockCost{0};
    uint32_t s1GBlock{0U};
    uint32_t s2Loop{0U};
    uint32_t oriS1GBlock{0U};
    int64_t oriS1GCost{0};
    int64_t oriS1GLastBlockCost{0};
    int64_t oriS1GNormalBlockCost{0};
    uint32_t cmpS1GBlock{0U};
    int64_t cmpS1GCost{0};
    int64_t cmpS1GLastBlockCost{0};
    int64_t cmpS1GNormalBlockCost{0};
    int64_t oriS2TailSize{0};
    int64_t cmpS2TailSize{0};
    uint32_t actOriS2Size{0U};
    uint32_t actCmpS2Size{0U};
    uint32_t reductionBlockSize{0U};
};

// 分核功能模块内部使用：记录分配过程中，当前核的负载信息
struct CoreCache {
    int64_t costLimit{0}; // 负载上限
    int64_t cost{0};      // 已分配负载
    uint32_t block{0U};   // 已分配块数
    uint32_t s2Loop{0U};  // 已分配s2计算轮次数量
};

// 分核功能模块内部使用：记录分配过程中的上下文信息
struct AssignContext {
    uint32_t curBIdx{0U};
    uint32_t curBN2Idx{0U};
    uint32_t curS1GIdx{0U};
    uint32_t curS2Idx{0U};
    uint32_t curCoreIdx{0U};
    int64_t unassignedCost{0};
    uint32_t curKvSplitPart{1U};
    uint32_t preFdDataNum{0U};

    int64_t bN2Cost{0};
    uint32_t bN2Block{0U};
    uint32_t bN2S2Loop{0U};
    bool isFinished{false};
    BatchCache batchCache{};
    S1GCache s1GCache{};
    CoreCache coreCache{};
};

class SparseFlashMlaMetadataCpuKernel : public CpuKernel {
public:
    SparseFlashMlaMetadataCpuKernel() = default;
    ~SparseFlashMlaMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

public:
    bool Prepare(CpuKernelContext &ctx);
    int32_t GetQueryBatchSize();
    void CalcOriMaskMode();
    void CalcCmpMaskMode();
    ValidSocVersion ProcessSocVersion();
    bool ParamsCheck();
    int32_t GetSumOfQuerySeq();
    bool ParamsInit();
    bool BalanceSchedule(SplitResult &splitRes);
    bool GenMetadata(SplitResult &splitRes);
    // util
    uint32_t GetS1SeqSize(uint32_t bIdx);
    uint32_t GetOriS2SeqSize(uint32_t bIdx);
    uint32_t GetCmpS2SeqSize(uint32_t bIdx);
    uint64_t GetRevertS2Size(uint32_t bIdx);
    uint32_t GetS1Idx(uint32_t s1Size, uint32_t s1GIdx);
    uint32_t GetBsStride(uint32_t bIdx, uint32_t s1Idx);
    uint32_t GetOriTopkLength(uint32_t bsStride);
    uint32_t GetOriTopkLength(uint32_t s1GIdx, const BatchCache &batchCache) const;
    uint32_t ReadOriTopkLengthAtRow(uint32_t bIdx, uint32_t s1Idx) const;
    uint32_t GetCmpTopkLength(uint32_t bsStride);
    int64_t CalcOriPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    int64_t CalcOriNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    int64_t CalcCmpPreTokenLeftUp(uint32_t s1Size, uint64_t s2Size);
    int64_t CalcCmpNextTokenLeftUp(uint32_t s1Size, uint64_t s2Size);
    Range<int64_t> CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache, bool isCmpKv);
    int64_t OriCalcCost(uint32_t basicM, uint32_t basicS2);
    int64_t CmpCalcCost(uint32_t basicM, uint32_t basicS2);
    void CalcCostTable(uint32_t s1GTailSize, uint32_t reductionBlockSize, uint32_t oriS2TailSize,
                       uint32_t cmpS2TailSize);

    // cache calculation
    void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
    void CalcOriBlockRange(const Range<int64_t> &oriS2TokenRange, const BatchCache &batchCache, S1GCache &s1GCache);
    void CalcCmpBlockRange(const Range<int64_t> &cmpS2TokenRange, const BatchCache &batchCache, S1GCache &s1GCache);
    void CalcOriS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo);
    void CalcCmpS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo);
    void GatherOriAndCmpCache(S1GCache &s1GCache);
    void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache,
                      S1GCache &s1GCache);

    // preprocess
    void CalcSplitInfo(SplitContext &splitContext);
    void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
    void CalcCostInfo(SplitContext &splitContext);

    // assign
    void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
    int64_t CalcCurBlockCost(const AssignContext &assignContext);
    uint32_t CalcCurBlockS2Loop(const AssignContext &assignContext);
    void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext);
    void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignBlocksToCore(const SplitContext &splitContext, AssignContext &assignContext, SplitResult &result);

    // FD
    bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes);
    bool IsFirstReductionBlock(const AssignContext &assignContext, const SplitResult &splitRes);
    void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result);

    // main
    void SplitFD(SplitResult &splitRes);
    void CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext, SplitResult &result);

public:
    // input
    Tensor *cuSeqlensQ_ = nullptr;
    Tensor *cuSeqlensOriKv_ = nullptr;
    Tensor *cuSeqlensCmpKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedOriKv_ = nullptr;
    Tensor *sequsedCmpKv_ = nullptr;
    Tensor *cmpResidualKv_ = nullptr;
    Tensor *oriTopkLength_ = nullptr;
    Tensor *cmpTopkLength_ = nullptr;

    // output
    Tensor *metadata_ = nullptr;

    // attributes
    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = 0;
    int32_t numHeadsQ_ = 0;
    int32_t maxSeqlenOriKv_ = 0;
    int32_t maxSeqlenCmpKv_ = 0;
    int32_t numHeadsKv_ = 1;
    int32_t headDim_ = 0;
    int32_t oriTopK_ = 0;
    int32_t cmpTopK_ = 0;
    int32_t cmpRatio_ = 0;
    int32_t oriMaskMode_ = static_cast<int32_t>(SparseMode::BAND);
    int32_t cmpMaskMode_ = static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL);
    int64_t oriWinLeft_ = 127;
    int64_t oriWinRight_ = 0;
    std::string layoutQ_ = "BSND";
    std::string layoutKv_ = "PA_BBND";
    bool hasOriKv_ = true;
    bool hasCmpKv_ = true;
    uint32_t aicCoreNum_ = optiling::AIC_CORE_MAX_NUM;
    uint32_t aivCoreNum_ = optiling::AIV_CORE_MAX_NUM;
    bool isBatchConsistency_ = false;

    // attr
    std::string socVersion_ = "Ascend950";
    int64_t oriPreToken_ = 0;
    int64_t oriNextToken_ = 0;
    int64_t cmpPreToken_ = 0;
    int64_t cmpNextToken_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = 0;
    uint32_t s2BaseSize_ = 128U;
    bool isS1G_ = true;
    bool supportFd_ = false;
    uint32_t oriAttentionMode_ = HAS_MASK;
    uint32_t cmpAttentionMode_ = HAS_MASK;
    BlockCost<int64_t> typeCost_ = {};
    bool isSplitG_ = false;
    bool isSparseOriKv_ = false;
    bool isSparseCmpKv_ = false;
    bool hasOriTopkLength_ = false;
    uint32_t remainedBlockNum_ = 0U;

public:
    enum class ParamId : uint32_t {
        // input
        cuSeqlensQ = 0,
        cuSeqlensOriKv = 1,
        cuSeqlensCmpKv = 2,
        sequsedQ = 3,
        sequsedOriKv = 4,
        sequsedCmpKv = 5,
        cmpResidualKv = 6,
        oriTopkLength = 7,
        cmpTopkLength = 8,
        // output
        metaData = 0,
    };
};
} // namespace aicpu

#endif
using namespace optiling;
namespace aicpu {
uint32_t SparseFlashMlaMetadataCpuKernel::Compute(CpuKernelContext&){return 0;}
int32_t SparseFlashMlaMetadataCpuKernel::GetSumOfQuerySeq()
{
    int32_t batchSize = GetQueryBatchSize();
    // 如果sequsedQ_ 传了，使用sequsedQ_获取 BsSize
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        if (sequsedQ_->GetTensorShape() != nullptr) {
            const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
            int32_t queryBsSize = 0;
            for (int i = 0; i < batchSize; i++) {
                queryBsSize += seqUsedPtr[i];
            }
            return queryBsSize;
        }
    }
    // sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_获取 BsSize
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            if (cuSeqlensQ_->GetTensorShape() != nullptr) {
                const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
                return s1Ptr[batchSize];
            }
        }
    }
    // 如果不是 TND，或者 cuSeqlensQ_ 为空，使用shape信息计算 BsSize
    return batchSize_ * maxSeqlenQ_;
}

int32_t SparseFlashMlaMetadataCpuKernel::GetQueryBatchSize()
{
    // 1. 如果sequsedQ_ 传了，使用sequsedQ_获取BatchSize
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        if (sequsedQ_->GetTensorShape() != nullptr) {
            return sequsedQ_->GetTensorShape()->GetDimSize(0);
        }
    }
    // 2. sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_获取BatchSize
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            if (cuSeqlensQ_->GetTensorShape() != nullptr) {
                return cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
            }
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用batchSize_
    return batchSize_;
}

void SparseFlashMlaMetadataCpuKernel::CalcOriMaskMode()
{
    if (oriMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK)) {
        oriPreToken_ = INT64_MAX;
        oriNextToken_ = INT64_MAX;
        oriAttentionMode_ = NO_MASK;
    } else if (oriMaskMode_ == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        oriPreToken_ = INT64_MAX;
        oriNextToken_ = 0;
        oriAttentionMode_ = HAS_MASK;
    } else { // SparseMode = 4
        oriPreToken_ = (oriWinLeft_ > -1) ? oriWinLeft_ : INT64_MAX;
        oriNextToken_ = (oriWinRight_ > -1) ? oriWinRight_ : INT64_MAX;
        oriAttentionMode_ = HAS_MASK;
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcCmpMaskMode()
{
    if (cmpMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK)) {
        cmpPreToken_ = INT64_MAX;
        cmpNextToken_ = INT64_MAX;
        cmpAttentionMode_ = NO_MASK;
    } else if (cmpMaskMode_ == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        cmpPreToken_ = INT64_MAX;
        cmpNextToken_ = 0;
        cmpAttentionMode_ = HAS_MASK;
    } else { // SparseMode = 4
        cmpPreToken_ = (oriWinLeft_ > -1) ? oriWinLeft_ : INT64_MAX;
        cmpNextToken_ = (oriWinRight_ > -1) ? oriWinRight_ : INT64_MAX;
        cmpAttentionMode_ = HAS_MASK;
    }
}

ValidSocVersion SparseFlashMlaMetadataCpuKernel::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    } else {
        return ValidSocVersion::ASCEND910;
    }
}

bool SparseFlashMlaMetadataCpuKernel::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    CalcOriMaskMode();
    CalcCmpMaskMode();
    isS1G_ = (layoutQ_ == "BSND" || layoutQ_ == "BSH" || layoutQ_ == "TND");
    if (numHeadsKv_ == 0) {
        KERNEL_LOG_ERROR("num_heads_kv should not be 0.");
        return false;
    }
    ValidSocVersion validSocVersion = ProcessSocVersion();
    groupSize_ = numHeadsQ_ / numHeadsKv_;
    if (hasOriKv_ && oriTopK_ != 0) {
        isSparseOriKv_ = true;
    }
    if (hasCmpKv_ && cmpTopK_ != 0) {
        isSparseCmpKv_ = true;
    }
    if (isBatchConsistency_) {
        supportFd_ = true;
    }
    if (validSocVersion == ValidSocVersion::ASCEND910) {
        if (isSparseOriKv_ && !isSparseCmpKv_) {
            mBaseSize_ = groupSize_;
        } else {
            mBaseSize_ = isSparseCmpKv_ ? groupSize_ : (256U / groupSize_) * groupSize_;
        }
        s2BaseSize_ = 512U;
    } else if (validSocVersion == ValidSocVersion::ASCEND950) {
        if (groupSize_ > 64U) {
            isSplitG_ = true;
            aicCoreNum_ /= 2;
        }
        mBaseSize_ = groupSize_;
        s2BaseSize_ = 128U;
    } else {
        mBaseSize_ = groupSize_;
        s2BaseSize_ = 128U;
    }
    return true;
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetS1Idx(uint32_t s1Size, uint32_t s1GIdx)
{
    uint32_t s1GToken = s1GIdx * mBaseSize_;
    uint32_t s1Idx = 0;
    if (isS1G_) {
        s1Idx = s1GToken / static_cast<int64_t>(groupSize_);
    } else {
        s1Idx = s1GToken % static_cast<int64_t>(s1Size);
    }
    return s1Idx;
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetBsStride(uint32_t bIdx, uint32_t s1Idx)
{
    uint32_t bsStride = 0;
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
            bsStride = s1Ptr[bIdx] + s1Idx;
            return bsStride;
        }
    }
    bsStride = bIdx * static_cast<uint32_t>(maxSeqlenQ_) + s1Idx;
    return bsStride;
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetOriTopkLength(uint32_t bsStride)
{
    // 尝试使用 oriTopkLength_
    if (oriTopK_ != 0 && oriMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
        oriTopkLength_ != nullptr && oriTopkLength_->GetData() != nullptr) {
        const int32_t *oriTopkPtr = static_cast<const int32_t *>(oriTopkLength_->GetData());
        return static_cast<uint32_t>(oriTopkPtr[bsStride]);
    }
    // 如果不是 DEFAULT_MASK，使用 oriTopK_
    return static_cast<uint32_t>(oriTopK_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::ReadOriTopkLengthAtRow(uint32_t bIdx, uint32_t s1Idx) const
{
    if (!hasOriTopkLength_) {
        return static_cast<uint32_t>(oriTopK_);
    }
    const int32_t *topkLenPtr = static_cast<const int32_t *>(oriTopkLength_->GetData());
    if (layoutQ_ == "TND") {
        uint32_t tIdx = s1Idx;
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *cuPtr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
            tIdx = static_cast<uint32_t>(cuPtr[bIdx]) + s1Idx;
        } else {
            tIdx = bIdx * static_cast<uint32_t>(maxSeqlenQ_) + s1Idx;
        }
        return static_cast<uint32_t>(topkLenPtr[tIdx * static_cast<uint32_t>(numHeadsKv_)]);
    }
    return static_cast<uint32_t>(
        topkLenPtr[(static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(maxSeqlenQ_) + s1Idx) *
                   static_cast<uint64_t>(numHeadsKv_)]);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetOriTopkLength(uint32_t s1GIdx, const BatchCache &batchCache) const
{
    if (!hasOriTopkLength_) {
        return static_cast<uint32_t>(oriTopK_);
    }
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
                                    static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) -
                           1;
    int64_t s1First = s1GFirstToken / static_cast<int64_t>(groupSize_);
    int64_t s1Last = s1GLastToken / static_cast<int64_t>(groupSize_);
    uint32_t maxLen = 0U;
    for (int64_t s1Idx = s1First; s1Idx <= s1Last; ++s1Idx) {
        maxLen = std::max(maxLen, ReadOriTopkLengthAtRow(batchCache.bIdx, static_cast<uint32_t>(s1Idx)));
    }
    return std::min(maxLen, static_cast<uint32_t>(oriTopK_));
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetCmpTopkLength(uint32_t bsStride)
{
    // 尝试使用 cmpTopkLength_
    if (cmpTopK_ != 0 && cmpMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
        cmpTopkLength_ != nullptr && cmpTopkLength_->GetData() != nullptr) {
        const int32_t *cmpTopkPtr = static_cast<const int32_t *>(cmpTopkLength_->GetData());
        return static_cast<uint32_t>(cmpTopkPtr[bsStride]);
    }
    // 如果不是 DEFAULT_MASK，使用 cmpTopK_
    return static_cast<uint32_t>(cmpTopK_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    // 1. 如果 sequsedQ_ 传了，直接使用
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用 maxSeqlenQ_
    return static_cast<uint32_t>(maxSeqlenQ_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetOriS2SeqSize(uint32_t bIdx)
{
    // 如果 sequsedOriKv_ 传了，直接使用
    if (sequsedOriKv_ != nullptr && sequsedOriKv_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedOriKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // sequsedOriKv_ 没传，判断 Layout
    if (layoutKv_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensOriKv_
        if (cuSeqlensOriKv_ != nullptr && cuSeqlensOriKv_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t *>(cuSeqlensOriKv_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 如果是PA场景，或 max_seqlen_ori_kv 没传入，且 ori_kv 为稀疏的，则尝试从 topk 中获取
    if ((layoutKv_ == "PA_BBND" || maxSeqlenOriKv_ == 0) && isSparseOriKv_) {
        return UINT32_MAX;
    }
    // 使用 max_seqlen_ori_kv
    return static_cast<uint32_t>(maxSeqlenOriKv_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetCmpS2SeqSize(uint32_t bIdx)
{
    // 如果 sequsedCmpKv_ 传了，直接使用
    if (sequsedCmpKv_ != nullptr && sequsedCmpKv_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedCmpKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // sequsedCmpKv_ 没传，判断 Layout
    if (layoutKv_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensCmpKv_
        if (cuSeqlensCmpKv_ != nullptr && cuSeqlensCmpKv_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t *>(cuSeqlensCmpKv_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 如果是PA场景，或 max_seqlen_cmp_kv 没传入，且 cmp_kv 为稀疏的，则尝试从topk中获取
    if ((layoutKv_ == "PA_BBND" || maxSeqlenCmpKv_ == 0) && isSparseCmpKv_) {
        return UINT32_MAX;
    }
    // 使用 max_seqlen_cmp_kv
    return static_cast<uint32_t>(maxSeqlenCmpKv_);
}

uint64_t SparseFlashMlaMetadataCpuKernel::GetRevertS2Size(uint32_t bIdx)
{
    uint32_t cmpS2Size = GetCmpS2SeqSize(bIdx);
    if (cmpResidualKv_ != nullptr && cmpResidualKv_->GetData() != nullptr) {
        const int32_t *residualPtr = static_cast<const int32_t *>(cmpResidualKv_->GetData());
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_) + residualPtr[bIdx];
    } else {
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_);
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        splitInfo.s1GBaseNum[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_) % mBaseSize_;
        if (hasOriKv_) {
            uint32_t curOriS2Size = GetOriS2SeqSize(bIdx);
            splitInfo.oriS2BaseNum[bIdx] = (static_cast<uint64_t>(curOriS2Size) + s2BaseSize_ - 1U) / s2BaseSize_;
        }
        if (hasCmpKv_) {
            uint32_t curCmpS2Size = GetCmpS2SeqSize(bIdx);
            splitInfo.cmpS2BaseNum[bIdx] = (static_cast<uint64_t>(curCmpS2Size) + s2BaseSize_ - 1U) / s2BaseSize_;
        }
        if (splitInfo.s1GBaseNum[bIdx] != 0U &&
            (splitInfo.oriS2BaseNum[bIdx] != 0U || splitInfo.cmpS2BaseNum[bIdx] != 0U)) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcOriPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    if (mode == SparseMode::BAND) {
        return oriPreToken_ == INT64_MAX ? INT64_MAX :
                                           static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + oriPreToken_;
    }
    return oriPreToken_;
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcOriNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return oriNextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return oriNextToken_ == INT64_MAX ?
                       INT64_MAX :
                       static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + oriNextToken_;
        default:
            return oriNextToken_;
    }
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcCmpPreTokenLeftUp(uint32_t s1Size, uint64_t s2Size)
{
    auto mode = static_cast<SparseMode>(cmpMaskMode_);
    if (mode == SparseMode::BAND) {
        return cmpPreToken_ == INT64_MAX ? INT64_MAX :
                                           static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + cmpPreToken_;
    }
    return cmpPreToken_;
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcCmpNextTokenLeftUp(uint32_t s1Size, uint64_t s2Size)
{
    auto mode = static_cast<SparseMode>(cmpMaskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return cmpNextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return cmpNextToken_ == INT64_MAX ?
                       INT64_MAX :
                       static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + cmpNextToken_;
        default:
            return cmpNextToken_;
    }
}

int64_t SparseFlashMlaMetadataCpuKernel::OriCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t oriAlignCoefM = 16U;
    uint32_t oriAlignCoefS2 = 64U;
    uint32_t oriAlignBasicM = (basicM + oriAlignCoefM - 1U) / oriAlignCoefM;
    uint32_t oriAlignBasicS2 = (basicS2 + oriAlignCoefS2 - 1U) / oriAlignCoefS2;
    return static_cast<int64_t>(COST_WEIGHT_M * oriAlignBasicM + COST_WEIGHT_S2 * oriAlignBasicS2);
}

int64_t SparseFlashMlaMetadataCpuKernel::CmpCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t cmpAlignCoefM = 16U;
    uint32_t cmpAlignCoefS2 = 64U;
    uint32_t cmpAlignBasicM = (basicM + cmpAlignCoefM - 1U) / cmpAlignCoefM;
    uint32_t cmpAlignBasicS2 = (basicS2 + cmpAlignCoefS2 - 1U) / cmpAlignCoefS2;
    return static_cast<int64_t>(COST_WEIGHT_M * cmpAlignBasicM + COST_WEIGHT_S2 * cmpAlignBasicS2);
}

void SparseFlashMlaMetadataCpuKernel::CalcCostTable(uint32_t s1GTailSize, uint32_t reductionBlockSize,
                                                    uint32_t oriS2TailSize, uint32_t cmpS2TailSize)
{
    uint32_t normalS2Size = isBatchConsistency_ && reductionBlockSize > 0U ? reductionBlockSize : s2BaseSize_;
    // ori 部分 cost
    if (hasOriKv_) {
        typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK] = OriCalcCost(mBaseSize_, normalS2Size);
        typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK] =
            (s1GTailSize == 0U) ? 0U : OriCalcCost(s1GTailSize, normalS2Size);
        typeCost_[ORI_NORMAL_BLOCK][ORI_TAIL_BLOCK] = (oriS2TailSize == 0U) ? 0U :
                                                                              OriCalcCost(mBaseSize_, oriS2TailSize);
        typeCost_[ORI_TAIL_BLOCK][ORI_TAIL_BLOCK] = (s1GTailSize == 0U || oriS2TailSize == 0U) ? 0U :
                                                                                                 OriCalcCost(s1GTailSize, oriS2TailSize);
    }
    // cmp 部分 cost
    if (hasCmpKv_) {
        typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] = CmpCalcCost(mBaseSize_, normalS2Size);
        typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] =
            (s1GTailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, normalS2Size);
        typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] = (cmpS2TailSize == 0U) ? 0U :
                                                                              CmpCalcCost(mBaseSize_, cmpS2TailSize);
        typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] = (s1GTailSize == 0U || cmpS2TailSize == 0U) ? 0U :
                                                                                                 CmpCalcCost(s1GTailSize, cmpS2TailSize);
    }
}

Range<int64_t> SparseFlashMlaMetadataCpuKernel::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache,
                                                                 bool isCmpKv)
{
    // actual seq == 0
    if (!isCmpKv) {
        if (batchCache.s1Size == 0U || batchCache.oriS2Size == 0U) {
            return std::make_pair(0, 0);
        }
        if (isSparseOriKv_ && !hasCmpKv_) {
            uint32_t oriTopkSize = GetOriTopkLength(s1GIdx, batchCache);
            if (oriTopkSize == 0U) {
                return std::make_pair(0, 0);
            }
            return std::make_pair(0, static_cast<int64_t>(oriTopkSize) - 1);
        }
    } else {
        if (batchCache.s1Size == 0U || batchCache.cmpRevertS2Size == 0U) {
            return std::make_pair(0, 0);
        }
    }

    // no mask
    uint32_t hasMask = 1;
    int64_t s2Size =
        isCmpKv ? static_cast<int64_t>(batchCache.cmpRevertS2Size) : static_cast<int64_t>(batchCache.oriS2Size);
    hasMask = isCmpKv ? cmpAttentionMode_ : oriAttentionMode_;
    if (!hasMask) {
        return std::make_pair(0, s2Size - 1);
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
                                    static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) -
                           1;

    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (isS1G_) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
        s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = 0;
    int64_t s2LastToken = 0;
    if (!isCmpKv) {
        s2FirstToken = s1FirstToken - batchCache.oriPreTokenLeftUp;
        s2LastToken =
            batchCache.oriNextTokenLeftUp == INT64_MAX ? INT64_MAX : s1LastToken + batchCache.oriNextTokenLeftUp;
    } else {
        s2FirstToken = s1FirstToken - batchCache.cmpPreTokenLeftUp;
        s2LastToken =
            batchCache.cmpNextTokenLeftUp == INT64_MAX ? INT64_MAX : s1LastToken + batchCache.cmpNextTokenLeftUp;
    }
    return std::make_pair(s2FirstToken, s2LastToken);
}

void SparseFlashMlaMetadataCpuKernel::CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext,
                                                     BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    if (hasOriKv_) {
        batchCache.oriS2Size = GetOriS2SeqSize(bIdx);
        batchCache.oriPreTokenLeftUp = CalcOriPreTokenLeftUp(batchCache.s1Size, batchCache.oriS2Size);
        batchCache.oriNextTokenLeftUp = CalcOriNextTokenLeftUp(batchCache.s1Size, batchCache.oriS2Size);
    }
    if (hasCmpKv_) {
        batchCache.cmpRevertS2Size = GetRevertS2Size(bIdx);
        batchCache.cmpPreTokenLeftUp = CalcCmpPreTokenLeftUp(batchCache.s1Size, batchCache.cmpRevertS2Size);
        batchCache.cmpNextTokenLeftUp = CalcCmpNextTokenLeftUp(batchCache.s1Size, batchCache.cmpRevertS2Size);
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcOriS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo)
{
    // 处理 ori 部分 block 信息
    if (s1GCache.oriS2Start >= s1GCache.oriS2End) {
        // ori 范围无效, 则整个 s1g 行等效为空行
        s1GCache.oriS1GBlock = 0;
        s1GCache.oriS1GCost = 0;
        s1GCache.oriS1GLastBlockCost = 0;
        s1GCache.oriS1GNormalBlockCost = 0;
    } else {
        // 计算 ori 方向 Block 数量及 Cost
        s1GCache.oriS1GBlock = s1GCache.oriS2End - s1GCache.oriS2Start;
        // 判断 ori S2 方向是否包含尾块
        uint32_t curOriTailS2Num = (s1GCache.oriS2TailSize != 0U) ? 1U : 0U;
        uint32_t curOriNormalS2Num = s1GCache.oriS1GBlock - curOriTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
            splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.oriS1GCost = typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK] * curOriNormalS2Num +
                                  typeCost_[ORI_TAIL_BLOCK][ORI_TAIL_BLOCK] * curOriTailS2Num;
            s1GCache.oriS1GLastBlockCost = curOriTailS2Num > 0U ? typeCost_[ORI_TAIL_BLOCK][ORI_TAIL_BLOCK] :
                                                                  typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK];
            s1GCache.oriS1GNormalBlockCost = typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK];
        } else {
            s1GCache.oriS1GCost = typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK] * curOriNormalS2Num +
                                  typeCost_[ORI_NORMAL_BLOCK][ORI_TAIL_BLOCK] * curOriTailS2Num;
            s1GCache.oriS1GLastBlockCost = curOriTailS2Num > 0U ? typeCost_[ORI_NORMAL_BLOCK][ORI_TAIL_BLOCK] :
                                                                  typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK];
            s1GCache.oriS1GNormalBlockCost = typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK];
        }
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcCmpS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo)
{
    // 处理cmp部分block信息
    if (s1GCache.cmpS2Start >= s1GCache.cmpS2End) {
        // Cmp范围无效, Cost保持为 0
        s1GCache.cmpS1GBlock = 0;
        s1GCache.cmpS1GCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
    } else {
        // 计算 cmp 方向 Block 数量及 Cost
        s1GCache.cmpS1GBlock = s1GCache.cmpS2End - s1GCache.cmpS2Start;
        // 判断 Cmp S2 方向是否包含尾块
        uint32_t curCmpTailS2Num = (s1GCache.cmpS2TailSize != 0U) ? 1U : 0U; // Updated check using local var
        uint32_t curCmpNormalS2Num = s1GCache.cmpS1GBlock - curCmpTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
            splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.cmpS1GCost = typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                                  typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] :
                                                                  typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
        } else {
            s1GCache.cmpS1GCost = typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                                  typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] :
                                                                  typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
        }
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcOriBlockRange(const Range<int64_t> &oriS2TokenRange,
                                                        const BatchCache &batchCache, S1GCache &s1GCache)
{
    int64_t oriS2FirstToken = oriS2TokenRange.first;
    int64_t oriS2LastToken = oriS2TokenRange.second;
    s1GCache.oriS2Start = 0;
    // ori 部分 s2 起止和 tailSize
    if (oriS2FirstToken >= static_cast<int64_t>(batchCache.oriS2Size) || oriS2LastToken < 0 ||
        oriS2LastToken < oriS2FirstToken) {
        s1GCache.oriS2End = 0;
        s1GCache.oriS2TailSize = 0;
    } else {
        oriS2FirstToken =
            Clip(oriS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.oriS2Size - 1U));
        oriS2LastToken = Clip(oriS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.oriS2Size - 1U));
        // oriS2LastToken 与 topk 取最小
        uint32_t s1Idx = GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx);
        uint32_t bsStride = GetBsStride(s1GCache.bIdx, s1Idx);
        uint32_t oriTopkSize = isSparseOriKv_ ?
                                   GetOriTopkLength(s1GCache.s1GIdx, batchCache) :
                                   GetOriTopkLength(GetBsStride(s1GCache.bIdx, GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx)));
        s1GCache.actOriS2Size = isSparseOriKv_ ?
                                    std::min(static_cast<uint32_t>(oriS2LastToken - oriS2FirstToken + 1), oriTopkSize) :
                                    static_cast<uint32_t>(oriS2LastToken - oriS2FirstToken + 1);
        s1GCache.oriS2End = s1GCache.actOriS2Size == 0 ?
                                0 :
                                (s1GCache.actOriS2Size - 1U) / s2BaseSize_ + 1U;
        s1GCache.oriS2TailSize = s1GCache.actOriS2Size % s2BaseSize_;
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcCmpBlockRange(const Range<int64_t> &cmpRevertS2TokenRange,
                                                        const BatchCache &batchCache, S1GCache &s1GCache)
{
    int64_t cmpRevertS2FirstToken = cmpRevertS2TokenRange.first;
    int64_t cmpRevertS2LastToken = cmpRevertS2TokenRange.second;
    s1GCache.cmpS2Start = s1GCache.oriS2End;
    // cmp 部分 s2 起止和 tailSize
    if (cmpRevertS2FirstToken >= static_cast<int64_t>(batchCache.cmpRevertS2Size) || cmpRevertS2LastToken < 0 ||
        cmpRevertS2LastToken < cmpRevertS2FirstToken) {
        s1GCache.cmpS2End = s1GCache.cmpS2Start;
        s1GCache.cmpS2TailSize = 0;
    } else {
        cmpRevertS2FirstToken =
            Clip(cmpRevertS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.cmpRevertS2Size - 1U));
        cmpRevertS2LastToken =
            Clip(cmpRevertS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.cmpRevertS2Size - 1U));
        // 如果压缩后长度为0，则直接返回
        if ((cmpRevertS2LastToken + 1) / cmpRatio_ == 0) {
            s1GCache.cmpS2End = s1GCache.cmpS2Start;
            s1GCache.cmpS2TailSize = 0;
            return;
        }
        // 获取压缩后的 token 索引
        uint64_t cmpS2FirstToken =
            (cmpRevertS2FirstToken + 1) / cmpRatio_ == 0 ? 0 : (cmpRevertS2FirstToken + 1) / cmpRatio_ - 1U;
        uint64_t cmpS2LastToken = (cmpRevertS2LastToken + 1) / cmpRatio_ - 1U;
        // cmpS2LastToken 与 topk 取最小
        uint32_t s1Idx = GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx);
        uint32_t bsStride = GetBsStride(s1GCache.bIdx, s1Idx);
        uint32_t cmpTopkSize = GetCmpTopkLength(bsStride);
        s1GCache.actCmpS2Size = isSparseCmpKv_ ?
                                    std::min(static_cast<uint32_t>(cmpS2LastToken - cmpS2FirstToken + 1), cmpTopkSize) :
                                    static_cast<uint32_t>(cmpS2LastToken - cmpS2FirstToken + 1);
        s1GCache.cmpS2End = s1GCache.actCmpS2Size == 0 ?
                                s1GCache.cmpS2Start :
                                s1GCache.cmpS2Start + (s1GCache.actCmpS2Size - 1U) / s2BaseSize_ + 1U;
        s1GCache.cmpS2TailSize = s1GCache.actCmpS2Size % s2BaseSize_;
    }
}

void SparseFlashMlaMetadataCpuKernel::GatherOriAndCmpCache(S1GCache &s1GCache)
{
    s1GCache.s2Start = 0;
    if (s1GCache.cmpS1GBlock > 0) {
        s1GCache.s1GLastBlockCost = s1GCache.cmpS1GLastBlockCost;
        s1GCache.s2End = s1GCache.cmpS2End;
    } else {
        s1GCache.s1GLastBlockCost = s1GCache.oriS1GLastBlockCost;
        s1GCache.s2End = s1GCache.oriS2End;
    }
    s1GCache.s1GBlock = s1GCache.oriS1GBlock + s1GCache.cmpS1GBlock;
    s1GCache.s1GCost = s1GCache.oriS1GCost + s1GCache.cmpS1GCost;
}

void SparseFlashMlaMetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext,
                                                   const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 如果s1G是空行，则直接返回
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.oriS1GNormalBlockCost = 0;
        s1GCache.oriS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.s1GBlock = 0;
        s1GCache.s2Loop = 0;
        s1GCache.s2Start = 0;
        s1GCache.cmpS2Start = 0;
        s1GCache.s2End = 0;
        return;
    }
    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;
    s1GCache.actOriS2Size = 0U;
    s1GCache.actCmpS2Size = 0U;
    s1GCache.reductionBlockSize = 0U;
    // 计算 ori_kv 有效负载起止
    if (hasOriKv_) {
        // 计算 ori_kv 的 s2Token 起止
        auto oriS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache, ORI_KV);
        // 计算 ori_kv 的 s2Block 起止
        CalcOriBlockRange(oriS2TokenRange, batchCache, s1GCache);
    } else {
        // ori_kv s2Token 起止初始化为0
        s1GCache.oriS2Start = 0;
        s1GCache.oriS2End = s1GCache.oriS2Start;
        s1GCache.oriS2TailSize = 0;
    }
    // 计算 cmp_kv 有效负载起止
    if (hasCmpKv_) {
        // 计算 cmp_kv 的 s2Token 起止
        auto cmpRevertS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache, CMP_KV);
        // 计算 cmp_kv 的 s2Block 起止
        CalcCmpBlockRange(cmpRevertS2TokenRange, batchCache, s1GCache);
    } else {
        // cmp_kv s2Token 起止初始化为0
        s1GCache.cmpS2Start = s1GCache.oriS2End;
        s1GCache.cmpS2End = s1GCache.cmpS2Start;
        s1GCache.cmpS2TailSize = 0;
    }
    if (isBatchConsistency_) {
        // The reduction block only depends on this row, so its reduction tree is independent of the surrounding batch.
        uint64_t actTotalS2Size = static_cast<uint64_t>(s1GCache.actOriS2Size) + s1GCache.actCmpS2Size;
        if (actTotalS2Size > 0U) {
            uint64_t rawReductionBlockSize = actTotalS2Size / BATCH_CONSISTENCY_MAX_REDUCTION_PARTS;
            s1GCache.reductionBlockSize = static_cast<uint32_t>(
                (rawReductionBlockSize + s2BaseSize_ - 1U) / s2BaseSize_ * s2BaseSize_);
            s1GCache.reductionBlockSize = std::max(s1GCache.reductionBlockSize, s2BaseSize_);
            s1GCache.oriS2End = s1GCache.actOriS2Size == 0U ?
                                    0U :
                                    (s1GCache.actOriS2Size - 1U) / s1GCache.reductionBlockSize + 1U;
            s1GCache.oriS2TailSize = s1GCache.actOriS2Size % s1GCache.reductionBlockSize;
            s1GCache.cmpS2Start = s1GCache.oriS2End;
            s1GCache.cmpS2End = s1GCache.actCmpS2Size == 0U ?
                                    s1GCache.cmpS2Start :
                                    s1GCache.cmpS2Start + (s1GCache.actCmpS2Size - 1U) / s1GCache.reductionBlockSize + 1U;
            s1GCache.cmpS2TailSize = s1GCache.actCmpS2Size % s1GCache.reductionBlockSize;
        }
    }
    // 计算基本块负载
    CalcCostTable(splitInfo.s1GTailSize[s1GCache.bIdx], s1GCache.reductionBlockSize, s1GCache.oriS2TailSize,
                  s1GCache.cmpS2TailSize);
    // 计算 ori 和 cmp 部分的 cost 和 block 信息
    CalcOriS1GCache(s1GCache, splitInfo);
    CalcCmpS1GCache(s1GCache, splitInfo);
    // 汇总 ori 和 cmp 部分的 cost 和 block 信息
    GatherOriAndCmpCache(s1GCache);
    s1GCache.s2Loop = static_cast<uint32_t>(
        (static_cast<uint64_t>(s1GCache.actOriS2Size) + s2BaseSize_ - 1U) / s2BaseSize_ +
        (static_cast<uint64_t>(s1GCache.actCmpS2Size) + s2BaseSize_ - 1U) / s2BaseSize_);
}

void SparseFlashMlaMetadataCpuKernel::CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2S2LoopOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx) == 0U) {
        return;
    }
    if (!hasOriKv_ && !hasCmpKv_) {
        return;
    } else if (!hasOriKv_) {
        if (GetCmpS2SeqSize(bIdx) == 0U) {
            return;
        }
    } else if (!hasCmpKv_) {
        if (GetOriS2SeqSize(bIdx) == 0U) {
            return;
        }
    } else {
        if ((GetOriS2SeqSize(bIdx) == 0U) && GetCmpS2SeqSize(bIdx) == 0U) {
            return;
        }
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        costInfo.bN2S2LoopOfEachBatch[bIdx] += s1GCache.s2Loop;
        // 更新最大S1G行开销
        if (s1GCache.s1GCost > costInfo.maxS1GCost) {
            costInfo.maxS1GCost = s1GCache.s1GCost;
        }
        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcCostInfo(SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算 batch 的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * numHeadsKv_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * numHeadsKv_;
    }
}

void SparseFlashMlaMetadataCpuKernel::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) { // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx == batchSize_ * numHeadsKv_) { // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / numHeadsKv_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / numHeadsKv_;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    // Update Cache
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.bN2S2Loop = costInfo.bN2S2LoopOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = supportFd_ ? assignContext.s1GCache.oriS2Start : 0U;
    }
}

void SparseFlashMlaMetadataCpuKernel::AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.coreCache.s2Loop += assignContext.bN2S2Loop;
        assignContext.curBN2Idx++;
        // to the end
        if (assignContext.curBN2Idx == batchSize_ * numHeadsKv_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / numHeadsKv_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / numHeadsKv_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.bN2S2Loop = costInfo.bN2S2LoopOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void SparseFlashMlaMetadataCpuKernel::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;
        assignContext.coreCache.s2Loop += assignContext.s1GCache.s2Loop;
        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                    assignContext.bN2Cost - assignContext.s1GCache.s1GCost :
                                    0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                     assignContext.bN2Block - assignContext.s1GCache.s1GBlock :
                                     0U;
        assignContext.bN2S2Loop = assignContext.bN2S2Loop > assignContext.s1GCache.s2Loop ?
                                      assignContext.bN2S2Loop - assignContext.s1GCache.s2Loop :
                                      0U;
        // 计算新一行的信息
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcCurBlockCost(const AssignContext &assignContext)
{
    int64_t curCost = 0;
    if (assignContext.curS2Idx < assignContext.s1GCache.cmpS2Start) {
        curCost = assignContext.s1GCache.oriS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.cmpS2Start - 1U)) {
            curCost = assignContext.s1GCache.oriS1GLastBlockCost;
        }
    } else {
        curCost = assignContext.s1GCache.cmpS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
            curCost = assignContext.s1GCache.cmpS1GLastBlockCost;
        }
    }
    return curCost;
}

uint32_t SparseFlashMlaMetadataCpuKernel::CalcCurBlockS2Loop(const AssignContext &assignContext)
{
    if (!isBatchConsistency_) {
        return 1U;
    }
    const S1GCache &s1GCache = assignContext.s1GCache;
    uint32_t blockSize = s1GCache.reductionBlockSize;
    if (assignContext.curS2Idx < s1GCache.cmpS2Start) {
        if (assignContext.curS2Idx + 1U == s1GCache.cmpS2Start && s1GCache.oriS2TailSize != 0) {
            blockSize = static_cast<uint32_t>(s1GCache.oriS2TailSize);
        }
    } else if (assignContext.curS2Idx + 1U == s1GCache.s2End && s1GCache.cmpS2TailSize != 0) {
        blockSize = static_cast<uint32_t>(s1GCache.cmpS2TailSize);
    }
    return (blockSize + s2BaseSize_ - 1U) / s2BaseSize_;
}

void SparseFlashMlaMetadataCpuKernel::AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished || !supportFd_) {
        return;
    }

    int64_t curCost = CalcCurBlockCost(assignContext);
    uint32_t curS2Loop = CalcCurBlockS2Loop(assignContext);

    // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + curCost)) {
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.coreCache.s2Loop += curS2Loop;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
        assignContext.bN2S2Loop =
            assignContext.bN2S2Loop > curS2Loop ? assignContext.bN2S2Loop - curS2Loop : 0U;
        assignContext.s1GCache.s2Loop =
            assignContext.s1GCache.s2Loop > curS2Loop ? assignContext.s1GCache.s2Loop - curS2Loop : 0U;
        curCost = CalcCurBlockCost(assignContext);
        curS2Loop = CalcCurBlockS2Loop(assignContext);
    }
}

void SparseFlashMlaMetadataCpuKernel::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = CalcCurBlockCost(assignContext);
    uint32_t curS2Loop = CalcCurBlockS2Loop(assignContext);

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.coreCache.s2Loop += curS2Loop;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    assignContext.bN2S2Loop =
        assignContext.bN2S2Loop > curS2Loop ? assignContext.bN2S2Loop - curS2Loop : 0U;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    assignContext.s1GCache.s2Loop =
        assignContext.s1GCache.s2Loop > curS2Loop ? assignContext.s1GCache.s2Loop - curS2Loop : 0U;
    UpdateCursor(splitContext, assignContext);
}

bool SparseFlashMlaMetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext &assignContext,
                                                         const SplitResult &splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

bool SparseFlashMlaMetadataCpuKernel::IsFirstReductionBlock(const AssignContext &assignContext,
                                                            const SplitResult &splitRes)
{
    if (assignContext.curCoreIdx == 0U || splitRes.s2End[assignContext.curCoreIdx - 1U] == 0U) {
        return true;
    }
    return assignContext.curBN2Idx != splitRes.bN2End[assignContext.curCoreIdx - 1U] ||
           assignContext.curS1GIdx != splitRes.gS1End[assignContext.curCoreIdx - 1U];
}

void SparseFlashMlaMetadataCpuKernel::RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext,
                                                   SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / numHeadsKv_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize =
        (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
            (static_cast<uint64_t>(s1Size) * groupSize_ - static_cast<uint64_t>(splitS1GIdx) * mBaseSize_) :
            mBaseSize_;
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.fdBN2Idx[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdMIdx[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.preFdDataNum;
    result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize;
    result.numOfFdHead++;
}

void SparseFlashMlaMetadataCpuKernel::AssignBlocksToCore(const SplitContext &splitContext, AssignContext &assignContext,
                                                         SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
        assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
    int64_t avgCost = assignContext.unassignedCost / (aicCoreNum_ - assignContext.curCoreIdx);
    assignContext.coreCache = {};
    if (!supportFd_) {
        assignContext.coreCache.costLimit = std::max(avgCost, costInfo.maxS1GCost);
    } else {
        assignContext.coreCache.costLimit = avgCost;
    }
    // 1、按整batch分配
    AssignByBatch(splitContext, assignContext);
    // 2、按行分配
    AssignByRow(splitContext, assignContext);
    // 3、按块分配
    AssignByBlock(splitContext, assignContext);
    // 4、强制分配
    if (assignContext.coreCache.block == 0 && supportFd_) {
        ForceAssign(splitContext, assignContext);
    }
    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
    result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
    assignContext.unassignedCost -= assignContext.coreCache.cost;
    result.maxS2LoopNum = std::max(assignContext.coreCache.s2Loop, result.maxS2LoopNum);
    // 对之前的归约信息进行记录并清理
    if (IsNeedRecordFDInfo(assignContext, result)) {
        if (isBatchConsistency_ && remainedBlockNum_ > 0U) {
            // curKvSplitPart already reserves one slot for the core that finishes this row.
            assignContext.curKvSplitPart += remainedBlockNum_ - 1U;
        }
        RecordFDInfo(splitContext, assignContext, result);
        assignContext.preFdDataNum += assignContext.curKvSplitPart;
        assignContext.curKvSplitPart = 1U;
        remainedBlockNum_ = 0U;
    }
    // 更新S2切分信息
    if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
        assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
        if (isBatchConsistency_) {
            if (IsFirstReductionBlock(assignContext, result)) {
                assignContext.curKvSplitPart++;
            } else {
                assignContext.curKvSplitPart +=
                    result.s2End[assignContext.curCoreIdx] - result.s2End[assignContext.curCoreIdx - 1U];
            }
            remainedBlockNum_ = assignContext.s1GCache.s1GBlock;
        } else {
            assignContext.curKvSplitPart++;
        }
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext,
                                                    SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    if (aicCoreNum_ == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    assignContext.bN2S2Loop = costInfo.bN2S2LoopOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    // 负载分配
    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlocksToCore(splitContext, assignContext, result);
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void SparseFlashMlaMetadataCpuKernel::SplitFD(SplitResult &splitRes)
{
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    // 计算当前最大冗余vec核数
    uint32_t emptyVectorNum = aivCoreNum_ - splitRes.numOfFdHead;
    // 计算每个核处理的load
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_; // 向上取整，避免核负载为0
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        // 冗余vec核数为0，此时规约任务无法进行更小的切分，只能1个vec核计算1个规约任务
        if (emptyVectorNum == 0U) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = 0U;
            splitRes.fdRes.fdMNum[curCoreIndex] = splitRes.fdRes.fdMSize[i];
            curCoreIndex++;
            continue;
        }
        // 计算当前归约任务所用核数，向下取整，避免使用核数超出总核数
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        curFDVectorNum = std::max(1U, curFDVectorNum);
        // 计算当前归约任务每个核的行数，向上取整，避免行数为0
        uint32_t curAveMSize = (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
        curFDVectorNum = (splitRes.fdRes.fdMSize[i] + curAveMSize - 1U) / curAveMSize;
        // 需要使用的vec核数与当前剩余可用vec核数取最小
        curFDVectorNum = std::min(curFDVectorNum, emptyVectorNum + 1U); // 1: Fd任务自身带一个核
        // FD负载分配
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
        // 更新冗余vec核数
        emptyVectorNum -= (curFDVectorNum - 1U); // 1: 空余核不包含FD自身的核，要-1
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

bool SparseFlashMlaMetadataCpuKernel::BalanceSchedule(SplitResult &splitRes)
{
    SplitContext splitContext(batchSize_);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = batchSize_ * numHeadsKv_;
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;

    CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
    // 3、存在FD任务，对FD进行负载均衡分配
    if (splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U); // 至少使用1个core
    return true;
}

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

}

using namespace aicpu;
struct Result {std::vector<int32_t> out;uint32_t used,fdHeads,fdVec;bool split,tailKept;};
Result run(const char* name,int q,bool batch,bool cmp,int ratio,bool empty=false) {
 SparseFlashMlaMetadataCpuKernel k;
 Tensor qs{{0,q},{{2}}},ori{{empty?0:32768},{{1}}},cm{{empty?0:32768/ratio},{{1}}},out;
 out.data.assign(1024,0x5a5a5a5a);out.shape.dims={1024};
 k.cuSeqlensQ_=&qs;k.sequsedOriKv_=&ori;k.sequsedCmpKv_=cmp?&cm:nullptr;k.metadata_=&out;
 k.numHeadsQ_=64;k.numHeadsKv_=1;k.headDim_=512;k.batchSize_=1;k.maxSeqlenQ_=q;
 k.maxSeqlenOriKv_=32768;k.maxSeqlenCmpKv_=32768/ratio;k.cmpRatio_=ratio;
 k.cmpTopK_=cmp?512:0;k.oriTopK_=0;k.oriMaskMode_=4;k.cmpMaskMode_=cmp?3:0;
 k.layoutQ_="TND";k.layoutKv_="PA_BBND";k.hasOriKv_=true;k.hasCmpKv_=cmp;
 k.aicCoreNum_=24;k.aivCoreNum_=48;k.socVersion_="Ascend910_9382";k.isBatchConsistency_=batch;
 if(!k.ParamsInit())std::abort();SplitResult r(24,48);
 if(!k.BalanceSchedule(r)||!k.GenMetadata(r))std::abort();
 bool split=false;for(uint32_t i=0;i<r.usedCoreNum;++i)split|=r.s2End[i]!=0;
 bool tail=true;for(int i=900;i<1024;i++)tail&=out.data[i]==0x5a5a5a5a;
 std::cout<<"{\"case\":\""<<name<<"\",\"q\":"<<q<<",\"batch_consistency\":"<<batch<<",\"ratio\":"<<ratio<<",\"used_cores\":"<<r.usedCoreNum<<",\"fd_heads\":"<<r.numOfFdHead<<",\"fd_vec\":"<<r.fdRes.fdUsedVecNum<<",\"S2_boundary_nonzero\":"<<split<<",\"tail124_sentinel_untouched\":"<<tail<<",\"endpoints\":[";
 for(uint32_t i=0;i<r.usedCoreNum;i++){if(i)std::cout<<",";std::cout<<"["<<r.bN2End[i]<<","<<r.gS1End[i]<<","<<r.s2End[i]<<"]";}
 std::cout<<"]}"<<std::endl;return {out.data,r.usedCoreNum,r.numOfFdHead,r.fdRes.fdUsedVecNum,split,tail};
}
int main(){
 auto c2=run("C2_q3_default",3,false,true,2);auto c1=run("C1_q3_default",3,false,true,1);
 auto fd=run("C2_q3_level3",3,true,true,2);auto large=run("C2_q24_default",24,false,true,2);
 auto largefd=run("C2_q24_level3",24,true,true,2);auto zero=run("emptyKV_q3_default",3,false,true,2,true);
 auto swa=run("denseSWA_q3_default",3,false,false,1);
 bool defined_equal=std::equal(c2.out.begin(),c2.out.begin()+900,c1.out.begin());
 std::cout<<"{\"checks\":true,\"C1_C2_defined900_equal\":"<<defined_equal<<",\"default_noFD\":"<<(c2.fdHeads==0&&!c2.split)<<",\"level3_crossS2_with_FD\":"<<(fd.fdHeads>0&&fd.split)<<"}"<<std::endl;
 return !(defined_equal&&c2.fdHeads==0&&!c2.split&&fd.fdHeads>0&&fd.split&&c2.tailKept&&fd.tailKept&&zero.used==1);
}
