#pragma once

#if not defined(__CCE__) or defined(__CLION_IDE__)
#define __HOST_ONLY__
#endif

#ifdef __CLION_IDE__
#define __CCE_STUB_DISABLE_ADDRESS_SPACE_QUALIFIERS__
#define __CCE_STUB_USE_LOCAL_WRAPPER__
#define __CCE_AICORE_SUPPORT_SIMT__
#define __clang__
#include <cce_stubs/cce_stubs.h>
#endif

#if not defined(__CCE__)
#define __aicore__
#define __gm__
#define __ubuf__
#define __global__
#define __vector__
#define __mix__(a, b)
#define __simt_vf__
#define __simt_callee__
#define __simd_vf__
#define __simd_callee__
#define __forceinline__ inline
#endif

#include <cstdint>

namespace deep_ep {

using sf_pack_t = int16_t;

static constexpr int kNumMaxRanks = 256;
static constexpr int kNumMaxExperts = 2048;
static constexpr int kNumMaxVecCores = 64;
static constexpr int kNumUbAlignmentBytes = 32;
static constexpr int kNumVectorRegisterBytes = 256;
static constexpr int kNumSectorBytes = 512;
static constexpr int kNumSectorInts = kNumSectorBytes / sizeof(int);
static constexpr int kNumMaxContexts = 8;
static constexpr int kNumRDMAAlignmentBytes = 32;
static constexpr int kNumTMAAlignmentBytes = 32;
// After dispatch / combine, ensure the sqebb_idx is aligned to 4
// to prevent a WQE to be warp-split across the queue's end and begin
static constexpr int kSQEBBAlignment = 4;
static constexpr int64_t kNumAllocationAlignmentBytes = 2097152;

}  // namespace deep_ep
