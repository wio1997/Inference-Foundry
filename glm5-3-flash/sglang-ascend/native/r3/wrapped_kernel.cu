#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR comb_handle,  GM_ADDR out_handle, int64_t n, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> comb;
  comb.SetGlobalBuffer((__gm__ float*)comb_handle);
  AscendC::GlobalTensor<float> out;
  out.SetGlobalBuffer((__gm__ float*)out_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto value = ascend_ub.GetWithOffset<float>(32, 0);
  auto col_sum = ascend_ub.GetWithOffset<float>(8, 128);
  auto column_row = ascend_ub.GetWithOffset<float>(8, 160);
  auto col_broadcast = ascend_ub.GetWithOffset<float>(32, 480);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(288, 192);
  auto row_sum = ascend_ub.GetWithOffset<float>(8, 608);
  auto row_div = ascend_ub.GetWithOffset<float>(32, 640);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    AscendC::PipeBarrier<PIPE_ALL>();
    if (((cid * 2) + vid) < n) {
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 32);
      AscendC::Duplicate<float, false>(value[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      for (int32_t i = 0; i < 4; ++i) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_ub<float, 4, 1>(value[(i * 8)], comb[(((cid * 32) + (vid * 16)) + (i * 4))], 4, 1, 4, 0.000000e+00f);
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
      tl::ascend::copy_ub_to_ub<float, float, 8>(col_sum[0], value[0], 1, 8, 8, 1, 8, 8);
      for (int32_t column_i = 1; column_i < 4; ++column_i) {
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, float, 8>(column_row[0], value[(column_i * 8)], 1, 8, 8, 1, 8, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
        AscendC::Add<float, false>(col_sum[0], col_sum[0], column_row[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
      {
      AscendC::Adds<float, false>(col_sum[0], col_sum[0], 1.000000e-06f, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      tl::ascend::Broadcast<float, 2, 0, false>(col_broadcast[0],col_sum[0],tmp_ub[0],(uint32_t[]){4, 8}, (uint32_t[]){1, 8});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)4294967295);
      AscendC::Div<float, false>(value[0], value[0], col_broadcast[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
      for (int32_t __1 = 0; __1 < 19; ++__1) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 4, 4, -1, 8, true>(row_sum[0], value[0], tmp_ub.ReinterpretCast<float>()[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
        {
        AscendC::Adds<float, false>(row_sum[0], row_sum[0], 1.000000e-06f, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
        }
        for (int32_t i_1 = 0; i_1 < 4; ++i_1) {
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetMaskCount();
          AscendC::SetVectorMask<uint8_t>((uint64_t)0, 8);
          AscendC::Duplicate<float, false>(row_div[(i_1 * 8)], row_sum.GetValue(i_1), AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)4294967295);
        AscendC::Div<float, false>(value[0], value[0], row_div[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, float, 8>(col_sum[0], value[0], 1, 8, 8, 1, 8, 8);
        for (int32_t column_i_1 = 1; column_i_1 < 4; ++column_i_1) {
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<float, float, 8>(column_row[0], value[(column_i_1 * 8)], 1, 8, 8, 1, 8, 8);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetMaskNorm();
          AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
          AscendC::Add<float, false>(col_sum[0], col_sum[0], column_row[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
        {
        AscendC::Adds<float, false>(col_sum[0], col_sum[0], 1.000000e-06f, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        tl::ascend::Broadcast<float, 2, 0, false>(col_broadcast[0],col_sum[0],tmp_ub[0],(uint32_t[]){4, 8}, (uint32_t[]){1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)4294967295);
        AscendC::Div<float, false>(value[0], value[0], col_broadcast[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
      }
      for (int32_t i_2 = 0; i_2 < 4; ++i_2) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<float, 4, 1>(out[(((cid * 32) + (vid * 16)) + (i_2 * 4))], value[(i_2 * 8)], 4, 1, 4);
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}

void main_kernel_tiling(int64_t n) {
}

extern "C" void call(uint8_t* comb_handle, uint8_t* out_handle, int64_t n, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling(n);
  main_kernel<<<((n + 1) / 2), nullptr, stream>>>(comb_handle, out_handle, n, fftsAddr);
}
