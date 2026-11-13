// SPDX-License-Identifier: Apache-2.0
// Standard CANN9.2 five-op-input ABI with exact standard-library resolver.
// CANN common helper keeps its original BSD3 notice; current Torch2.13 headers.
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <climits>
#include <cstdlib>
#include <string>
#include <tuple>
#include <algorithm>
#include <limits>
#include <unistd.h>
#include "include/aclnn_inplace_partial_rotary_mul_core5.h"
#include "include/op_api_common_213.h"
#include <type_traits>
using Core5Workspace = aclnnStatus (*)(const aclTensor *,const aclTensor *,const aclTensor *,int64_t,const aclIntArray *,uint64_t *,aclOpExecutor **);
static_assert(std::is_same<decltype(&aclnnInplacePartialRotaryMulGetWorkspaceSize),Core5Workspace>::value,"Unexpected core5 header ABI");
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset=0;

void InplaceRoPE(at::Tensor &x,const at::Tensor &cos,const at::Tensor &sin,
                c10::string_view rotary_mode,at::IntArrayRef partial_slice)
{
    TORCH_CHECK(x.dim()==4,"RoPE x must have4 dimensions");
    TORCH_CHECK(rotary_mode=="interleave","This V41 adapter accepts interleave only");
    TORCH_CHECK(partial_slice.size()==2,"Expected two partial-slice bounds");
    const int64_t mode=1;
    // No negate_sin exists in this exact standard header. Native callers
    // already pass negative sine for inverse RoPE; no arithmetic is changed.
    EXEC_NPU_CMD(aclnnInplacePartialRotaryMul,x,cos,sin,mode,partial_slice);
}
TORCH_LIBRARY(sglang_v41_rope_core5_g3,m) {
 m.def("inplace_partial_rotary_mul(Tensor(a!) x, Tensor cos, Tensor sin, str rotary_mode, int[] partial_slice) -> ()");
 m.impl("inplace_partial_rotary_mul",torch::kPrivateUse1,&InplaceRoPE);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {}
extern "C" __attribute__((visibility("default")))
void *sglang_v41_rope_core5_resolve_symbol(const char *name) { return RopeCore5GetOpApiFuncAddr(name); }
