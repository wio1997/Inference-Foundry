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
#include "include/op_api_common_213.h"
#include "include/aclnn_moe_gating_top_k_hash_current.h"
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset=0;
std::tuple<at::Tensor,at::Tensor,at::Tensor> current_norm2(
    const at::Tensor& x,int64_t k,int64_t kg,int64_t gc,int64_t gsm,int64_t rn,
    int64_t nt,bool of,double scale,double eps,const c10::optional<at::Tensor>& bias_opt) {
    TORCH_CHECK(nt==2 && kg==1 && gc==1 && gsm==0 && rn==0,"Unexpected task router mode");
    TORCH_CHECK(x.dim()==2 && x.scalar_type()==at::kFloat,"Expected native FP32 router input");
    TORCH_CHECK((x.size(1)==384 && k==6)||(x.size(1)==128 && k==3),"Unexpected target/draft expert geometry");
    const at::Tensor& bias=c10::value_or_else(bias_opt,[] {return at::Tensor();});
    at::Tensor none;
    at::Tensor y=at::empty({x.size(0),k},x.options());
    at::Tensor ids=at::empty({x.size(0),k},x.options().dtype(at::kInt));
    at::Tensor scores=at::empty({x.size(0),x.size(1)},x.options().dtype(at::kFloat));
    // CURRENT header: six tensor inputs, no image sentinel scalar attributes.
    // All four optional routing/visual inputs are absent: ordinary biased
    // sqrtsoftplus selection, not token-hash or vision routing.
    EXEC_NPU_CMD(aclnnMoeGatingTopKHash,x,bias,none,none,none,none,
        k,kg,gc,gsm,rn,nt,of,scale,eps,y,ids,scores);
    return {y,ids,scores};
}
TORCH_LIBRARY(sglang_v41_router_g2,m) {
    m.def("moe_gating_top_k(Tensor x, int k, int k_group, int group_count, int group_select_mode, int renorm, int norm_type, bool out_flag, float routed_scaling_factor, float eps, Tensor? bias_opt=None) -> (Tensor,Tensor,Tensor)");
    m.impl("moe_gating_top_k",torch::kPrivateUse1,&current_norm2);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {}
extern "C" __attribute__((visibility("default")))
void *sglang_v41_router_resolve_symbol(const char *name) { return RouterGetOpApiFuncAddr(name); }
