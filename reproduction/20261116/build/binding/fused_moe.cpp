// Standalone SGLang binding for the pinned CANN DispatchFFNCombine INT8 ABI.
// Scheduling, routing, checkpoint loading and shared experts remain with SGLang.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include <cstdlib>
#include <cstring>
#include <string>

namespace sglang_dsv41_fused {

// Resolve the two versioned fused entry points from the explicit provider.
// Other ACL helpers continue to use the installed TorchNPU implementation.
void* GetOpApiFuncAddr(const char* name) {
  if (std::strcmp(name, "aclnnDispatchFFNCombine") != 0 &&
      std::strcmp(name, "aclnnDispatchFFNCombineGetWorkspaceSize") != 0) {
    return ::GetOpApiFuncAddr(name);
  }
  static void* handle = []() {
    const char* path = std::getenv("SGLANG_DSV41_FUSED_MOE_OPAPI");
    TORCH_CHECK(path && path[0] == '/',
                "Set SGLANG_DSV41_FUSED_MOE_OPAPI to the pinned absolute library path");
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(h, "Cannot load fused provider: ", dlerror());
    return h;
  }();
  void* fn = dlsym(handle, name);
  TORCH_CHECK(fn, "Pinned fused provider is missing ", name);
  return fn;
}

std::string provider_path() {
  Dl_info info{};
  TORCH_CHECK(dladdr(GetOpApiFuncAddr("aclnnDispatchFFNCombineGetWorkspaceSize"),
                    &info) != 0 && info.dli_fname,
              "Cannot identify the fused provider");
  return std::string(info.dli_fname);
}

std::tuple<at::Tensor, at::Tensor> dispatch(
    const at::Tensor& x, at::TensorList w1, at::TensorList w2,
    const at::Tensor& expert_idx, at::TensorList scale1,
    at::TensorList scale2, const at::Tensor& probs, const std::string& group,
    int64_t max_output_size, double swiglu_limit, int64_t world_size,
    const c10::optional<at::Tensor>& active_mask,
    at::Tensor out, at::Tensor expert_counts) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
              x.scalar_type() == at::kBFloat16 && x.dim() == 2,
              "Expected a two-dimensional NPU BF16 source tensor");
  TORCH_CHECK(w1.size() == 1 && w2.size() == 1 &&
              scale1.size() == 1 && scale2.size() == 1,
              "This candidate uses grouped expert tensors, not per-expert lists");
  TORCH_CHECK(w1[0].scalar_type() == at::kChar &&
              w2[0].scalar_type() == at::kChar &&
              scale1[0].scalar_type() == at::kLong &&
              scale2[0].scalar_type() == at::kLong,
              "Expected INT8 weights and FP32-bit-pattern INT64 scales");
  TORCH_CHECK(expert_idx.scalar_type() == at::kInt &&
              probs.scalar_type() == at::kFloat &&
              expert_idx.sizes() == probs.sizes() &&
              expert_idx.size(0) == x.size(0), "Invalid trained top-k payload");
  TORCH_CHECK(swiglu_limit == 10.0 && world_size > 1 &&
              max_output_size > 0 && !group.empty(),
              "Expected explicit 1017 clamp10 and distributed capacity/group");
  TORCH_CHECK(out.sizes() == x.sizes() && out.scalar_type() == x.scalar_type() &&
              expert_counts.scalar_type() == at::kInt &&
              expert_counts.numel() == w1[0].size(0), "Invalid output buffers");
  if (active_mask.has_value()) {
    TORCH_CHECK(active_mask->scalar_type() == at::kBool &&
                active_mask->dim() == 1 && active_mask->numel() == x.size(0),
                "active_mask must have one BOOL per source row");
  }
  const c10_npu::NPUGuard guard(x.device());
  char* group_ptr = const_cast<char*>(group.c_str());
  at::Tensor mask = active_mask.has_value() ? *active_mask : at::Tensor();
  EXEC_NPU_CMD(aclnnDispatchFFNCombine, x, w1, w2, expert_idx, scale1,
               scale2, probs, mask, group_ptr, max_output_size, swiglu_limit,
               world_size, out, expert_counts);
  return {out, expert_counts};
}
} // namespace sglang_dsv41_fused

TORCH_LIBRARY(sglang_dsv41_fused, m) {
  m.def("provider_path() -> str", &sglang_dsv41_fused::provider_path);
  m.def("dispatch(Tensor x, Tensor[] w1, Tensor[] w2, Tensor expert_idx, "
        "Tensor[] scale1, Tensor[] scale2, Tensor probs, str group, "
        "int max_output_size, float swiglu_limit, int world_size, "
        "Tensor? active_mask, Tensor(a!) out, Tensor(b!) expert_counts) "
        "-> (Tensor(a!), Tensor(b!))");
}
TORCH_LIBRARY_IMPL(sglang_dsv41_fused, PrivateUse1, m) {
  m.impl("dispatch", &sglang_dsv41_fused::dispatch);
}
