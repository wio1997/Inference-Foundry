#include <pybind11/pybind11.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <acl/acl_rt.h>
#include <string>
#include <utility>

void enqueue_tag(std::string payload) {
    // Capture by value: torch_npu's queue may execute after Python returns.
    at_npu::native::OpCommand::RunOpApi(
        "BoundNativeTag", [payload = std::move(payload)]() -> int {
            return aclrtCacheLastTaskExtendInfo(payload.data(), payload.size());
        }, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("enqueue_tag", &enqueue_tag);
}
