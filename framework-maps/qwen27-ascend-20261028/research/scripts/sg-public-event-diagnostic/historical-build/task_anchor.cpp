// PREPARE ONLY. Never compiled by import or the fixture.
// Build against the exact torch_npu 2.10 / CANN 9.0 installation only after root review.
#include <acl/acl_rt.h>
#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <sstream>
#include <string>
#include <thread>

struct AnchorResult {
  std::string producer_thread;
  std::string callback_thread;
  uint32_t task_id = 0;
  int32_t acl_stream_id = -1;
  int task_rc = -1;
  int stream_rc = -1;
  int capture_rc = -1;
  int capture_status = -1;
  bool model_ri_nonnull = false;
  int callback_rc = -1;
};

static std::string thread_id_string(std::thread::id id) {
  std::ostringstream out;
  out << id;
  return out.str();
}

std::shared_ptr<AnchorResult> q27_anchor(const std::string& label) {
  TORCH_CHECK(label == "after_add" || label == "after_mul", "unexpected anchor label");
  // Public torch_npu C++ accessors; capture the producer's actual current stream
  // before enqueue. Do not call getCurrentNPUStream() from the consumer callback:
  // current_streams is thread_local in the pinned implementation.
  const auto npu_stream = c10_npu::getCurrentNPUStream();
  const aclrtStream acl_stream = npu_stream.stream(false);
  TORCH_CHECK(acl_stream != nullptr, "active NPU stream handle is null");

  auto result = std::make_shared<AnchorResult>();
  result->producer_thread = thread_id_string(std::this_thread::get_id());
  at_npu::native::PROC_FUNC callback = [result, acl_stream]() -> int {
    result->callback_thread = thread_id_string(std::this_thread::get_id());

    // This one value is the only task anchor emitted. Query it first, before
    // other APIs; it is not a range, per-op inventory, or completion receipt.
    result->task_rc = static_cast<int>(aclrtGetThreadLastTaskId(&result->task_id));
    if (result->task_rc != ACL_ERROR_NONE) { result->callback_rc = result->task_rc; return result->callback_rc; }

    result->stream_rc = static_cast<int>(aclrtStreamGetId(acl_stream, &result->acl_stream_id));
    if (result->stream_rc != ACL_ERROR_NONE) { result->callback_rc = result->stream_rc; return result->callback_rc; }

    aclmdlRICaptureStatus status{};
    aclmdlRI model_ri = nullptr;
    result->capture_rc = static_cast<int>(aclmdlRICaptureGetInfo(acl_stream, &status, &model_ri));
    if (result->capture_rc != ACL_ERROR_NONE) { result->callback_rc = result->capture_rc; return result->callback_rc; }
    result->capture_status = static_cast<int>(status);
    result->model_ri_nonnull = (model_ri != nullptr);
    result->callback_rc = ACL_ERROR_NONE;
    return result->callback_rc;
  };

  at_npu::native::OpCommand::RunOpApiV2(std::string("q27_anchor_") + label, callback, false);
  return result; // Python caller must synchronize this same NPU stream before reading fields.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<AnchorResult, std::shared_ptr<AnchorResult>>(m, "AnchorResult")
      .def_readonly("producer_thread", &AnchorResult::producer_thread)
      .def_readonly("callback_thread", &AnchorResult::callback_thread)
      .def_readonly("task_id", &AnchorResult::task_id)
      .def_readonly("acl_stream_id", &AnchorResult::acl_stream_id)
      .def_readonly("task_rc", &AnchorResult::task_rc)
      .def_readonly("stream_rc", &AnchorResult::stream_rc)
      .def_readonly("capture_rc", &AnchorResult::capture_rc)
      .def_readonly("capture_status", &AnchorResult::capture_status)
      .def_readonly("model_ri_nonnull", &AnchorResult::model_ri_nonnull)
      .def_readonly("callback_rc", &AnchorResult::callback_rc);
  m.def("anchor", &q27_anchor, "Queue one exact-thread task-ID anchor after prior work on current stream");
}
