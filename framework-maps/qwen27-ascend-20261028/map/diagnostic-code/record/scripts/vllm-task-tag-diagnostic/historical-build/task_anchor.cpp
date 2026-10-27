// CPU-only recompile of reviewed task-anchor queries for vLLM torch_npu 2.10.0.post7 / CANN9.2; no operator/event insertion.
#include <acl/acl_rt.h>
#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>
#include <utility>
#include <algorithm>

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
  aclmdlRI model_ri = nullptr;  // opaque borrowed handle; query before owning graph reset
  int callback_rc = -1;
  std::string annotation_json;
  int extend_info_rc = -2;       // -2 means omitted or API not compiled in
  uint32_t task_id_after_annotation = 0;
  int task_after_rc = -2;
  bool annotation_task_id_unchanged = false;
};

static std::string thread_id_string(std::thread::id id) {
  std::ostringstream out;
  out << id;
  return out.str();
}

std::shared_ptr<AnchorResult> q27_anchor(const std::string& label,
                                         const std::string& annotation_json = "") {
  TORCH_CHECK(label == "after_add" || label == "after_mul", "unexpected anchor label");
  TORCH_CHECK(annotation_json.size() <= 4096, "annotation JSON exceeds 4KiB");
  TORCH_CHECK(annotation_json.find('\0') == std::string::npos,
              "annotation JSON contains NUL");
  TORCH_CHECK(annotation_json.empty() ||
              (annotation_json.front() == '{' && annotation_json.back() == '}'),
              "annotation must be a compact JSON object");
  TORCH_CHECK(std::all_of(annotation_json.begin(), annotation_json.end(),
              [](unsigned char c) { return c < 0x80; }),
              "annotation must be ASCII-safe UTF-8 JSON");
  // Public torch_npu C++ accessors; capture the producer's actual current stream
  // before enqueue. Do not call getCurrentNPUStream() from the consumer callback:
  // current_streams is thread_local in the pinned implementation.
  const auto npu_stream = c10_npu::getCurrentNPUStream();
  const aclrtStream acl_stream = npu_stream.stream(false);
  TORCH_CHECK(acl_stream != nullptr, "active NPU stream handle is null");

  auto result = std::make_shared<AnchorResult>();
  result->producer_thread = thread_id_string(std::this_thread::get_id());
  result->annotation_json = annotation_json;
  at_npu::native::PROC_FUNC callback = [result, acl_stream]() -> int {
    result->callback_thread = thread_id_string(std::this_thread::get_id());

    // This one value is the only task anchor emitted. Query it first, before
    // other APIs; it is not a range, per-op inventory, or completion receipt.
    result->task_rc = static_cast<int>(aclrtGetThreadLastTaskId(&result->task_id));
    if (result->task_rc != ACL_ERROR_NONE) { result->callback_rc = result->task_rc; return result->callback_rc; }

    aclmdlRICaptureStatus status{};
    aclmdlRI model_ri = nullptr;
    result->capture_rc = static_cast<int>(aclmdlRICaptureGetInfo(acl_stream, &status, &model_ri));
    if (result->capture_rc != ACL_ERROR_NONE) { result->callback_rc = result->capture_rc; return result->callback_rc; }
    result->capture_status = static_cast<int>(status);
    result->model_ri = model_ri;
    result->model_ri_nonnull = (model_ri != nullptr);

    // Optional exact-task tag, only for a proven active graph capture. Compile
    // this branch only after verifying the installed CANN 9.2 header exports
    // this API. JSON was prepared before capture; this callback does no JSON
    // encoding, file I/O, synchronization, or tensor inspection.
    if (result->capture_status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE &&
        result->model_ri_nonnull && !result->annotation_json.empty()) {
#if defined(Q27_HAS_ACLRT_CACHE_LAST_TASK_EXTEND_INFO)
      result->extend_info_rc = static_cast<int>(aclrtCacheLastTaskExtendInfo(
          result->annotation_json.data(), result->annotation_json.size()));
      result->task_after_rc = static_cast<int>(aclrtGetThreadLastTaskId(
          &result->task_id_after_annotation));
      result->annotation_task_id_unchanged =
          result->task_after_rc == ACL_ERROR_NONE &&
          result->task_id_after_annotation == result->task_id;
#else
      result->extend_info_rc = -3; // compile-time API unavailable/unverified
#endif
    } else if (!result->annotation_json.empty()) {
      result->extend_info_rc = -4; // not an active capture or no model RI
    }

    result->stream_rc = static_cast<int>(aclrtStreamGetId(acl_stream, &result->acl_stream_id));
    if (result->stream_rc != ACL_ERROR_NONE) { result->callback_rc = result->stream_rc; return result->callback_rc; }

    result->callback_rc = ACL_ERROR_NONE;
    return result->callback_rc;
  };

  at_npu::native::OpCommand::RunOpApiV2(std::string("q27_anchor_") + label, callback, false);
  return result; // Python caller must synchronize this same NPU stream before reading fields.
}


// Called only after the graph capture is closed and the exact result callback
// has drained, while the same owning graph still retains the model RI. This
// never runs from the capture callback itself.
pybind11::dict q27_query_model_ri(const std::shared_ptr<AnchorResult>& result) {
  namespace py = pybind11;
  py::dict out;
  out["query_phase"] = "post_capture_before_graph_reset";
  out["capture_status"] = result->capture_status;
  out["capture_rc"] = result->capture_rc;
  out["callback_rc"] = result->callback_rc;
  out["extend_info_rc"] = result->extend_info_rc;
  out["task_id_before_annotation"] = result->task_id;
  out["task_after_rc"] = result->task_after_rc;
  out["task_id_after_annotation"] = result->task_id_after_annotation;
  out["annotation_task_id_unchanged"] = result->annotation_task_id_unchanged;
  out["annotation_json"] = result->annotation_json;
  out["physical_stream_id"] = result->acl_stream_id;
  out["model_ri_nonnull"] = result->model_ri != nullptr;
  if (result->model_ri != nullptr) {
    out["model_ri_handle_token"] = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(result->model_ri));
  } else {
    out["model_ri_handle_token"] = py::none();
  }
  if (result->model_ri == nullptr || result->capture_status != ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE ||
      result->capture_rc != ACL_ERROR_NONE || result->callback_rc != ACL_ERROR_NONE) {
    out["status"] = "unknown_query_precondition_failed";
    out["model_ri_get_id_rc"] = -1;
    out["get_streams_rc"] = -1;
    out["model_ri_id"] = py::none();
    out["model_streams"] = py::list();
    return out;
  }

  uint32_t model_ri_id = 0;
  const aclError id_rc = aclmdlRIGetId(result->model_ri, &model_ri_id);
  out["model_ri_get_id_rc"] = static_cast<int>(id_rc);
  if (id_rc == ACL_ERROR_NONE) out["model_ri_id"] = model_ri_id;
  else out["model_ri_id"] = py::none();

  // Header contract: numStreams is input capacity and output count. Bound the
  // query and preserve overflow/error as unknown; never guess a missing suffix.
  constexpr uint32_t kMaxStreams = 1024;
  std::vector<aclrtStream> streams(kMaxStreams, nullptr);
  uint32_t stream_count = kMaxStreams;
  const aclError streams_rc = aclmdlRIGetStreams(result->model_ri, streams.data(), &stream_count);
  out["get_streams_rc"] = static_cast<int>(streams_rc);
  out["stream_capacity"] = kMaxStreams;
  out["reported_stream_count"] = stream_count;
  py::list model_streams;
  bool stream_id_failure = false;
  if (streams_rc == ACL_ERROR_NONE && stream_count <= kMaxStreams) {
    for (uint32_t i = 0; i < stream_count; ++i) {
      py::dict row;
      row["stream_index"] = i;  // diagnostic array position only, not an identity join
      row["stream_handle_nonnull"] = streams[i] != nullptr;
      int32_t stream_id = -1;
      const int stream_rc = streams[i] == nullptr
          ? -1 : static_cast<int>(aclrtStreamGetId(streams[i], &stream_id));
      row["stream_get_id_rc"] = stream_rc;
      if (stream_rc == ACL_ERROR_NONE) row["model_stream_id"] = stream_id;
      else { row["model_stream_id"] = py::none(); stream_id_failure = true; }
      model_streams.append(std::move(row));
    }
  }
  out["model_streams"] = std::move(model_streams);
  if (streams_rc != ACL_ERROR_NONE) out["status"] = "unknown_get_streams_failed";
  else if (stream_count > kMaxStreams) out["status"] = "unknown_stream_capacity_exceeded";
  else if (id_rc != ACL_ERROR_NONE) out["status"] = "partial_model_ri_id_failed";
  else if (stream_id_failure) out["status"] = "partial_stream_id_query_failed";
  else if (stream_count == 0) out["status"] = "unknown_empty_model_stream_list";
  else out["status"] = "queried_ids_unvalidated_against_same_profile";
  return out;
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
  m.def("anchor", &q27_anchor, pybind11::arg("label"),
        pybind11::arg("annotation_json") = "",
        "Queue exact-thread anchor and optionally attach JSON to current last task");
  m.def("query_model_ri", &q27_query_model_ri, "Query the retained model RI after capture closes, before graph reset");
}
