// Isolated synthetic probe; never inject into the inference service.
// It does not verify dump files; see expert_dump_graph_probe.md.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <initializer_list>
#include <unistd.h>
#include <string>

namespace {
constexpr std::size_t kCount = 24;
constexpr std::size_t kBytes = kCount * sizeof(int32_t);

void fail(const char *step, int rc) {
  std::fprintf(stderr, "FAIL %s rc=%d; no compatibility conclusion\n", step, rc);
  // An error during capture leaves the stream state unknown. Exit without
  // synchronizing, resetting the device, or touching any other process.
  std::fflush(nullptr);
  std::_Exit(1);
}
void check(const char *step, int rc) { if (rc != 0) fail(step, rc); }
void fill(int32_t *host, int base) {
  for (std::size_t i = 0; i < kCount; ++i) host[i] = base + static_cast<int>(i);
}
void announce(const char *phase, int base) {
  std::printf("EXPECTED phase=%s count=%zu first=%d last=%d\n",
              phase, kCount, base, base + static_cast<int>(kCount) - 1);
  std::fflush(stdout);
}
}

int main(int argc, char **argv) {
  if (argc != 5 || std::strcmp(argv[1], "--execute-isolated") != 0) {
    std::fprintf(stderr, "Disabled by default. Use the guarded launcher; no device call made.\n");
    return 2;
  }
  const std::string mode(argv[2]);
  if (mode != "eager" && mode != "graph") return 2;
  char *end = nullptr;
  long device = std::strtol(argv[3], &end, 10);
  if (*argv[3] == '\0' || *end != '\0' || device != 0) return 2;
  if (!std::getenv("PROBE_LAUNCHED_BY_GUARD") ||
      std::strcmp(std::getenv("PROBE_LAUNCHED_BY_GUARD"), "1") != 0 ||
      !std::getenv("ASCEND_RT_VISIBLE_DEVICES") || std::getenv("LD_PRELOAD")) return 2;
  const std::string out(argv[4]);
  // Restrict even direct invocation to the previously established task mount.
  constexpr const char *kTaskRoot = "/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029";
  char root_resolved[4096], out_resolved[4096];
  if (!::realpath(kTaskRoot, root_resolved) || !::realpath(out.c_str(), out_resolved)) return 2;
  const std::string root_prefix = std::string(root_resolved) + "/";
  if (out != out_resolved || out.rfind(root_prefix, 0) != 0) return 2;
  // Refuse unsafe JSON string characters even if the launcher was bypassed.
  if (out.empty() || out[0] != '/' || out.find_first_of("\"\\\n\r") != std::string::npos) return 2;
  for (const char *name : {"ASCEND_DUMP_PATH", "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "ASCEND_CACHE_PATH", "NPU_COLLECT_PATH", "TMPDIR"}) {
    const char *v = std::getenv(name);
    if (!v || out != v) { std::fprintf(stderr, "Refusing unsafe environment: %s\n", name); return 2; }
  }
  if (std::getenv("ASCEND_DUMP_SCENE")) return 2;

  const std::string config = out + "/synthetic-dump-config.json";
  const std::string body = "{\"dump\":{\"dump_path\":\"" + out +
      "\",\"dump_list\":[{}],\"dump_mode\":\"output\",\"dump_op_switch\":\"on\",\"dump_data\":\"tensor\"}}\n";
  int fd = ::open(config.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0) { std::perror("open config"); return 2; }
  ssize_t wrote = ::write(fd, body.data(), body.size());
  int close_rc = ::close(fd);
  if (wrote != static_cast<ssize_t>(body.size()) || close_rc != 0) return 2;

  check("aclInit", aclInit(nullptr));
  check("aclrtSetDevice", aclrtSetDevice(static_cast<int32_t>(device)));
  aclrtStream stream = nullptr;
  check("aclrtCreateStream", aclrtCreateStream(&stream));
  void *device_mem = nullptr;
  void *host_mem = nullptr;
  check("aclrtMalloc", aclrtMalloc(&device_mem, kBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
  check("aclrtMallocHost", aclrtMallocHost(&host_mem, kBytes));
  const int64_t dims[] = {static_cast<int64_t>(kCount)};
  const int64_t strides[] = {1};
  aclTensor *tensor = aclCreateTensor(dims, 1, ACL_INT32, strides, 0,
                                      ACL_FORMAT_ND, dims, 1, device_mem);
  if (!tensor) fail("aclCreateTensor", -1);
  check("aclmdlInitDump", aclmdlInitDump());
  check("aclmdlSetDump", aclmdlSetDump(config.c_str()));
  aclTensor *outputs[] = {tensor};
  auto *host = static_cast<int32_t *>(host_mem);

  if (mode == "eager") {
    fill(host, 1000);
    check("eager H2D", aclrtMemcpyAsync(device_mem, kBytes, host, kBytes,
                                        ACL_MEMCPY_HOST_TO_DEVICE, stream));
    check("eager dump", aclDumpOpTensors("SyntheticExpertCount", "synthetic_eager", outputs, 0, 1, stream));
    check("eager synchronize", aclrtSynchronizeStream(stream));
    announce("eager", 1000);
  } else {
    // Capture H2D and dump using pinned host memory. Host values change only
    // after the previous replay has synchronized. Eager verification is an
    // explicit caller prerequisite and is not inferred from file existence.
    aclmdlRI model = nullptr;
    check("capture begin", aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    check("capture H2D", aclrtMemcpyAsync(device_mem, kBytes, host, kBytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE, stream));
    check("capture dump", aclDumpOpTensors("SyntheticExpertCount", "synthetic_graph", outputs, 0, 1, stream));
    check("capture end", aclmdlRICaptureEnd(stream, &model));
    for (int replay = 0; replay < 3; ++replay) {
      const int base = 2000 + replay * 100;
      fill(host, base);
      check("replay", aclmdlRIExecuteAsync(model, stream));
      check("replay synchronize", aclrtSynchronizeStream(stream));
      char phase[32];
      std::snprintf(phase, sizeof(phase), "replay_%d", replay);
      announce(phase, base);
    }
    check("aclmdlRIDestroy", aclmdlRIDestroy(model));
  }
  check("aclmdlFinalizeDump", aclmdlFinalizeDump());
  check("aclDestroyTensor", aclDestroyTensor(tensor));
  check("aclrtFreeHost", aclrtFreeHost(host_mem));
  check("aclrtFree", aclrtFree(device_mem));
  check("aclrtDestroyStream", aclrtDestroyStream(stream));
  check("aclFinalize", aclFinalize());
  std::puts("API_CALLS_COMPLETE; inspect dump files and compare every replay before concluding PASS");
  return 0;
}
