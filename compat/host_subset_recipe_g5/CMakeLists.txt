cmake_minimum_required(VERSION 3.16)
project(sglang_v41_four_type_host LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  message(FATAL_ERROR "Build only on the parent's target aarch64 CPU")
endif()
foreach(name DONOR_SOURCE_ROOT CURRENT_CANN_ROOT)
  if(NOT IS_DIRECTORY "${${name}}")
    message(FATAL_ERROR "Set explicit ${name}")
  endif()
endforeach()
set(A "${DONOR_SOURCE_ROOT}/csrc/attention")
set(Q "${A}/quant_lightning_indexer_v2")
set(S "${A}/sparse_flash_mla")
set(M "${A}/quant_lightning_indexer_v2_metadata")
# Explicit host-only lists, matching actual selected op_host/CMakeLists.txt.
# OpDef *_def.cpp and all op_api/device files are intentionally absent.
set(TILING_SOURCES
  "${Q}/op_host/quant_lightning_indexer_v2_tiling.cpp"
  "${Q}/op_host/quant_lightning_indexer_v2_infershape.cpp"
  "${S}/op_host/sparse_flash_mla_tiling.cpp"
  "${S}/op_host/sparse_flash_mla_infershape.cpp"
  "${M}/op_host/quant_lightning_indexer_v2_metadata_infershape.cpp"
  "${S}/op_host/checkers/base_checker.cpp"
  "${S}/op_host/checkers/checker_runner.cpp"
  "${S}/op_host/checkers/common_checker.cpp"
  "${S}/op_host/checkers/mask_checker.cpp"
  "${S}/op_host/checkers/metadata_checker.cpp"
  "${S}/op_host/checkers/paged_attention_checker.cpp"
  "${S}/op_host/checkers/seq_len_checker.cpp"
  "${S}/op_host/checkers/sinks_checker.cpp"
  "${S}/op_host/checkers/softmax_lse_checker.cpp"
  "${S}/op_host/checkers/sparse_compression_checker.cpp"
  "${S}/op_host/checkers/sparse_flash_mla_checker.cpp")
foreach(path IN LISTS TILING_SOURCES)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Pinned host TU missing: ${path}")
  endif()
endforeach()

# These locations come from actual csrc/common/CMakeLists.txt and the current
# toolkit inventory. No CANN source tree or device compiler is required.
set(INCLUDES "${DONOR_SOURCE_ROOT}" "${DONOR_SOURCE_ROOT}/csrc/common/include")
foreach(prefix "${CURRENT_CANN_ROOT}" "${CURRENT_CANN_ROOT}/aarch64-linux")
  foreach(suffix include include/base include/op_common include/aclnn
      include/aclnn/opdev include/experiment/metadef include/experiment/slog
      include/experiment/runtime include/experiment/msprof)
    if(IS_DIRECTORY "${prefix}/${suffix}")
      list(APPEND INCLUDES "${prefix}/${suffix}")
    endif()
  endforeach()
endforeach()
if(IS_DIRECTORY "${CURRENT_CANN_ROOT}/asc/include")
  list(APPEND INCLUDES "${CURRENT_CANN_ROOT}/asc/include")
endif()
set(LIB_PATHS "${CURRENT_CANN_ROOT}/lib64" "${CURRENT_CANN_ROOT}/aarch64-linux/lib64")
set(HOST_LIBS)
foreach(lib register exe_graph graph graph_base platform error_manager ascendalog unified_dlog ops_base metadef mmpa c_sec tiling_api)
  find_library(HOST_${lib} NAMES ${lib} PATHS ${LIB_PATHS} NO_DEFAULT_PATH REQUIRED)
  list(APPEND HOST_LIBS "${HOST_${lib}}")
endforeach()

find_library(HOST_RT2_REGISTRY NAMES rt2_registry PATHS ${LIB_PATHS} NO_DEFAULT_PATH REQUIRED)
if(NOT HOST_RT2_REGISTRY MATCHES "\\.a$")
  message(FATAL_ERROR "CANN plugin requires the private static rt2_registry archive")
endif()

# Detect the toolkit C++ string ABI with a link-only compile test; no program
# runs and no op is registered. Do not inherit Torch ABI from its wheel.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${INCLUDES})
set(CMAKE_REQUIRED_LIBRARIES ${HOST_LIBS})
foreach(abi 0 1)
  set(CMAKE_REQUIRED_DEFINITIONS "-D_GLIBCXX_USE_CXX11_ABI=${abi}")
  check_cxx_source_compiles("#include <graph/operator.h>
    #include <string>
    int main(){std::string n(\"v41_link_only\"),t(\"NoRegistration\"); ge::Operator o(n,t); return o.GetName().empty();}"
    CANN_ABI_${abi}_LINKS)
endforeach()
if(CANN_ABI_0_LINKS AND NOT CANN_ABI_1_LINKS)
  set(CANN_ABI 0)
elseif(CANN_ABI_1_LINKS AND NOT CANN_ABI_0_LINKS)
  set(CANN_ABI 1)
else()
  message(FATAL_ERROR "Toolkit C++ ABI link test ambiguous/failed; inspect CMakeError.log, do not guess")
endif()
message(STATUS "Detected CANN C++ string ABI=${CANN_ABI}")

add_library(v41_proto SHARED proto_subset.cpp)
add_library(v41_tiling SHARED ${TILING_SOURCES})
set_target_properties(v41_proto PROPERTIES OUTPUT_NAME cust_opsproto_rt2.0)
set_target_properties(v41_tiling PROPERTIES OUTPUT_NAME cust_opmaster_rt2.0)
foreach(target v41_proto v41_tiling)
  target_include_directories(${target} PRIVATE ${INCLUDES})
  target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=${CANN_ABI}
    LOG_CPP PROCESS_LOG BUILD_OPEN_PROJECT
    OPS_UTILS_LOG_PACKAGE_TYPE="[Custom]")
  target_compile_options(${target} PRIVATE -O2 -fvisibility=hidden -fvisibility-inlines-hidden
    -fstack-protector-all -include "${DONOR_SOURCE_ROOT}/csrc/common/include/cann_compat.h")
  target_link_libraries(${target} PRIVATE -Wl,--whole-archive ${HOST_RT2_REGISTRY} -Wl,--no-whole-archive -Wl,--no-as-needed ${HOST_LIBS} -Wl,--as-needed dl)
  target_link_options(${target} PRIVATE -Wl,--no-undefined -Wl,-z,relro,-z,now
    "-Wl,-Map,${CMAKE_BINARY_DIR}/${target}.map")
endforeach()
target_compile_definitions(v41_proto PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO")
target_compile_definitions(v41_tiling PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/selected_targets.txt"
  CONTENT "v41_proto=$<TARGET_FILE:v41_proto>\nv41_tiling=$<TARGET_FILE:v41_tiling>\nCANN_ABI=${CANN_ABI}\n")
