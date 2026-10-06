# Run254 saved CPU evidence and the concrete MC2 predicate

本阶段没有新增代码级性能 KEEP。这里继续归约 Run249 和一次已完成的最小 Run254，不安排新的 profile、参数扫描或大规模 NPU Run。H6 是唯一正在准备验证的性能候选；isolated native build 尚不是模型 correctness。

## What the minimal diagnostic actually establishes

Run254 对原语义 P249/D253 发出一个 2334-prompt/8-token PD 请求。原始输出仍为相同八个 token、chunks 1/2/2/2/1，2334 external KV hits，finish=length，非完整自然 EOS；256.793ms/token 是诊断请求结果，不是新收益。16 个 D 主线程共 3223 个 user CPU-clock IP/callchain 样本，zero lost；采样启用窗口 2.241002328s，observer CPU 38.419030ms。没有 NPU profiler、源码变更或服务重载。

原始 17,277,931-byte raw 留在服务器 Run254/cpu_samples_raw.json，SHA256 `63be953de17ebefce743738bae3b6d7780a84373d52a147d936f480c8e70d803`。ELF 的 file-offset/load-segment 映射和 exact symbol-size containment 由 [resolver](reduce_cpu_instructions.py) 实现；无 symbol 的 nearest FDE 不能默认命名。去重 resolved evidence SHA256 为 `902e500c1c7c213c44fb826d80551307647da8d00841c73b5aa0c5e5902b0381`。[第二个离线 reducer](reduce_cpu_paths.py) 对同一证据独立复算出 [紧凑分组](cpu_paths_reproduced.json)。

| Disjoint sampled call-stack context | Samples | Share |
|---|---:|---:|
|Python kernel body/callback without inner boxed operator|1853|57.493%|
|Nested native operator under Python kernel|679|21.067%|
|Other user CPU, Python source function unknown|381|11.821%|
|MC2 V4 capability query|248|7.695%|
|Operator without sampled Python kernel frame|62|1.924%|

这些是统计 stack contexts，不是互斥的 step wall 预算，也不是可删除比例。第一项可能含未通过 boxed dispatcher 的 native callback。2780/3223 samples 包含 PythonKernelHolder ancestor，但仅一个 leaf 落于该函数；删除外层 wrapper 不会删除 handler body 与内部 native 操作。主线程 before/after schedstat 跨 reader setup 和 P 请求，不能除以 2.241s 推断 Decode CPU occupancy。event 使用固定 10,000,000ns CPU-clock period，不是 adaptive wall-frequency。没有 Python source stack 或四个控制层 entry/exit 标记。

## Exact repeated code path

248 个 GetOpApiFuncAddr 样本全部位于实际 availability predicate `0x490a9d0`，其 caller 只有 MC2 dispatch 的 FDE `0x31ed7c4`/return PC `0x31edf44`（134）及 combine 的 `0x31ba1d0`/`0x31ba4ec`（114）。[Compiled path](mc2_compiled_capability_path.json) 对实际 installed libtorch_npu.so 核验两 V4 字面量和 BL；[统计 attribution](cpu_capability_attribution.json) 保留每 rank 与 leaf 明细。不能泛化为所有 NPU 算子每步都重新搜索 API。

源码 `check_aclnn_kernel_available` 在两个 V4 分支前重新查询 API 与 GetWorkspaceSize。后续 EXEC 的 static function pointer 并不缓存这个先行 bool。GetOpApiFuncAddr 的 custom/default 路径及 handlers 在 worker 初始化后固定，但查询仍执行 realpath/dlopen/dlsym/owner 检查。它没有 dlclose；不能推断每次查找都会使 TLS generation 失效。其余 loader leaf 也不能全归到这个补丁。

[Production patch](mc2_capability.patch) 仅把两处 V4 能力结果存为函数局部 static const bool。V3/V2 fallback、输出分配、动态路由、workspace/executor、kernel/communication 参数与输出保持相同。负缓存只适用于库生命周期固定的已就绪 worker；换库必须重启。

## Evidence and the next gate

[Native CPU helper result](mc2_native_cache_summary.json) 使用实际 installed exported getter 和源码相同的 C++ predicate。8 threads×1000 次 positive/missing-API/missing-workspace case 均通过，初次查询分别仅 2/1/2 次。11×100 native batch 的原查询中位 dispatch 51.02684us/combine 50.53403us；cached loop 约 1/1.5ns。计时在 C++ 内，无逐 predicate Python/ctypes 往返。380+380 的 38.593ms 机械量级参考仍不是模型 critical-path 预算或 E2E saving。

两个 MC2 frontend object 已能编译，但 queue/stream/workspace 等约 30 个内部依赖不导出。用独立插件重实现这些依赖会扩大语义变化，已排除。当前用实际 torch_npu commit `5dd8ef3f9b375b5ae4a83538d5785754148c3302` 和 op-plugin `8b9c8534fa41eff367a41c155843daa530ab3a08` 构建隔离 native 产物。所有 kernel/CANN/communication 实现保持现场版本；原 Tensorpipe shared library 配同 commit headers；DVM 使用 pinned PRE_ASCEND binaries。

C 的代码生成缺 ACL 头文件、D 的编译缺 super_kernel.h 均保留原日志，E 只在隔离 clone 补现场缺失头文件并复用对象继续 CPU 构建。没有安装 candidate、初始化 NPU 或提交新模型请求。通过编译后须先独立 CPU import/ABI 检查，再模型 correctness，随后 common-binary matched A/B/A/B 和相同自然 EOS 完整请求。不能用源 diff、CPU helper 或 object compile 代替这些门槛。

H6 有已定位的具体可删查询，不宣称它是全局最大可删 OFF wall gap。最大已定位 region 仍是模型内部 eager producer，主要在 MoE；其余 handler/native preparation 正在通过实际源码及已有 trace 下钻。控制层精确四分法和完整 SLA 有效容量仍 unknown。
