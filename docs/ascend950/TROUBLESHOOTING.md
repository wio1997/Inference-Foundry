# 950DT / DeepInfer 实际问题与修复说明

本页对应2026-10-21的真实功能部署，固定提交与验收结果见[部署手册](README.md)。修复范围是这组源码、CANN9.2和本机拓扑；更新版本后先确认上游是否已有等价修复。

## 补丁顺序与构建边界

`deploy/ascend950/apply_patches.py` 按以下顺序处理：

1. `deepselect-ascend-build-import.patch`：DeepSelect构建平台选择。
2. `flashmla-cann-rint-api.patch`：FlashMLA RINT入口。
3. `engram-direct-storage.patch`：最终NPU存储直接加载。
4. `server-local-ubmem.patch`：本机层与Engram UBMEM-only开关。
5. `npu-from-blob.patch`：外部NPU存储的格式描述。
6. `deep-gemm-assert-include.patch`：JIT断言声明。
7. `candidate-short-context.patch`：短上下文候选宽度。

顺序有依赖：Engram直接存储补丁新加了from_blob调用，后续NPU补丁才将其一起替换。不能乱序，也不能只修Python而继续使用旧的DeepEP/DeepGEMM二进制。

复现目录执行：

```bash
python3 apply_patches.py
python3 apply_patches.py --check-only
```

脚本只接受固定原始源码；完成后记录补丁哈希与最终源文件哈希。重复执行检查最终状态；手改源码后会报告drift。要升级或试验其他改动，请另建源码目录，保留原始可运行版本。

## 1. 源码和Python依赖下载失败

大仓库递归克隆、嵌套submodule和大wheel下载是不同问题。历史部署采用固定commit归档分层填充vendor；DeepJIT同时出现在DeepEP和DeepGEMM的不同子目录。缺一个嵌套依赖可能在JIT或首次模型加载时才暴露。

Python依赖先下载成wheelhouse并核对SHA，再从本地安装。历史torch wheel约155MB，曾使用HTTP range并发获取，最终按完整SHA验证；不能拼接未经检查的分片。

模型历史下载使用ModelScope snapshot和分片下载，并逐文件核对SHA；整理后的下载器按文件revision固定内容。下载结束不等于权重完备，应检查48个分片、索引引用和84个文件全部校验。损坏partial、HTTP代理返回错误页、磁盘满和中断需分别处理，不跳过校验继续启动。

## 2. DeepSelect构建时导入CUDA测试依赖

现象：显式指定ASCEND平台，setup仍因测试辅助模块引入CUDA/Triton相关依赖失败。

原因：平台覆盖变量被处理前，setup已导入 `tests.kernelkit`。安装额外CUDA依赖不能解决目标平台选择顺序。

修复：先读 `DEEP_SELECT_BUILD_TARGET_PLATFORM`；显式ASCEND时不导入CUDA测试工具。未覆盖时保留平台探测，CUDA分支仍加载原来的辅助模块。

验收：在不传NPU的构建容器完成wheel构建，再实际运行candidate TopK检查。CPU构建成功不能替代NPU数值检查。

## 3. GCC13与Bisheng设备头文件冲突

现象：主机扩展需要GCC13/C++20，直接把整条工具链切到GCC13后，Bisheng设备编译在受约束std::pair等头文件处失败。

修复：容器内做编译器分流：`.cpp/.cc/.cxx` 由GCC13编译；`.asc/.cce` 或对应语言标志由真正Bisheng编译，并指定只提供GCC11头文件的toolchain前缀。先用包含 `<utility>` 的小Ascend C源做语法probe。

`configure_compilers.py` 只修改专用构建容器里的Bisheng入口，记录真实入口和分流规则；宿主编译器不变。该wrapper依赖 `/work/bisheng-gcc11` 和 `compiler-entrypoint.json`，恢复时必须保留目录。

若出现找不到真实Bisheng、symlink目标失效或头文件不匹配，先核对镜像和挂载，再运行probe。更换CANN版本时不能默认该分流仍必要或兼容。

## 4. FlashMLA RINT转换入口不兼容

现象：当前CANN头文件没有上游代码调用的 `asc_float2bfloat16_rn_v2_impl`。

修复：使用当前公开的 `asc_float2bfloat16_rn_v2` RINT转换入口，未改成另一舍入模式。

验收：四组真实NPU sparse-prefill数值检查，包含不同q/kv长度、sink与topk_length。补丁解决的是这一API兼容问题，不表示所有FlashMLA路径都已验证。

## 5. Engram初始化峰值显存OOM

现象：模型materialize阶段约70.94GiB，随后还申请约23.61GiB的Engram存储，超出剩余容量。

原因：HF加载先物化一份本地Engram表，通信buffer又分配最终NPU表；两份完整存储在初始化峰值同时存在。减小运行时batch不一定消除这份权重级重复分配。

修复：先创建最终Engram buffer，暴露其storage tensor view；HF加载直接写入最终分片。保持shape、dtype及scale数据，避免同地址的device-to-device copy；持有context确保外部分配生命周期有效。

观测：materialize约降到48.05GiB；实际ready后每卡进程显存约79,266MB，整卡约84,125–84,128/98,304MB。计量阶段和单位不同，不能把materialize值当作稳态总显存。

验收：8rank FP8/BF16数据和scale逐位一致，随后完整模型真实生成。不是把Engram关闭或换成主存存储；本固定Ascend后端不支持CPU-backed Engram。

## 6. 单机通信错误使用外部CLOS层

现象：HcclChannelAcquire超时；某些初始化操作可走通，但大尺寸AllGather不能完成。

原因：固定实现默认选第二层网络，当前单机任务的外部CLOS通路不适用。小collective通过不足以证明大payload或另一通信模块的路径正确。

修复分两部分：

- 从实际DCMI信息派生任务专有layer0 rootinfo，保留真实EID/port；只挂载到任务容器。
- DeepEP与DeepGEMM增加显式层索引覆盖 `DEEPINFER_UBMEM_NET_LAYER=0`；Engram使用 `DEEPINFER_ENGRAM_UBMEM_ONLY=1` 跳过未用Jetty。

默认层行为仍保留；单机开关不自动推广到跨节点。原查询、过滤前结果和endpoint配置在部署者私有目录保留，公共仓库不发布设备地址。

验收：8rank Engram、完整权重加载、真实前向和生成都通过。新拓扑重新生成并重验；不要通过改宿主全局拓扑把其他任务一起切换。

## 7. `unknown format type` 与外部NPU内存包装

现象：外部通信分配内存用通用 `torch::from_blob` 包成张量后，后续HCCL/NPU路径报格式未知。

原因：内存地址、dtype和device正确，仍不足以建立torch_npu要求的设备格式描述。

修复：包含 `torch_npu/csrc/aten/common/from_blob.h`，使用 `at_npu::native::from_blob`，保留原本的deleter、context持有和offset/size检查。覆盖Engram、PP、EP、bucket和DeepGEMM相关外部buffer视图。

修改后重新构建DeepEP-Ascend和DeepGEMM-Ascend，再确认Python实际加载的 `.so` 路径；不能只看源码已经改了。历史二进制哈希随验收快照提供，新构建记录自己的哈希。

## 8. DeepGEMM JIT缺少断言声明

现象：设备JIT报找不到ascendc_assert声明。

修复：在Ascend公共设备头显式包含 `utils/debug/asc_assert.h`。保留断言，不用关闭检查或替换为常量表达式绕过。

验收：真实JIT能执行，scale layout三组参考检查通过，并完成后续模型前向。

## 9. 短上下文的candidate TopK断言

现象：context4096时，实际候选块少于配置的2048输出列，上层断言拒绝执行。

错误的直觉修法是把2048改成短上下文块数；这样会改变下游稀疏MQA期望的固定输出宽度。

本次修复保留固定2048列，用DeepSelect已有的有效 `end` 和 `-1` 填充契约处理无效部分。仅改断言的适用条件，仍检查配置上界。

验收覆盖空有效长度、短输入、block8/16、4096上下文以及超过2048候选块的筛选；检查有效索引、对应分数、无效填充及输出前后哨兵区域。完整模型随后通过。不把这五组推广为全部上下文与dtype证明。

## 10. API可见EOS、mock及ready状态

现象：模型token正确，但最后把EOS可见字符串一并返回；或者只确认API进程200，尚未确认模型ready。

修复：从模型config取得EOS ID，从tokenizer_config取得对应文本；只有真实最后token为EOS且文本末尾对应时才去掉该末尾标记。原始token证据和usage保留。结合真实EOS和长度决定finish reason。

本次适配器通过 `create_app(infer=...)` 注入真实RPC，backend不可用时返回503。`/healthz` 与 `/readyz` 分别代表API活性和引擎就绪；不向用户返回官方mock示例的固定答案。

验收：数字和中文题两轮通过；各rank token哈希可追溯。当前仍只支持文本、非流式、单请求功能链路。

## 查看证据与下一步

- [历史实卡结果及native哈希](../../deploy/ascend950/evidence/validated-deployment.json)
- [FlashMLA检查](../../deploy/ascend950/evidence/flashmla-correctness.json)
- [DeepGEMM布局检查](../../deploy/ascend950/evidence/dg-layout-correctness.json)
- [短上下文候选检查](../../deploy/ascend950/evidence/candidate-short-context-correctness.json)
- [真实中文验收](../../deploy/ascend950/evidence/chinese-verification.json)

性能研究另行记录无profiler基线、独立DP请求和时间线。上述“通过”都是明确用例与环境下的正确性/部署证据，不自动升级成吞吐领先或生产可用结论。
