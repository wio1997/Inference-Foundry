# M017/M018：Ascend 长度镜像、dtype 所有权与 staging

本轮定位了保留 CPU 长度的具体消费者，并否定两种无条件省复制的改法。安装源码固定在 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`。本地 Torch 2.8.0 与目标镜像 Torch 2.10.0+cpu 的探针结果一致；**整体服务证据仍是 E0，没有 NPU 或 PD 性能数据，也没有修改生产源码。**

## 已有实现与字段依赖

下表针对固定安装源码的 Ascend backend 和 spec_v2 分支。它闭合了“不能直接删除/共享”的反证；不是对所有模型和可选后端的穷尽审计。源码行与哈希见 [source-excerpts.json](source-excerpts.json)、[source-index.json](source-index.json)。

| 字段/能力 | 生产与门控 | 读取或变更 | 生命周期结论 |
| --- | --- | --- | --- |
| `FutureMap.new_seq_lens_buf` / `batch.seq_lens` | verify 后发布真实新长度；resolve 先等待 publish，再按 future_indices gather | 设备长度始终解析；关闭 CPU 镜像也保留此更新 | 接受结果是真实依赖，不能用旧长度 + 1 代替 |
| `batch.seq_lens_cpu` / `seq_lens_sum` | `decide_needs_cpu_seq_lens` 对后端声明取 OR；缺失声明默认 True，TBO/NGRAM 强制 True | AscendAttnBackend 继承 AttentionBackend 的 True；多步 draft wrapper 未声明，仍默认 True；EAGLE 同时列出 target/draft/draft_extend | 当前消费者存在，不能直接声明 False；实际进程对象尚未 introspect |
| 非 CUDA 长度回传 | 私有 D2H stream 与 pinned mirror 仅在 `_is_cuda` 构造分支分配 | 无私有 stream 时 `.cpu()` 后 CPU sum；Ascend metadata/graph 仍读 CPU 长度 | 不能把 CUDA 私有流当作已启用 NPU 实现；未测 NPU 等待成本 |
| `forward_metadata.seq_lens_cpu_int` | eager metadata 从共享 CPU 长度执行 `.int()` | 非 compact verify 加 draft_token_num；decode+spec 加 step+1；MLA/non-MLA MTP 均有 CPU 长度读取 | int64→int32 时得到独立副本；共享 int32 输入会 alias，原地偏移会污染源及先前 snapshot |
| 图模式 CPU 长度 | graph metadata 使用 CPU list；DFlash 的预规划 replay 分支也从 forward_batch CPU 长度更新 buffers | graph.update 可重绑 list；实际分支和 kernel 未执行 | 不能凭 eager 单处读点消失就关闭镜像；图模式没有被本轮 CPU 探针验证 |
| P `prefill_input_ids_cpu` | prepare_for_extend 调 `flatten_arrays_to_pinned_cpu(parts, _pin)`；NumPy concatenate 复制输入，torch.from_numpy 共享新数组 | forward 入口 `.to(device, non_blocking=True)` 后清空 staging 引用 | concatenate 提供准备时快照；借用调用方 view 需要额外不可改写所有权合同 |
| 显式 pin 能力 | common wrapper 委派 current_platform；in-tree `NPUDeviceMixin.is_pin_memory_available` 返回 False | `_pin` 控制 `.pin_memory()`，不能由变量名或注释推断已 pin | 该平台声明下“启用 pinned pool”子候选条件不成立；不是硬件普遍不支持 pin 的结论，实际 platform 对象仍待观察 |

## 实际最小实验

主控冻结代码、检查 AST 提取和断言；ZCode 使用已核验的 `deepseek-flash` 配置别名执行一次冻结命令。四轮外层 CLI 和内层 controller 均为 exit 0，输入哈希保持不变。完整命令、哈希和模型别名边界见 [zcode-audit.json](zcode-audit.json) 与 [receipts](receipts)。后端实际发布版本未知，不由别名推断。

| 检查 | 每个环境的实测结果 | 可支持的结论 |
| --- | --- | --- |
| 6 种算法配置 × TBO × 三个后端的 4 态声明 | 768 个实际门控调用匹配 oracle；NEXTN 映射 EAGLE | 门控不是“一律可删”，缺失声明保持保守 |
| int64/int32 × 算法 × mode × step × draft 数 | 180 组通过预期断言；共享 int32 在 48 组非零偏移用例修改源长度 | 否定无条件 dtype 替换；不否定 clone、out-of-place 等所有权重新设计 |
| 同一 batch 连续两个 draft step | int64 保存 `[11,101]`、`[12,102]`，源 `[10,100]`；int32 两份最终都变 `[13,103]` | 同址别名使早先 metadata 被后续 step 修改；wrapper 源码确实复用同一 forward_batch，但完整 NPU method 未执行 |
| concatenate staging | 5 组，含单段、多段、空段、512 tokens；修改调用方数组后快照不变，pin=False | 当前复制隔离已存在；不能当作新增优化 |
| 延迟 resolve 前注入调用方改写 | 原路径保留 `[11,22,33]`；borrowed view 变为 `[999,22,33]` | 缺少不可改写合同的无条件借用不等价；未证明真实服务会在该窗口改写 |

[本地完整结果](local-cpu-result.json)和[目标镜像 CPU 结果](target-image-cpu-result.json)除时间和环境字段外完全一致。实际执行的是原函数或明确标记的 AST 语句切片：metadata 只取 `.int()` 赋值及偏移 block，`compact=None`；未运行整段 NPU 初始化、FIA、graph、DMA、event 或 allocator。resolve 的目标设备为 CPU，不能证明 H2D 完成语义。

目标镜像 CPU 容器使用精确 image ID `sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263`，加载固定安装源码快照到 /tmp；未把镜像内置源码与安装 HEAD 混同。runc、无网络、只读根、无 device/bind、cap-drop ALL、1 核/2 GiB/64 PID 的 inspect 字段已断言；探针确认无加速设备节点且未加载 torch_npu。容器退出后按任务 label 和精确 ID 删除，已有服务 ID、启动时间、重启计数前后相同。见[容器回执](target-image-container-receipt.json)。宿主机与本地时钟存在约 99 秒偏差，只保留各自时间，不跨时钟计算耗时；本轮没有性能比较。

## 观察配置及先前 CPU 压力用例的范围

R001 GET200，但最初按嵌套 server_args 取值，得到空 whitelist；这个不完整观察保留。读取 endpoint 的 flat resolved_dict schema 后，R002 捕获了[限定配置字段](observed-config.json)：page_size=128、chunked_prefill_size=16384、TP/EP=16、PP/DP=1、Ascend、EAGLE steps=3/topk=1/draft_tokens=4、TBO=False。`disaggregation_mode` 是字符串 `"null"`，观察进程没有 P/D 角色；仅有 transfer_backend=mooncake 不能说明 PD 传输被执行。KV dtype=`auto`，实际 dtype 仍未知。该进程属于观察对象，未作为本任务资源或基线。

前几轮 descriptor 的 N 是 **page 索引数**。以观察到的 page_size=128 换算，单请求 32768-token 上下文约 256 页；N=32768 页对应 4194304 tokens，超过观察模型配置的 max_position_embeddings=1048576，不能称为该配置单请求代表用例。并发 aggregate 是否能出现该 N 尚未测量；大 N 仅保留为函数压力测试。下一轮性能实验先从真实 PD 调用采集每次 N、连续段 R、层数和暴露时间，再决定候选优先级。

## 决策、必要工作与下一步

- **保留**当前 Ascend CPU 长度镜像。任何 device-only 改造必须先改完 target/draft/graph 等全部消费者，再证明实际后端组合不再需要它。
- **拒绝**无条件共享 int32 和借用 caller view；反例针对未改变所有权的候选，未宣布所有 int32 或复用设计不可行。
- 显式 pinned-memory=True 子候选对 in-tree NPU 声明**条件不适用**；通用 staging 复用仍 **INCONCLUSIVE**，需要实际 NPU copy/event 的最后读取与安全回收证据。
- 对 B 个请求，当前 int64 CPU 长度镜像的有效载荷是 8B 字节，int32 metadata 输出是 4B 字节；生产等待、拷贝启动、CPU sum/list/转换和竞争成本未测。T 个输入 tokens 的 int64 staging 含 8T 字节 payload，复制快照有语义用途。字节量只给出必要工作的建模起点，不能替代实际可达带宽和关键路径测量。
- 本轮没有速度数据和生产候选入选。后续转到 M021/NPU graph 的形状/元数据边界，并继续等待已提出的 44/48 资源归属信息；独立 CPU/source 工作继续推进，PD+MTP 主线约束保留。

## 复现与原始证据

`scripts/probe_metadata_support.py` 可使用 `--source-root source-inputs --stdout --expected-hashes <local-cpu-result.json中的source_hashes JSON>` 在无加速设备、无 torch_npu autoload 的独立 CPU 环境运行；结果内有每个 AST 节点哈希和源行。默认目录依赖本地任务布局。remote controller 属于本任务专用访问配置，不是通用部署脚本，不应在别的环境直接运行。

原始 source collection JSON 和 ZCode stdout 保留本地，发布其哈希、限定字段和 CPU 原始结果，不发布配置或 model I/O。[artifact-index.json](artifact-index.json)列出发布文件哈希与本地原始入口。`source-inputs` 是固定源码的最小输入集，保留原文件头；其他阅读位置用 excerpt+完整文件哈希定位。
