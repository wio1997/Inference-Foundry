# 910C W8A8 适配说明

本文说明交付版本的执行路径与实现约束。完整源码以 [SOURCE.md](SOURCE.md) 中的归档和逐文件 SHA256 为准；`code/` 保存从该归档提取的适配文件，便于阅读。

## 1. 检查点与量化计算

入口为 [modelslim910c.py](code/deepinfer/hugging_face/modelslim910c.py)。加载器读取 `config.json`、`quant_model_description.json` 和 `quant_model_weights.safetensors.index.json`，按索引定位原始分片。

`W8A8_DYNAMIC` 线性层保持 signed INT8 权重常驻 NPU；激活动态量化后进入 INT8 计算。每通道 weight scale 按固定镜像的运行时精度处理，FLOAT 层保留相应浮点类型。本版本仅接受受支持的量化布局；非零 weight offset 等不匹配配置直接报错，不进行隐式转换。

算子和专家布局见 [w8a8.py](code/deepinfer/npu_ops/w8a8.py) 与 [ascend_w8a8_packed.py](code/deepinfer/model_executor/models/ascend_w8a8_packed.py)。TP 切分同时处理权重和对应 scale；路由专家采用 EP16 分布。主干与专家权重不执行 CPU prefetch。

## 2. 主干、注意力与缓存

完整目标模型保留 40 层、mHC、RoPE、路由专家和共享专家。mHC 使用固定镜像的 `npu_hc_pre_v3` 与 `npu_hc_post`；该融合不等同于上游 lazy mHC 的跨子层融合。

注意力模块分别位于：

| 模块 | 源码 |
|---|---|
| CANN DSA 扩展加载与稀疏 MLA | [dsa910c.py](code/deepinfer/npu_ops/dsa910c.py) |
| 索引器 | [w8a8_indexer910c.py](code/deepinfer/model_executor/models/w8a8_indexer910c.py) |
| C1/C2 压缩 | [w8a8_compressor910c.py](code/deepinfer/model_executor/models/w8a8_compressor910c.py) |
| 分页缓存与 SWA 页环 | [w8a8_paged_cache910c.py](code/deepinfer/model_executor/models/w8a8_paged_cache910c.py) |

缓存按实际检查点的跨层拓扑建立。KV 源层为 2、8、14、20；索引源层还包括 24、28、32、36。消费者复用对应源层结果，不能按每层独立生成缓存处理。C2 压缩保留奇数尾部与待配对状态；投机拒绝后的恢复同时处理这些状态。

`dsa910c.load_ops()` 在首次设备初始化之前配置 custom OPP 并加载编译扩展。该模块直接调用固定镜像的算子 ABI；vLLM Python 模型执行器和调度器不参与目标模型计算。镜像内 tokenizer 在 CPU 接口中复用。

## 3. TP、EP 与 MC2

worker 建立两个 TP8 组及覆盖 16 个 rank 的独立 MC2 通信组。预填充使用 sequence parallel、context parallel 与 KV token parallel；目标词表头按 TP 分片，避免每 rank 重复执行完整词表投影。

[mc2_910c.py](code/deepinfer/npu_ops/mc2_910c.py) 调用固定镜像的 `dispatch_ffn_combine`，路由专家执行融合 dispatch、INT8 FFN 和 combine。普通通信组与 MC2 的缓冲区独立配置，本版本 MC2 缓冲区为 1024 MiB。

融合算子的容量参数采用：

```text
max_output_size = max(capacity_floor, global_bs × topk)
capacity_floor = 4096
```

`global_bs × topk` 覆盖所有有效 token 路由均落到同一 rank 的上界。容量不依赖平均专家负载；不能为了减少内存而省略该上界。C8 的目标验证实例中，`global_bs=48`、`topk=6`，完整路由上界为 288；预填充使用各自实际形状计算。

本次容量调整降低了 NPU 进程内存，但未观察到吞吐提升。该参数也是 vLLM Ascend 可调项，不属于 DeepInfer 独有收益。

## 4. 图执行与 DSpark

目标图输入、缓存坐标、通信掩码及常量保持持久地址，图捕获后执行实际 replay。图执行代码见 [w8a8_graph910c.py](code/deepinfer/model_executor/models/w8a8_graph910c.py) 与 [dspark_varlen910c.py](code/deepinfer/model_executor/models/dspark_varlen910c.py)。

DSpark 计算见 [models/dspark910c.py](code/deepinfer/model_executor/models/dspark910c.py)，服务调度见 [engine/dspark910c.py](code/deepinfer/engine/dspark910c.py) 和 [dspark_confidence910c.py](code/deepinfer/engine/dspark_confidence910c.py)。其流程为：

1. 从目标层 37、38、39 提取草稿输入特征。
2. 使用检查点的 `mtp.0`–`mtp.2` 学习式模块生成最多 5 个草稿 token，执行对应的非因果草稿注意力、Markov 修正和置信度头。
3. 依据上一轮可用的接受置信度调度验证长度，避免用本轮尚未验证的结果反向选择本轮计算预算。
4. 由完整目标模型验证草稿前缀，确定接受长度并生成目标 token。
5. 拒绝时恢复缓存长度、Engram 历史及 C2 待配对状态，并执行必要的修复计算。

本版本的每 DP 目标 query 图档位为 4、7、24。C8 全部选择 5 个草稿时，每个 DP 的 4 个请求各提交 6 个目标 query，共 24 个；这里的“6”来自 5 个草稿及 1 个目标位置，不是 64-token 验证方案。

提示查找关闭。早期“提示查找 63 + 目标验证 64”的性能记录仅属于历史实验，不能用于证明 DSpark 完成或推导其收益。

预填充末段 FFN 裁剪按精确依赖关系保留所需位置。该实现不等同于上游含近似 SWA 的完整有界重放，其收益不能按上游生产数据直接折算。

## 5. Engram CPU 表与共享读取

Engram 表保留原始 INT8 数据与 E8M0 group-32 scale。CPU 侧维护历史并计算 hash；解码时 NPU 通过注册的 host 映射读取所需行并反量化，不将整张表展开到 NPU。

实现见 [npu_ops/engram_uva910c.py](code/deepinfer/npu_ops/engram_uva910c.py) 和 [models/engram_uva910c.py](code/deepinfer/model_executor/models/engram_uva910c.py)。24 个 hash head 按 TP8 分片，每 rank 负责 3 个；两个 DP 中相同 TP rank 使用同一 memfd，并以 `MAP_SHARED` 映射。共享表通过同一 NPU 容器的 `/proc/<pid>/fd/<fd>` 访问，不受该容器 64 GiB `/dev/shm` 配额限制。

原始文件的 `MAP_PRIVATE` 注册曾触发写时复制，导致两个 DP 各保留一份物理页，因此不能用于证明共享。本版本通过 rank 0 与 rank 8 的 smaps 核对共享页和 PSS；约 202.8 GB 为所有 TP 分片合计的原始表数据，两个 DP 不再各自复制一份。

该实现仍包含 CPU hash 与图入口前的依赖等待，未完成全流程 ZOS。设备 hash 单点探针通过尚不代表已集成服务；该探针代码未纳入本次部署源码。

## 6. 实现状态

| 状态 | 内容 |
|---|---|
| 已接入并经本负载验证 | W8A8 原生加载、TP8/EP16、分页 DSA、学习式 DSpark、置信度调度、目标图、目标词表并行、Engram 共享读取、MC2 完整路由容量保护 |
| 已测试，本配置关闭 | Engram CPU 选行预取、pinned staging；小幅差异不足以确认稳定收益 |
| 尚需适配或完整验证 | 全流程 ZOS、共享专家与路由专家重叠、更多图档位、完整有界重放、lazy mHC 跨子层融合、目标词表头的进一步存储分片等 |
| 本次部署不覆盖 | 多机 PD 分离、3FS 前缀缓存、EP64–128 规模下的系统调度与生产负载结果；NVIDIA 专用内核的直接运行 |

上游机制说明参见固定提交的 [DeepInfer 部署问答](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md)。上游生产数字和单算子结果不替代本机同配置实测。
