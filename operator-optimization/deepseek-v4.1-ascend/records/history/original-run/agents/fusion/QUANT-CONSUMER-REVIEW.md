# Loop-006：量化元数据和 Q/KV 共同输入的边界

## 本地量化描述能确定的范围

用 `python3 agents/fusion/summarize_quant_metadata.py`，对本地已保存的 `source172/model-metadata/quant_model_description.json` 核对历史 SHA 后只读取键值，输出 `quant-metadata-summary.json`。40 个 target 层中：

- `layers.{i}.attn.wq_a.weight`：40/40 标记 `FLOAT`。
- `layers.{i}.attn.wq_b.weight`：40/40 标记 `W8A8_DYNAMIC`。
- `layers.{i}.attn.wkv.weight`：40/40 标记 `FLOAT`。

例如第 0 层 Wq_b 的 `weight_scale` 与 `weight_offset` 键也标记 `W8A8_DYNAMIC`；这只说明描述文件的标签，不证明数值、实际执行的 dtype 或内部计算如何使用 offset。当前文件是 **172 资料的本地冻结副本**；本次没有访问 172。stage 084 已核对 169 基线容器 checkpoint 描述文件与之 SHA 一致，详见下文；这仍不能单独证明进程实际加载对象或历史执行分支。`model.py:1859-1860` 明确有 `.attn.`→`.self_attn.` 的权重名称重映射，但量化 scheme 的解析和每个运行时对象类型尚未获得完整证据。

## 对 F05 的收紧

冻结包装源码 `CVLinearWrapper`：无通信且实例识别为 W8A8 Dynamic 时，Q norm 的结果 `qr` 分别传入 `npu_dynamic_quant` 和 `npu_quant_matmul`（详见 `WRAPPER-REVIEW.md`）。本地描述对全部 40 个 Wq_b 指定了动态 W8A8，和该条件一致；也和已捕获目标层的 INT8 矩阵算子 shape/类型证据一致。但**一致不等于证明**该元数据被 169 实际实例解析成此类、该实例没有通信、该窗口采用多流，以及 trace 中每对 norm/quant/matmul 操作拥有同一 tensor。F05 仍为条件候选。

第 0 层 compression ratio=0，`dsa_v41.py:393-395` 的 indexer 准备跳过，因此同层 indexer 不读取 `qr` 数值；index source 层则有 `indexer.project_query(qr)` 消费。非 source CP 分支还读取 `qr.shape[0]`，元数据读取并非读取整块数值。量化、输出和元数据消费者均需保证，未证明其他执行器/alias 的全部消费者。

## 对 F06 的修正：不能笼统称“消掉一次动态量化”

`dsa_v41.py:451-466` 的共享分支需要 `share_inputs`、两个 `_quant_method` 的类型相同、两者通信条件相同。`CVLinearWrapper.quantize` 在没有 W8A8/MXFP8 动态量化方法或有通信时可返回原输入和空 scale。由于本地描述 40 层的 Wq_a/Wkv 均为 FLOAT，**如果** 169 的对应对象确实走 wrapper 的非动态分支，则这里两个 `quantize` 操作并不各自生成 INT8 数据，共享分支可能只是复用输入引用；不能将它记为已经节省了一次动态量化或一次读写。具体对象类型、代码分支与所有别名仍未直接观察；Wq_a/Wkv 权重的 GEMM 输入共享、读取与布局是另外的问题。

## 应向主控补充的精确只读请求

1. 在已授权 169 的目标基线容器，仅获取启动脚本指定 checkpoint 下 `quant_model_description.json` 的文件大小与 SHA-256，与本地 `2f4020a0...3774bf` 比较；不读权重、不改服务、不向 172 发请求。若文件不在该位置，按现有加载日志/配置只读定位；哈希不一致时分别保留并重新判断，不把两套元数据混用。
2. 在 baseline/stage 两容器中以只读目录搜索 `class AscendW8A8DynamicLinearMethod` 和量化配置中将 `W8A8_DYNAMIC` 映射到对应方法的文件；有界回传相应文件路径、SHA 和必要源码，检查加载前后 weight_scale/offset、rounding、`apply`、TP custom_op。无须导入包或执行模型。两个容器内容不同则分别保存。
3. 如果现有 trace 无 tensor/storage 标识，不能靠 task 名称及时间接近确认 F05 的数据链，保持 UNKNOWN；是否需要新 trace 由主控在不占用他人 NPU 的前提下按整体流程判断。

## 追加：169 checkpoint 描述文件与本地快照一致

ZCode stage `operator-ceiling169-v2-084-quant-hash` 退出码 0，执行凭据验证为 `deepseek-flash`。只读结果保存在 `agents/fusion/checkpoint-metadata-comparison.json`：169 基线容器中启动路径所指的 `quant_model_description.json` 与本地参考副本大小均为 8,677,636 字节、SHA-256 均为 `2f4020a0a12daa621cfd2ff9d09a2c73e5674af7c624ef47df5c66796c3774bf`。所以本节对 40 层 Wq_a/Wq_b/Wkv 描述标签的统计，现在可确认为**两处描述文件字节一致**，不再只是未经校验的参考文件。

边界仍然是文件哈希与源码：这不是运行中的量化方法对象、逐次图分支、张量 dtype 或 scale/offset 实际数值的直接观测，也不能排除模型启动之后 checkpoint 文件发生改变。F05 的候选状态不变；F06 关于 FLOAT 标记对应 wrapper 非动态分支的结论仍需解析器和实际实例绑定支持。对历史基线是否读取完全相同的文件，需要当时加载/哈希证据，不能仅凭现在文件相同推定。
