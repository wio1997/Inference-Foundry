# EP9 选定 MC2 的路由与模型输入元数据

## 可确认的记录

沿用 [EP9 调度与物理 M 记录](EP9-SCHEDULER-MC2-WORKLOAD-SCOPE.md) 中的 752-50 runner、752-51 target 第 0 层 scope 和摘要 `8c18c293…`，检查已保存 `events-752.jsonl` 和经 trace flow 唯一关联的设备任务：

| 同 scope、同 stream 47 记录 | 导出输入 | 导出输出 |
| --- | --- | --- |
| `MoeGatingTopKHash`，task 46754 | `[643,384] FLOAT; [384] FLOAT; [643] INT64; 未定义; [384] FLOAT` | `[643,6] FLOAT; [643,6] INT32; [643,384] FLOAT` |
| `DispatchFFNCombine`，task 46755 | 含 `[643,6] INT32`、`[643,6] FLOAT`、`[643] BOOL` | `[643,5120] BF16; [24] INT32` |

前者设备记录结束到后者开始的 trace 间隔约 **1.080 µs**。流内顺序以及两组 `[643,6]` 的 shape/dtype 相容，为继续寻找路由生产者—消费者关系提供线索；**没有 tensor 地址、storage ID、实际值或已绑定的历史 ABI 槽位，不能证明是同一张量或直接数据依赖。** `[643] BOOL` 只是一个导出输入的形状和类型，不能直接证明它是当次有效 mask、该 mask 的值或 true 的数量。`[24] INT32` 输出也不能凭 shape 唯一认定是每专家计数；冻结源码中有相容的输出契约，但历史实例和值尚未绑定。

752-50 的 sidecar `tensor_metadata` 列表为空；752-51 的记录只含三个 tensor 描述符：`[5144] INT64`、`[643,4,5120] BF16`、`[643,4] FLOAT`，没有内容。采集器的冻结 `_tensor_metadata` 只检查前 8 个 positional 参数中的 tensor 并紧凑追加 shape/dtype/device，不保存原参数位置，kwargs 不在该摘要；因此不能单凭三个数组位置将它们认作指定模型参数或关联到 MC2 的 mask、专家输出。这是该次 sidecar 的有限事实，不证明其他未检查的数据源无内容。

## 批次和工作量判定

runner 记录的 `scheduled_total=5144` 是主机侧 scheduler 统计；MC2 物理首维为 643。它们与上表 gating shape 一致地描述了一次有界的导出记录，但无法单靠这些元数据判定此层具体样本是 prefill、decode 或混合，也不能推出每个 token 的 expert ID、有效行数、mask 或各 expert 接收量。调度数、物理 M、层输入 shape 及 `[24]` 输出之间看似匹配不构成数值绑定；host 与设备活动仍可能异步，采集受到 profiler 干扰。

官方 [CANN 9.1 beta2 aclprofEventAttributes/aclprofTensorInfo](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_1371_2.html) 定义了 tensor 类型、格式、dtype 与 shape 的元数据字段，未列出元素值；该不同版本结构不证明当前采集文件格式或全局无其他数据。对 EP3 指定 raw tensor_info 使用已安装解析器的实证及目标绑定限制见 [tensor_info 契约](MC2-EP3-TENSORINFO-CONTRACT.md)。

## 复算

在既有 stage 容器运行 `python3 agents/diagnosis/query_ep9_gating_metadata_binding.py`，得到 [ep9-gating-metadata-binding.json](ep9-gating-metadata-binding.json)。脚本校验固定目标 JSONL 和 EP9 metadata 的 SHA，核对 scope、digest、同一 pid/tid、两个任务及 stream 唯一性，仅输出 shape/dtype 和导出起止；不读取模型权重、请求内容或 tensor 值。该结果不能确定低活动原因，也无法替代真实同次专家输出及阶段计时。未启动新 NPU 负载或修改模型实现。
