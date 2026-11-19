# D 侧 GMM：专家工作量能否从现有资料恢复

## 当前判断

**不能用当前已检查的归档判定 rank0 的真实专家负载。** 对 [G240 诊断](018-matched-SG-all16-model-profile.md)，D 侧 G181 readiness、G182 和 G204 三份更早的服务配置快照中，`expert_distribution_recorder_mode=null`、`expert_balancedness_report_mode=off`。G240 是同一 G181 服务 epoch，但这些是采集前的快照，不证明 G240 期间绝无动态变更或其他独立采集。

[D 原始文件双扫索引](../evidence/matched_SG_G181/profile_G240/CPU_d/review_g252/raw_stability.json)中有 1207 项，按文件名检查无包含 `expert` 或以 `.pt` 结尾的条目；已检查的控制及 rank 解析摘要也没有逐次 group-list 实值。**这只覆盖该索引和已检查的本地资料，不证明服务器其他目录绝不存在数据。** 本次 profiler 的控制脚本设 `record_shapes=False`，它是 CPU 侧采集配置，不否认设备 CSV 已有的 shape 字段；shape 字符串也不是张量内容。三个快照和索引身份可用[检查脚本](../scripts/verify_expert_045.py)复算。

## 静态源码中的候选计数路径

固定底座 [SGLang 上游 commit](https://github.com/sgl-project/sglang/tree/8a0edcf799d06bdee8832356d382495e7fb50f22) 的 [DeepEP dispatcher](https://github.com/sgl-project/sglang/blob/8a0edcf799d06bdee8832356d382495e7fb50f22/python/sglang/srt/layers/moe/token_dispatcher/deepep.py) 将低延迟接收计数作为 `masked_m` 返回；[Ascend runner](https://github.com/sgl-project/sglang/blob/8a0edcf799d06bdee8832356d382495e7fb50f22/python/sglang/srt/layers/moe/moe_runner/ascend.py) 的 `deepep_ll` pre-permute 将其转换为 int64 `expert_tokens`，后续传给 GMM。仓内 [W8A8 量化方法](../evidence/profile_mechanism_G287/SG_source/sglang/srt/hardware_backend/npu/quantization/moe_methods.py) 和 [GroupedMatmul 包装](../evidence/profile_mechanism_G287/SG_source/sglang/srt/hardware_backend/npu/moe/matmul.py)显示 `expert_tokens` 最终作为 `group_list` 传入算子。

上述三个上游文件与检查过的本地固定底座副本逐字节 SHA 一致；这是**静态可能路径**，没有 G240 每次事件到具体函数的关联或全部现场源码逐文件证明，不能据此反推历史每个 `group_list` 的值或实际专家不均衡。

## 再采样需注意的两个陷阱

固定底座的 [expert_distribution.py](https://github.com/sgl-project/sglang/blob/8a0edcf799d06bdee8832356d382495e7fb50f22/python/sglang/srt/eplb/expert_distribution.py) 在 NPU+DeepEP 的非 `per_token` 分支选择按有效 TopK 路由统计，不是直接 dump 已进入 GMM 的 `masked_m/group_list`；`per_pass` 收集数据后有设备到 CPU 的 clone，并按 rank 落文件。因此采集开销、图捕获/重放语义、padding 与各层 pass 的对应关系都需实验校验，不能把 TopK 分布直接当 GMM 的实际工作量。另一个源码分支的 `per_token` 会复制 `input_ids`、`positions` 和 TopK 信息，**未经授权与脱敏不得直接启用或发布原始录制**。MTP draft 的记录覆盖也不能仅据 target 分支推定。

厂商公开的 [msopprof 字段说明](https://github.com/Ascend/msopprof/blob/ce5a1158cebf0ba3b9f8a48793304660c22ece36/docs/zh/user_guide/msopprof_performance_data.md)中，MTE1 指令描述为 L1→L0A/L0B 数据搬运且不包含搬运等待，MTE2 指令描述为 GM→AI Core 搬运。该文件描述的是 msopprof 模式，不是已核对的本次 CANN 9.1 `kernel_details.csv` 同版本字段契约，**只能用于理解概念，不能当作本次字段已逐项校准**；不同 pipe 可能重叠，更不能用 MTE2 ratio 宣称 HBM 带宽瓶颈。

## 最小补证

1. 先寻找同一 epoch 的合法现存组列表和 profiler 事件关联，不对当前服务操作；无就明确缺失。
2. 仅在授权且隔离的窗口中，先核对运行源码和图模式；以受控、无真实请求文本的数据验证 per-pass TopK 计数与实际接收 group-list 的差异、MTP 覆盖及探针开销。
3. 取得同一层/forward、实际 group-list、有效 token、权重布局与单事件耗时后比较八 rank；再用单变量 A/B 检查正确性、TTFT 和吞吐。跨 rank 先建立步骤映射，不能按各自序号假装同步。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_expert_045.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

此页没有发新请求、没有改服务，也没有新的收益或根因结论。
