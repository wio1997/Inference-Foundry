# EP8 Indexer：metadata 复制、共享缓冲与调度汇总的绑定边界

继续 `INDEXER-EP8-METADATA-PRODUCER-SEARCH.md`：对已识别的 EP8 第 24 层三个 Indexer，验证六条历史 metadata producer 的连接记录，并检查既存 benchmark 前后匿名指标。只读原 SQLite、指标文件和冻结源码；未启动 NPU，也未读取请求文本或张量内容。

## 重要区别：数据链可以跨不同存储

冻结源码 `sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:1227–1278` 展示了两种 metadata 构造分支：得到 producer 的 `value` 后，均执行 `self._qli_metadata.copy_(value)`；再将 `self._qli_metadata` 以 `qli:c{ratio}` 键交给 `_publish_task`。其 `:879–900` 中，如果同一个 `shared` 字典已有该键，则返回现有 buffer；否则发布该 buffer，按条件将工作加入设备 metadata task 或执行构造函数。`:956–967` 的注释说明 runner 在每次 build 重置共享字典。

因此，**如果历史执行使用此冻结代码及相应分支**，producer 输出与 Indexer 输入可通过一次 copy 建立数据关系，不要求是同一 storage；反过来，仅看到相同 shape、地址或一个 Python buffer 名称，也不足以证明其内容对应某次 replay，因为 buffer 可能共享、复用或被后续构造覆盖。需绑定 producer 的本次输出 → copy 的输入与目标 → 已发布 buffer → graph/replay 中该次 Indexer 实参，并核验事件、完成、复用与生命周期。冻结源码 SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`；历史进程真实加载源码、二进制及运行分支尚未绑定，以上不是对历史实际数据流的断言。

## 已保存连接不能闭合该链

此前选出的六个 producer launch connectionId 为 78002、78017、158316、158331、164003、164018；这次各在 `CONNECTION_IDS` 表找到一条关联。三轮 Indexer 共享的设备 `connectionId=184330` 在此表查得零行，且目标 `globalTaskId=19466` 在 TASK 中重放复用；没有找到从上述六个 producer 到指定 Indexer 的连接记录。CANN 9.1 [官方 PyTorch profiler 表说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_1145.html)将 `CONNECTION_IDS` 描述为框架 API 与 CANN/task 等关联，而非张量内容或存储身份；9.2 的历史 exporter 实现尚未独立绑定。故这既不能证明不消费，也不能证明消费或推算任何长度。

目标和候选 producer 分别用精确设备起止与 `globalTaskId` 确认，具体输入形状、前后时间和 host launch/API 见前一份报告；这次没有用时间邻近当作数据依赖。

## 既存调度匿名汇总的限制

只读 `diagnostic/runtime/profile-client-results` 的 `measure1-metrics-before.txt`、`measure1-metrics-after.txt`，只提取四组固定 metric 的标签**名称和行数**，不读取/输出请求或标签值。两份都包含 `vllm:iteration_tokens_total_bucket`（各 28 行），其标签名集合为 `engine`、`le`、`model_name`；相关 count/sum 使用 `engine`、`model_name`。所查 `num_requests_running`、`request_prompt_tokens` 和 `prompt_tokens_total` 字段也没有 replay、层号或请求标识标签。它们是两次汇总快照，不能将直方图中任意一次 iteration 的 token 数归属于这三个 Indexer 调用，更不能提供各次压缩 key 长度或候选数。这仅覆盖两份文件、上述固定指标，不排除未核查的其他材料含有匿名逐次记录。

**结果：三个 Indexer 的有效 Q/K 长度和动态 metadata 值仍为 UNKNOWN。** 最小需要的是可验证的同进程 batch/replay 标识、上述 copy 和缓冲生命周期、受控匿名长度或分块汇总；此外需核验采集到的真实代码/二进制和 metadata 版本。采集/运行约束未变化，本次没有实施新的诊断。

复现：`audit_indexer_ep8_binding_metrics.py` 以 SQLite 只读查询固定连接值并核验 DB hash；仅对两份指标文件按固定 metric 白名单统计标签名，保存 `indexer-ep8-binding-metrics.json`，其中包含文件 SHA 而不包含指标值、标签值或请求内容。
