# F09 数值证书与 169 现有 trace 的对应关系

## 初步结论

现有记录能确认冻结源码定义了 F09 非 packed 量化路径、该路径的源码级 scale 消费者，以及 profile 汇总记录了同名设备任务；但尚不能仅凭这些事实把证书的 BF16 输入、具体有效行数、在线初态或 bit-exact 中间 scale 输出要求证明为 169 整模型的普遍条件。更不能把证书中的条件 probe 次数当作当前 HBM 流量。

## 源码与配置能说明什么

`agents/fusion/frozen-source/vllm_ascend/ops/triton/quantize_indexer_query.py:40-69` 对进入该 helper 的张量检查三维且最后一维为 128，并按源码分配 INT8 同形量化输出和形状为前两维的 FP16 scale。**源码没有断言输入 dtype 是 BF16**；模型配置的 `dtype=bfloat16` 也不能单独证明某次 F09 投影输出的实际 dtype、输入值或有效行数。

`sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py:50-55,167-179` 根据配置设置 heads/width，project 后将末维拆为 `(n_heads,width)`，`packed_cache_ops is None` 时 `quantize_query` 才选择 F09，否则选择另一个量化路径。冻结 `evidence/model-config.json` 给出的 `index_n_heads=32`、`index_head_dim=128`，因此该配置真正绑定到该实例且 F09 分支实际运行时，源码预期逻辑形状为 `[有效或捕获 token 行数,32,128]`；第一个维度的具体值仍须从实际调用确认。这个逻辑推断不等于逐任务 tensor 身份。

同文件 `indexer.py:254-280` 在非 packed 正常路径中，把 `quantized_query` 和 `query_scale` 作为参数传给 `npu_quant_lightning_indexer_v3`；若此前没有准备量化结果，就在这里调用 `quantize_query`。`attention/dsa_v41.py:394-430,582-592` 显示当前源码还允许提前在辅助流计算并通过事件传递这两个张量，消费前主流等待事件。故**源码级消费者关系有条件成立**，但已采集 task 的量化输出与某次 QLI 的输入张量地址、具体层绑定和下游是否对 scale 差异敏感未由该源码 alone 证明。

helper 检查的是输入张量 shape；graph capture 的缓冲维度、padding 和具体有效 token 数可能不同。空 query、空 cache、packed 模式均有不同分支，不可把任一窗口存在 QLI 推断为所有情况下都执行同样量化。

## 已保存 trace 的证据

`evidence/TRACE-SUMMARY.json` 对 16 rank 均出现 `_quantize_indexer_query_kernel` 同名类型，分别 72 或 80 条、合计 1200 条；这份是 observer-affected 的整段 profile 汇总，名称统计本身不提供逐条 shape、dtype、请求、执行分支或 tensor 对应关系，不等于完整服务中的调用数。源码同名是对应线索，不自动证明某条历史事件执行了当前快照的某一次调用。

`agents/diagnosis/indexer-runtime-metadata-ep8.json` 的 EP8、target 层 24、三次 source-attributed 内部采样窗口中，每次 QLI 记录的输入 shape 序列含 `3,32,128` 与两个 `3,32`，dtype 序列含 INT8 和 FLOAT16。它们与源码所期待的量化 query 与 scale 形状/类型**相容**；但该导出未附逐参数命名/张量标识，不能仅凭形状把某一项直接命名为 F09 scale，不能得出 F09 输入为 BF16，也不能把固定形状 3 当作每轮实际有效 query 数。更不能把这三次窗口关联到具体原始请求或明确 prefill/decode 阶段；这三次是已归属的 target 图采样窗口，非对完整端到端负载的阶段计数。

## 对条件证书的判断

`agents/scheduler/F09-NUMERIC-DEPENDENCY.md` 的证明明确要求 scale bit-exact 可观察、指定的 FP32/FP16 舍入和不冲刷最小 subnormal、指定输入集合以及在线初态不能预存 query 摘要。CPU witness `f09-single-coordinate-witness.json` 核对的是这些假设下的算术，不是 NPU 精度观测。当前缺具体历史调用的输入 dtype/值、查询输出的 tensor ID、精度实现、真实请求是否可取所需 129 个独立输入、window 初态以及最终下游观察契约；因此既不能认定条件已满足，也不能从未观测到数据断言条件不成立。

作为理论算子接口的**条件证书**仍有效；面向 169 本次真实 workload，它尚不能转成已证实的逐 head 必要读数、物理内存流量、时间或 TPS 界。

## EP8 原始 CSV 补查：部分 shape 和同窗线索

ZCode stage `operator-ceiling169-v2-105-f09-trace` 退出码 0，`model_verified=true`，模型 `deepseek-flash`。只读脚本 `query_f09_ep8_trace.py` 扫描了 EP8 已保存 kernel CSV，结果 `f09-ep8-trace.json` 的 SHA-256 为 `d3f772093d1102355c084e5f806e99c008ffd18ca27c149665b31ed1e3d05636`。脚本和窗口 SHA 与本机一致，kernel CSV 哈希 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7` 与冻结清单一致。没有设备运行。

- EP8 整段 profile 的 80 条 `_quantize_indexer_query_kernel` 同名事件中，stream 35 的 24 条报告输入 dtype `DT_BF16`，shape 分别为 `[3,32,128]`、`[800,32,128]`、`[834,32,128]`，各 8 条；输出报告 INT8 同形和 FP16 `[行数,32]`。这些记录有明确导出 shape/dtype，但尚未关联到具体请求、prefill/decode、F09 源码实例或目标层，不能拿来代替下面的目标窗口。
- 选定的 24 个 target 图窗口为 EP8 的三次采样 × 八个 index-source 层。每个窗口中有一条主流 `VllmAscendQuantLightningIndexerV2`，并恰好有一条同名 F09 事件与窗口相交；该 F09 事件记录在 stream 87，结束时间均早于窗口内 stream 89 的 QLI 开始。对应 F09 记录的输入、输出 shape 和 dtype **均为 `N/A`**，而 QLI 本身报告了 INT8 `[3,32,128]` 和 FP16 `[3,32]` 等输入项。这支持源码所述先量化、后使用的**时序线索**，但仅凭同窗、先后及形状相容，不能证明这两条事件的 tensor 是同一对象，也不能确定 F09 的有效输入 shape 或 BF16 dtype。
- 80 条 F09 同名事件均显示 model ID `4294967295`；不能把它解读为已知请求或具体 target 图身份。24 个目标窗口自身属于之前按源码角色归属的 target 图采样；每个 F09 跨流事件的实际请求阶段、计算资源、graph 调用归属和量化精度仍需更强证据。

因此，对“本次服务是否有 BF16 F09 同名样本”可以更具体回答：**EP8 的 profile 中有报告 BF16、形状 `[3,32,128]`、`[800,32,128]`、`[834,32,128]` 的同名事件；而在这 24 个 target 图窗口中，F09 事件的 shape/dtype 仍未导出，无法确定是 BF16。** 不论哪一组，F09 证书所要求的 bit-exact 中间观察、候选输入可达、初态及受限读取模型均未由 trace 证明。
