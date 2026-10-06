# 历史复用与避免重复

checkpoint158之后按[AGENTS](AGENTS.md)使用本页。以下历史事实、Run引用和当时下一候选原样保留；它们是证据，不是后续必须执行的路线。历史失败/假设可挑战，新解释另记，不改旧raw/裁决。Full Replica的prefill-aware、prefill-work、work-seconds只保留为研究知识，进入最终产品必须重新裁决，不叫标准PD优化。

现有仓库已含133条`performance_knowledge/entries.jsonl`，以及TaskCtl、evidence、RESULTS和Framework Bound。复用这些索引和持久化机制，不重新总结全部历史或另造同内容数据库。

## 候选检查

昂贵执行前按机制/消费者/失败原因检索旧知识与[当前点索引](records/points.jsonl)，先读少量相关摘要，再按需要打开raw。根`performance_knowledge/README.md`及`scripts/performance_knowledge.py`可用；不要为一次检索默认sync整个私有历史。直接rg本地知识目录也是有效入口。

在优化点简短记录：旧source/immutable ref、环境和真正证明的事实；当前关键差异；旧证据是否足够；本次实验新增哪条决定性信息。没有相关记录也只记一句。若索引遗漏，按实际证据修正。

| 类型 | 用法 |
|---|---|
| 通用方法/记录工具 | 直接沿用；无需为“同样方法可用”重跑性能 |
| 条件相容的旧事实 | 引用原artifact与适用范围，避免重复采集；新归约可离线完成 |
| 稳定buffer、metadata、缓存、Graph、设备状态等机制 | 先映射GLM实际路径/语义/已实现能力，只验证差异与未知 |
| 旧失败/反例 | 复用失败原因和边界；同条件无新增信息不重跑，条件改变时做区分性验证 |
| DSpark专属状态机/计数/固定组织 | 不直接移植到GLM MTP；仅借问题与证据方法 |

DeepSeek与GLM、910B3与910C、W4A8与W8A8、TP8与TP16及不同PD/Graph的旧TPS不可直接迁移。KEEP不等于GLM KEEP；REJECT也不否定所有新条件。模型IndexShare/KVShare已经存在时计入Current，先核对框架是否落地，避免把已有节省重复计为新收益。

## 初始映射

[复用索引](records/reuse.jsonl)是指向已有来源的适用性映射，既非复制完整知识库，也非GLM实测成果。

- Run673–675：输入编码/初始prefix-hash/metadata Graph，GLM可核对重复准备与key有效范围。
- PK-080、PK-122：资源竞争和局部收益未转E2E的反例，用于选择测量，不把数值搬来。
- PK-120/121/123/127：提交/设备ready、观察器、跨rank等待与轨迹可比性，复用证据纪律。
- DSpark专属执行代码：明确不直接port；GLM MTP重新识别消费者与提交。
- 旧GLM7月P端profile：模型线索可复用，TP8/DP2且版本/量化/关联D缺失，不能充当当前0.27基线。

若同机制已有有效GLM Run，后续优先引用；只有新软件/布局/负载/状态路径或旧结论缺项会影响判断时才重验。实质重复且无新增信息时标`REUSED`，不创建伪新性能成果。


## 2026-10-01 native MTP动态临时内存边界

Run8/9建立两个合法E2E反例：Graph/MTP3、budget16384、gmu0.92能初始化并完成短本地请求，但mixed D与原生PD长输入P的MTP draft/DCP临时buffer分别520MiB和1.13GiB OOM。启动profiling/KV分配并非覆盖所有动态draft峰值；失败配置不当硬件极限。D回到native budget128后Run9完整router能力通过，长prefill延迟明显依赖cache/调度预算。降低P gmu0.87是Run11待验证候选，同时减少KV缓存容量，不能预先宣称吞吐收益。复用时按actual argv、native原始错误/usage和所有尝试裁决。

原生HTTP200可能携带SSE error500并DONE，不能当成功或仅凭HTTP access/lease释放判后端健康；Run8真实错误报文已进入有界observer/zero-credit合同。原样转发、无重试；5秒cooldown仅临时隔离，压缩/超限观测unknown，永久native健康监督仍未实现。证据见GLM-OPT-0002 Run8/9和inband_contracts.json。
