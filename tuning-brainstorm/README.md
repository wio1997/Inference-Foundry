# 调优头脑风暴

集中保存 SGLang、vLLM 与 Ascend 推理链路的优化想法。先把现有做法、候选改法、收益来源和代价讲清楚，再用同条件实验判断是否值得实现或保留。

本目录中的初始记录均为**待验证假设**。源码定位说明某段工作确实存在，不代表它已成为当前负载的瓶颈，也不代表候选改法已经有效。

## 从这里开始

- [SGLang 36 机制任务：逐点讲解、最小实验与物理上界](sglang-pd-spec-36/README.md)（PD＋推测解码；安装审计、M035局部CPU候选与M017/M018目标镜像CPU反证、M021形状策略反例、M019布局/包装层、M001前处理候选CPU结果、M031接受合同/metadata CPU组件、M025 NPU seed状态模型、M023/M024条件KV/index接口、M026/M028 MoE状态与W8加载方法及M030工作边界、M005/M006 PD prefill预算/提交状态、M009/M010/M011 默认 unified cache 的 300 项 CPU 合同已记录；PD服务器基线待资源分配）
- [PD 分离下的 chunk 优化：9 项价值筛选、6 条候选与 2 个配置对照](2026-10-18-sglang-pd-chunk.md)
- [缓存管理逻辑排查总表：21 个新增检查项、12 个排除项、48 格覆盖](2026-10-18-sglang-cache-audit.md)
- [可筛选的离线排查地图](diagrams/cache-audit.html)
- [第一份笔记：SGLang 前缀缓存与 CPU 调度开销](2026-10-18-sglang-prefix-cache.md)
- [减少重复工作：共享路径锁维护与插入游标（008 / 009）](2026-10-18-sglang-equivalent-work.md)
- [两处候选的交互图解](diagrams/equivalent-work.html)（教学操作量，非服务性能实测）
- [可操作图解：当前 LRU 做法与候选改法](diagrams/lru-current-vs-candidate.html)（HTML 下载后可在本地浏览器打开；GitHub 文件页显示源码）
- [新增记录模板](TEMPLATE.md)
- [记录约定](AGENTS.md)
- [仓库调优方法](../FOUNDRY_METHOD.md)

## 想法索引

| ID | 想法 | 希望减少的成本 | 当前状态 |
| --- | --- | --- | --- |
| [CACHE-001](2026-10-18-sglang-prefix-cache.md#cache-001) | 持久维护淘汰候选，减少反复建堆 | 淘汰时 CPU 开销 | 已有相关 PR #38371；目标适用性待验证 |
| [CACHE-002](2026-10-18-sglang-prefix-cache.md#cache-002) | 批量回收，保留空槽余量 | 淘汰启动次数 | 待验证假设 |
| [CACHE-003](2026-10-18-sglang-prefix-cache.md#cache-003) | 抽样近似 LRU | 候选检查量 | 待验证假设；会改变淘汰选择 |
| [CACHE-004](2026-10-18-sglang-prefix-cache.md#cache-004) | 对照已有 LFU / SLRU / priority / T-LRU | 后续未命中与重复前向 | 已有策略；适用性待验证 |
| [CACHE-005](2026-10-18-sglang-prefix-cache.md#cache-005) | 合并同批共享路径的访问记录刷新 | 重复路径维护 | 源码已定位；时序与语义待验证 |
| [CACHE-006](2026-10-18-sglang-prefix-cache.md#cache-006) | 简单缓存场景的轻量匹配路径 | Python 分配与回调 | 源码已定位；改法待验证 |
| [CACHE-007](2026-10-18-sglang-prefix-cache.md#cache-007) | 减少可避免的索引拼接与复制 | 索引分配、复制和小操作提交 | 已有相关 PR #42923 / #43023 / #43600；剩余点待验证 |
| [CACHE-008](2026-10-18-sglang-equivalent-work.md#cache-008) | 已有 FULL 引用时省掉重复集合删除尝试 | 共享锁路径的 Python 集合操作 | 有限状态对照通过；完整正确性 / 性能待验证 |
| [CACHE-009](2026-10-18-sglang-equivalent-work.md#cache-009) | 插入遍历使用 offset，减少临时 token 后缀复制 | 长输入、多节点的 CPU 复制 | 纯遍历对照通过；完整插入状态机待验证 |
| [CACHE-010](2026-10-18-sglang-cache-audit.md#cache-010) | 缓存关闭后，先返回再切输入 | 关闭缓存路径的一次 CPU 输入复制。 | 等价方向；未测真实推理性能 |
| [CACHE-011](2026-10-18-sglang-cache-audit.md#cache-011) | 分裂取页键时，避免临时复制整个后缀 | 一次临时 token 后缀复制与 RadixKey 对象。 | 等价方向；未测真实推理性能 |
| [CACHE-012](2026-10-18-sglang-cache-audit.md#cache-012) | 已对齐的键，避免再复制一次 | 重复的全长 token 复制。 | 条件候选；未测真实推理性能 |
| [CACHE-013](2026-10-18-sglang-cache-audit.md#cache-013) | 合并 checkpoint 前的多次输入裁剪 | 进入插入遍历之前的重复全长 / 前缀复制。 | 条件候选；未测真实推理性能 |
| [CACHE-014](2026-10-18-sglang-cache-audit.md#cache-014) | FULL 匹配只记长度，减少临时段列表 | 列表 append、切片与末尾再遍历。 | 条件候选；未测真实推理性能 |
| [CACHE-015](2026-10-18-sglang-cache-audit.md#cache-015) | 树上子节点只查一次字典 | 一次重复字典查找；长 page 键还涉及重复哈希。 | 等价方向；未测真实推理性能 |
| [CACHE-016](2026-10-18-sglang-cache-audit.md#cache-016) | 没有主机缓存时，省去主机叶子集合维护 | host 检查、方法调用和空集合删除尝试。 | 条件候选；未测真实推理性能 |
| [CACHE-017](2026-10-18-sglang-cache-audit.md#cache-017) | FULL 锁入口避免组件内外重复刷新 | 重复资格检查与集合操作。 | 条件候选；未测真实推理性能 |
| [CACHE-018](2026-10-18-sglang-cache-audit.md#cache-018) | 重叠段确实有消费者时，再切索引视图 | 临时 Tensor view、框架小操作与默认回调。 | 条件候选；未测真实推理性能 |
| [CACHE-019](2026-10-18-sglang-cache-audit.md#cache-019) | 独立索引副本是否可延后到树实际收养时 | 中间的全长索引复制；不删除树长期持有的独立副本。 | 条件候选；未测真实推理性能 |
| [CACHE-020](2026-10-18-sglang-cache-audit.md#cache-020) | 同一次容量检查，没有变化就少查一次 | 重复的 allocator 元数据查询。 | 等价方向；未测真实推理性能 |
| [CACHE-021](2026-10-18-sglang-cache-audit.md#cache-021) | 关闭淘汰指标时，省去起始计时 | 无指标路径的时钟调用。 | 等价方向；未测真实推理性能 |
| [CACHE-022](2026-10-18-sglang-cache-audit.md#cache-022) | free group 只有一段时，省一次 cat | 单段输出分配、复制与提交。 | 条件候选；未测真实推理性能 |
| [CACHE-023](2026-10-18-sglang-cache-audit.md#cache-023) | SWA 已回收区间为空时，跳过逐段拆分 | 逐段 min/max、临时列表和重复分类。 | 等价方向；未测真实推理性能 |
| [CACHE-024](2026-10-18-sglang-cache-audit.md#cache-024) | 全是空回收区间时，先返回再索引请求行 | 空回收的 Tensor 行选择、视图与区间准备。 | 条件候选；未测真实推理性能 |
| [CACHE-025](2026-10-18-sglang-cache-audit.md#cache-025) | 维护设备驻留子节点数，减少反复扫描 | 高扇出且很多子节点不驻留时的扫描。 | 有取舍；未测真实推理性能 |
| [CACHE-026](2026-10-18-sglang-cache-audit.md#cache-026) | 只读匹配按结构版本复用 | 同一输入、同一结构版本的重复匹配。 | 有取舍；未测真实推理性能 |
| [CACHE-027](2026-10-18-sglang-cache-audit.md#cache-027) | 复用固定页内偏移向量 | 固定常量的分配与小操作。 | 条件候选；未测真实推理性能 |
| [CACHE-028](2026-10-18-sglang-cache-audit.md#cache-028) | FULL-only 叶子资格走直接字段判断 | 生成器与未启用组件槽的检查。 | 条件候选；未测真实推理性能 |
| [CACHE-029](2026-10-18-sglang-cache-audit.md#cache-029) | FULL-only 节点减少辅助对象与动态字典 | CPU 堆对象、内存占用和可能的缓存未命中。 | 有取舍；未测真实推理性能 |
| [CACHE-030](2026-10-18-sglang-cache-audit.md#cache-030) | 持久节点复用已经算出的首 page 键 | 重复页切片、整数装箱与 tuple 构造。 | 有取舍；未测真实推理性能 |

## PD 分离与 chunk 想法索引

本组仅讨论独立 P/D 资源；源码审阅不代表用户 Ascend PD 已取得收益。优先检查真实交接等待，再选择是否修改 CPU 准备、粒度或准入。

| ID | 想法 | 希望减少的成本 | 当前状态 |
| --- | --- | --- | --- |
| [PD-CHUNK-001](2026-10-18-sglang-pd-chunk.md#pd-chunk-001) | Ascend 连续区间只保留起点与长度 | 页号装箱、中间列表和描述符准备 | 源码已定位；已有通用 PR #45003；CPU描述符等价/成本已测，NPU/PD待验证 |
| [PD-CHUNK-002](2026-10-18-sglang-pd-chunk.md#pd-chunk-002) | 检查 KV 就绪后的发送等待和交接空隙 | 未被计算隐藏的等待 | 已有重叠机制及 PR #43811；待建立目标时间线 |
| [PD-CHUNK-003](2026-10-18-sglang-pd-chunk.md#pd-chunk-003) | 在后端约束下选择计算块与传输段 | 提交固定成本或传输尾巴 | 有取舍；Ascend / staging 限制已记录；待基线 |
| [PD-CHUNK-004](2026-10-18-sglang-pd-chunk.md#pd-chunk-004) | 续块重复准备的 PD 适用性检查 | 切片、索引与组批重复工作 | 入口已定位；关联 CACHE-013 / 019；具体冗余待确认 |
| [PD-CHUNK-005](2026-10-18-sglang-pd-chunk.md#pd-chunk-005) | 对照 D KV 预留时机 | 未完成 P 的请求占住 D 容量的时间 | 现有策略 / 后端适配研究；非普遍可用开关 |
| [PD-CHUNK-006](2026-10-18-sglang-pd-chunk.md#pd-chunk-006) | 针对已证实积压的有界准入反馈 | 在途 KV 滞留、竞争与无效重试 | 条件性策略；先证明已有控制不足 |

## 怎样继续记录

1. 复制 `TEMPLATE.md`，按 `日期-主题.md` 命名；同主题可在原笔记中增加有独立 ID 的条目。
2. 优先用白话和一个小例子解释：当前怎样做，准备改变哪一步。
3. 明确会省什么、会新增什么成本，适合和不适合什么请求流。
4. 记录源码固定提交、实际运行版本与证据位置。未测的微秒数和加速比留空。
5. 有实验后追加结果与适用条件，保留被否定的想法，并更新本索引。

建议状态：`待查源码`、`源码已定位`、`待建立基线`、`待实验`、`已验证`、`已否定`、`暂缓`。已有实现与候选新实现分别标注。

确认有效的结果按目标研究线规则进入对应实验记录或 `performance_knowledge/`；本目录保留思路和证据入口。
