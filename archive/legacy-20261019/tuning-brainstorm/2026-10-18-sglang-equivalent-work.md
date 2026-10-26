# 2026-10-18｜减少重复集合维护与 token 后缀复制

## 结论与边界

记录两个具体候选，优先追求行为等价并删除重复工作。**均未修改生产 SGLang，未进行真实推理性能对照，不能保证端到端收益非负。**

- 固定上游提交：[`57db5089c207ae4ce189cc092a89a2027c8aea29`](https://github.com/sgl-project/sglang/tree/57db5089c207ae4ce189cc092a89a2027c8aea29)。
- 范围：Python UnifiedTreeCore / FULL 缓存。不会把旧 GLM W8A8C8 教学镜像、普通 W8A8 与 Flash 研究线混成一个基线。
- [交互图解](diagrams/equivalent-work.html)：第一个看引用数与重复删除尝试，第二个看复制剩余输入与移动位置标记。
- [有限对照结果](checks/2026-10-18-equivalent-work/validation.json)、[检查脚本](checks/2026-10-18-equivalent-work/check_equivalent_work.py)、[源码版本与哈希](checks/2026-10-18-equivalent-work/source-version.json)。
- 脚本从固定源码提取真实方法 / RadixKey，但外围树和存储对象为轻量替身；游标只验证匹配轨迹与最终后缀，不验证完整插入副作用。结果不是框架测试或 benchmark。

## CACHE-008

**仅在 FULL 引用数从 0 变 1 时，尝试移出可淘汰叶子集合**

- **状态 / 类型**：源码已定位、有限状态对照通过；等价代码候选，完整集成与性能待验证。
- **当前**：`FullComponent.acquire_component_lock` 沿路径增加每个节点的 `lock_ref`；不管增加前是否已经大于 0，都执行 `evictable_device_leaves.discard(cur)`。
- **不变量**：`_is_device_leaf` 排除任意组件 `lock_ref > 0` 的节点。因此集合状态一致时，一个 FULL 已加锁的节点肯定不在可淘汰叶子集合。
- **候选**：将上述 `discard` 移到已有 `if cd.lock_ref == 0` 分支。引用数、保护量账本与其他处理照常执行。

```python
# 候选示意；不是已合入的实现
if cd.lock_ref == 0:
    # 原有保护量转账……
    self.tree_core.evictable_device_leaves.discard(cur)
cd.lock_ref += 1
```

- **白话例子**：A 已保护公共前缀；B、C 又使用它时，引用数仍必须从 1→2→3，但可以省下对同一集合的重复删除尝试。
- **收益来源**：共享路径已有引用时，少一次 Python 方法调用与集合查找。与 LRU 排序、命中率、KV 所有权的变化无关。
- **新增成本 / 回退**：不增加新分支，复用既有零引用判断，但改变分支内执行布局；浅树、低共享或极少锁操作时收益可能不可见。不能用操作数直接推出耗时严格不增。
- **正确性条件**：依赖集合与引用数不变量；保留锁的每次递增、递减、分裂后保护传播及多组件约束。该改动不修复本就不一致的集合状态。
- **有限验证**：提取生产 FULL acquire / release 方法，对候选仅移动一条 `discard`。4 组外围替身状态覆盖共享路径、设备已淘汰的叶子尾段、额外组件保护标志，累计 4080 次锁状态事件；每步引用数、叶子集合、保护 / 可淘汰量与 delta 相同。未验证真实多组件生命周期、并发插入 / 分裂。
- **教学计数**：4 个公共节点 + A/B/C 三个独立叶子，三次 acquire 循环删除尝试 15→7。只计 FULL acquire 内循环，外围刷新仍有工作；不是整条请求路径操作量减少比例。
- **最小下一步**：真实缓存单测覆盖 split / session / auxiliary / pin；随后固定树深、共享率与并发，测总 acquire / release CPU 时间，再确认是否缩短设备等待。
- **否定条件**：任何状态 / 回收资格差异；完整 CPU 时间不下降，或收益过小、不值得维护。
- **源码**：[FULL acquire](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/components/full.py#L307)、[叶子资格](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L2264)。

## CACHE-009

**把匹配已有的 offset 读法推广到插入遍历，减少临时 token 后缀复制**

- **状态 / 类型**：源码已定位、纯遍历轨迹对照通过；插入状态机候选，生产集成与性能待验证。
- **当前**：普通 `_match_prefix_helper` 已使用 `child_key_at` / `match_at` 和偏移；但 `_insert_walk_step` 每步仍执行 `state.key = key[prefix_len:]`。固定版本 `RadixKey.__getitem__` 会切片底层 `array('q')`，复制剩余 token ID 并构造包装对象。
- **白话例子**：读长文章，读完一段就把剩余全文再抄一份。候选是原文不动，只更新“读到第几个字”。
- **候选**：插入状态保存原始 key 和逻辑偏移，通过已有 `*_at` 方法读取当前段；到新增持久节点、分裂或确实需要独立后缀时，再按原有所有权规则物化。`state.value` 是另一个索引 Tensor，不能把其切片也描述成 token 数组全量复制。
- **收益来源**：输入长、经过节点多时，减少反复复制临时 token ID 后缀和构造 RadixKey。不是减少模型 KV 复制；普通匹配已经有这项思路。
- **教学上界**：65536 token、64 段、每段 1024 token、元素 8 B，原循环后缀复制载荷共 16515072 B（15.75 MiB）；完全命中时 offset 遍历这部分可为 0。这里不计页对齐、匹配比较内切片、持久节点 key、Tensor 索引操作；更不是实际服务节省了 15.75 MiB 的结论。
- **新增成本 / 回退**：每次寻址维护偏移；短输入、单节点命中几乎没有后缀复制可省。状态机和组件接入更复杂，可能抵消收益。
- **正确性条件**：保留 action barrier / resume 时机、LRU 与命中计数、重复槽回收和组件 overlap 行为；检查 MTP bigram 边界、limit、页对齐、namespace / salt、部分匹配和 rotation。持久节点不能直接保留导致请求缓冲被长期固定或复用破坏的视图。
- **有限验证**：使用固定版本真实 RadixKey，在 2400 个页大小 / bigram / namespace / salt / limit / 分歧用例中，后缀切片遍历与 offset 遍历的节点匹配轨迹、最终剩余 token 相同。**未运行完整 `_insert_walk_step` 状态机的 before/after 对照。**
- **最小下一步**：先落单独 opt-in 插入实现，运行完整树与分配器回归、split / cancel / evict / session / MTP / HiCache 保护用例；测复制量、总插入 CPU、设备空洞与同条件 E2E。
- **否定条件**：状态机副作用或存储所有权不等价，或真实树太浅 / 成本已被覆盖、总时间无改善。
- **源码**：[插入尾部切片](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1357)、[已有 offset 匹配](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L909)、[RadixKey 切片与 match_at](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/radix_cache.py#L119)。

## 查重记录与优先级

2026-10-18 查询仓库所有状态 PR，关键词包含 `evictable_device_leaves lock_ref`、`evictable_device_leaves discard`、`lock redundant cache`、`insert offset radix`、`match_at insert`、`state.key slice`。本次检索未发现直接对应上述两个具体改动的 PR，**不构成全网原创性证明，也不排除描述不同或尚未公开的实现**。

已发现并明确区分：

| 相关 PR | 查询时状态 | 与本记录关系 |
| --- | --- | --- |
| [#26943](https://github.com/sgl-project/sglang/pull/26943) | closed，未合并 | 通过改变 RadixKey 存储来做零拷贝切片；与 009 同一成本方向，009 选择复用现有 offset API，不能宣称首次提出减少后缀复制 |
| [#38371](https://github.com/sgl-project/sglang/pull/38371) | open，未合并 | 已有持久惰性淘汰堆实现；原 CACHE-001 不作为新发明 |
| [#42923](https://github.com/sgl-project/sglang/pull/42923) | merged | 匹配接口返回长度，按需收集索引；避免把已完成的匹配拼接优化再次当作新点 |
| [#43023](https://github.com/sgl-project/sglang/pull/43023) | merged | 回载接口返回长度，部分内部路径仍物化索引 |
| [#43600](https://github.com/sgl-project/sglang/pull/43600) | open，未合并 | 已有 full_device_path_len 与结束请求跳过无用映射重写；原 CACHE-007 的“只求长度”方法已有先例 |

008 改动更小，适合先做完整正确性与 CPU 对照。009 在长输入、多节点遍历下可删除的复制量可能更大，但集成风险与验证面也更大。两者都需要目标路径调用频率与稳定基线，不能直接声称对当前 GLM 提速。

复现有限检查：在 `checks/2026-10-18-equivalent-work/` 内运行 `python3 check_equivalent_work.py --fetch-source`；只下载固定公开源码并校验哈希，不连接设备、不启动模型。
