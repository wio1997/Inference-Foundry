# SGLang 前缀缓存：逻辑排查总表

Python Unified FULL 设备缓存的键准备、匹配、引用、索引、插入、淘汰与分配回收边界。HiCache / SWA / Mamba 仅检查与此主线的合同边界；不宣称覆盖其全部实现、Rust 核心或 Ascend 内核。

48 个预先列出的逻辑问题逐格给出处置；这不是数学上穷尽全部程序优化，也不是全网原创性证明。

固定源码：[`57db5089c207ae4ce189cc092a89a2027c8aea29`](https://github.com/sgl-project/sglang/tree/57db5089c207ae4ce189cc092a89a2027c8aea29)。

**结果：复用原 9 条记录，新增 21 个检查项，另列 12 个排除项。新增检查项不等于已证实的新优化。**

[打开离线交互总图](diagrams/cache-audit.html) · [结构化目录](cache-audit-catalog.json) · [48 格覆盖表](checks/2026-10-18-cache-audit/coverage.json)

没有真实设备 benchmark、没有候选端到端加速比。逻辑等价与删除某类操作不保证所有机器 / 规模下耗时不增。

## 优先验证顺序

先做局部且边界清楚的 008、011、020、023、024；009、013、019 可删除的复制量可能更大，但需要更多所有权与状态机验证。010、015、021、027 可能收益很小或目标配置不触发，不应先于瓶颈诊断。

## 局部检查与查重边界

本轮源码提取检查：010 的 6 个关闭分支、011 的 2400 个页键用例、020 的 540 个容量 / donor 状态、023 的 600 个区间分类用例通过。外围依赖为轻量替身，尚未验证整个 SGLang、异步设备或真实模型。

[脚本](checks/2026-10-18-cache-audit/check_audit.py) · [结果](checks/2026-10-18-cache-audit/validation.json) · [源码哈希](checks/2026-10-18-cache-audit/source-version.json) · [PR 搜索记录](checks/2026-10-18-cache-audit/pr-search.json)

PR 查重覆盖 24 组有界题目 / 正文查询，另读取相关 PR 状态并检查 #38371 / #43600 的文件补丁。未命中不代表无人做过；未遍历所有 PR diff、私有分支或其他框架。006 / 007 / 009 的细分项标明关系，不能把同一收益重复相加。

## 新增检查项一览

| ID | 检查项 | 类别 | 首次验证优先级 |
| --- | --- | --- | --- |
| [CACHE-010](#cache-010) | 缓存关闭后，先返回再切输入 | 等价方向 | P3 |
| [CACHE-011](#cache-011) | 分裂取页键时，避免临时复制整个后缀 | 等价方向 | P1 |
| [CACHE-012](#cache-012) | 已对齐的键，避免再复制一次 | 条件候选 | P2 |
| [CACHE-013](#cache-013) | 合并 checkpoint 前的多次输入裁剪 | 条件候选 | P2 |
| [CACHE-014](#cache-014) | FULL 匹配只记长度，减少临时段列表 | 条件候选 | P2 |
| [CACHE-015](#cache-015) | 树上子节点只查一次字典 | 等价方向 | P3 |
| [CACHE-016](#cache-016) | 没有主机缓存时，省去主机叶子集合维护 | 条件候选 | P2 |
| [CACHE-017](#cache-017) | FULL 锁入口避免组件内外重复刷新 | 条件候选 | P2 |
| [CACHE-018](#cache-018) | 重叠段确实有消费者时，再切索引视图 | 条件候选 | P2 |
| [CACHE-019](#cache-019) | 独立索引副本是否可延后到树实际收养时 | 条件候选 | P2 |
| [CACHE-020](#cache-020) | 同一次容量检查，没有变化就少查一次 | 等价方向 | P1 |
| [CACHE-021](#cache-021) | 关闭淘汰指标时，省去起始计时 | 等价方向 | P3 |
| [CACHE-022](#cache-022) | free group 只有一段时，省一次 cat | 条件候选 | P2 |
| [CACHE-023](#cache-023) | SWA 已回收区间为空时，跳过逐段拆分 | 等价方向 | P1 |
| [CACHE-024](#cache-024) | 全是空回收区间时，先返回再索引请求行 | 条件候选 | P1 |
| [CACHE-025](#cache-025) | 维护设备驻留子节点数，减少反复扫描 | 有取舍 | P3 |
| [CACHE-026](#cache-026) | 只读匹配按结构版本复用 | 有取舍 | P3 |
| [CACHE-027](#cache-027) | 复用固定页内偏移向量 | 条件候选 | P3 |
| [CACHE-028](#cache-028) | FULL-only 叶子资格走直接字段判断 | 条件候选 | P2 |
| [CACHE-029](#cache-029) | FULL-only 节点减少辅助对象与动态字典 | 有取舍 | P3 |
| [CACHE-030](#cache-030) | 持久节点复用已经算出的首 page 键 | 有取舍 | P2 |

## CACHE-010

**缓存关闭后，先返回再切输入** · 输入与键准备 · 等价方向 · P3

- 当前：checkpoint 在 disable 判断前切一份 token 数组。
- 候选改法：把这次切片移到原有 disable 返回之后，保留结束清理。
- 想省：关闭缓存路径的一次 CPU 输入复制。
- 成立条件：只改变 disable 路径；仍执行既有组件 cleanup。
- 新增成本 / 回退：启用缓存路径布局变化；关闭缓存时才有这项收益。
- 最小验证 / 当前证据：6 个关闭路径对照通过；启用路径及真实组件清理待集成回归。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_radix_cache.py:1132](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_radix_cache.py#L1132)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-011

**分裂取页键时，避免临时复制整个后缀** · 插入与分裂 · 等价方向 · P1

- 当前：为取后缀首个 page 的键，先切出整段后缀；稍后还会切 child.key。
- 候选改法：这里改用 key.child_key_at(split_len, page_size)。
- 想省：一次临时 token 后缀复制与 RadixKey 对象。
- 成立条件：保持 namespace、salt、bigram、limit 与页边界；持久 child.key 切片照常保存。
- 新增成本 / 回退：改动小；分裂很少时收益很小。
- 最小验证 / 当前证据：真实 RadixKey 的 2400 个局部结果对照通过；完整分裂锁 / 动作回归待做。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1448](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1448)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-012

**已对齐的键，避免再复制一次** · 输入与键准备 · 条件候选 · P2

- 当前：page_size 大于 1 时，即使长度已对齐，仍返回一次完整切片。
- 候选改法：内部只读调用可复用原键；或提供显式借用接口。
- 想省：重复的全长 token 复制。
- 成立条件：需要证明调用者不依赖独立 key 或底层数组；bigram 模式存在原地转换。
- 新增成本 / 回退：别名合同、可变字段与生命周期维护。
- 最小验证 / 当前证据：核对所有调用者；覆盖同 key 多次对齐、模式转换、后续修改和持久节点保存。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/radix_cache.py:147](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/radix_cache.py#L147)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-013

**合并 checkpoint 前的多次输入裁剪** · 输入与键准备 · 条件候选 · P2

- 当前：先裁 up_to，再裁 effective_cache_len，再按页裁；begin_insert 又对齐。
- 候选改法：前期传原输入与长度界限，到必须独立保存的位置再物化。
- 想省：进入插入遍历之前的重复全长 / 前缀复制。
- 成立条件：各组件仍按相同长度计算；持久节点须保持独立、合适大小的所有权。
- 新增成本 / 回退：改 API 与长度边界传播，可能固定更大的输入缓冲。
- 最小验证 / 当前证据：统计各裁剪真实字节，覆盖长度截断、MTP、页边界与组件组合。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_radix_cache.py:1152](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_radix_cache.py#L1152)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。
- 关系：细化 / 关联 CACHE-009，不可把相同工作重复计入收益。

## CACHE-014

**FULL 匹配只记长度，减少临时段列表** · 树匹配 · 条件候选 · P2

- 当前：匹配中收集 value 列表，末尾再切列表并累加长度。
- 候选改法：已确认的 FULL-only / 无 HiCache 路径维护等价长度与节点，省临时列表。
- 想省：列表 append、切片与末尾再遍历。
- 成立条件：不能影响辅助组件 finalizer、分裂与命中边界；普通匹配已按需收集索引。
- 新增成本 / 回退：专用路径维护与每节点计数可能抵消收益。
- 最小验证 / 当前证据：对照匹配结果、访问记录与 action；逐次测纯 CPU，总时间下降才保留。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1041](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1041)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。
- 关系：细化 / 关联 CACHE-006，不可把相同工作重复计入收益。
- 相关先例：[#42923](https://github.com/sgl-project/sglang/pull/42923)

## CACHE-015

**树上子节点只查一次字典** · 树匹配 · 等价方向 · P3

- 当前：先判断 child_key 在字典中，再用同一个键取节点。
- 候选改法：在循环内用一次 get，未命中即退出；插入的同型位置也可评估。
- 想省：一次重复字典查找；长 page 键还涉及重复哈希。
- 成立条件：Unified children 是普通 dict，合法值为非 None 节点；中途无树变更。
- 新增成本 / 回退：控制流布局变化；字典查找通常很小，不能默认更快。
- 最小验证 / 当前证据：覆盖空树、未命中、盐隔离、部分命中、split；测总 matcher 时间。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:951](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L951)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-016

**没有主机缓存时，省去主机叶子集合维护** · 访问与引用 · 条件候选 · P2

- 当前：每次都检查 device 和 host 叶子，再分别 add / discard。
- 候选改法：无 host 驻留能力且 host 集合恒空时，选择只维护 device 的路径。
- 想省：host 检查、方法调用和空集合删除尝试。
- 成立条件：保证不持有 host_value / host 锁；配置启用或重配时正确切换。
- 新增成本 / 回退：增加配置分支 / 绑定与维护范围；单次 host 检查原本很短。
- 最小验证 / 当前证据：覆盖 reset、启用 HiCache、buffer-only、外部 linker 和 host 锁。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1568](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1568)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-017

**FULL 锁入口避免组件内外重复刷新** · 访问与引用 · 条件候选 · P2

- 当前：Full acquire / release 已沿路径维护候选，TreeCore 入口又刷新锚点。
- 候选改法：FULL-only、无 host、正常组件锁合同下，研究省掉外层重复刷新。
- 想省：重复资格检查与集合操作。
- 成立条件：不可用于 host-only 尾段或混合组件；skip receipt、split 与根节点需核对。
- 新增成本 / 回退：增加专用条件；误删刷新会留下错误回收资格。
- 最小验证 / 当前证据：逐事件比较引用数、候选集合、账本，覆盖 pin、空锚点和中途分裂。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:725](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L725)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。
- 关系：细化 / 关联 CACHE-008，不可把相同工作重复计入收益。

## CACHE-018

**重叠段确实有消费者时，再切索引视图** · 插入与分裂 · 条件候选 · P2

- 当前：已有节点重叠时先切 Tensor 视图、调用组件 hook，再判断是否有重复槽要回收。
- 候选改法：FULL-only 且原保护前缀已覆盖该节点时，跳过无消费者的 view 与空 hook。
- 想省：临时 Tensor view、框架小操作与默认回调。
- 成立条件：Full 默认 hook 不消费；混合组件可能需要读取或接管该段。
- 新增成本 / 回退：多维护专用路径；view 常是小固定成本。
- 最小验证 / 当前证据：比较重复释放范围、adopted_ranges、action、锁及插入返回。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1318](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1318)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-019

**独立索引副本是否可延后到树实际收养时** · 索引与所有权 · 条件候选 · P2

- 当前：checkpoint 先复制整段索引；Python 新节点 / 恢复节点又 clone 自己收养的部分。
- 候选改法：核对后尝试只做必要 dtype 转换，由树在收养处生成独立副本。
- 想省：中间的全长索引复制；不删除树长期持有的独立副本。
- 成立条件：必须穷查所有收养点、prompt 二次 insert、延迟 action 和异步消费者。
- 新增成本 / 回退：所有权审计面大；Rust、session、linker 合同不能直接类推。
- 最小验证 / 当前证据：以请求行复用、取消、分裂、恢复、双 insert 和异步读取验证；任何悬空别名即否定。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_radix_cache.py:1196](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_radix_cache.py#L1196)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。
- 关系：细化 / 关联 CACHE-007，不可把相同工作重复计入收益。

## CACHE-020

**同一次容量检查，没有变化就少查一次** · 淘汰选择 · 等价方向 · P1

- 当前：第一次容量未达标后，普通分支未修改容量，却再次查询并比较。
- 候选改法：普通无变更分支直接返回 False；Mamba donor 准备后仍重查。
- 想省：重复的 allocator 元数据查询。
- 成立条件：该 getter 在两次调用之间纯且稳定；保留 donor、联合回收回调的行为。
- 新增成本 / 回退：需要核对自定义 allocator；查询很便宜时收益很小。
- 最小验证 / 当前证据：提取真实闭包的 540 个容量 / donor 情况通过，局部共少 208 次读取。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_radix_cache.py:930](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_radix_cache.py#L930)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。
- 相关先例：[#33091](https://github.com/sgl-project/sglang/pull/33091), [#36713](https://github.com/sgl-project/sglang/pull/36713)

## CACHE-021

**关闭淘汰指标时，省去起始计时** · 观测与配置 · 等价方向 · P3

- 当前：_evict 无条件读起始时钟，update_eviction_metrics 在 collector 为 None 时不使用它。
- 候选改法：仅在确实收集指标时启动计时，保留启用时的统计边界。
- 想省：无指标路径的时钟调用。
- 成立条件：指标启用状态 / collector 在调用期间符合冻结合同。
- 新增成本 / 回退：增加条件判断；小于噪声时不值得维护。
- 最小验证 / 当前证据：指标开 / 关、零驱逐、异常路径；比较事件和统计含义。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_radix_cache.py:791](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_radix_cache.py#L791)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-022

**free group 只有一段时，省一次 cat** · 分配与回收 · 条件候选 · P2

- 当前：free_group 已克隆输入取得所有权；结束时单段也会 cat 成新缓冲。
- 候选改法：单段交出已拥有的快照，多段保持拼接。
- 想省：单段输出分配、复制与提交。
- 成立条件：下游 free 不依赖另一份独立存储；各 allocator 子类需要核对。
- 新增成本 / 回退：新增单段分支；组内 snapshot 仍必须保留。
- 最小验证 / 当前证据：0 / 1 / 多段、异常、输入行随后覆写、各后端子类回归。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/allocator/base.py:220](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/allocator/base.py#L220)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。
- 关系：细化 / 关联 CACHE-007，不可把相同工作重复计入收益。
- 相关先例：[#26988](https://github.com/sgl-project/sglang/pull/26988)

## CACHE-023

**SWA 已回收区间为空时，跳过逐段拆分** · 分配与回收 · 等价方向 · P1

- 当前：即使 dead_lo == dead_hi，也逐段求交并构造 alive / dead 列表。
- 候选改法：空 dead 区间直接交给原 allocator.free_segments。
- 想省：逐段 min/max、临时列表和重复分类。
- 成立条件：相同 segments 顺序、零段处理与 allocator 页不重叠合同。
- 新增成本 / 回退：非空分支多一次判断；不会删除必要页校验。
- 最小验证 / 当前证据：真实分类函数 + 轻量 allocator 的 600 个空 / 非空区间对照通过。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/common.py:160](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/common.py#L160)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-024

**全是空回收区间时，先返回再索引请求行** · 分配与回收 · 条件候选 · P1

- 当前：free_kv_row 先取 Tensor 行，再为零长度区间构造空切片。
- 候选改法：合法 ranges 全为空时，在触碰请求行之前结束。
- 想省：空回收的 Tensor 行选择、视图与区间准备。
- 成立条件：只跳过确实空的范围；自定义 allocator 不能依赖空调用的副作用。
- 新增成本 / 回退：额外空区间判断；不能吞掉真实无效范围与必要资源释放。
- 最小验证 / 当前证据：完整命中、结束 / 取消、空列表、混合空非空范围、页与 DCP 边界。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/base_prefix_cache.py:497](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/base_prefix_cache.py#L497)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。
- 相关先例：[#43183](https://github.com/sgl-project/sglang/pull/43183)

## CACHE-025

**维护设备驻留子节点数，减少反复扫描** · 淘汰选择 · 有取舍 · P3

- 当前：判断设备叶子时用 any 扫描子节点，找到设备驻留项就结束。
- 候选改法：维护 resident child count，设备子节点为空与否用计数判断。
- 想省：高扇出且很多子节点不驻留时的扫描。
- 成立条件：所有新增、split、恢复、降层、删除都更新计数；不能只改一个入口。
- 新增成本 / 回退：每次结构变化多维护一个计数；普通短路扫描可能更便宜。
- 最小验证 / 当前证据：扇出与驻留比例扫描；逐事件和原 any 对照，累计维护成本也纳入。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:2276](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L2276)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-026

**只读匹配按结构版本复用** · 树匹配 · 有取舍 · P3

- 当前：重复只读探测仍从根匹配。
- 候选改法：对明确只读 API 用 key / namespace / tree epoch 复用结果。
- 想省：同一输入、同一结构版本的重复匹配。
- 成立条件：该 API 目前主要用于分层边界；不能跳过有 LRU / split 副作用的完整 match。
- 新增成本 / 回退：维护版本、缓存与失效；短队列可能更贵，HiCache 关闭时未必触发。
- 最小验证 / 当前证据：覆盖 insert、split、evict、restore、salt、句柄失效；测命中率和版本维护总成本。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:984](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L984)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-027

**复用固定页内偏移向量** · 分配与回收 · 条件候选 · P3

- 当前：普通 paged alloc 每次构造固定的 0..page_size-1。
- 候选改法：在配置稳定时只读复用页内偏移向量，或由后端融合生成。
- 想省：固定常量的分配与小操作。
- 成立条件：只针对调用这个 alloc 的路径；实际 Ascend 子类、dtype、device 与图捕获另查。
- 新增成本 / 回退：持久常量占少量设备内存；可能不在热路径。
- 最小验证 / 当前证据：alloc 输出一致；页大小 / device / resize / graph 生命周期检查，先统计调用量。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/allocator/paged.py:199](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/allocator/paged.py#L199)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-028

**FULL-only 叶子资格走直接字段判断** · 访问与引用 · 条件候选 · P2

- 当前：节点为全部组件编号分配数据，叶子资格遍历整个 component_data 检查引用。
- 候选改法：明确 FULL-only 时直接检查 FULL 引用；混合组件保留通用判断。
- 想省：生成器与未启用组件槽的检查。
- 成立条件：不能漏掉任何实际持锁组件；组件配置变化后切回正确路径。
- 新增成本 / 回退：专用分支 / 绑定与维护；组件少时收益仍小。
- 最小验证 / 当前证据：不同组件集、host / device 锁、split 后资格逐项对照。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:2273](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L2273)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。
- 关系：细化 / 关联 CACHE-006，不可把相同工作重复计入收益。

## CACHE-029

**FULL-only 节点减少辅助对象与动态字典** · 观测与配置 · 有取舍 · P3

- 当前：每个节点准备全组件 ComponentData、辅助 LRU 槽和动态字段。
- 候选改法：研究 FULL 专用布局、slots 或惰性辅助数据，保留接口合同。
- 想省：CPU 堆对象、内存占用和可能的缓存未命中。
- 成立条件：所有字段读写、调试、序列化、组件切换与 Rust 适配都要核对。
- 新增成本 / 回退：辅助访问可能增加间接层；工程和兼容成本大。
- 最小验证 / 当前证据：实测每节点内存、GC 和遍历时间；大树与混合配置回归。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:112](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L112)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## CACHE-030

**持久节点复用已经算出的首 page 键** · 插入与分裂 · 有取舍 · P2

- 当前：节点首 page 键在插入、分裂、删除等位置重新构造。
- 候选改法：在节点中保留与父字典一致的 child key，必要时随 split 更新。
- 想省：重复页切片、整数装箱与 tuple 构造。
- 成立条件：节点 key / namespace / salt 改变时失效；复用同一个父字典键对象更合适。
- 新增成本 / 回退：每节点额外引用和维护；page=1 或很少复用时可能不值。
- 最小验证 / 当前证据：split 前后父子查找、删除、salt 隔离和节点数量变化；测总 CPU 与额外内存。
- 否定条件：状态或服务合同不等价，或总成本未降低；目标路径不触发 / 改善被覆盖时，不能声称端到端提速。
- 源码：[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1479](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1479)
- 查重：本轮有界题目 / 正文检索未发现直接对应改动；不是原创性证明。

## 排除项与已有能力


### REJECT-ROOT · 跳过所有根节点访问记录

根不被淘汰不足以证明其字段、全局访问编号与所有观测合同可忽略；不作为无条件优化。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1027)

### REJECT-REF · 共享前缀只计一次引用

每个请求仍有独立生命周期；少计会导致过早回收。008 只省集合操作。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/components/full.py#L317)

### REJECT-COPY · 删除全部 clone / copy=True

延迟回收快照与持久树所有权需要独立存储；019 必须逐收养点验证。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/allocator/base.py#L225)

### REJECT-BARRIER · 跨所有 action barrier 延迟回收

动作提交关系到下一阶段容量与所有权；不能只为合并操作跳过边界。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1262)

### REJECT-UNIQUE · 通用 free 一律删 unique

通用输入不保证页不重复；page-exact free_segment 已存在，应复用既有合同。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/allocator/paged.py#L333)

### REJECT-ZERO · 零回收量仍建堆

驱动已跳过满足目标的组件；不能再把这一假设当作已找到的冗余。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_radix_cache.py#L979)

### REJECT-EVENT · 关闭事件还生成完整 token 日志

record_* 已在重计算 / payload 之前返回；调用本身的轻量成本另论。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/events.py#L219)

### REJECT-HASH · 禁用存储仍无条件计算所有节点哈希

已有配置守卫；启用 KV 事件时可能另有必要哈希消费者。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1532)

### REJECT-HIT · 普通 LRU 时随意删 hit_count

LFU / SLRU 与 HiCache 使用计数；需要策略与观测合同，不能笼统称无用。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1105)

### REJECT-TAIL · 所有多轮请求只插新增尾部

已有重叠段仍有访问、优先级、命中与回收副作用；相关 Rust continuation 已有先例 #38636。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1284)

### REJECT-ROW · 结束请求无用映射重写作为新发明

相关 PR #43600 已实现；应评估复用与本地版本兼容，不重复声称新点。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/unified_radix_cache.py#L1249)

### REJECT-SYNC · 只看 CPU 时间就认定设备提速

已有调试同步守卫；局部成本、是否暴露在关键路径、设备竞争与 E2E 需分开测。
[源码位置](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/mem_cache/memory_pool.py#L110)

## 48 个逻辑问题的覆盖

| 阶段 | 提前结束 | 重复维护 | 多余数据 | 临时复制 | 专用或合并 | 时序 / 所有权 |
| --- | --- | --- | --- | --- | --- | --- |
| 输入与键准备 | CACHE-010 | CACHE-012/CACHE-013 | CACHE-009/CACHE-013 | CACHE-011/CACHE-012/CACHE-013 | CACHE-006/CACHE-013 | CACHE-012/CACHE-013 |
| 树匹配 | CACHE-006 | CACHE-015/CACHE-026 | CACHE-014 | CACHE-014/CACHE-007 | CACHE-006/CACHE-014 | CACHE-005/CACHE-026 |
| 访问与引用 | CACHE-008/CACHE-017 | CACHE-008/CACHE-017 | CACHE-028 | CACHE-029 | CACHE-016/CACHE-028 | CACHE-005/CACHE-008/CACHE-017 |
| 索引与所有权 | CACHE-007 | CACHE-019 | CACHE-007/CACHE-019 | CACHE-007/CACHE-019/CACHE-022 | CACHE-022 | CACHE-019 |
| 插入与分裂 | CACHE-018 | CACHE-011/CACHE-030 | CACHE-009/CACHE-013 | CACHE-011/CACHE-018/CACHE-019 | CACHE-018/CACHE-030 | CACHE-009/CACHE-019 |
| 淘汰选择 | CACHE-020 | CACHE-001/CACHE-020 | CACHE-003/CACHE-025 | CACHE-001 | CACHE-001/CACHE-002/CACHE-025 | CACHE-004 |
| 分配与回收 | CACHE-024 | CACHE-020 | CACHE-023 | CACHE-022/CACHE-027 | CACHE-023 | CACHE-022/CACHE-024 |
| 观测与配置 | CACHE-021 | CACHE-021 | CACHE-029 | CACHE-029 | CACHE-016/CACHE-028/CACHE-029 | CACHE-021/CACHE-029 |

## 开始真实实验前

冻结实际模型 / 镜像 commit / 并行与缓存配置，先补每个函数调用次数、树深、叶子数、分裂与回收频率；测完整 CPU 管理时间及是否挡住设备提交。保持请求 trace、缓存容量、页大小、有效输入输出与图 / MTP 条件相同，再评价端到端首字与生成延迟。

复现本轮局部检查：在检查目录运行 `python3 check_audit.py --fetch-source`。只下载固定公开源码并校验哈希；不连接服务器。
