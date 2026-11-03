# Stage 098：动态 K 与 DSpark 草稿边界

## 证据核验

ZCode 回执 `operator-ceiling169-v2-098-draft-constraints.exit-code` 为 0，`*.verified.json` 记录执行模型 `deepseek-flash`。`draft-constraints-snapshot-status.json` 记录 baseline 与 stage 两容器镜像 ID 相同，五个指定文件没有覆盖挂载、逐对 SHA 相同；本地重算五个快照 SHA 与状态文件一致。以下是这些源码静态语义，不证明运行中对象未被其他代码修改。

## 已收紧的约束

1. **动态调度表不会把 K 增加到配置最大值以上。** `draft-constraints-snapshot/dynamic_utils.py:37-55,77-146` 校验表项为非负数，构表时每个配置区间、间隙及尾部都写入 `min(vllm_num_speculative_tokens, 该区间或前一区间的值)`，初始元素为 0。因此成功构表且 `vllm_num_speculative_tokens=5` 时，所有表项都在 `[0,5]`，包括用户输入更大的表项。`metadata-path-snapshot/vllm_scheduler.py:263-277,1392-1396` 确认配置分支把该值用于每批 K，未启用动态表时直接取配置 K。范围限定：实际调用的是这些函数、表在构造后未被篡改，且运行配置对应 K=5。
2. **DSpark 本路径的输出列宽有明确条件界。** `draft-constraints-snapshot/ascend_dspark_proposer.py:39-46` 按初始化的 K 创建宽 `1+K` 缓冲；`ascend_base_proposer.py:1046-1058,1581-1588,1637-1644` 在运行 K=0 返回 0 列，否则 DSpark parallel-drafting 的返回 `buffer[:,1:]`。若实例初始化 K=5 且成功运行并走该返回，返回宽为 5；对应 K=0 时返回宽为 0。即使运行 K=1–4，该 slice 仍是五列，不能声称减小 K 会自动缩窄 tensor。`spec_config.py:1461-1462` 的 DSpark 配置处理设置 `parallel_drafting=True`，但需完整实际实例化和调度链仍按这些文件执行。`metadata-path-snapshot/ascend_runner.py:2098` 将 scheduler 指定的 K 传给 `_propose`；`:2712-2760` 的跳过 drafting 分支创建宽 `self.num_spec_tokens` 的零占位张量，因此若该值为 5，此分支也是 5 列，不是 0 列。
3. **无结构化输出时，validator 不增加 draft。** `draft-constraints-snapshot/structured_output.py:294-297` 对 `request.use_structured_output=False` 直接返回输入列表。因此在该条件下，`vllm_scheduler.py:2404-2459` 的直接存储不改变长度；异步路径先截到原 placeholder 长度，随后补至该长度，不能扩大该长度。`measure1.py` 的 benchmark 配置未显式列出结构化输出，但仅此不能证明实际 HTTP 请求和 runtime request 标志为 false。
4. **结构化输出分支仍有例外待证。** 同一 validator 的 `:299-312` 调用了 `strip_speculative_padding` 和外部 `grammar.validate_tokens` 后拼接，尽管文档说返回前缀，本次没有核验被调后端或运行对象，不能仅凭方法名断定对于所有 grammar 对象不会扩张。

## 尚未闭合的条件

- 调度字典入口、请求 draft 更新和 runner 的 tensor/list 分支已在 `loop010-draft-stage095-review.md` 列出；要将静态局部约束归纳为**所有真实运行请求**的长度 ≤5，仍要证实实际 proposer 注册选择、初始化 K、无其他注入或替换、请求是否启用 structured output，及所有更新发生于上述已核验路径。不能把文件存在视为运行时证明。
- `metadata-path-snapshot/vllm_runner.py:4941-4955` 的 list 路径直接返回列表，不能用 DSpark tensor 的列宽证明不相关 proposer/注入路径。
- 即使推得所有 metadata draft ≤5，sampler 的每序列 ≤6 槽还不是每次 **客户端**有效 token 数 ≤6：需要确认 sampler、后续前端与计数的真实调用链；8192 是结果文件所报值，其计算方式与时间窗口映射尚未核实。
- 严格有效的 TPS 数字上界还需要可证明的事件间最短时间及资源约束；测得的最短间隔、带宽或利用率均不是物理硬上限。

本次没有 NPU 操作、没有修改目标服务或代码。
