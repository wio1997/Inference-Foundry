# 基线的 proposer、结构化输出与草稿注入：证据边界

## 本次基线已确定的事实

- `benchmark-20261028/raw-results/service-log-tail.txt:24,100` 记录 API 启动参数和 DP0 的引擎配置均为 `method=dspark, num_spec_tokens=5`、PP=1、DP=2；日志 `:966-1081` 显示两个 DP 的多个 worker 加载 method=dspark 的 draft 模型。这是对**该次启动配置和加载活动**的直接证据。
- 双容器 SHA 一致的 `metadata-path-snapshot/ascend_runner.py:750-765` 表明对应源码在配置 DSpark 且是最后 PP rank 时构造 drafter，并对其 `isinstance(AscendDSparkProposer)` 做断言；`draft-constraints-snapshot/spec_config.py:1910-1924` 显示 DSpark 属于 `use_eagle()`，因此 runner 的 `:2009-2110` 属于 `_propose` 路径，而非普通 ngram、suffix 或 medusa 分支。配置日志、DSpark 模型加载日志和源码共同支持该次按 DSpark 运行的结论，强于仅凭配置猜测。这里仍不等于通过对象快照核实每次函数调用：代码可能被运行时替换，断言也受 Python 优化模式影响；日志没有每个事件的类名。
- 冻结测量配置 `benchmark-20261028/raw-results/measure1.py` 和 `measure2.py` 的 generation_kwargs 只明确指定 `ignore_eos=True, temperature=0`，未显式设置结构化输出；两份冻结 workload 的 8 行各仅包含 `answer, max_tokens, question`。服务日志 `:100` 的 `StructuredOutputsConfig(backend='auto', ...)` 是服务能力配置，**不能证明每个请求启用或关闭结构化输出**。实际 HTTP payload 的构造插件当前没有双容器快照，仍可能附加参数；因而每个 `request.use_structured_output=False` **未获证明**。

## 已快照文件中的输入与修改点

- `metadata-path-snapshot/vllm_scheduler.py:593,812-827,1268-1270,2404-2459` 的显式映射赋值来自已有 request draft、按 K 补位，及异步 draft 更新。对同一文件中 `request.spec_token_ids` 的赋值检索还发现 `:1499` 的抢占清空，未见额外赋值点；对已快照的 Ascend runner 中 `_draft_token_ids` 赋值检索，看到的是 proposer 结果、skip 分支的 K 列零张量及 None 重置。上述扫描仅是**这些文件**的局部事实。
- 双容器快照 `structured_output.py:294-312` 中无结构化输出直接原样返回；有结构化输出调用外部 grammar，再拼接结果。只有前一种直接证明不扩张。
- 因为 scheduler 接受外部传入的 `DraftTokenIds`、异步输出字典是可变对象、scheduler/worker 有可能由其他代码替换或修改，不能凭这两个文件的赋值点排除所有运行时草稿注入；静态证据没有给出全系统的排他性证明。上游基类 `metadata-path-snapshot/vllm_runner.py:4941-4955` 还有直接返回 Python list 的分支，必须先证明其数据在这次从 DSpark tensor 而来，才能排除外部列表。

## 最小闭环路径（不改服务，必要时只读）

1. 优先查现有 benchmark 产物是否保存了 16 个实际 HTTP 请求的非敏感参数摘要；若存在，只摘取 response_format/structured_outputs/guided 等键是否存在及其非内容状态，不输出 prompt、回答或密钥。无保存则不能反证实际全部请求标志；历史请求无法通过事后一个请求观测替代。
2. 若主控仍需要进一步缩小静态缺口，针对唯一已识别的 `vllm_ascend.spec_decode.get_spec_decode_method` 定义文件，以及实际 benchmark payload 构造文件，分别获取两容器/实际执行环境的路径和哈希；确认注册、payload 默认与版本。不做全仓宽泛重复搜索。
3. 已有日志比只读进程当前内存更能证明**历史**基线；若要观测新一轮请求，仅在授权且无他人占卡情况下，通过已有低开销统计/trace 记录匿名的 request structured 标志、drafter 类型、K、draft 长度和提交数量等；不记录 token 内容。统计只能验证观察到的运行，不能证明从未观察到的运行或数学上的所有实现。

当前全局 5+1 上限、客户端计数与窗口对应、不可超越的数字 TPS 上界均仍为 UNKNOWN。上述排查不改变服务、不操作 NPU。
