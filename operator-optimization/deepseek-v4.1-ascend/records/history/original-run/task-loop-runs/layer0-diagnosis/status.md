# Task 状态

- Task：`layer0-diagnosis`
- 标题：第零层 MC2 低活动原因
- 算子：`DispatchFFNCombine / layer0`
- 开发框架：`vLLM-Ascend 910C`
- 状态：`ACTIVE`
- 阶段：`DESIGNING`
- 活动 Loop：`loop-019`
- 已接受基线：`NONE`
- 证据成熟度：`E3_PROFILE_SUPPORTED_PARTIAL_MECHANISMS`
- 用例数：`1`
- 当前知识条目：`0`
- 下一步：本轮研究已发布GitHub commit 9e2870bb9982d83377b180f9523adadf8e5343bd；当前无在跑实验。历史未闭合理论/语义问题保留为未知，不据此重复采集；后续具体实现/定向取证属于下一阶段。
- 更新时间：`2026-10-30T09:36:43.340577+00:00`

## 最近 Loop

共 `19` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-010` | `ACCEPTED` | `ACCEPTED` | 分组证据支持该样本的主流内时长比例，不能直接判断因果和可节省时间。 |
| `loop-011` | `ACCEPTED` | `ACCEPTED` | 该 rank 时间重叠不证明属于同一请求、实际通信隐藏或关键路径。 |
| `loop-012` | `ACCEPTED` | `ACCEPTED` | 窗口起点非完整 forward 完成；通信行可能汇总，不能推定纯通信和关键路径。 |
| `loop-013` | `ACCEPTED` | `ACCEPTED` | 计数为累计汇总，偏差来源、逐请求与时间线未证明。 |
| `loop-014` | `ACCEPTED` | `ACCEPTED` | 只证明同卡时间相交，无法确定同请求、因果及有用通信隐藏。 |
| `loop-015` | `INCONCLUSIVE` | `INCONCLUSIVE` | 同窗先后不证明数据身份或输入值，也不能应用受限算子访问界。 |
| `loop-016` | `INCONCLUSIVE` | `INCONCLUSIVE` | first SSE data不保证是token，计数差不证明实际重算量，不能给prefill TPS。 |
| `loop-017` | `PIVOTED` | `PIVOTED` | 历史占位 Run 作废；已有相关报告独立保存，本轮转向当前更具体的排查。 |
| `loop-018` | `INCONCLUSIVE` | `INCONCLUSIVE` | 已关联选定queue和主机调用，但固定外层行无调用栈，不能从捕获活动解释主流长缺口；转向可比较层分组。 |
| `loop-019` | `EVALUATING` | `PENDING` | 审查 Run run-011 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
