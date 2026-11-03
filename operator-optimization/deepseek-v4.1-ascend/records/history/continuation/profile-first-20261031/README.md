# Profile 优先分析：当前证据与下一步

用户最新约束：充分使用 profile；只有 profile 无法单独定案时，再定点查补代码。

本轮只读取保留的 profile 汇总、索引和已有研究记录，没有读取新的目标源码，没有连接服务器或新增设备负载。主控准备固定脚本，ZCode 执行机械复核，主控审查结果。

## 本轮已经确认

- 旧批次的 16 rank 共 481,548 条记录，87 类算子。每个 rank 的 family、operator type 计数均与总行数相等。
- 官方批次 `official-20261030T084726Z-cb252d1d` 的两窗口共 980,002 条导出记录；窗口一有 87 类，窗口二有 83 类。每个保留 sample/window 的 type、core、model 计数一致。
- 共 48 项一致性检查全部通过。完整聚合保存在 `aggregate-recheck.json`，展示没有截掉类型或 rank；这仍是对保留聚合的复核，并非重新解析全部原始 trace。
- 旧 family 分类的 `other_unattributed` 有 357,314 条。它是 family 分类器的余项；这些记录仍有 operator type，不能把它等同于完全没有算子名，更不能据此直接算语义模块未覆盖率。
- 旧批次时间单位为微秒，官方表为纳秒。两个窗口、不同 rank、两批 profile，以及 166 observer 的计数不拼成同次调用；只累计计数作一致性检查，不累计时长声称请求关键路径。

## Profile 可以先排除的重复收益说法

旧批次和官方两个窗口均直接观察到 `HcPre`、`RmsNormCast`、`SparseFlashMla`、`DispatchFFNCombine`、`DequantSwigluQuant`。因此，F01/F04/F13 的既有复合边界、F21 的 MC2 外部复合边界，以及 F22 中已有激活量化复合段，不能当作尚未发生的融合重复计收益。这只是对“已经有复合调用”的判断；内部拆分重组或延伸边界是否值得做，仍需进一步分析。

## 候选逐项执行门槛

1. 先检查相关采集批次、phase、rank、模块映射、算子/kernel、shape/dtype、host scope、stack、flow/connection、stream 与事件。保留同调用绑定的依据和不确定性；先用完整聚合，再读有界事件切片。
2. Profile 能支持或排除的具体命题直接落结论；相邻、同形、同流或同一个调用栈都不能独自证明融合数值合法。
3. 相关 profile 证据仍不能定案时，才写出一个影响判断的最小问题，查对应生产者、消费者、精度/舍入、alias 或事件边界的最小源码范围。已经保存的局部合同先复用，避免重复阅读。
4. 实际历史 tensor 身份、运行时别名或事件代际缺失时，源码只能说明可能的语义，不能替代那次调用的证据；缺口继续保留，不自动扩展采集。

## 下一轮继续检查的 profile 内容

- 全部 F01–F32 及 26 个暂定模块表逐项关联；本轮尚未完成这些关联，不能用聚合复核代替整个候选分析。
- 先复用已有的 F08 host-scope 链、F20 路由准备栈/flow、F21 到达和跨流证据、F22 既有复合段，检查它们分别支持哪些局部命题；对其余条目同样保留 profile 查询状态，不只检查 TopN。
- 纳入官方双窗口全模型/全 stream 导出、框架 CPU 栈，以及 target/draft/Engram 的已有独立证据；不能只沿用旧 captured-target 子集。
- 关键路径与资源意义单独记录。混合通信复合任务、导出空洞、CPU 等待栈、PMU 比率都不能直接转换为可消除耗时。

当前 Loop004 保持进行中。`profile_exhausted=false`、`whole_modules_closed=0`、`overall_complete=false`；原始全范围目标未完成。
