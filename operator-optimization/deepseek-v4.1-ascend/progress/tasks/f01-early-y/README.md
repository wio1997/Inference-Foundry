# F01 · HcPre 提前交付 Y 的逻辑与验证

## Task 契约

- Task ID：`f01-early-y`
- 算子：`HcPre EARLY-Y`
- 开发框架：`vLLM-Ascend / Ascend C / CANN9.2.0`
- 源码根目录：`/Users/wio/work/Inference-Foundry-operator-optimization/operator-optimization/deepseek-v4.1-ascend`
- 创建时间：`2026-11-03T12:04:53Z`

## 任务目标

针对有外部pre_mix的指定路径，记录源码依赖、当前可实现性、数值要求及同条件性能结论；不能因记录成稿标记性能方案完成。

## 允许修改的路径

- `/Users/wio/work/Inference-Foundry-operator-optimization/operator-optimization/deepseek-v4.1-ascend/progress`
- `/Users/wio/work/Inference-Foundry-operator-optimization/operator-optimization/deepseek-v4.1-ascend/tools/review_f01_archive.py`

## 非目标

- 本次只记录归档复核和未来计划；不操作服务器/NPU/容器，不实施kernel，不采新profile，不启动子代理。
- 不关闭其他F01子方案或其他41个候选，不把旧166时序迁移到171。

## 正确性与性能契约

当前只冻结归档审查域：有外部FP32 pre_mix的A3 HcPre源码分支；示例case为BF16 X[3,4,5120]。缺少pre_mix的分支不在独立性结论内。硬件/模型案例仍沿用项目README中的W8A8、A3、CANN9.2.0配置；旧166计时和171图回放分开记录。

Y按原转换、逐项乘法、4路归约和BF16舍入产生；保留全部新pre/post/comb、原残差、其他消费者和事件代际。新post/comb在当前HcPost使用前就绪，新pre在后续FFNPre使用前就绪。没有删除投影、统计或Sinkhorn。

尚未冻结native实现、当前二进制身份、允许误差、输入覆盖、并发接口和同条件测量协议；因此不进入实现接受或性能验收。未来正确性要覆盖全部有效输出和最终合并，性能按相同case完整边界比较A原版、B拆分顺序执行、C拆分重叠，且说明噪声。当前没有已接受的新实现基线，机器状态E0不表示历史profile不存在。

本轮结果：[归档复核摘要](../../evidence/F01-review-20261103.json)、[设计审查决定](loops/L001-early-y-review/decision.json)、[可读候选页](../../F01.md)。设计Loop为INCONCLUSIVE：源依赖得到支持，native可执行性和收益未建立；Task保持未完成。

## 恢复入口

先读[恢复包](resume-pack.json)。执行 `taskctl.py resume --task-dir <当前clone>/operator-optimization/deepseek-v4.1-ascend/progress/tasks/f01-early-y` 只恢复记录；下一步按当时用户指令和环境限制开展。

机器文件保留本次Run的真实本地路径作为来源；其他clone阅读时使用本页相对链接。待办不自动启动服务器任务或子代理。
