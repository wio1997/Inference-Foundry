# 171 设备空档深入归因

2026-11-10，用户授权深入排查设备空档中的调度工作。

- 介绍：复用balanced-profile-171已有32worker/2前端trace与冻结代码，区分Host输入/metadata生产、队列提交、跨进程协调、设备/流依赖及尚未采到的调度内部。
- 到了哪里：原16卡时序分解完成；一次原生阶段标记定点补采完成，16卡trace离线导出和分析成功。去掉首尾不完整迭代，各卡11个闭合循环；16卡组间结论一致。
- 结果：16卡原steady空档约97%在目标整图执行区间外。按每卡Free占比再取16卡平均，75.61%在下一项计算任务提交API开始前、4.39%在调用中、14.93%在返回后；3.10%图内，剩余1.97%未精确绑定。是时序定位，尚不把全部信用算作可删Host成本。单线程张量操作与Free重叠很大，但缺Python主线程栈及阶段标签。
- 定点结果：prepare input占全部Free平均96.21%，draft_token 1.89%，forward 1.77%，其余约0.13%；DP0/DP1输入准备分别95.99%/96.44%。全窗口保留，仍约94%，不与原样本绝对耗时混算。代表EP0/8各11轮，协调后至首个metadata提交前约31ms，其中约27ms为空档、243次where。该边界不是精确Python子函数边界，CP metadata共同构造是源码支持的首查方向，净收益/实现仍待确认。
- 恢复：CAPTURED_RESTORED；原服务health200，8-token请求成功，serve/model/Attention/runner冻结SHA一致，原Config无新增scope开关。临时实例已停止，本轮不再持续采集。实现代码未改，凭据不落盘。
- 交付：REPORT.md、findings.json、scoped-capture/scope-analysis.json；全局性能图新增“空档在做什么”。后续先沿已有metadata-host / BatchFrame方向价共同构造与交付，不重复旧CPU-plan/slot-mapping融合信用；EngineCore调度仍未直接采集，不编造耗时。
