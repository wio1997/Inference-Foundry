# SG P/D bootstrap：边际分布对照与请求级证据范围

## 新观察：长桶计数相同，但快桶不同

复算[SG正式直方图脱敏摘要](../evidence/stage_snapshots_056/summary_v7.json)，在每侧标签稳定且窗口内无未观察重置的前提下，32K/64K的cold/hot四个窗口中，P `prefill_bootstrap` 和D `decode_bootstrap` 在**大于精确桶界0.029282294342350097秒**的每个对应非累计桶，观测数恰好相同；较短桶的分布不同。P、D快照采集不原子，且两个系列没有请求身份，因此这是两个**边际分布**的事实，不是同一批请求或P/D各次等待相等的证明。

|输入 / 轮次|P观测总数|D观测总数|P / D 大于精确桶界约0.029282秒|每个长桶计数|
|---|---:|---:|---:|---|
|32K cold|33|33|22 / 22|逐桶相同|
|32K hot|35|35|22 / 22|逐桶相同|
|64K cold|35|35|27 / 27|逐桶相同|
|64K hot|34|33|27 / 27|逐桶相同|

例如32K cold中，(25.108,40.675]秒的P/D各15个，(40.675,65.894]秒各7个；但小于等于原始桶界约0.011158秒的P只有1个、D有10个。即使强行假设两边33个观测可以任意一一对应，也至少有10对必须落在不同原始桶；**实际请求身份与是否存在这样的双射未知**。桶相同仍允许桶内时长相差很大：64K的(106.748,172.932]秒桶宽就超过66秒。

慢观测各自的均值也可以只从各侧sum和桶上下界求出保守范围。按同一0.029282秒阈值，32K hot的D慢观测均值减P慢观测均值落在 **−64.264至−58.791毫秒**；64K hot落在 **20.414至22.584毫秒**。这是两个边际均值相减的数学范围，**不是逐请求差值、跨服务时钟偏移、共同等待时间或TTFT贡献**。冷窗口P首个非空桶下界未知，相应范围很宽；不能把缺失下界设成0来获得好看的精度。既有SG快照缺process-start和`_created`，未排除中途重置及额外/旧请求；阶段只在退出时观察，尚未离开阶段的请求可能不计入。

## 启动观测记录：不是正式全程开关证明

G181后验归档的P/D启动server-info回读均记录 `enable_metrics=true`，同时 `enable_request_time_stats_logging=false`、`enable_trace=false`、`log_requests=false`、`export_metrics_to_file=false`、`log_requests_level=2`。这是**启动时点**的信息；已检查的正式轮公开清单未给这些开关的正式全程回读，候选代码也存在运行时配置入口，因此正式阶段开关实际值仍是UNKNOWN。

后来一次128K失败诊断归档的P/D发布版服务日志中，精确候选标记 `ReqTimeStats(` 的行数分别为0；这些是经过转换/脱敏的两份特定日志，不证明原始日志字段保真、其他目录不存在或正式时段未开启。**不能直接启用逐请求日志或OTLP以补证**：候选实现的输出包含请求/room标识、长度和绝对时间，其他导出路径还可能包含输入相关字段；需独立授权、隐私审查与资源隔离。

## 复核与下一步

[边际桶和均值界](../evidence/bootstrap_marginals_064/summary_v2.json) SHA `1af7fd694b87cfd877403c8021c4c4a6aa6fd330935fea81c73ff599ecffb8b4`；[启动开关与发布日志标记汇总](../evidence/observability_064/summary.json) SHA `442a3f86ec048996e9a2053c4af520a7a8362f17709673e99a9548afc8b07928`。边际计算只读已脱敏阶段摘要；开关/标记复算器会读取归档server-info响应和发布版日志到内存，只输出白名单布尔/计数，不输出原文、身份或绝对时间。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize_bootstrap_marginals_064.py --check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_summarize_bootstrap_marginals_064.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_summarize_observability_064.py
```

最有判别力的后续证据仍是同服务epoch的真实import/路由版本，以及同一匿名attempt的P/D状态、锁后有效预算与实际队首判定。缺失时保持UNKNOWN，不从相似边际分布反推因果或优化收益；新埋点、请求和服务改动需另行授权。
