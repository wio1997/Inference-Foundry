# 4P2D性能基线

状态：六组基线完成（2026-10-08）。固定0.93请求/秒的C16/C32各48请求，不限速C16/C32/C64/C96各384请求，全部成功且每条输出600 token。严格SLA没有全部通过；详见实际结果，不能把请求成功等同于SLA达标或设备满载。

此基线由用户指定：4个P节点组成DP4×TP16/EP64，2个D节点组成DP16×TP2/EP32，完整P→KV→D服务。先保持当前实现、不合入q1修复，建立并发16/32/64/96下的实际吞吐、延迟和SLA基准。

- [实际结果](RESULTS.md)
- [输入、输出及缓存 token 吞吐](TOKEN_THROUGHPUT.md)
- [深入参数实验、依赖与验收顺序](PARAMETER_EXPERIMENTS.md)
- [C64/C72 瓶颈诊断、采样缺口与验收](BOTTLENECK_DIAGNOSTICS.md)
- [完整性能报告、节点参与情况与参数优化建议](PERFORMANCE_REPORT.md)
- [逐轮缓存命中与P/D压力位置](CACHE_AND_BOTTLENECK.md)
- [不限速、并发和稳态统计定义](METRIC_DEFINITIONS.md)
- [原始证据包哈希与复算验证](PROVENANCE.json)
- [部署记录](DEPLOYMENT.md)
- [测量计划](TEST_PLAN.md)
- [全程指标复算脚本](verify_results.py)
- [独立问题专题：q1触发全局NONE](../../issues/pd-q1-global-none/README.md)

执行时冻结实际代码及既有改动哈希、镜像、模型、MTP/图模式、通信配置、cache状态、输入输出合同及到达模型。当前已有临时诊断补丁；基线明确记录其存在且逐步日志关闭，不称未经改动的stock版本。

每档结果单独保留全量完成/失败、有效输出、TTFT/TPOT分位数、实际并发和TPS。有限pilot样本不声称长期尾延迟稳定达标。后续修复/配置候选须链接具体基线Run，不能拿不同拓扑或负载混算收益。

拥有私有证据包时，可在复制出的单个run目录执行 `python verify_results.py RUN_DIR --concurrency 32 --expected-requests 384 --expected-output 600`。需要原始details JSONL和对应SQLite时间数据库；脚本将聚合结果写入该目录的analysis.json。公开仓库不包含现场逐请求数据，无法仅凭公开文件完整复算。
