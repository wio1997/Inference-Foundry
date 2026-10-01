# GLM — HANDOFF

2026-10-01正式自主恢复。研究、开发、执行和raw在166，Mac仅SSH入口；仓库现场路径`/data/tiankuan/wio/Inference-Foundry`，基线c1361b81。

- **Current**：无正式可重复KEEP，数值界/稳定容量unknown。旧c3六条长度通过、SLA失败；原始吞吐包含取消attempt，不能作为独立六条有效吞吐。
- **活动点**：[GLM-OPT-0001](records/points/GLM-OPT-0001/point.md)。恢复CLI/Result/bridge通过，P/D/proxy健康；没有恢复旧队列。新唯一controller双flock、boot/PID/start ticks、阶段状态和5秒心跳已通过synthetic fixture，子进程真实退出/超时与duplicate拒绝有证据。直接legacy lifecycle仍可绕过锁，未知中断不自动重放。
- **新E2E**：[Run0001](records/points/GLM-OPT-0001/runs/GLM-RUN-0001/summary.md)：81932→256真实通过，冷TTFT38.42s/TPOT14.63ms。[Run0002](records/points/GLM-OPT-0001/runs/GLM-RUN-0002/summary.md)：81932→2048，并发1/2/1共4条全有效；c1 TPOT22.82/20.45ms，c2 21.87/26.81ms。两点均诊断INCONCLUSIVE，不提升Current。
- **当前执行**：GLM-RUN-0003新并发2完整E2E（4×81920原始输入/61440输出、90%公共前缀），准备/启动状态以服务器Run目录state.json及task controller-owner.json为准。唯一controller负责必要重启清缓存、prefix warmup、长请求、phase退出与完整请求校验；没有后台优化/完成的虚假声明。
- **代码**：runtime单owner执行与流观测；task-local AISBench adapter堵住tee掩盖exit及单请求失败继续full的缺口，fixture已通过。现役引擎和计算实现无代码改动。原始脚本/日志按Run source/artifact hash追溯。
- **下一问题**：短输出下c1也未稳定达18ms；D MTP接受率随生成轨迹变，未观察容量排队。P引擎prefix命中而响应cached_tokens=0；恢复时按engine指标解释缓存。高价值未知是TTFT的prefill/transfer/双端接口开销及长输出decode成本；准备定向阶段观测，继续证据驱动。

现场连接/操作沿用deploy/HANDOFF.md、SSH_AND_OPERATIONS.md、PD_START_AND_TEST_GUIDE.md。localhost与内网HTTP显式绕过宿主机代理。Raw和凭据不提交Git。P=166:9081、D=167:9900、proxy=166:8000；glm52-single；P MTP1 eager、D MTP3 FULL_DECODE_ONLY、TP16/EP16/DCP16。
