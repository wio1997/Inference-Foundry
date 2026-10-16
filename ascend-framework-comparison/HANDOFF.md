# 当前状态：Docker资源准备；两个对话分轨接手

最新资源核验：2026-10-17，证据时点以[新快照](inventory/20261017/docker-asset-checkpoint.json)为准。先读[服务器分工](SERVER_ALLOCATION.md)、[Docker目录与资源门控](DOCKER_ASSET_LAYOUT.md)、[执行规则](EXECUTION_CONTRACT.md)、[测试合同](BENCHMARK_CONTRACT.md)。

- 15台Docker27.1.2/ARM64/Ascend runtime和本地/data/wio目录均已验证。Docker层仍在/docker_data/data，未迁移daemon根目录。原GLM本地193文件大小/config与统一入口已核，未再次哈希全部764GB payload。
- Flash14/15/46/47使用源1同一精确快照，97文件332970562139bytes全部复制SHA通过，83索引分片头/范围检查完成；父独立复核四台文件大小/configSHA和本地NVMe，未第二次重读全部333GB。
- vLLM14/15/30/31/43/45已导入image090eeee6，父独立核实真实linux/arm64。旧source tar有blobSHA错误已阻止导入，现用固定源image重新导出并全层校验的新archive。
- SGLang B103仍由Mac压缩断点上传源1，再校验SHA、分发并导入34/35/36/41/46/47。已有后台任务正常；不要重复传输/导入/重放launcher。partial/计划容量不算就绪。B103是候选基础环境，不是Flash适配或两模型稳定认证。
- 原GLM vLLM P30/31 D43/45；SGLang P34/35 D36/41，均2P2D。Flash vLLM14P15D；SGLang46P47D。备用32/33/37；38/42排除。每角色32die须新冻结并行/真实KV/MemCache配置，不能照搬旧EP64/TP64或借备用缓存。
- 用户计划自行开两个GPT6.1 Sol High对话：A负责Flash适配，B负责原GLM53对比和优化。两对话所有实际构建、部署、启动服务、模型请求、压测和profile由Mac本机ZCode DeepSeek执行。详见EXECUTION_CONTRACT；不要自动创建/消息其他对话或恢复定时任务。
- 可先读源码和准备实现；实际模型工作等本轨道对应节点资源与独占门控通过后开始，不必等另一轨道。新Flash/2P2D尚未启动，无新模型请求或性能。满载测试初期错峰，避免共享链路干扰。

本机资源管线交接：`/Users/wio/work/sglang-glm53-33-41-deploy/docker-assets-unification-20261016/HANDOFF.md`；每台实际状态：`/data/wio/ascend-framework-comparison/assets/unify-20261016/STATUS.json`。公开仓库不提交凭证/provider/权重/镜像。服务启动前接手者须读真实状态、进程和锁；不能因历史PID存在而认为仍活，也不能因receipt较迟而重复启动。

接手提示：[Flash](tracks/flash-adaptation/START_PROMPT.md)、[原GLM53](tracks/glm53-2p2d/START_PROMPT.md)。共用v2合同：C16/32/64，每档C条独立74400前缀预热后四轮新后缀、4C条不同80000/600，预热排除，保留自然缓存。无合格基线前不得声称收益。

## 停机与历史证据边界

旧SGLang33容器（12模型20辅助1Meta）已按授权停止并保留对象/镜像/源码/配置/日志；43AISBench工具与原K8s资源声明保留。父停机后曾独立核实16台256die健康、无NPU进程（42排除前）。原K8s预留不等于free，部署前确认归属。42曾用自有临时管理Pod核实host健康，后按用户排除撤除，未改SSH/原任务；38未参与停止。见[停机证据](inventory/20261016/stop-and-reallocation.json)。不要自动恢复旧4P2D。

archive/20261016保持旧六机4P2D日更27b/官方SDK1.3/MFLOCAL/640GiB/P1D5/图/buffer_only/SMG及实际源码SHA。旧自然暖C64的失败和缺字段不合格；旧强制L3的448合格结果不能当自然暖或新2P2D性能。旧readiness.json是早期快照，不覆盖当前资源准备。历史一次性launcher不能重放。
