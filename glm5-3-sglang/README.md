# GLM-5.3 / SGLang / Ascend 910C

> 本分支默认优化入口已转到 [glm5-3-vllm](../glm5-3-vllm/README.md)。本目录保留 SGLang 研究与部署资料；切换研究基础不否定已有证据，也不改写其适用范围。

独立研究线：复用已有 SGLang 环境，研究产生同样正确服务的更好执行方式。参数搜索由用户另行安排，SGLang 是工程起点，不限定研究只能修改其参数或模块。

| 入口 | 用途 |
| --- | --- |
| [MISSION](MISSION.md) | 目标、当前约定与探索空间 |
| [AGENTS](AGENTS.md) | 少量协作规则与设备操作边界 |
| [HANDOFF](HANDOFF.md) | 本线事实、未知项与下一决定 |
| [PLAN](PLAN.md) | 按需参考的研究判断与最小验证方法 |
| [Ascend MemCache PD部署](ASCEND_MEMCACHE_PD_DEPLOYMENT.md) | 独立部署工单的安装、4P2D配置、80K/600验证与回滚 |
| [AISBench MemCache 前缀测试](benchmarks/aisbench_memcache_prefix/README.md) | 可保存复用的 80K/600、约93%独立前缀命中、16/32/64并发四批测试脚本 |

研究入口最初建立于 2026-10-14；之后本目录已加入独立部署工单的验证快照和测试脚本，范围见上表对应资料。旧 HANDOFF 的日期快照、部署时的进程状态和固定负载结果不代表当前现场，也不转为新 vLLM 线的基线或收益。

旧经验按问题查阅 [历史审计](../glm5-3/docs/RESEARCH_RESET.md) 和 [补丁证据](../glm5-3/patches/20261007/README.md)。引用时保留原环境与适用范围；旧 Current、运行任务和收益不转为本线事实。

续接提示：`继续 glm5-3-sglang 研究；按 AGENTS.md，从 MISSION.md 和 HANDOFF.md 恢复，按需读取相关代码与证据。`
