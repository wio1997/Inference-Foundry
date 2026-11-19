# GLM-5.3-Flash W8A8：910B 上的 SGLang / vLLM P/D 对比

本分支按当前任务持续归档测量、机制、源码与失败证据。**性能对齐尚未完成；32K首次冷热观测相对旧SG输出吞吐高16.01%/18.25%，独立归因和可重复收益仍待验证。** 所有时间为 UTC；日期为 2026-11-17。

**复现 GLM-5.3-Flash W8A8 的 SGLang 环境** · [六项适配、补丁源码、固定镜像和 P/D 启动配置](docs/038-GLM53-G32-adaptation-and-reproduction.md)

| 编号 | 对象 | 一句话结论 | 关键数字 | 状态 | 入口 |
|---|---|---|---|---|---|
| 001 | 固定镜像、W8A8、TP8/EP8/MTP3 | vLLM 四档、原 SGLang 32K 的冷热原始测量已保存 | SG 32K 冷/热输出 128.3895 / 126.3610 token/s | 有测量；最终公平对齐未验收 | [测量与边界](docs/001-baseline.md) |
| 002 | 当前双框架 P/D profile | 每框架各16个rank有真实模型NPU事件 | SG 每8步出现28,080次除法；窗口差异已列明 | 诊断证据；不能作墙钟份额 | [profile地图](docs/002-profile.md) |
| 003 | 20轮 Sinkhorn 融合 | 32K首次正式冷热观测改善，最终对齐未完成 | 输出148.9474 / 149.4161 token/s | 独立归因/重复性待验证 | [首次端到端观测](docs/004-first-end-to-end.md) |

正式与预热分账；未通过稳定门槛的预热不改写为正式结果。权威数据在 [metrics.json](records/metrics.json)；[逐Run索引](records/runs.json) 保留种类与裁决。

[环境与合同](environment/contract.json) · [离线验档](scripts/validate_archive.py) · [原始数据与脱敏](docs/003-evidence.md)

[G43 完整模型融合 profile](docs/005-fused-model-profile.md)

[vLLM G4 32K 未计分预热](docs/006-vllm-G4-warming.md)

[vLLM G4 首轮32K正式冷热](docs/007-vllm-G4-first-32K.md)

[vLLM G4 首轮64K正式冷热](docs/008-vllm-G4-first-64K.md)

[vLLM G4 首轮128K正式冷热](docs/009-vllm-G4-first-128K.md)

[同一vLLM G4运行时四档首轮冷热完整基线](docs/010-vllm-G4-four-shape-first-cold-hot.md)

[G4 profiling真实设备窗口验收与D触发修正](docs/011-vllm-G4-profiler-device-gate.md)

[G4 同一采集批次十六 rank 真实模型设备窗口](docs/012-vllm-G4-all16-model-profile.md)

[同机 SGLang 迁移、控制器目录失败及修正](docs/013-matched-host-SG-migration.md)

[同机 SGLang 实际加载、完整图与 PD 功能门槛](docs/014-matched-SG-real-PD-readiness.md)

[同机 SGLang 首遍完整32K/C16/1024未计分预热](docs/015-matched-SG-first-full-warmup.md)

[同机 SGLang 七轮独立预热与编译门槛](docs/016-matched-SG-seven-warmups.md)

[同机 SGLang 32K 首轮正式cold/hot及待定位差异](docs/017-matched-SG-first-32K-cold-hot.md)

[同机 SGLang 与G4匹配输入的十六rank模型profile](docs/018-matched-SG-all16-model-profile.md)

[同机 SGLang 64K 五轮独立预热](docs/019-matched-SG-64K-five-independent-warmups.md)

[匹配 profile 的机制差异与高潜力候选](docs/020-profile-mechanisms-and-priority-candidates.md)

[同机 SGLang 64K 首轮正式冷测与热测](docs/021-matched-SG-first-64K-cold-hot.md)

[首次128K预热失败与原生超时修复](docs/022-first-128K-warmup-failure-and-timeout-repair.md)

[G316 128K 五轮独立预热与正式提交门槛](docs/023-G316-128K-five-independent-warmups.md)

[同机 SGLang 128K 首轮正式冷测与热测](docs/024-matched-SG-first-128K-cold-hot.md)

[G316 D 编译缓存验收更正：128K完整协议未验证](docs/025-G316-D-compile-root-correction.md)

[当前G316优化候选与独立D编译采集修正](docs/026-current-G316-candidates-and-collector.md)

[256K五轮预热与当前P/D独立编译证明，首次正式已失败留证](docs/027-G316-256K-five-warmups-current-cache.md)

[256K首次cold32：18成功、14路由503，hot未提交](docs/028-G316-first256-cold-router-failure.md)

[32K单窗口差异地图与32K/64K客户端配对：非因果](docs/029-single-window-difference-map.md)

[D侧rank0/1逐事件分布：诊断窗口、未匹配工作负载](docs/030-decode-rank-event-distributions.md)

[AISBench base API 源码归档更正及语义检查](docs/031-aisbench-base-api-source-correction.md)

[P侧逐事件与形状差异：诊断窗口、尚未完成因果归因](docs/032-p-prefill-event-analysis.md)

[256K→32 未计分诊断与 D rank0 阶段证据](docs/033-256k-diagnostic-stage-evidence.md)

[vLLM G4 混合缓存容量口径和历史 D 采样](docs/034-vllm-hybrid-cache-scope.md)

[256K 冷轮执行器工作目录错误与 claim 保全](docs/035-cold-executor-cwd-failure.md)

[32K P侧rank0独立局部时间轴与取整边界](docs/036-p-rank0-observed-timeline.md)

[32K D侧rank0独立局部时间轴与显示轴边界](docs/037-d-rank0-observed-timeline.md)

[32K P/D 全32 rank设备 kernel 分布与脱敏范围](docs/039-all-rank-kernel-distributions.md)

[32K D侧同物理shape/dtype事件分布：仍非同step因果](docs/040-d-physical-shape-distributions.md)

[32K P侧全量记录输入shape字段交集与空槽边界](docs/041-p-recorded-shape-overlap.md)

[32K D侧逐rank GMM记录形状与事件分布](docs/042-d-gmm-by-recorded-shape.md)

[32K D侧GMM独立序列分段与原始计数器](docs/043-d-gmm-ordinal-counters.md)

[32K D侧GMM主stream原始字段与专家工作量待验证项](docs/044-d-gmm-counter-scope.md)

[专家工作量的现有证据、源码与补采隐私边界](docs/045-expert-observability-scope.md)

[32K P侧同记录shape/dtype的单事件分布](docs/046-p-shape-dtype-distributions.md)

[32K P侧动态量化的记录格式与设备字段](docs/047-p-quant-format-counters.md)
