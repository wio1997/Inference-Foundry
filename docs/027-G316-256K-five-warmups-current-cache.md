# 256K五轮预热与当前P/D编译证明

同一G316 epoch完成五轮独立C16、每轮16请求，共80请求，实际262144输入/1024输出全部通过父级原始JSONL、CSV、汇总算术与SHA复核；实际流式server-usage源码hash逐项重查。预热不计分，首轮原始SHA与暂停前记录相同，恢复时未重发已完成请求。

当前D411个缓存文件的独立基线取得于warm3期间，严格早于完整warm4和warm5；五轮结束后的当前D末态与其完全相同。P636项独立末态与首轮预热后基线完全相同，新增、删除和大小/mtime变化均为零。原采集器误读旧G181的快照和passed字段完整保留，不能作为当前D证明；本次采用另外的物理目录、实际容器/挂载身份和独立区间证据。最后两轮duration变化0.398799%、TTFT0.626998%、TPOT1.793722%；预注册主门槛duration/TTFT不超过5%，TPOT作为guardrail同时报告。

修正采集器从独立目录恢复，只改编译证据收集；原manifest、输入、客户端、服务ID、五轮预热与最初08:42:32.657316UTC截止时间保持。首次cold32于06:24:35UTC实际开始，06:29:56观察仍在运行；hot尚未提交。现在没有任何256K正式指标或收益通过结论，不以预热吞吐替代正式结果。正式阶段仍需当前D前后快照、原始usage/五项指标/prefix计数和独立P末态验收。

此前128K的历史warm manifest也显式补充了D编译未验证资格，原始测量不变；完整无编译干扰验收没有恢复。

- [父级80请求与当前P/D证明](../evidence/matched_SG_G316/256K/warmups/G521_parent_256K_allwarm_D_acceptance.json)
- [当前D跨完整warm4+5区间](../evidence/matched_SG_G316/256K/warmups/current_D/D_256K_warm_current_cache_proof_g496.json)
- [P独立末态](../evidence/matched_SG_G316/256K/warmups/G520_P256_allwarm_cache_2201_parsed.json)
- [首次正式实际句柄](../evidence/matched_SG_G316/256K/warmups/G522_first256_corrected_formal_2202_parsed.json)
- [实际cold阶段观察](../evidence/matched_SG_G316/256K/warmups/G524_first256_formal_actual_2202_parsed.json)
- [五轮原始JSONL、SQLite与指标索引](../records/runs.json)
- [更正后的独立采集器](../code/SG_current_D_collector_G488/run_aisbench_cold_hot.py)
- [128K验收更正](025-G316-D-compile-root-correction.md)

后续06:57:41UTC首次cold因14次路由503验收失败，hot未提交；最新结论与原始证据见[失败记录](028-G316-first256-cold-router-failure.md)。上文开始时观察保留为历史时间点。
