# 首次128K预热失败与有界超时修复

G181原始128K/C16/1024首遍预热最终为failed_requires_review：16个请求全部131072输入，其中8个输出1024，8个输出0。AISBench为这16行都标success=true，但父级按实际token验证拒绝，不能将该轮报成完成或有效性能。128K正式cold/hot均未发送，32K/64K已完成的首轮结果不重发、不替换。

P原始日志含8个唯一room的bootstrap300秒超时；D含对应8个room的waiting300秒超时。原生early allocation的P bootstrap计时从注册后开始，等待D发回KV索引，未扣除D容量排队；D waiting计时在实际metadata发出后重置。原始日志显示D早期只有约两个128K上下文的驻留容量并持续排队。该证据支持先检验deadline不足，尚未分解排队、compute、wire及因果贡献；不跨机相减时间。

原始全日志、请求JSONL/SQLite、状态、配置与失败判断已冻结。确认客户端无ais_bench子进程、父级已终态后，通过原controller、container label/image和8卡owner身份校验发出stop.request，等待两端实际释放全部8卡。没有终止未知进程。

G316创建新服务epoch，计算argv与G181完全相同，仍为TP8/EP8/MTP3/完整D图/BF16KV/相同DeepEP与chunk，仅原生SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT及WAITING_TIMEOUT均300→3600秒。新epoch重新充分预热；该调整仅是可用性修复候选，尚无128K成功或性能收益结论。G181的32K/64K与后续G316结果必须明确不同服务epoch，失败记录长期保留。

- [实际失败判断与8个唯一room](../evidence/matched_SG_G181/128K_failed_warm/assessment.json)
- [原始父级终态](../evidence/matched_SG_G181/128K_failed_warm/2202/capture/run/state.json)
- [P原始服务日志](../evidence/matched_SG_G181/128K_failed_warm/2201/capture/service_server.log)
- [D原始服务日志](../evidence/matched_SG_G181/128K_failed_warm/2202/capture/service_server.log)
- [P实际释放](../evidence/matched_SG_G181/128K_failed_warm/stop_2201.json)
- [D实际释放](../evidence/matched_SG_G181/128K_failed_warm/stop_2202.json)
- [G316 P参数](../code/sg_pd_g316/candidate_p.json)
- [G316 D参数](../code/sg_pd_g316/candidate_d.json)


## G316 新 epoch 的实际门槛

新P/D两端各16份原生dispatch ledger全部完整，无missing/unexpected；8份原生PD注册每端returncode=0。实际server_info核对TP8/EP8/PP1、上下文263296、C16、MTP3、overlap/prefix开启；D端8rank目标verify完整图完成，草稿decode和extend图完成。固定镜像与原计算argv保留。

G324已实际结束，8项数学/边界/OpenAI/native PD检查通过，最长32768输入、32输出。它只验收选定语义、长度与PD路径；没有新C16/1024性能或完整质量结论。G327据此准备新的128K/C16/1024五遍预热，首遍后暂停采集P编译基线，warmup_only=true。原G291失败与G181正式32K/64K保持不变。

- [P真实就绪与功能结果](../evidence/matched_SG_G316/readiness/2201/readiness.json)
- [D完整图/配置/原生注册](../evidence/matched_SG_G316/readiness/2202/readiness.json)
- [新epoch 32份装载ledger的逐项原始SHA与计数](../evidence/matched_SG_G316/load_ledger_index.json)
- [实际功能任务终态](../evidence/matched_SG_G316/functional/state.json)
- [完整功能原始结果](../evidence/matched_SG_G316/functional/result.json)
- [新128K预热构造与门槛](../code/sg_pd_g316/prepare_G316_first128_g327.py)
