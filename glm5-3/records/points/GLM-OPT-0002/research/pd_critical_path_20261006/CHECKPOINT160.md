# checkpoint160 — actual source/structural research and CPU diagnostic

本阶段没有新增代码级性能 KEEP。真实GLM-5.3完整PD E2E Gain=unknown；Current=None；active PERF_KEEP空。项目未完成；没有模型加载、设备/生成/profile/benchmark Run或服务切换。

已执行：从GitHub/Mac/166同一6deda28恢复；Zcode0.16.5；Sol亲读实际PD/core/Ascend runner/聚合/模型加载源码与历史Run101 raw，独立Astra Challenger完成。最大可信可删除性能Gap/time unknown；唯一H3问题及反证见[Reset](RESET.md)。历史helper结束后的包装dispatch成本仅0.58/0.58/4.54ms，不能宣称秒级控制收益。

两机权重**结构完整性**已证明，覆盖索引全部182个文件/177474tensor：header/index成员一致、连续offset、shape/dtype字节数、精确长度、stat未变、两机header/size相同。metadata总量差恰等`rot.weight`额外payload；tokenizer、量化描述与其hash相同。payload hash、上传工具终态unknown，不能补成成功加载或bitwise payload identity。实际Ascend覆盖类已正确跳过rot，5.3 index pattern与现有frequency/offset源码匹配；注册/源码支持不代替fit或功能。

现场原始证明：JobB observations；JobC reduction与metadata。native roots仍是166TP8PP2/167TP4PP4旧GLM-5.2完整副本，public仍Run241V14/Run242 restored；三health200、metrics running/waiting0，32个native worker。未见活跃controller，owner文件仍为已失败Run244的历史记录，不按其PID或旧队列执行。没有清理/重启observer或STORE。

实际代码：[host-boundary observer](pd_completion_trace.py)与[reducer](reduce_pd_completion_trace.py)，只作为DIAGNOSTIC保留；不是产品路径或性能patch。8项CPU检查在Mac及166实际服务Python3.12通过，包含真实安装版本tracker/aggregator类：一次消费、跨批次完成、input mutation前捕获、返回/异常保持、卸载、trace截断/IO错误、缺rank/失败、时钟/计数/时序异常。无模型/kernel导入或live hook。设备安全ready、合法错过机会、真正执行及公开有效输出仍缺；不称完整device诊断已准备好。

Job→Result→bridge：

- [A](../../jobs/GLM53-READINESS-20261006A/bridge.json)：真实CLI480s超时/143，INVALID，raw保留，不伪造Result；raw包含大量历史zombie，超时原因未完全归约。后续固定范围命令代替task自由扫描。
- [B](../../jobs/GLM53-READINESS-20261006B/result.json)：CLI/inner0，VALID；全部header通过，但总量严格相等假设报warning；原Result/raw不改。
- [C](../../jobs/GLM53-READINESS-20261006C/result.json)：CLI/inner0，VALID；从B raw精确解析rot差额，不重复header scan；补tokenizer/quant/实际root身份。
- [CPU](../../jobs/PD-DIAGNOSTIC-CPU-20261006/result.json)：CLI/inner0，VALID；真实8CPU检查及两机源/package/cachemetadata核验，model calls=0。

Job VALID数和CPU检查数不计性能进展。大raw留166原Job，Git保存短结果/原退出/源码hash与关键片段。权重payload读/哈希=0。旧8dirty与两个Mac untracked JSON不加入提交，不reset/clean/force。

下一必要证据：确认上传已终态或取得可信现场transfer-complete证明；在新鲜所有权/资源条件下建立一个5.3真实标准PD功能/fit参考，再冻结H3唯一短诊断的device-ready与合法机会证明。授权已涵盖必要实现/验证/实验，无需重新批准研究；未知事实须先补，不按旧启动或PID重放。若H3不命中最大Gap立即关闭，不扩模块或进行cap/部署扫描。

持久化：code/evidence提交`5468f4204f3ca5a0bff94e59ff19dc443104c9a0`已由[真实同步Job](../../jobs/GLM53-CHECKPOINT160-SYNC-20261006/result.json)在166安全FF；8个历史dirty、index和其他untracked保留，22个已有相同新文件逐字节校验/备份后纳入Git。同步不是服务操作或性能Run。

GitHub发布：首次push被自动审批拒绝，原因为提交包含安装源码片段和主机/Run证据、目标可信/私有身份及payload授权未建立。只读API确认`wio1997/Inference-Foundry`为public，向用户说明后，用户明确回复“允许”，授权本次研究代码、安装源码片段与现场证据公开同步。随后实际push成功，GitHub branch-ref API核验`26e1c6377392872db5cd4ce3cbd4483e9939ccf6`。没有绕过审批；历史Job里“pending”保留当时身份，本批材料的待授权状态已解除。公开上传许可不代替权重上传完成事实或新模型设备验收。

Sol补读下载器：[实际完成链源码](download_completion_sources.json)显示`.msc`在单文件move后写入，`.mv`在snapshot download返回后写入；但cache也可能来自此前下载或复制。两机补读[166](snapshot_cache_166.json)/[167](snapshot_cache_167.json)的全部182引用均有cache entry，`.mv`mtime却早于末次引用权重mtime，两机还保留`._____temp`。这些不证明当前上传进行或结束，也不证明payload校验和；上传终态仍unknown。补读只读取小metadata与stat，无payload/model/service操作，不伪称为同步Job的模型验收Result。
