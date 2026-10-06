# Performance Research Reset — checkpoint160

2026-10-06；新的执行授权已生效，不继承早前“先调整方案”的仅资料阶段限制。

身份：`glm5-3-autonomous-20261001`；恢复 HEAD/rule commit `6deda28e01718ed28a6abe2d03f28ca48a42bcfe`，GitHub/Mac/166一致；`GLM-RESEARCH-RULES-v2`；active point `GLM-OPT-0002`。166的8个历史tracked dirty与本机两个既有untracked JSON保留。新记录不改历史Run或裁决。

产品与Current：GLM-5.3 W8A8动态、完整功能、标准 `P Prefill→KV Transfer→D Decode/MTP→有效用户输出`，P/D各自DP/TP/EP；既有TTFT P50/P75/P90/P99<4/8/12/30s、TPOT P50/P90<18/40ms。Current=None；5.3真实完整PD E2E、稳定有效容量与Gain unknown。现役仍是GLM-5.2完整副本；新模型没有加载/设备测试。合同见MISSION，现场见JobC reduction与JobB原始observations。

Active KEEP stack：CURRENT_PERFORMANCE_STACK为空；本次CPU诊断保留不进入产品stack。

最大代码Gap：**最大可信可消除性能Gap及时间均unknown**。最大的已证明产品缺口仍为optional-PD/local fallback和当前无PD驻留。唯一条件性能问题H3：全体必需贡献者已拥有成功、设备安全KV后，晚到的completion observation是否错过一个本来合法、尚未commit的D admission机会？这不是已证明的Gap，也不是out-of-band通知patch。

决定性证据（Sol亲读）：

1. `critical_path_sources.json`：实际Mooncake线程transfer/reformat→tracker、worker get_finished、Scheduler remote-wait/promotion、EngineCore queue/commit；whole-file SHA与行定位保留。
2. `native_consumers.json`、`native_aggregator_class.json`：实际Ascend V1 no-forward polling、target-forward capture与后续MTP输出；聚合跨调用等待expected contributor count，不能以rank0代替。
3. `run101_pd_events.json`：历史实际三请求的P helper wall7.48956/7.51535/60.03485s，helper completed→D dispatch仅0.582865/0.575905/4.544121ms；没有transfer-ready/D eligibility/device trace，不补成5.3性能。
4. Job `GLM53-READINESS-20261006B/C`：全部182引用文件/177474tensor的头部映射、连续offset、shape/dtype字节数、精确文件长度、stat稳定及两机header/size一致；不读tensor payload。metadata总量差75497472B恰等rot payload，保留B原始警告并由C另作归约。tokenizer/量化元数据解析通过；上传工具终态/payload校验和仍unknown。
5. `model_support_sources.json` + `PD-DIAGNOSTIC-CPU-20261006/reduction.json`：真实Ascend注册覆盖类合法跳过`rot.`，不能根据基类lookup宣称rot加载不支持。两机同模型覆盖类/Mooncake hash与包版本；这些不证明fit/功能或E2E。

代码假说与patch：**无性能patch**。实现opt-in `pd_completion_trace.py`、`reduce_pd_completion_trace.py`，观测现有调用一次，保存原return/exception，不消费future、不额外读completion、不改Scheduler/Graph/计算/通信策略。明确CPU marker≠device ready；所有请求含失败/缺失保留，时钟域隔离，丢trace/缺rank/计数或时序异常拒绝计算。reducer只归约host envelope，removable_ms/E2E gain保持null。新增成本为诊断JSON写入/字段提取，正式性能必须关闭且实际开销unknown。未装入驻留进程。

最小matched A/B：尚未冻结性能A/B。先在确认上传终态或可信现场transfer-complete证据、安装支持/fit/原生PD功能、唯一controller所有权后，冻结一个短标准PD baseline diagnostic：idle-D receive + receive overlapping ongoing decode；不跑配置扫描或大型E2E。必须补设备安全ready、合法未commit机会、执行/commit/有效公开输出证明；当前host诊断代码单独不充分。

Decision table：首个合法batch已进入→关闭H3；late observation造成重复可证的合法机会遗漏→再定义唯一通知/派发候选；rank/device/committed work/budget/KV/grammar阻塞→按真实原因归因并换题；身份/设备/合法性缺失→INCONCLUSIVE，不把host间隔当节省。后续候选才做风险相称correctness及至少两次matched完整E2E或预定义稳定窗口，近噪声用bracketed比较；所有请求/错误/拒绝/有效输出及分位SLA一并审计。

Astra：YES。触发为Current长期None、最大PD Gap不清和现有错误产品前提。独立Review见`ASTRA_REVIEW.md`；Sol全接受证据纪律和单诊断建议，拒绝把排名当已测性能大小，也不激活Top3队列。

H1保持旧窗口scoped REJECT；H2保持PARKED且非当前PD路线。没有新设备Run，不创建虚假Run ID。**本阶段没有新增代码级性能 KEEP。**

补证据边界：`download_completion_sources.json`与两机`snapshot_cache_*.json`不能把历史cache补成当前上传终态；`.mv`mtime早于引用权重末次mtime。加载gate未解除，用户的上传工具状态问题尚无回复。用户已明确授权本次研究代码、安装源码片段与现场证据公开同步；实际push成功，GitHub API核验`26e1c637`，本机/166已为同一证据提交。后续身份以fresh核验为准，不按历史Job的待授权状态重新阻塞本批上传。
