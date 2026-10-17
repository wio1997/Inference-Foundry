结论：候选应收窄为**把 D 初始有效 query 从最后一个 prompt token 推进到 P 已知首输出 y1**。这在状态齐全时有条件可行；保持现有 MTP5 补齐规则，首批仍是每请求6行，C4仍可为24行，不能称为省掉一次完整模型迭代。当前原生 wire 尚不能传入这份初始输出 history，最后 prompt 行及 MTP 前缀的物理有效性也未证实，本轮不强行实现新设备原型。全程仅本机文件核验，无 SSH、设备请求、续租或在线压测。

令 prompt 长度为 N。存档的 P metadata 声明 `remote_num_prompt_tokens=N`，原生 D 两轮冷 C1 每6请求的 external hits 分别为1536/6144，即每请求256/1024，支持“声明接收 N”。原生 `_update_waiting_for_remote_kv` 在完整命中时把 computed=N 回退到 N−1，为重新采样留下最后 query；这不是物理 KV 少一行的证明。若先用原生 append 把 y1 记为输出，则 all_tokens=N+1、outputs=1、computed仍N，原生晋升方法不再触发该回退。把 computed 直接加到 N+1 则错误地把 y1 的 target KV 当成已经算出。

| 边界 | 已核实的具体限制 |
|---|---|
| Worker 新请求 | `NewRequestData` 无初始 output IDs；`GPUModelRunner._update_states:1318` 构造 `CachedRequestState(output_token_ids=[])`。只改 scheduler history，worker仍缺 y1。|
| Query、位置与合批 | runner按 computed+query offset生成位置；推进后首有效位置由N−1变N。但原生K5 padding条件仍真，6→6，非删掉一批计算。混合批次还需实际确认。|
| Async归属 | P已知输出不是本轮D预留的输出。直接套普通异步完成扣减段会把0个placeholder减为−1。不能用伪造placeholder或stale标志掩盖。|
| max_tokens/stop | 保持prompt=N、native output history含y1、max_tokens=64，余下为63。把y1追加为prompt而不改预算，frontend已有64时core仍只有63且不会length-stop；改63又改变prompt/output分类、min/penalty及统计，非通用替代。|
| 原v1去重协议 | 采纳后D首结果应是y2，不能再核对/丢弃“重复y1”。对原4f42984源码块输入存档y2=198、已知y1=20的有限反例，会置FINISHED_ERROR；若ID巧合相同还可能误删有效输出。|
| 晋升与结束 | 在KV晋升回调内直接把max1置FINISHED，外层原生晋升会再改回WAITING。`finish_requests`也会跳过已经finished的请求。终止请求必须连同队列、KV/connector与in-flight释放完整处理；仅计数检查不证明生命周期正确。|

C8/Indexer 不能只看 aggregate finished。源注册逻辑按实际层、group/spec、地址、page stride、block length和scale描述缓存；packed main C8的latent、存在时的RoPE/量化scale及Indexer的K/scale均须覆盖。P/D日志各有16条注册摘要，样本num_layers=num_regions=101（cache条目，不是decoder层数）；数量不能识别具体MTP/Indexer层。接收映射还会取本地/远端块数的较小值，完成标记描述已提交任务完成，不替代最后有效行的内容证明。原生 D 会重新计算最后 prompt query，并可能重写相关cache行，既有正确性结果不能单独证明该行可跳过。当前未存档完整运行时 handshake、D各group块表或最后行readback；不推广到非对齐边界或stateful模型。

MTP caller明确使用 `target_hidden_states`、token/position及attention metadata，现有ReqMeta没有原始hidden/draft状态。由D实际查询y1产生新hidden，可作为继续draft的候选路径，但它保留了target forward和drafter工作，还要求MTP持久cache前缀有效。匹配的proposer内部源码不在本机归档，不能宣称P MTP1到D MTP5的尾部状态已经可直接续用，也不能由KV接收完成推导raw hidden已传输。

**验证层级**：本机完整原生导入被缺少兼容torch阻断，msgspec/vLLM也未安装；Linux aarch64离线wheel不能当Darwin运行时。仅执行未改函数体的原类定义/方法段：使用8条真实存档token ID检查计数、native check_stop、DTO结构及上述两个反例。Request构造器被绕过，SamplingParams、read-only view、KV manager为明确标注的fixture/回调；未执行真实InputBatch、KV分配/哈希、GPU scatter、MTP或资源回收。**这些不是完整原生CPU状态回放通过，更不是设备正确性证明。**

时序口径补充：64输出时 `TPOT=(t64−t1)/63=(gap12+span2..64)/63`。沿用原SHA核验样本，首二gap倍率2.4613是真实呈现节奏变化；第二至末token区间的run-p50几何均值倍率为1.00897，不能把平均TPOT的1.12897直接写成稳定decode变慢13%。八组等权**均值**的TPOT增加1.48368ms，其中1.30817ms（约88.2%）来自首二gap项、0.17550ms来自第二至末项；不能对中位数直接做该加法。后一段仍含batch、draft、传输及续写差异，且不同D实例/续写混合未消除，不能据此给出稳定计算变慢的因果结论。实际draft轮数3183→3240表明该次运行未减少这项工作，不等于新增57次target kernel。此前16/16窗口的负phase计数保留；缺匹配frontend stats源码与完整依赖，本轮未把事件推演冒充已验证的phase修复。

下一次最小设备判别实验（仅说明，未安排）：在新的允许窗口复用一次完整P/D加载，先用冷C1的N=256/257，核对每个target C8/Indexer及MTP层的实际metadata和最后prompt行有效性；只收到N−1或MTP前缀不齐就否定直接推进。状态齐全后，显式把y1初始输出history送入worker，保持computed=N、prompt=N、预算64，记录前两个D步骤的输入ID/位置、有效/补齐query行、hidden来源与实际步骤数，检查第二token、max1/64及取消的计数/归属。首token采样策略须对齐（现有P强制min_tokens1、启用发布的D为0，EOS等价性未证明）。只有第二token和完整请求的实际工作/质量检查支持时再扩大C4与性能测量；不得从旧trace直接预算新工作组织的收益。

可核查证据：[原生方法段及导入限制](../evidence/20261018-p-first-state-audit/native-contract-run-v3/result.json)、[原始接收计数](../evidence/20261018-p-first-state-audit/observed-transfer-counts-v1.json)、[时序分解](../evidence/20261018-p-first-state-audit/full-token-phase-decomposition-v1.json)、[源码SHA与行号索引](../evidence/20261018-p-first-state-audit/source-evidence-index-v1.json)。固定源码保存在本机 `glm53-execution-evidence-backup/20261018-offline-state-audit/source`；两个检查脚本为 `tools/offline_p_first_state_audit.py` 与扩展后的 `tools/compare_pd_publication.py`。约定精度和原有算子未改；完整模型精度仍未验收。
