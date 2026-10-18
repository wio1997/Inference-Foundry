结论：候选应收窄为**把 D 初始有效 query 从最后一个 prompt token 推进到 P 已知首输出 y1**。这在状态齐全时有条件可行；保持现有 MTP5 补齐规则，首批仍是每请求6行，C4仍可为24行，不能称为省掉一次完整模型迭代。当前原生 wire 尚不能传入这份初始输出 history，最后 prompt 行及 MTP 前缀的物理有效性也未证实，本轮不强行实现新设备原型。全程仅本机文件核验，无 SSH、设备请求、续租或在线压测。

首批6→6只说明首批未删，不证明整请求绝不可能少一次target forward；最终须比较同预算/同采样完整请求的实际前向次数。[下一步审阅包](../diagnostics/p-first-state-v1/README.md)已准备边界、两输入和身份核验，**设备捕获适配器尚未实现，不应据此申请窗口**；当前建议先做无需占卡的镜像源码身份核验。包内Mac检查也不是设备状态/事件回放验证。

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

2026-10-18补取官方固定提交的12个必要源码文件，共425191字节，未下载完整框架或安装运行环境。vLLM固定在`ced6857`、Ascend固定在`5f828ad`。**新增文件均仅为固定提交参考，未找到可配对的现场逐文件SHA；不能称为实际镜像源码已验证。** 原runner/scheduler/engine仍使用此前实际镜像捕获且SHA核对的文件。下载URL、SHA、24个定义行号、证据等级及限制见[补充索引](../evidence/20261018-p-first-state-audit/fixed-source-followup-v1/source-and-state-findings.json)。

沿实际配置而非类名猜测：存档config是`glm_moe_dsa` / `GlmMoeDsaForCausalLM`，78个主层、1个nextn层；launch是`deepseek_mtp`、D K5。固定提交配置将别名归一为`mtp`、draft架构改为`DeepseekV32MTPModel`，不满足Gemma4、Step3.5或`glm5_next_mtp`分支，因此[factory](https://github.com/vllm-project/vllm-ascend/blob/5f828ad638ee4f9b12db87f9d5481f29076d37f9/vllm_ascend/spec_decode/__init__.py#L42)选`AscendEagleProposer`。其核心实现是`AscendSpecDecodeBaseProposer`；模型注册将该draft架构指向`AscendDeepSeekMTP`。这是固定提交加存档配置的路径判定，**没有现场proposer对象类型记录，新增factory/config自身也未与镜像SHA配对**。

[首轮输入逻辑](https://github.com/vllm-project/vllm-ascend/blob/5f828ad638ee4f9b12db87f9d5481f29076d37f9/vllm_ascend/spec_decode/llm_base_proposer.py#L1782)在默认非parallel、传hidden的路径中左移token IDs，以next token替换各请求有效采样槽，保持target positions并复制对应hidden。由此可明确**移位关系**：若P的MTP首轮已运行并转移完成，位置N−1的MTP行由`y1 + h(prompt[N−1])`形成；推进后的D查询`y1@N`产生新target hidden并采样y2，其首个有效MTP行用`y2 + h(y1)`写入位置N。因而不必推断必须传P raw hidden，但必须已有正确的前缀至N−1。模型只有1个物理nextn层，按draft step对层数取模，P K1/D K5不代表需要五套层历史。匹配runner在drafter之后才finalize connector，但仍有input-fits/K等分支；这说明可能的初始化顺序，**不证明现场P尾行或转移内容**。缓存注册还可能含共享/Indexer项，必须按真实层名和映射核对。

该源码没有否定直接推进，却没有解除物理状态门槛：MTP模型会调用decoder层，使用attention/cache状态；新hidden和已知token ID不能替代缺失前缀。padded输入路径保留拒绝行、通过索引选择有效采样行，也进一步反对“推进query就删掉首批6行”的推断。没有执行新InputBatch、scatter、MTP或新模型原型。

**验证层级**：本机完整原生导入被缺少兼容torch阻断，msgspec/vLLM也未安装；Linux aarch64离线wheel不能当Darwin运行时。仅执行未改函数体的原类定义/方法段：使用8条真实存档token ID检查计数、native check_stop、DTO结构及上述两个反例。Request构造器被绕过，SamplingParams、read-only view、KV manager为明确标注的fixture/回调；未执行真实InputBatch、KV分配/哈希、GPU scatter、MTP或资源回收。**这些不是完整原生CPU状态回放通过，更不是设备正确性证明。**

时序口径补充：64输出时 `TPOT=(t64−t1)/63=(gap12+span2..64)/63`。沿用原SHA核验样本，首二gap倍率2.4613是真实呈现节奏变化；第二至末token区间的run-p50几何均值倍率为1.00897，不能把平均TPOT的1.12897直接写成稳定decode变慢13%。八组等权**均值**的TPOT增加1.48368ms。有两种成立的分解：直接按首/末时刻分，首token前移项`−Δt1/63=1.29091ms`（87.0%），末token项0.19277ms；按gap/后段分，首二gap项1.30817ms（88.2%），第二至末项0.17550ms。88.2%还包含第二token时刻变化的0.01727ms，两者不能混称；不能对中位数做这些加法。后段仍含batch、draft、传输及续写差异，且不同D实例/续写混合未消除，不能给出稳定计算变慢的因果结论。实际draft轮数3183→3240表明该次运行未减少这项工作，不等于新增57次target kernel。

负phase的来源现在有固定提交源码解释：[OutputProcessor](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/output_processor.py#L664)先以`is_prefilling=True`处理提前发布，随后置False；[stats](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/metrics/stats.py#L498)因此保存发布时间为first_token_ts。实际镜像scheduler异步接收分支在记录SCHEDULED前continue，v1先在finished_recving回调发y1；后续真正运行才记录SCHEDULED。stats虽继续接收这个事件，却保留原first_token_ts，完成时计算`prefill_time=first_token_ts−scheduled_ts`，此顺序必为负。原方法的[定义级检查](../evidence/20261018-p-first-state-audit/fixed-source-followup-v1/stats-definition-result.json)对显式10.00/10.25时间fixture得到−0.25秒，普通顺序得到+0.25秒。**不是现场事件回放、完整frontend验证或phase修复通过**，新增stats文件也没有实际镜像SHA配对，原16/16负增量保留。修复需把对客发布时刻与target/prefill计算phase的开始/完成事件分开，原生重复输出被抑制时也须传递计算事件；截零或伪造更早SCHEDULED不能建立正确口径。

下一次最小设备判别仍有价值，但限于检验上述剩余物理前提，**不是直接扩大性能测试或承诺收益**（仅说明，未申请/安排窗口）：在新的允许窗口复用一次完整P/D加载，先用冷C1的N=256/257，核对每个target C8/Indexer及MTP层的实际metadata和最后prompt行有效性；只收到N−1或MTP前缀不齐就否定直接推进。状态齐全后，显式把y1初始输出history送入worker，保持computed=N、prompt=N、预算64，记录前两个D步骤的输入ID/位置、有效/补齐query行、hidden来源与实际步骤数，检查第二token、max1/64及取消的计数/归属。首token采样策略须对齐（现有P强制min_tokens1、启用发布的D为0，EOS等价性未证明）。只有第二token和完整请求的实际工作/质量检查支持时再扩大C4与性能测量；不得从旧trace直接预算新工作组织的收益。

可核查证据：[原生方法段及导入限制](../evidence/20261018-p-first-state-audit/native-contract-run-v3/result.json)、[原始接收计数](../evidence/20261018-p-first-state-audit/observed-transfer-counts-v1.json)、[时序分解](../evidence/20261018-p-first-state-audit/full-token-phase-decomposition-v1.json)、[源码SHA与行号索引](../evidence/20261018-p-first-state-audit/source-evidence-index-v1.json)。固定源码保存在本机 `glm53-execution-evidence-backup/20261018-offline-state-audit/source`；两个检查脚本为 `tools/offline_p_first_state_audit.py` 与扩展后的 `tools/compare_pd_publication.py`。约定精度和原有算子未改；完整模型精度仍未验收。
