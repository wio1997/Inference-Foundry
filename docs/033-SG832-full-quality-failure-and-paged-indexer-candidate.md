# SG832完整质量失败、实际设备诊断与分页索引候选

固定daily A2镜像、910B1/aarch64、CANN9.1.0、torch2.10/torch_npu2.10post6，TP8/EP8、DSPARK五个draft加root、P eager/D target与draft full graph。未换镜像、权重、量化、题目、提示、预算或严格grader。当前[权威状态](../metadata/full_model_G832.json)未通过完整质量、四档AISBench和最终匹配profile。

## 实际功能与质量失败

[父级功能检查](../evidence/quality_failure_G893/selected_functional_parent.json)核实两个固定答案及18-input/1024-output的原始SSE/DONE；它是限定功能范围。D八rank实际完成target batch4/8/12/16、width6和draft batch8/16、width5的捕获，不代表完整质量或性能通过。原生路由注册仍可能调用会生成fake-bootstrap请求的/health，即使周期端点设为/ready；此前瞬态注册失败已恢复，保留[实际源码与日志](../evidence/quality_failure_G893/native_router_startup_health_contract.json)。

同官方1319题的完整GSM8K于17:46:43UTC实际启动，C16、T0/seed0、thinking=false、max2048、无重试，worker AST与原生vLLM一致。随后14条请求各自实际达到600秒传输超时。17:58:56UTC精确结束本任务CPU客户端中的质量Python，停止新提交，保留[实际终态](../evidence/quality_failure_G893/actual_quality_stop.json)及[全部原始请求/响应](../evidence/quality_failure_G893/raw/sglang832_full_test/results.jsonl)。共36个请求文件、20行完成记录、6个实际响应、14个传输失败；仅2个响应满足原严格####答案规则。其余四个响应数学数值虽对，但格式不满足原grader，不能放宽计为正确。该结果不是1319题准确率，也不构成质量验收。与vLLM同题同body的局部响应不同，尚未证实原因，不能归咎于提示丢失。

## 实际短诊断及归因边界

在质量负载仍运行时，17:55:35–17:55:56UTC启动两step D侧诊断并由SDK自动结束/解析，八rank真实设备DB各15392条compute。该21秒诊断会干扰同期质量延迟，必须披露；不能将全部600秒失败直接归因于该窗口。它没有提交额外模型请求，也未消耗预留最终16请求匹配profile cohort。

[八rank只读设备检查](../evidence/quality_failure_G893/actual_all8_short_profile_review.json)包含实际device/model组和算子统计；CPU有step[VERIFY bs=16]与step[DRAFT bs=16]，但g_sq/c_sq聚合注释为零，不能称逻辑工作覆盖完整。SDK完成后冻结1048个文件共602151671字节，manifest及SHA见[冻结记录](../evidence/quality_failure_G893/idle_and_profile_SHA_freeze.json)，大型trace留受控存储。

D0约7秒窗口内，Index AI_CPU累计1.926秒、FP4 logits累计0.776秒、custom sparse attention累计2.971秒、MoE dispatch累计4.451秒。算子求和含重叠和等待，不是墙钟贡献，也不能相加推导收益。Index+FP4累计约2.702秒仅给出约39%的乐观无重叠上界；真实关键路径和端到端收益未验证。

[实际FloorDiv输入形状](../evidence/quality_failure_G893/actual_capacity_shapes_and_fixed_source_contract.json)为96×263435；同期短质量提示约51–81 tokens。固定nonpaged decode源码按最大上下文展开int64 slot表，再索引请求表和计算FP4分数。原FP4 tile内已有不可见tile跳过，不能把重复增加该判断当新优化。

## 单一机制候选与当前状态

[候选及验收计划](../code/paged_indexer_candidate_G901/mechanism_plan.json)只替换NPU FullTopKIndexer decode：复用镜像已有paged FP4内核，从同一请求表直接读取可见位置，初始化同容量分数为-inf，保留同一torch full-width top-k，并复用原生paged finisher映射物理KV slot。FP4编码、BF16舍入、head权重、输出容量与选择规则保持。

原D832在两次实际P/D队列为零且客户端仅sleep后[受控退休](../evidence/quality_failure_G893/D832_actual_retirement.json)，八卡清空、owner released，旧容器和全部raw保留，P与负载guard保留。[八卡组件门](../evidence/quality_failure_G893/paged_gate902_actual_start.json)已实际启动，覆盖ratio1/2、空/奇数/ragged长度、capacity263435、96行及变q/长度/请求表/packed K的NPU图重放，要求FP32分数逐位和512个物理slot完全相同。此刻结果待审；候选未接入模型，不宣称机制成立、质量通过或端到端收益。

SG正式AISBench四档仍未提交。G878仅完成runner/真实Prometheus多标签选择器准备；D侧prefill计数为零不能冒充cache miss，日志周期批量刷新的计数也必须在正式测量前核对稳定。质量修复后仍需全1319题、四档充分独立预热和首次cold/replay、最终双方P/D匹配profile及收益验证。

## 后续组件实际结果

902在原生Triton finisher编译失败；912修订后仅32个小形状case通过，随后基线TopKV2在96×131073触发设备异常，两次均已释放。完整门未通过、模型未激活，详见[034](034-paged-indexer-compiler-and-stock-topk-failures.md)。
