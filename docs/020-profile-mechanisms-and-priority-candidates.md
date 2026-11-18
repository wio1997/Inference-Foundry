# 匹配 profile 的机制差异与优化优先级

同一对8卡910B1、相同输入SHA、C16下，32K首轮正式测量中SGLang cold输出吞吐低28.13%、TTFT高58.03%，TPOT接近。该差异已实测，原因贡献尚不能仅从profile总时间确认。P/D共32份原生profiler_info显示两框架均为Level1、CPU+NPU、关闭record_shapes/stack/memory，公共采集配置相同；data_simplification为vLLM true、SGLang false。固定vLLM镜像为CANN9.2.0/torch_npu2.10.0.post7，固定SG镜像为CANN9.1.0/post6，依赖栈差异不能归因给框架本身，也没有跨镜像移植二进制。

## 已确认的实现差异

vLLM固定镜像的A2硬件策略为CAPACITY_AND_EXPERT_DENSITY。其原生selector只有在EP世界大小至少16等条件成立时才选择MC2；当前TP8/EP8/DP1/PP1使该条件不成立，因此按实际硬件与启动配置推导为AllGather。AllGatherCommImpl仍持有num_local_experts，使用NPU routing/unpermute和对应prepare/finalize，不能把此路径理解为关闭EP。这是实际源码与配置推导，尚无新运行时hook计数或因果收益实测。

SGLang使用DeepEP auto，P的设备trace实际出现CamMoeDispatchNormal/CombineNormal及其AI CPU核，D出现LowLatencyDispatch/Combine。SG原生none路径明确拒绝局部专家分片，因为其Ascend routing按全局expert范围索引；直接改成none或EP1会改变任务要求或产生错误，不作为对齐方案。

|P rank0观察|vLLM原始事件|SGLang原始事件|判断范围|
|---|---|---|---|
|MoE通信|源码推导AllGather|344次CamMoe normal dispatch/combine；8896次SDMA累计2779.76ms|优先查token所有权、payload及通信临界路径；累计时长有重叠|
|Gate/up+激活量化|360次GroupedMatmulSwigluQuant，累计309.14ms|688次GroupedMatmul，累计369.35ms，另344次swiglu_quant累计172.45ms|存在融合与工作量差异，未按同一shape/逻辑step配对|
|KDA prefill|272次原生ChunkKdaFwd，累计885.69ms|272次Triton chunk_kda_fwd，累计1088.75ms|同调用数不证明同shape；可核对状态/布局与fused入口|
|mHC相关|720次HcPre累计159.39ms、720次HcPost累计71.51ms|720次sinkhorn累计83.73ms，另有大量Reduce/Mul/Cast|需按CPU调用关联拆分，不能把全部Reduce归到mHC|

P采集窗口SG明确含8个bs1/tokens8192 EXTEND和draft步骤；D含bs1/query4 VERIFY。vLLM和SG尚未证明捕获了完全相同的逻辑forward单位。record_shapes关闭，不能从op名猜测shape；344/360等计数差异必须保留。profile开启的时延不参与正式性能计分。

## 下一步的干预顺序

1. **保持EP8的prefill通信路径**：先核对真实token分片、输入/输出张量所有权与原生局部expert编号，复用已确认存在的原生ascend_ops AllGather路由候选；用逐专家/边界token参考验证后做同机单变量诊断。现有DeepEP AI CPU/SDMA事件规模与vLLM selector构成高潜力线索，不构成可宣称的收益上限。完整基线四档首轮结果采齐前不修改服务。
2. **W8A8 GMM1+SwiGLU+quant融合**：核对固定CANN9.1实际op的INT8权重、scale、group-list与输出契约；SG现有v2 wrapper用于block-scaled路径，不能仅换类名套入W8A8，也不从CANN9.2复制二进制。数值、形状和graph门槛通过后测端到端。
3. **KDA与mHC融合/布局**：在同一逻辑step内核对shape、状态读写和CPU关联，按实测临界路径筛选；不能以总kernel sum比值作为端到端收益预测。

Prefix hot差异、排队和PD时间仍须用原始请求/计数与模型步骤拆分，未认定单一根因。不能相减跨主机未校准时间戳，也不能把WAIT、SDMA或AI CPU累计时长当作可消除的墙钟损失。本页是有证据的候选排序与机制地图起点，未验收最终因果地图或优化收益。

- [32份原生采集配置、原始报告SHA、观察与边界](../evidence/profile_mechanism_G287/observations.json)
- [P两框架原生profiler_info](../evidence/profile_mechanism_G287/role_2201.json)
- [D两框架原生profiler_info](../evidence/profile_mechanism_G287/role_2202.json)
- [vLLM原生A2策略](../evidence/profile_mechanism_G287/vllm_ascend/device/hardware_profile.py)
- [vLLM selector与阈值](../evidence/profile_mechanism_G287/vllm_ascend/ascend_forward_context.py)
- [vLLM AllGatherCommImpl](../evidence/profile_mechanism_G287/vllm_ascend/ops/fused_moe/moe_comm_method.py)
- [SG none路径EP保护](../evidence/profile_mechanism_G287/SG_source/sglang/srt/layers/moe/fused_moe_triton/layer.py)
- [正式32K原始结果与边界](017-matched-SG-first-32K-cold-hot.md)


## G309：已找到固定镜像中的原生 EP AllGather

G309旧G181候选已被[G464当前G316候选](026-current-G316-candidates-and-collector.md)替代；旧记录保留，不能部署恢复300秒超时。

后续检查实际固定7da镜像的ascend_ops dispatcher、MoE utils与hook，三文件与GLM实际服务runtime快照逐字节相同。原生ascend_ops支持prefill=allgather、decode=low_latency；它先gather实际行数、处理不等长/零行，再收集hidden和top-k，按连续局部专家编号路由，以零权重屏蔽非本地专家，最后FP32加权并reduce-scatter回原行分片。TP8/EP8、288专家、每rank36专家符合构造器约束，无须关闭EP或新增通信kernel。

G309准备了P main+draft改用ascend_ops的候选，D配置逐字节保持G181。原生draft resolver强制草稿也使用ascend_ops，故不能声称只改变target。CPU检查执行实际原生Config/constructor接口和draft resolver，验证TP=EP、连续分片、无fused shared-expert约束；未创建NPU张量、未部署服务、未验证语义或收益。实际MTP/GLM辅助hidden捕获、全图及PD仍须验证。待四档原始基线完成后再进入该优化，当前G316仅处理原生PD超时。

- [原生代码逐字节来源清单](../evidence/native_ascend_ops_G309/manifest.json)
- [候选及未验证范围](../code/native_allgather_candidate_G309/preflight_contract.json)
- [P候选参数](../code/native_allgather_candidate_G309/candidate_p.json)
- [保持原参数的D候选](../code/native_allgather_candidate_G309/candidate_d_unchanged.json)
