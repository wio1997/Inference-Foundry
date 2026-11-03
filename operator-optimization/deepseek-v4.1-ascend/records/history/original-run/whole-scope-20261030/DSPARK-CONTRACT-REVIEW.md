# DSpark 与固定负载的条件合同：继续补全整体范围

**状态：局部源码与配置推导；全部消费者、真实调用身份和整体覆盖仍OPEN。** 本文不修改目标实现，不报告模型提速。旧F27–F31及新WS-VOCAB/REJECT方向继续保留其条件与未知；以下是细化，不增加互相独立的收益项。

## 纠正一处取证误读

此前输出头记录将V4.1的DSpark依赖误写成 `vllm.models.qwen3_dspark`，继而登记“legacy导入重定向未知”。冻结 `models/deepseek_v41/dspark.py:31` 实际直接导入 `vllm.model_executor.models.qwen3_dspark`；`spec_decode/llm_base_proposer.py:29` 也是legacy路径。此前查找的新路径是错误假设，不是缺失运行依赖。原错误记录保留于版本快照，当前已纠正。

新补采保留了8个源码文件、固定两库commit与tracked clean；DSpark头文件现在有正式快照，见[补采manifest](contract-source-supplement/frozen-source-snapshot/SOURCE-MANIFEST.json)。另两个猜测的 `entrypoints/openai/engine/{serving,protocol}.py` 路径不存在，前端所需基类依赖仍需按真实import继续追踪，不能说前端范围已经闭合。

## 固定配置与实际观测分开

源文件事实为DOCUMENTED；由保存的serve/client/config及源码推出的路径为DERIVED，条件是采集使用这些输入、默认值未被其它已知或未知入口覆写。运行数据形状/计数为MEASURED，但形状相容不等于张量或实例身份。

| 条件 | 已有依据及当前判断 | 保留的限制 |
|---|---|---|
| target头布局 | 保存serve为TP8，PCP/DCP未指定；ParallelConfig默认均1；FinegrainedTPConfig的LMHead大小默认0，serve未提供该子配置。因此标准初始化可推导STANDARD，Markov的disable_tp优先推导REPLICATED | 未直接读取历史worker实例、权重和LP对象；动态patch/初始化旁路仍需闭合 |
| target greedy/logprobs | 官方client显式temperature=0，未请求logprobs、prompt_logprobs、logprob_token_ids。CompletionRequest将这些字段传给SamplingParams；InputBatch依SamplingType登记greedy并以空num_logprobs映射返回None | 此推导只针对固定客户端请求；历史batch是否含其它请求和动态覆写需结合采集隔离与请求身份，不把所有processor默认删除 |
| entropy/block | RejectionSamplerConfig默认均false，保存additional_config无该覆写 | 仍应绑定最终AscendConfig；clone局部证明中的无额外可观察alias条件保持 |
| draft模式 | 保存speculative_config为dspark/K=5/enforce_eager=true；SpeculativeConfig对dspark设parallel_drafting=true，draft_sample_method默认greedy | 不是K次decoder forward；随机分支留作有证据排除的条件范围，不能删除其语义记录 |
| dynamic confidence | DynamicSpecConfig.method默认None；保存additional_config无dynamic_spec_config，DSparkProposer仅method=dspark时创建scheduler，merged draft仅其非None才update | 头被构建/载入不等于运行执行；实际对象仍待直接或同次证据确认 |
| packed cache | 当前服务定名_build_info.py为A3；A3 profile选择STANDARD adaptor，BaseDeviceAdaptor返回packed_cache_ops=None，A5覆盖返回packed ABI | 当前磁盘metadata不是历史加载对象的直接观测；硬件适用排除与历史调用绑定分开 |

[定名metadata回执](fixed-contract-metadata.json)记录 `_build_info.py` 的SHA与源码commit。指定checkpoint的 `generation_config.json` 未找到。不能由“文件不存在”断定所有生成参数均为中性默认；ModelConfig还经HF辅助函数取得默认值，完整前端依赖与其回退仍待追踪。显式temperature与不经过generation whitelist的logprobs字段不受这一个未知项同等影响。

## DSpark真实局部图与值/状态边

冻结主配置：40目标层、384目标专家/top6；draft是3层、128专家/top3、K=5、Markov rank256、目标辅助层ID列表[37,38,39]。draft_config patch修改其私有配置中的专家数与topk，不能套用目标384/top6作draft计算账。辅助状态的配置ID、runner表达和真实生产层还需完整核对；target源码用 `layer.layer_idx+1 in aux_hidden_state_layers` 判断，并在进入该层前对HC维mean取状态。不得仅凭数字相同称为“第37–39层输出”。

1. `combine_hidden_states`：拼接的辅助状态经main_proj与main_norm；各draft层的context投影分别读取该状态，按各自wkv、kv_norm、RoPE和cache槽写入。名称shared_kv不证明不同层权重相等或可删多层计算。
2. draft forward：embedding后扩展HC，3个decoder层顺序变换hidden/pre_mix，最后collapse；sequence-parallel分支还分片/汇集并裁剪padding。三层输出存在真正的数据依赖，不能只拼成一个“并行三层”节点。
3. merged proposer一次生成B×K的基础logits（标准模式），view为[B,K,V]。draft_buffer[:,0]复制本轮seed。每个位置i读取token_i→Markov embedding→复制的Markov词表投影→对logits[:,i]原位add→argmax（或概率采样）→写token_(i+1)。循环结束返回draft_buffer[:,1:]，这是持久buffer的view，下一轮复用不能早于所有消费者完成。
4. greedy分支没有构建draft概率列表；概率模式会保留整词表概率供后续拒绝/恢复，不能用一个选中概率替代。映射词表分支还要保证target ID用于下一次embedding、draft ID用于相应base/bias索引。
5. dynamic_spec存在时才读取confidence相关信息；概率和confidence均关闭的条件下，不能把未执行的confidence投影时长算为可节省成本。
6. DSpark返回的tokens经target验证决定真实接受数；其结果又决定下一轮seed、有效seq_len、槽位和拒绝回滚。`set_inputs_first_pass`复制seed、context hidden并修订metadata，其中slot buffer按cache group分别生成。以上状态边不能由图中未画出而忽略。

源码入口：[draft model](frozen-source-snapshot/vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py)、[merged proposer](frozen-source-snapshot/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py)、[DSpark inputs](frozen-source-snapshot/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py)、[Markov/Confidence头](contract-source-supplement/frozen-source-snapshot/vllm/vllm/model_executor/models/qwen3_dspark.py)、[draft配置patch](sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/patch/platform/patch_speculative_config.py)。

V4.1 draft类自身将 `draft_id_to_target_id=None`，因此在标准构造且无覆写时可进一步推导无映射；目前分支台账仍保守保留实际对象未直接观测的限制。该限制不表示源码没有给出默认值。

## 融合、拆分与重组覆盖的新增细化

| 原边界/规则 | 具体分析 | 合法性与取证缺口 |
|---|---|---|
| F30，R01/R03/R09 | Markov投影→bias加法→argmax，可以按词表tile在消费端组合，避免完整bias/add结果物化；位置间仍传选中token | 原matmul、BF16 bias舍入、BF16 add舍入和tie/NaN语义须保持，直接FP32合并不是自动等价；base logits仍需保留至相应位置 |
| F30，R02/R05 | 基础投影的K个位置可按就绪次序供数，较早位置的Markov选择可能与较晚位置base投影重叠 | 拆小M会增加权重读取/启动和降低GEMM效率；真实tile-ready接口、流与资源竞争未证，不能把理论重叠当收益 |
| F30，R06 | 将复制Markov头按词表分片，局部应用base+bias后只归约候选，或在复制布局下保持无collective的逐位置链 | 前者每步有跨卡依赖且必须对齐base/head所有权；可能比当前复制更慢。已有disable_tp为当前事实，不能重复作为新增优化 |
| F30，R07/R08 | 对重复previous-token的Markov bias按权重版本、dtype、layout和精确算术合同复用，或按请求共享相同投影结果 | 需同次token重复率、cache工作集、命中/替换成本及全部消费者/缓冲生存期；当前未测，不能按“token重复常见”给数字 |
| F27，R03/R05/R07 | 辅助层状态逐步就绪后，分解main_proj的输入块并提前算部分贡献，再在最后状态到达后完成合并/norm | 改变FP32累加/量化和输出舍入可能改变值；原SP gather与草稿权重所有权不能删。辅助状态真实生产点/消费者未闭合 |
| F29/F31，R02/R04/R05 | 多个draft层context投影可共享输入读取、位置索引或cache组元数据，按层/行分块发布 | 每层wkv/norm/RoPE/cache各有语义，不能假定KV数值跨层相同；空输入、组blocksize、槽位-1及覆盖事件需逐项保留 |
| F28/F31，R08/R10 | 下一轮与接受结果无关的固定形状/常量索引可提前准备；seed、有效长度和拒绝相关槽位继续受结果依赖 | 要区分CPU镜像与device权威状态，上一轮buffer最后读和graph静态地址的约束仍未绑定 |

Markov源码给出逐位置依赖，只能约束这个直接实现族的排程；不能据此宣称所有允许算法都具有相同串行下界。枚举全部前驱token、函数复合、投机重算等算法会改变工作/存储域，需单独分析，不能无依据写“不可能并行”。

## WS-BONUS-01：按真实需求拆分bonus计算

普通greedy前缀拒绝kernel只有未拒绝请求会调用bonus_renew读取bonus token；已拒绝请求在首次不匹配处已有替代target token。因此，在无entropy/block/synthetic等改变验证行为、无额外logprobs或bonus消费者，且bonus处理无必须先发生的可观察副作用的条件下，可以研究先验证K行，再只为全接受请求生成bonus头/argmax，或以固定容量和mask实现同样的按需计算。原实现对所有请求预先产生bonus，故这是计算拆分与延迟执行边界，区别于WS-REJECT-01仅收敛整行返回值。

必须保留原bonus处理器所处的prefix状态、全局ID、padding、数值/tie与输出顺序。若任何RNG状态变化对外可见，还须保留或等价推进；当前讨论不能直接推广到随机模式。跨卡参与者的动态行数/顺序必须一致，不能由每rank自行筛选导致collective失配。拒绝结果引入新的依赖，拆小GEMM可能增加权重重读与启动；当前一次性[24,5120]头对[20,5120]加条件bonus小头的总成本没有实测，甚至可能回退。

固定R请求、K验证行的逻辑bonus比例为R/[R(K+1)]=1/(K+1)，K5时为1/6；它只是当前头行域中的比例。若所有请求全接受则没有bonus行可省；若有请求拒绝，是否省去物理tile仍取决于实现/布局。没有从接受率直接推导端到端收益，也没有把此项与词表物化、argmax或头通信的重叠收益相加。后续必须核实所有bonus消费者与同次每轮全接受请求数，再比较资源与关键路径。

## 当前的性能证据与下一步

同次16rank导出，Markov相容MatMul为每rank60次，基础draft头12次，BF16 Markov ArgMax也是60次；它们与12轮×K5相容，不能直接证明每个task的张量身份。Slice/Add/ViewCopy族需额外绑定stride与底层存储跨度，不能由base容量认定每次都复制整个[B,K,V]。原matmul/通信/复制时长不能相加为关键路径收益。

本次补齐了原F30的一段具体值链、条件模式和跨位置/跨迭代限制。M00/M01/M05/M16–M24仍未整体关闭；尚缺primitive内部、全体消费者、历史resolved对象、完整请求因果链、专家有效行/容量、Indexer长度和分层访存证据。整体目标保持active。
