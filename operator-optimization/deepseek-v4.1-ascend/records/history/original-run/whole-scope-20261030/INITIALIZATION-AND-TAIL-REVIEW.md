# 初始化、固定文本请求与目标尾部：范围和消费者增量审阅

**整体仍未完成。** 本轮补真实前端/配置依赖7份（两库commit与tracked clean核验），并只读服务中已安装Transformers的两个配置源文件；不导入目标模块、不运行推理、不修改部署。新增源码事实、配置推导和同次测量分别记录，不能以一个检查点替代完整覆盖。

## 1. 三项影响适用范围的证据

**DOCUMENTED：实际配置库版本与序列化元数据不同。** 服务distribution metadata返回Transformers **5.14.1**；checkpoint JSON中的`transformers_version=5.6.0`是文件保存元数据，不能拿它当已安装库版本。[安装库快照](installed-transformers-contract.json)保留两个路径、SHA与取证日期；它仍不是历史worker加载对象的直接快照。

**DERIVED：generation文件缺失后的普通默认链。** 新源`vllm.transformers_utils.config.try_get_generation_config`先`GenerationConfig.from_pretrained`，OSError时转`get_config→GenerationConfig.from_model_config`，再失败才返回None。5.14.1的from_model_config从模型配置转dict并提取生成字段，`to_diff_dict`相对默认GenerationConfig比较。保存的root/text配置均未提供ModelConfig采样白名单中的repetition_penalty/temperature/top_k/top_p/min_p/max_new_tokens，serve也未指定generation override；普通固定初始化下该白名单没有非默认覆写，CompletionRequest回到repetition_penalty=1等默认，显式temperature=0进一步将top_p/top_k/min_p设为greedy值。不能仅从“generation_config.json不存在”跳到此结论；现在补上了实际回退源码。自定义加载/动态覆写和历史同调用身份仍OPEN。

**DERIVED：辅助状态编号已具体化。** NPU runner先查上游eagle/dflash覆盖，再对DSpark的[37,38,39]加1，得到模型接口集合[38,39,40]；EagleModelMixin排序去重，不再改编号。目标forward在`layer.layer_idx+1`属于该集合时、执行该层前，对HC轴取mean并按SP恢复token行。因此普通固定配置下生产点是**进入零基编号37、38、39层前**的残差；不是这些层完成后的输出。冻结JSON没有eagle/dflash层号覆盖，V1 load_model在PP1且aux启用时配置该集合。其它fallback保留为不同条件路径。

F27的提前投影还受到**行索引就绪**约束：worker在拒绝结果之后按token_indices选取辅助行，再cat后给draft。即使辅助值已在末几层先后产生，也不能无条件提前计算“最终选择的行”。可研究全行投机计算后选择，或证明哪些行提前确定；两者的额外工作、buffer与精度合同要单列。

## 2. M05：请求不执行vision与初始化驻留是两件事

新`DeepseekV41Config`将text字段展开、将vision配置32层/维度1024等映射为vision_*字段。ModelConfig的language_model_only默认false，未指定的image limit在MultiModalConfig中默认**999**。冻结serve未给language-model-only或limit-mm-per-prompt，因此普通配置下VL wrapper的`image_enabled`为true，构建ViT、aligner和3个image向量。仅凭文本架构名称不能排除这些构造、载权或启动profile。

固定completions客户端只传文本prompt，没有patches/media payload；wrapper解析patches=None时返回None，embed_multimodal不进入vision forward。因此在**该请求阶段/固定输入合同**下，vision数学forward体可以有依据地排除。排除不适用于模块初始化、权重加载、启动dummy/profile、其它图像输入或模型能力承诺。

**WS-VISION-RESIDENT-01（条件工作负载特化）：**在只要求这些固定文本请求、允许不提供图像能力的域中，可研究不构造/加载vision塔，或按需延迟加载。需要保留完整多模态服务合同的域则不能直接关闭该能力。当前没有改变配置或部署。

源码使用普通nn.Linear/ModuleList，vision/aligner没有TP/DP分片。按保存维度的逻辑参数账：ViT **411,842,560**参数，aligner **73,410,560**参数，image向量**15,360**参数。若前两者BF16、向量保持声明FP32，合计970,567,680字节，约925.6MiB/逻辑rank。该账由[算术回执](initialization-tail-evidence.json)保存，**不是已测HBM占用/可释放分配量**；设备、加载/offload、allocator、graph引用和其它工作区仍需绑定。固定C8场景能否受益还取决于KV容量/内存瓶颈，不能从内存账直接推TPS。

## 3. F26的消费者条件：MTP缓冲并不自动等于DSpark数值输入

保存target源码中`needs_mtp_hidden_states`使用`use_eagle() or uses_draft_model()`，前者包含dspark。PP末组因此构造容量`[8192,4×5120]`的BF16 `_mtp_hidden_buffer`。forward在SP条件下先汇集完整HC residual与pre_mix，复制flatten residual到该buffer，再collapse/norm；这正是旧F26按“缓冲仍可观察”审阅的路径。

本轮对两份固定仓库的tracked Python/C++/头文件做[两符号定向审计](mtp-buffer-symbol-audit.json)，90个命中包含其它架构、测试和V2路径；它们不能全部算本次执行，也不能据字符串检索声称无反射/alias。当前V4.1 wrapper→language getter只返回buffer；V1 `NPUModelRunner.propose_draft_token_ids`无条件取得该对象，却只在`method=='mtp' && buffer!=None`时把它作为hidden值。DSpark实际分支通过独立aux_hidden_states和main_proj构造输入。

因此，在**冻结V1 DSpark请求域、无其它数值/调试/反射消费者且不要求保持内部getter可观察内容**的条件下，完整HC buffer的逐步写入可能是死数据；需要保留内部getter可观察值的域中，这个副本仍是必须消费者。旧F26针对后一个域的结论继续成立，本轮不把“存在getter”或“没有直接数值读”各自偷换成所有域的结论。V2和其它draft模式的消费者要另审，不能套用。

## 4. WS-TAIL-STATE-01：消除非必需状态后重新选择collapse/gather边界

这是F26/M15的条件细化，与旧collapse/norm融合、WS-VOCAB及其它尾部方向重叠，不计独立可相加收益。

若上节死数据条件成立，可研究不保存该全HC副本，并采用目标forward已有的无buffer结构：各rank先collapse局部token，再汇集collapsed hidden、裁剪padding、做norm。不能只删两次gather而仍将局部token当全局输出。

collapse源码为`sum_h(pre_mix_h * fp32(x_h)).to(x.dtype)`，只沿每token HC轴归约。若两种gather保持相同token所有权/顺序/裁剪，且shape变化不改变实际浮点归约、dtype、舍入和NaN/tie约定，则rowwise collapse与token行gather可以交换。语义上可交换不自动证明NPU实现位相等；还需保留aux消费者、graph地址/事件及后续norm/logits合同。

条件逻辑账（H5120、HC4、BF16 residual/hidden、FP32 mix）：原完整行载荷为40,960+16=**40,976B**，候选collapsed行是**10,240B**；另一个潜在消除项是最大容量**320MiB/rank**的HC缓冲，以及24行时983,040B的单向copy payload。这里没有计作实际链路字节、必要HBM流量或时间下界，collective算法、padding/复用、压缩和布局会改变物理成本。

## 5. 同次采集的增量测量

只读原官方later窗口16rank CSV，再核对原SHA，选取完整HC/copy兼容shape，得到下列每rank均12次的组。时间为“各rank调用中位数”的跨rank中位，单位µs：

| shape/dtype兼容族 | 时间 | 证据强度 |
|---|---:|---|
| BF16 TensorMove [24,20480] | 8.655 | 与HC副本相容，尚非直接存储/调用绑定 |
| BF16→FP32 Cast [24,4,5120] | 8.590 | 与full-HC collapse输入转换相容 |
| FP32 Mul [24,4,1]×[24,4,5120] | 17.250 | 与系数混合相容 |
| FP32 ReduceSum [24,4,5120]→[24,5120] | 11.895 | 与HC轴归约相容 |

[导出](tail-state-exports.json)和[核验/算术](initialization-tail-evidence.json)保留原文件SHA、dtype、完整形状、次数和时长直方图。HcPre的权重输入也含[24,20480]，命中1032次；这说明只搜尺寸会误把大量层内任务算为尾部，必须分开。上述四行也没有被相加为可节省时间；copy或通信是否暴露在请求关键路径仍UNKNOWN。

## 6. 余下工作

M00/M01/M05/M15/M18已有更具体的条件合同，但初始化/平台替换/全部消费者和运行实例闭包仍未完成。需要实际V1解析配置/权重及同调用buffer/event身份、vision驻留与启动profile范围、尾部gather通信组/顺序/字节及shape变化数值合同。MoE有效专家行/容量、Indexer真实长度与端到端关键路径仍是独立决定性缺口。整体目标保持active。
