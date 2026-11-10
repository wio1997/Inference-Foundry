# 2026-11-11 三份aux到context cache：解析真实receiver，共享准备按原预算停止

介绍：从target层37/38/39入口的HC均值，追三次SP交付、选行/concat、main投影/RMS和三个draft context cache，到各层SMLA实读。到了哪里：19当前源码身份、3份旧精确SHA回调源、15原Python/有限布局控制；24完整aux/main/三cache链、72panel。结果：A3 main投影不量化，RoPE有重复准备；同窗独立aux打包/准备共享沿原预算停止，更大的host/context共同producer保留。原122约束/34未定输入不变，native准入0，全链路未闭合。

真实字段派发：文本wrapper的language_model_cls→AscendDeepseekV41LLMForCausalLM，model_cls→DeepseekV41Model；PP1拥有40层，DSpark wrapper→DeepseekV41DSparkModel→三个DSparkDecoder。后者继承原Decoder.forward，attention_cls指向DSparkSWAAttention，原attention前向别名和CP实现保留。具名receiver与原工作表匹配边、额外展开body另列，不把“字段名字相同”当相同计算。

aux是真实三个不同时间的输入：模型在进入zero-based37/38/39之前取HC4均值、SP gather并裁full_num_tokens；不是三个相同Tensor，也不是层输出。runner的无spec metadata前缀分支先裁num_scheduled，验证分支按token_indices选行；随后按配置顺序concat，保留重复行、原顺序、discard/rejected与DP/SP尾部关系。DFlash继承的build_model_inputs_first_pass把固定context隐藏/位置buffer和各层slot交给三层cache producer，不能把B×5 query当作全部context行。

main数学边界：DSpark main_proj传入的quant_config仅在packed-cache设备且checkpoint标为fp8时非None；A3 packed=None，所以这里是原BF16/FLOAT unquantized GEMM，不能按模型W8A8标签省quant。main_proj前缀不命中特殊column op，ColumnParallelLinear保留K15360、TP8列640，feature gather回5120后原RMS；原return_bias=False。Replicated wkv经CustomReplicatedOp调用各自quant method，无这条feature TP gather。wkv的全局quant_config仍要按各prefix描述和实际native任务判断，不能从main不量化外推所有wkv。RMS还有bias_loaded分支，保持原norm/add边界；不把三份aux拆K后相加、改变M/分片权重或穿过RMS作代数折叠当作逐位等价。旧changed-M有4/2位差反例，原样配方不重跑。

实际少算点：每层_project_shared_kv都调用_apply_dsv4_rope；其get_cos_and_sin_dsa(context_positions)默认use_cache=False、layer_names=None，遍历全部已注册配置并构造新cos/sin，再只取本层名字。当前匹配链每层有4个表Index及其4个IndexCheck，三个panel共12次表Index；保留原第一份完整proxy、后两层共享准备，可删后8次及对应检查。它不要求三层config相同，但要求同一position/registry/full-cache代次、只读cos/sin与覆盖三消费者的租约。若再按layer_names限制配置，必须保留各层真实config，不能猜所有名字共用一套频率。15控制执行了原getter/proxy、不同config和缺层错误，并给出position变代次的否定例；没有原生数值验收资格。原consumer内cast、RotaryMul、KV/RMS数学和写入仍保留，不能把准备共享说成RoPE本体全免。

能少搬什么：先保持三个均值的BF16结果不动，考虑本卡aux按每row的三个H段打包，单次SP gather成全局[N,3H]，再选行；其值与原三次gather后选行/concat可做坐标对应。总aux逻辑payload不减少，仍要付pack、等体积collective和选行成本。三个legacy aux若以strided view暴露，还须闭合全部reader/别名、graph固定地址与native stride支持；只证明数值排列不足以准入。full_N裁切必须先于可能的负索引，TP8/N11与重复索引有限tag例保留了这点。

追到真实cache读者：_context_slot_mapping_buffers按_layer_group_idx重建，gid相同可引用同一slot buffer，三个cache Tensor仍是不同层的物理view。_store_standard_swa_kv在slot=None/empty时只停止store；上层仍已算过该层Wkv/norm/rope。A3一维linear slot先clamp/除模并把负数两列置-1；二维坐标保留。scatter_cache_sk把values转cache.dtype并contiguous，原SK算子保留cache实际page stride和[-1,-1]跳行，不能按shape假设平面连续。随后各DSpark层经显式dsa_v41_forward side-effect边界、query cache更新和metadata ATTENTION wait，把同一swa_cache_layer.kv_cache[0]作为SMLA ori_kv。packed A5、long-compressed reader不属于这个冻结draft角色。跨层/跨流须证明物理byte-span、cache/query读写次序及producer/consumer退役；不是前缀/gid不同就可并发。

按整段定价再剪枝：旧配平C8 EP0/8、M24/local3的完整aux均值+三AG+选行/main+三cache native分支，均0.893095%该窗；从最早aux到cache末写的墙钟跨度达5.374410%，其中夹着必需target末三层、head/rejection与metadata，不能当融合预算。aux AG/选行/concat全免仅0.158914%，context全部90任务0.583920%；三次全RoPE getter0.243367%，保留第一份而共享后两份的native信用0.161580%。新增pack10us+AG30us+选行25us、共享控制5us/call，二者不相交的条件净2.119–2.193ms/0.225392–0.235898%该窗；较贵110+10us仅0.161575–0.171357%，210+20us只0.021179–0.029368%，aux packing单项已转负。均为付成本情景，非实测新kernel收益。

历史/请求分开：原C8全16卡112周期中，全部90 context任务全免最多0.866501%周期，沿用原stage-confined row/RoPE/slot停止决定。固定所有此前迭代和请求开始、只推进当前carrier的旧context全免，TP/latest-DP最大个体1.759380%不代表P90；重算该模型P90只有2.175160ms/0.146108%，全16时钟宽包络为5.134580ms/0.344896%。它们都不是全迭代或客户端收益上界。旧逐波传播的更大host/context父段仍有P90全免23.118418ms/1.549643%、整批271.438628ms/1.460072%的宽包络，含必需工作且与metadata/frame重叠，不与本轮比例相加。六普通的完整新父段/净曝光仍缺，新增请求信用0；1%只是旧独立优先预算的引用，不是新用户硬阈值。

结果分档：已停止——相同预算下独立aux打包、context RoPE/slot/row小方案及既有changed-M反例配方，不做原样native重验。待确认——更大的aux/host/context共同producer、保持原数值tile的owner或megakernel，以及既有metadata/gid12/add-HC/sharedDFC/Engram/Markov候选；要扣必需数学、通信、query/cache写入及全部租约，再看六普通和请求曝光。已确认可实现——本轮0，原15协议/坐标控制不是逐位准入；旧封闭CPU原型资格独立保留。

下一条待扫：loop047下一Run：转DeepseekV41Compressor.forward及其实际设备实现，沿输入缓存/压缩KV、index-K发布到跨层SMLA实际消费者，先绑定冻结role与receiver，再按exact-SHA旧证据去重。覆盖合法ratio/short-input/append/reset/缓存复用分支和全部可见输出；先审整段数学/搬运/等待共享，再匹配原C8/六普通，低整段预算才结束同预算子项。保留大metadata/gid12、add-HC/sharedDFC、Engram owner、Markov/host-context具体缺口；每替换原语逐位一致，不重开本轮停止的小配方。本源码Run无新Torch/NPU/模型/profile/请求/服务/远端写入。
