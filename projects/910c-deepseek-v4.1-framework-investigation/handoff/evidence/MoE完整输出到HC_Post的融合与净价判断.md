# MoE完整输出到HC Post

介绍：沿真实AscendMoERunner/shared量化与归约追到HC Post输出完成。到了哪里：18当前源、冻结serve白名单、13原Python factory/layout/完整shared opaque控制；旧两代表卡1032完整交付、原C8全16卡2560 add-HC对。结果：SEDP复制权重且无shared TP通信，缩放在路由权重侧；add→HC保留中间BF16舍入的融合值得定点验证。原122源码约束/34动态位未变，全链路未扫全，新原语/框架准入0。

已停止：把disable_tp=is_sequence_parallel直接推成权重已复制；按默认SEDP=False给冻结负载补出shared TP gather/归约；寻找不存在的后置scale；擦掉ForeachAdd的BF16舍入再合并HC数学式；只开multistream flag就认为DFC与shared可重叠。实际serve enable_shared_expert_dp=True；OOT get_parallel_op按shared开关独立决策，shared.parallel_mode为SP_SEDP。patch factory默认apply_routed_scale_to_output=False，TopK原生ABI接收routed factor，runner factor1；DFC配置强制shared multistream=False。

待确认、值得开发验证：完整ForeachAdd→HC Post原语。共享/路由BF16两输入先做原add的BF16舍入，再按HC DSplit原FP32 Mul/Add顺序及最终CAST_RINT产生残差输出；不能分配乘法/使用FMA擦掉中间舍入。有限IEEE例(1+2^-8)*3，原BF16 add边界结果0x4040，直接塌缩0x4041；这只否定通用代数替换，不是native对拍。M3/D5120逻辑上可省中间BF16写回+读回61440B/卡/call及一次launch；残差读、HC输出写、系数/下一层pre_mix/aux与DFC counts/潜在expertIdx变异都保留。完整native Foreach舍入/NaN/零符号、格式/tiling/别名/图捕获和全部输出逐位、增量成本/六普通回归仍缺。

先把整段算全：上轮5.746362%止于Post开始，不能当完整输出上界。补Post执行后，旧EP0/8 global24窗保留DFC的宽松包络均6.126260%；shared开始→Post完成4.445635%，Down-ready→Post完成1.498290%。Foreach本体1.066954%、HC Post0.379898%；若保留原HC Post成本、每call新增5us，480次局部均净7.572826ms/0.810240%该窗；新增20us只剩0.040098%。这是条件价，不是实测融合收益，外流allgather未归给shared。global20另列，不迁移target价。

请求另算：原C8首token16卡2560对，按每层最大卡信用同时给全体、固定原carrier/请求次序串行累计的极宽松模型，Foreach全免最大0.371417% TTFT，新增5us为0.316674%；add+HC全部免最大1.353714%，但含必须HC数学/写出，不可实现为零。该预算不是下一EP arrival模型，也不用于预测decode/makespan。六普通/全请求曝光仍未知，实际请求信用默认0；不把两卡steady百分比外推。

更大父段仍保留：保持原量化数学的shared与DFC联合调度/megakernel、共同输入读与双quant格式交付；已有DequantSwigluQuant已融合activation+requant，不能当未融合重复计。DFC和shared的quant舍入政策不同，不能直接共用量化Tensor。新增重叠须改当前mutex、证明EP参与/资源与跨流读写租约，不能复用“已打开”的假设。优先继续剩余路径，留下以上具体native/净价缺口。

[本段判断](MoE完整输出到HC_Post的融合与净价判断.md) · [源/候选边界](moe-output-review.json) · [13原源码控制](moe-output-controls.json) · [1032局部交付及原C8请求价](moe-output-price.json)。

下一条待扫：loop047 Run184748Z PLANNED：转M08真实DSpark _run_merged_draft的选行/LMHead/confidence→采样/拒绝→下一轮调度交付，追全输入、实际消费者和合法动态分支。已扫MoE输出源边界保留add-HC完整原语native逐位/格式/捕获/净价与大shared-DFC联合调度缺口；不重做不存在的shared TP通信/后置scale或已停小配方。34原动态位及递归callee未闭合，保留metadata/gid12、Engram owner/input候选。源先定大交付后匹配旧C8/六普通，每替换原语逐位一致；本源审Run不新开Torch/NPU/模型/profile/请求/服务或远端写入。

原C8逐Task原件：moe-output-original-C8-tasks.json与同内容zlib；所有原数据SHA见price inputs。当前原语/框架替换0，原CPU封闭租约原型及历史有限native fixture保留原资格。
