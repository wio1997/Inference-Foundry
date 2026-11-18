# 主模型与草稿的原生混合通信dtype

G462实际8个worker的安装记录显示query两处、inverse output一处NPU RoPE分发修正已执行；完整target图再次在8rank完成。草稿图越过此前CUDA JIT位置后，在未量化MoE的GMM报错：INT8 activation与BF16 weight没有匹配模式。日志另记录3238002688 bytes工作区申请失败；此前target图后约3.07GB余量不足，不能把本轮全部错误只归为显存或只归为dtype。

固定源码的 `get_deepep_output_dtype` 首先检查全局CLI值；强制int8会覆盖每层已有的dtype声明。原生NPUW4A8Int8MoEMethod本来声明int8，NPUUnquantMoEMethod本来声明bf16。G472提取原始Enum/选择函数在CPU执行8个配置用例，normal/low_latency两种模式均复现强制int8覆盖草稿，auto则保留target int8和draft bf16。该检查不执行张量、CANN GMM或模型，不代替完整图验收。

G474使用原生auto参数，保留每层量化定义；没有把主模型通信统一降成BF16，也没有改动草稿权重。D预算0.82→0.80，为已经观察到的草稿工作区增加空间；实际池容量仍须至少263296。两端原生PD bootstrap/wait超时设3600，适应后续长C16队列，属于可用性预算。EP8、DSPARK5、完整target/draft图、权重、先前已验收的数学helper保持原要求。

新配置仍须实际混合dtype、全部图、容量、原生PD、语义和四档测量验收。调整三个可用性/契约设置不构成独立性能A/B或优化收益。

- [完整错误与dtype诊断](../evidence/full_model_G462/2204/server.log)
- [D退出与8卡清空](../evidence/full_model_G462/2204/state.json)
- [原始归档SHA与范围](../evidence/full_model_G462/2204/mixed_dtype_assessment.json)
- [原生选择函数CPU8用例](../evidence/native_dispatch_dtype_G472/contract.json)
- [原始固定镜像选择函数](../code/native_dispatch_dtype_G472/fixed_source/sglang/srt/layers/moe/utils.py)
- [原生量化/未量化方法dtype声明](../code/native_dispatch_dtype_G472/fixed_source/sglang/srt/hardware_backend/npu/quantization/moe_methods.py)
- [P配置](../code/full_model_candidates/sg_pd_g474/candidate_p.json)
- [D配置](../code/full_model_candidates/sg_pd_g474/candidate_d.json)
