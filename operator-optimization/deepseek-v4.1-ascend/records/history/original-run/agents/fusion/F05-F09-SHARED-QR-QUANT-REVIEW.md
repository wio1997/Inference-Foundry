# F05/F09 交叉边界：主 Q 与 Indexer Q 的输入量化复用（条件候选）

## 为什么不是重复 F06 或 F09

F06 的共享对象是 `query_hidden` 与 `cache_hidden` 上的 Wq_a/Wkv 输入，CP 下需要区别 local/global 行域。这里是 **同一个 Q norm 输出 qr** 的两条消费者：主 attention Wq_b 与 indexer Wq_b。F05 已提到 qr 有两个消费者和 norm→主 Wq_b 的融合，但没有分析这两个 Q 投影输入量化能否共享；F09 指 indexer **投影输出 → RoPE → 按 head 量化**，与这里投影之前按行输入量化是不同的数据和 scale。以上候选互相有关，不能将一次量化的潜在收益与 F05 重复记账。

## 有来源支持的数据依赖

冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:350–380,393–409,440–495` 的多流分支：`qr = attn.q_norm(q_a)`，主投影调用 `wq_b.quantize(qr)` 与 `wq_b.matmul(...)`；主 Q 输出经过原地 RoPE 后 `_prepare_indexer_inputs(..., qr, ...)`，且只有 `role.has_long_context && role.is_index_source` 时调用 `indexer.project_query(qr)`。`models/deepseek_v41/indexer.py:57–64,165–166` 中 indexer 的独立 `wq_b` 是 `ReplicatedLinear(... quant_config=quant_config)`，`project_query` 将 qr 直接送入该 linear，再按 head 还原形状；上述只是源码中的调用参数与顺序，不证明中间无其他别名写入或历史运行采用此分支。冻结角色映射中层 24 是 index source，但历史每个请求是否 long-context、是否该次进入上述分支仍未由此证明。

`agents/fusion/frozen-source/vllm_ascend/ops/cv_linear.py:93–119` 表明：若主 Q wrapper 识别 W8A8 dynamic 且无通信，执行 `torch_npu.npu_dynamic_quant(qr)`，否则可能退化为原输入或采用另一量化。**目前尚无 indexer 侧独立 ReplicatedLinear 的绑定量化实现及历史实例**，不能证明它也用同一 API、相同 scale 算法、输入 dtype、对称/非对称、rounding、零值/NaN 规则或局部 view，更不能认定实际可以重用主分支的输出。主 Wq_b 和 indexer Wq_b 的权重量化参数/布局各自保留，输入 scale 相同不意味着权重可合并。

参考量化描述文件 `source172/model-metadata/quant_model_description.json` 的 `layers.24.attn.wq_b.weight` 与 `layers.24.attn.indexer.wq_b.weight` 都标记 `W8A8_DYNAMIC`；文件 SHA-256 `2f4020a0a12daa621cfd2ff9d09a2c73e5674af7c624ef47df5c66796c3774bf`，先前 stage084 保存的只读比较显示当时 169 描述文件 hash 一致。标签不是历史实例或具体量化算子的证明。

## 现有一窗设备采样的相容性

已有 `f22-layer24-run2-evidence.json` 的 EP8、层 24、run2、model49、stream89 中：task1261 `DynamicQuant` 输入 BF16 `[3,1280]`，输出 INT8 `[3,1280]` 与 FLOAT `[3]`，时间 83.199–86.119 µs；task1264 `QuantBatchMatmulV3` 输出 BF16 `[3,32768]`，92.559–133.078 µs；task1267 相同形状 `DynamicQuant` 为 149.557–152.377 µs；task1268 量化矩阵输出 BF16 `[3,4096]`，153.577–167.517 µs。分别与主 Q 64×512、indexer Q 32×128 的源码投影维度相容；记录只保存形状、类型和时序，没有 tensor/storage ID 或这些设备 task 到 Python 调用的精确关联，因此**不能断言两次量化的输入是同一 qr、由相同代码触发，或实际可消除其中一次**。只选了一窗且事件列表经筛选，2.920/2.820 µs 并非收益预测，期间还有其他工作；不能作为端到端或完整关键路径的缩短。

## 可研究的重组与必备证明

若两个投影实际使用同一稳定 qr 行域，且逐行量化结果与 scale 的全部数值语义、dtype 和布局逐位等价，可保留生成一次的 INT8 与 per-token scale，让主 Q 和 indexer Q 使用各自权重、scale 和原 matmul 路径。可进一步研究在同一量化后将两个 N 输出作分组投影或 tile 交接，但两个结果各自的 output dtype、rounding、layout、RoPE 和独立的权重偏置/scale 仍须保留；分组合并改变 matmul 调度与舍入，不能由实数代数保证逐位等价。

该改动需证明：量化结果 buffer 活到 indexer 使用完成，主 Q 后的原地 RoPE 与 qr 和缓存无别名冲突；原 stream/event、图捕获与 buffer 生命周期得到维护；indexer 在条件不命中时不会被提前执行，且原本可重叠的 KV、投影、辅助流计算不被破坏。量化缓存额外占用、Cube/Vector/MTE 带宽竞争均可能抵消收益。普通分支、CP 其他包装方法以及非 W8A8 分支不能无条件照搬。

源文件 SHA-256：dsa_v41.py `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`；indexer.py `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33`；cv_linear.py `d2bf17c189f8748a9b7e5cbd42325c798eeb238da894755065792da982450b6d`；采样摘要 `0cae38a2a726f1471f651cd664f0dcb4f608734dbb59f8603d1315ce02084dfb`。这些来源可供只读复核；与历史编译二进制、调用栈和 tensor 身份的绑定尚缺，不计算可达提速、也不作实现。

## 同镜像旧 DSA CP 中的量化共享先例，不等同 V4.1 已实现

冻结 `attention/context_parallel/dsa_cp.py:1836–1854,1907–1914,2127–2147` 的旧 DSA CP `_forward` 在主 Q 使用 W8A8 动态线性方法的条件下，以 `npu_rms_norm_dynamic_quant` 将 Q norm 与动态量化合并，得到 `qr_local` 与 `qr_pertoken_scale_local`；主 Q 的 `npu_quant_matmul` 和（压缩比为 4、未跳过 topk 等条件下的）`_indexer_select_topk` 使用同一对象。在 indexer 量化方法也是 W8A8 动态、scale 非空且 `support_fp8_attention` 为 false 时，后者使用同一对输入与自身权重、scale、bias 调用 `npu_quant_matmul`；否则回退 `self.inderxer_wq_b(qr)`。这些是源码内的条件路径，非历史执行证据；fallback 获得的 qr 可能已量化，不能在未核实完整配置约束时推断所有混合配置行为正确。

这段实现是**相关先例，但不能称 V4.1 CP 已实现共享**：`attention/context_parallel/dsa_v41_cp.py:14–25,170–207,445–490` 的 V4.1 CP 和 PCP impl 继承 `AscendDSAV41Impl`，从旧 `dsa_cp.py` 导入 metadata builders 和 `restore_tp_heads`，而不继承旧 DSA CP 执行实现；其 `_prepare_inputs_and_caches` 经 V4.1 `_preprocess` 工作，V4.1 indexer 输入调用 `indexer.project_query(qr)`。所以仍须核对 V4.1 两端普通动态量化语义和运行实例，不能仅凭旧 DSA 先例证明可共享，亦不能将其假设为新增机会的已证明收益。

旧 CP 文件 SHA-256 `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305`；V4.1 CP 文件 SHA-256 `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`；V4.1 DSA 文件 SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。


## 固定样本数据库关联核对

对上述 EP8 层24、第2轮 task1261、1264、1267、1268，按 model、stream、task 与精确起止时间在同一已保存 profiler 数据库各唯一定位一行。四条都使用 connectionId 184330，而该 ID 在该库被 19971 条 TASK 共享；四条各自的 globalTaskId 均各出现 7 次。按这个 connectionId 精确查询 CANN_API 和 CONNECTION_IDS 未返回记录，因而在此关联路径上没有找到对应 PyTorch 调用链。TASK 与 COMPUTE_TASK_INFO 表字段没有 tensor 或 storage ID；两条 DynamicQuant 的属性字段均引用字典 ID 1249，说明导出元数据引用相同，但不证明其输入对象相同或两次具有相同运行上下文。数据及查询条件见 agents/diagnosis/qr-quant-task-links-ep8.json，脚本 agents/diagnosis/query_qr_quant_task_links.py 固定摘要和数据库 hash。

这次查询只能说明这些表及该关联键不足以绑定采样中两个 tensor 或精确 Python 调用；不能推断其他 profiler 文件不存在身份信息，也不能从未匹配判定不存在主机调用。已有同形状与源码共享 qr 的证据仍是条件候选，不能声称已证可复用或按 2.820 微秒推算性能收益。

## 追加：W8A8 线性方案选择及两条 API 的差异

本机 ZCode stage `operator-ceiling169-v2-200-indexer-quant-sources` 先获取来源清单，stage `operator-ceiling169-v2-203-indexer-quant-source-snapshot` 退出码 0 且 model_verified=true从 169 stage 容器定点只读快照了 `agents/fusion/frozen-source/vllm_ascend/quantization/` 下 modelslim config、method adapter、registry、base、methods init、`methods/w8a8/w8a8_dynamic.py` 六个文件；所有文件均按此前 stage 200 清单核对 SHA 和字节数。源码选用关系是条件性的：`configs/modelslim_config.py:265–318,322–345,655–706` 中，runtime prefix 经 model-specific mapper 和 packed mapping 得到描述中的 weight 标签，若对象是 `LinearBase` 且解析标签是 `W8A8_DYNAMIC`，则注册表构造对应 linear scheme 并交给 `AscendLinearMethod`；adapter `method_adapters.py:209–224` 转调其 `apply`。此前 Indexer `ReplicatedLinear(... quant_config=quant_config)` 与参考描述字段同标签为该路径提供相容证据，但缺历史运行时 config、最终映射 prefix 和对象类型绑定，不证明实例确实进入此方案。

`methods/w8a8/w8a8_dynamic.py:60–68,104–145` 的该 scheme 默认 `act_quant_type=torch.int8`，普通未预量化输入调用 `torch_npu.npu_dynamic_quant(x, dst_type=self.act_quant_type)`；另有 `PreparedLinearInput` 直接复用已量化结果和 scale 的分支，不能假设每次都重新量化。`cv_linear.py:103–109` 主 Q wrapper 的 W8A8 无通信分支调用同一个 `torch_npu.npu_dynamic_quant(x)`，但没有显式传 `dst_type`；**当前源码尚不足以证明底层默认值和上述显式 INT8 参数完全一致**。已有被筛选窗口的两个设备记录输出 INT8/FLOAT 只证明这两条记录的导出 dtype，不证明 Python 绑定、相同输入、rounding、饱和、零值/NaN 规则或 scale 位级等价。

两端后续矩阵乘还有值得保留的差异：scheme 普通 `apply` 用 `output_dtype=x.dtype`（预量化输入使用保存的 output dtype），CV wrapper 用 `output_dtype=self.linear.weight_scale.dtype`；scheme 的 bias 按 `act_quant_type==torch.int8` 条件传入，而 CV 使用调用方提供的 bias。`weight_offset` 在该 scheme 初始化并在加载后 flatten，但这两个所核对的 W8A8 `npu_quant_matmul` 调用没有作为独立参数传入；这不证明 checkpoint offset 值为零或所有路径不使用 offset。两边权重、weight_scale 和 bias 仍各自保留；dtype 差异及每侧的输出语义需要分别验证，不能仅因共用输入量化而合并矩阵乘结果。

这项补证收紧了前文“尚无 indexer 侧独立 ReplicatedLinear 的绑定量化实现”的表述：现在**已有条件性方法选择和实现源码**，尚缺历史运行实例、默认参数和张量身份绑定；同一 tensor、数值语义、生命周期、同步条件全部需进一步证明，候选仍不能视为已确定或可量化收益。

快照 SHA-256（按 stage 200 清单核对）：modelslim config `92c924abc0af1ae37d7614ee01cbae0b3be1da86ef6abc885b2208560f3dcbb1`；method adapter `341d93cae5774414a810444a50eb752cd425330156f404d1b9a10381e6bc0138`；methods init `cd0e11a198dc6b6eb49f261b7e25d73f42f11cb893d08e5dc126396dd5d887f2`；base `51a7aafcc49b64a53902eeb3619023e3d79f9c8f4aaf846205dc83dfd35b0d24`；registry `65b5cc268362c87d6b0e5cda5cf37ca5d54264ad54c3f2896dda35c8d9410be5`；w8a8_dynamic `a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910`。精确结果在 `agents/fusion/indexer-quant-source-inventory.json` 和 `agents/fusion/snapshot_indexer_quant_sources.py`。

## stage204：当前安装包说明中，省略 dst_type 按 INT8 处理

本机 ZCode 的 `operator-ceiling169-v2-204-*` 只读运行 `agents/fusion/query_dynamic_quant_installed_schema.py`，固定容器、镜像及 Python 3.12 site-packages，并检查该目录无覆盖挂载；未导入 torch_npu、未运行设备任务。容器安装元数据为 torch_npu `2.10.0.post7`（METADATA SHA-256 `3b2748974d953f767c8d3ee27f0e1b8278fedadd283ddba6aa2bed13b80b09f8`），与历史 EP8 profiler 自述版本字符串一致，但版本一致不证明历史安装包文件逐字节一致。

安装包 `torch_npu/utils/custom_ops.py:363` 静态赋值 `torch_npu.npu_dynamic_quant = torch.ops.npu.npu_dynamic_quant`，文件 SHA-256 `4b5d2cac3f851dc8001d0bba3ef94546e9cce7d25c2a3fecd43cc9e8b9a4897a`。安装的 `lib/libtorch_npu.so`（SHA-256 `7e5eb430fd0f1d438e9598cd89e609332d4d1eea1c458e8f42fd01060afa47a1`）包含同名 schema 字符串，`dst_type=None`、`quant_mode="pertoken"`、`dst_type_max=0.0`。同包 `_op_plugin_docs.py:4670,4677`（SHA-256 `636706a8a1ca65c5948c7da751c36448a9729c4bd764d093ee4015ae4f61511f`）明确说明 dst_type 传 None 当作 torch.int8；`op_plugin/meta/_meta_registrations.py:5240–5243`（SHA-256 `1f39ee98533f377d8a71cb457ca1f9d0ade2f3e82e6cdfda9707e05fe606787d`）另有 meta 实现默认值为 1 且注释为 torch.int8 enum。后者是 shape/meta 路径，不能代替设备执行语义的独立实证。

据此，**按当前安装包的接口文档**，主 Q 的省略参数与另一分支显式指定 `torch.int8` 在输出类型选择上相同；更早的“当前默认值未核实”只保留为该阶段历史记录，现被本段更新。静态 schema 的 `None` 到 INT8 的实际执行绑定未直接验证，历史运行对象、安装文件的历史 hash、张量身份、量化 scale / 舍入和特殊值处理是否逐位一致仍未知；不可据此断言两个量化结果可以复用，或推算性能收益。选中文件从安装包 RECORD 筛得 162 条静态文本路径，读取共 3712118 字节，另查询三个指定二进制名字中存在的文件，不构成全系统或所有运行分支扫描。完整路径、上下文、哈希及界限见 `agents/fusion/dynamic-quant-installed-schema.json`，该文件 SHA-256 `8808bbda84541b4f0028389ed1d4d5eb896edcc218fab6283dc0f11d9ad125ec`。

## 已存预量化输入路径及实际交接缺口

进一步追踪当前快照的接入链：如果 indexer 的实例确实被替换为 `AscendReplicatedLinear` 且 `disable_tp=false`，`ops/linear.py:546–550,596–603` 选择并调用 `CustomReplicatedOp`；`ops/linear_op.py:145–153,471–477` 不重算输入而将它转交量化方法；`quantization/method_adapters.py:209–224` 将它继续转交 inner scheme。若选中的是 `AscendW8A8DynamicLinearMethod`，该 scheme 在 `w8a8_dynamic.py:104–121` 收到 `PreparedLinearInput` 时直接使用其 `quantized`、`scale`、`output_dtype`，不会执行普通 Tensor 分支的第二次 `npu_dynamic_quant`。这是现有的**条件性方法链能力**，不是当前 V4.1 已复用输入的证据；`disable_tp=true` 会选择上游 fallback，其他注册、包装、graph lowering 和实际运行实例仍须核对。

V4.1 当前 `dsa_v41.py:477–495` 在 multistream 中生成 `q_b_quant/q_b_scale` 后只返回 `q,qr`；`:369–370` 的普通分支直接调用主 Wq_b，`:376–401` 将仍为 Tensor 的 `qr` 交给 indexer，`indexer.py:101–104,167–168` 再调用其线性层。这些源码说明当前同一快照的 V4.1 调用链**没有把该局部预量化对象交给 indexer**；历史是否执行这份代码仍未知。若研究交接这一对结果，应保留 `qr` 本身给现有返回及后续消费者，保证它和量化 buffer 在相关 stream 使用结束前都有效，并且不能在普通分支、无 indexer 分支或量化 wrapper 退化返回 `(x,None)` 的情况下假定已有可复用 INT8/scale。`prepare_input_for_overlap` 接口本身要求输入为二维 Tensor（`w8a8_dynamic.py:94–102`）；已有采样的 `[3,1280]` 相容，但不能推广到动态形状或认定设备任务绑定该源码。

构造预量化输入时必须保存**原 `qr.dtype`** 作为 indexer 的 `output_dtype`，而不是使用主 Q wrapper 传给自身 matmul 的 `weight_scale.dtype`；两者不一定相同，各自权重、scale 和 bias 也不可共用。作为仅用于反证的假设：若原 `qr` 为 BF16、主 Q scale 的 dtype 为 FP32，且某投影在最终转换前有精确值 `1+2^-8`，按 BF16 的最近偶数舍入是 1，而 FP32 可保留该值；以主侧 dtype 替代 indexer 原有 dtype 并不保证输出相同。这不是历史 dtype/数值或实际硬件舍入的观测。还须证明相同输入对象与行域、动态量化默认/显式参数与实际设备行为一致、量化结果布局和 scale 生命周期合法、位置编码原地操作无别名冲突，并支持编译/图捕获；仅有适配器透传不证明复用正确或性能改进。

本段涉及当前快照 `linear_op.py` SHA-256 `2a28240882788db363c14a0999b1cad68b4f9f63896d0ad9475f2d39051e818b`、`linear.py` `7654777d9252fc6a3cc2c95f92b67a42410a0de24a5912ef20e807cf64ab74e6`、`method_adapters.py` `341d93cae5774414a810444a50eb752cd425330156f404d1b9a10381e6bc0138`；只读，没有 NPU 工作或修改目标实现。

## 投影前逐行与投影后逐 head 量化的数值反例

除了不同的输入位置与归约范围，F09 对 scale 的中间 FP16 舍入还可能改变 INT8 值。给一个可复算的**条件性反例**：假设仅用于比较的两种量化恰好收到相同的一个 BF16 行/单 head，前两项为 `1.0078125`、`0.01190185546875`，其余为 0（两者均为 BF16 精确值）。根据 stage204 保存的安装文档，不带 smooth 时普通动态量化的数学式为 `scale=max(abs(x))/127`、`y=round(x/scale)`；假设该路径使用 FP32 scale 和除法、round 为最近偶数，则 FP32 scale 为 `0.007935531437397003`，第二项归一化约 `1.499818325`，得到整数 **1**。

所存 `quantize_indexer_query.py:26–35,59–69` 的非 packed F09 路径先将 scale 从 FP32 转 FP16 再转回 FP32，并做 `div_rn`、`nearbyint`：该行的 FP16 舍入后 scale 为 `0.0079345703125`，第二项归一化恰为 `1.5`，按源码注释的最近偶数得到 **2**。最小 scale 和截断在该例不改变这两个结果。CPU 使用 IEEE 浮点转换即可重算该示例；文件 SHA-256 `6065b79f0f764e99bafae71347c8c8dfd0625ad52b0d695639c47e71338d0bba`。

这否定的是**在上述普通路径 FP32 假设下**把两种量化无条件视为位等价；安装文档的公式没有完整给出普通路径的设备舍入/中间精度，实际 kernel 可能有其他细节，本例不是其硬件行为验证。F05 输入是投影前 `qr`、F09 输入是投影及 RoPE 后每 head query，真实两个输入一般也不同；本例人为令值和归约域相同，仅用于隔离 scale 舍入这一差异。未绑定历史运行或声称该值实际出现，也没有性能收益推断或 NPU 工作。

## 输入量化相容签名在固定 40 层采样中的覆盖

在先前 EP8 一窗观察基础上，`query_qr_quant_window_coverage.py` 对固定 hash 的 16 份原始 kernel CSV 执行只读定向筛选：16 rank × 3 轮 × 40 个已归属 target 层窗，各自使用 model49 对应主流，要求任务正时长且完整包含于层窗。对 `DynamicQuant` 的 BF16 `[3,1280]` → INT8 `[3,1280]` + FLOAT `[3]` 标记 D；对输入形状以 `[3,1280]` 开头且首两项为 INT8 的 `QuantBatchMatmulV3`，BF16 `[3,32768]` 输出标记 Q，BF16 `[3,4096]` 输出标记 I。该筛选不代表其他算子不存在。

八个此前记录为 indexer 投影输出相容的层（2、8、14、20、24、28、32、36）共 **384 个 rank/轮次/层窗口**，每窗均检出 **D、Q、D、I** 的筛选次序，四条均逐条不重叠；其余 32 层共 **1536 窗**均为 D、Q 各 1 条而 I 为 0。八层窗口的第二条 D 任务采样时长中位数 **2.86 µs**、范围 2.66–3.36 µs；Q 结束到第二 D 开始的相同卡流端点差中位 15.54 µs、范围 13.62–22.04 µs。时差并非空闲或未依赖：筛选忽略了可能位于其中的其他任务、未导出活动及其他流。本节统计及源码、清单和每张 CSV hash 记录在 `qr-quant-window-coverage.json`，脚本 SHA-256 `c80f58d0e97d2d9294df31ff09a9d6a91f7589d4f4f28ca0a3e80d21d3457c5f`。

这说明先前相容的双量化形状及先后次序在这批固定窗口中持续出现，提高候选的覆盖证据；**不证明两条 D 读取同一 qr、真正对应这些 Python 对象、权重身份或输入值相同**。相关窗口是源码推断归属，历史二进制、张量/storage、请求身份和跨流事件未绑定；不能排除不同任务、trace 过滤与其他执行路径。任务有 profiler 扰动，第二条 D 的时长不是可节约时间或端到端收益，跨 rank/轮次相加也不是墙钟时间。无新 NPU 工作，未修改目标实现。

## EP3 layer 36：两条量化投影有同 scope 调用栈和设备 flow 关联

以现有 `f32-ep3-layer36-scope.json` 已归属的 EP3、layer 36、decoder host scope `2128-42` 为入口，进一步只读该历史 profiler 数据库（路径与打开方式见 `query_f32_ep3_layer36_host.py`），在已知 globalTid 与该 scope 起止时间内查询 dynamic quant / quant matmul 的 host API，再沿 `CONNECTION_IDS` 查 `TASK` 和 `COMPUTE_TASK_INFO`。本轮未启动采集或 NPU 计算，未访问 tensor 内容。以下全是同一次 scope 的记录；宿主调用栈是 profiler 保存的堆栈，源码文件行对照仍假设运行时与保存版本一致。

| host connection → flow → 命名设备 task global ID | host API 栈中关键帧 | 设备输入 → 输出 | 命名设备记录持续 |
|---|---|---|---:|
| 431375 → 102427 → 3165 | `cv_linear.py:108 quantize` → `dsa_v41.py:478 multistream_preprocess` | BF16 `[1023,1280]` → INT8 `[1023,1280]` + FLOAT `[1023]` | 19.580 µs |
| 431379 → 102450 → 3169 | `cv_linear.py:143 matmul` → `dsa_v41.py:491 multistream_preprocess` | INT8 `[1023,1280]`，权重 INT8 `[1024,80,16,32]` → BF16 `[1023,32768]` | 307.920 µs |
| 431381 → 102469 → 3171 | `w8a8_dynamic.py:117 apply` → `indexer.py:168 project_query` → `dsa_v41.py:401 _prepare_indexer_inputs` | BF16 `[1023,1280]` → INT8 `[1023,1280]` + FLOAT `[1023]` | 19.300 µs |
| 431382 → 102474 → 3172 | `w8a8_dynamic.py:124 apply` → `indexer.py:168 project_query` → `dsa_v41.py:401 _prepare_indexer_inputs` | INT8 `[1023,1280]`，权重 INT8 `[128,80,16,32]` → BF16 `[1023,4096]` | 50.860 µs |

各命名 task 的四条设备记录都在 stream 47；两条矩阵 flow 还分别有同流、同 taskId、globalTaskId 不同且无 COMPUTE_TASK_INFO 名称的 TASK 行，不能简单说一条 flow 在数据库里只有一个 TASK。不得将无名行重复计入时长。四条 host API 的 `inputDtypes/inputShapes` 字段为空，表本身也没有提供它们的 tensor/storage ID。当前 trace 能把**主机栈、flow 和这四条命名 task**关联起来，强于仅按形状推测其角色；仍没有证明两个 DynamicQuant 读的是同一个对象、结果能直接复用、实际权重文件的身份或每步 tensor 边。相同输入形状仅是必要条件之一。`1023` 是导出物理维度，不自动代表有效 token 数。设备时长来自重 profiler 采集，不能按 19.300 µs 宣称收益；host 时间和 device 时间也不能直接相减。

冻结源码 `dsa_v41.py:477–478,491–495,376–401` 对应主 Q 与 indexer 都使用变量 `qr`，且主侧 `q_b_quant/q_b_scale` 没有经 `return q,qr` 交给 indexer。结合本记录的调用栈可条件推导：**若**历史运行实际载入同一源码且对应语义无 graph/包装改写，则本次采样的两条分支符合该源码的数据分叉，值得检查共享输入量化；还需 tensor/storage、offset/stride、调用期间是否修改、量化数值和生命周期作为实际复用证据。不得将条件源码关系当成已观测的 tensor 身份。

复核入口：该库 `PYTORCH_API` 以 globalTid `9139690408016` 和固定 scope `1793241329100209040–1793241329168402770 ns` 筛选名称 `aclnnDynamicQuantV2` / `aclnnQuantMatmulWeightNz`，再选择 connection `431375,431379,431381,431382`；`PYTORCH_CALLCHAINS` 对应 `1020827,1020862,1020884,1020891`；`CONNECTION_IDS.id` 精确查这四个 connection 后按返回 `connectionId` 关联 `TASK.connectionId` 和 `COMPUTE_TASK_INFO.globalTaskId`。这一查询范围不是全局缺失审计。此前 EP8 图重放 flow 无 host 起点的限定结论不应覆盖此 EP3 prefill 样本，两次采样的时长或 tensor 不能互相拼接。

### 同一选样的同流记录顺序和时间

复用该次采集的 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl`，按 EP3、scope `2128-42`、stream 47 筛选已匹配记录，以第一条量化开始为零，得到以下**导出区间**：

| 记录 | 开始–结束（µs） | 注释 |
|---|---:|---|
| 第一条 DynamicQuant，task 41419 | 0–19.580 | 上文主 Q 调用关联的候选 |
| 第一条 QuantBatchMatmulV3，task 41422 | 246.940–554.860 | 上文主 Q 投影关联的候选 |
| InplacePartialRotaryMul，task 41424 | 573.400–608.100 | 同流旋转记录；本表未证明它操作上述主 Q 输出 |
| 第二条 DynamicQuant，task 41425 | 608.120–627.420 | 上文 Indexer Q 调用关联的候选 |
| 第二条 QuantBatchMatmulV3，task 41426 | 628.800–679.660 | 上文 Indexer 投影关联的候选 |

这说明上述选中的同流记录并非两个相邻量化调用；主投影和旋转记录位于两次量化之间。筛选来源只包含已匹配设备任务，不能排除未匹配任务或其他流工作；记录间隔也不能解释为调度空闲、等待或可删除时间。如果研究提前发布和复用量化结果，需同时保留主 Q 输出旋转、Indexer 输入和各自消费者的依赖及原地别名；不能根据设备时间先后证明数据依赖或把第二次量化的时间当收益。源 JSONL 的 SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`，仅 CPU 读取已有聚合，未操作设备。

## EP3 两次量化的 host 参数与原始 op_range 身份覆盖

针对上节**同一个 EP3/L36 scope**，只读 profiler 数据库 `PYTORCH_API` 的外层 `npu::npu_dynamic_quant`：主侧 callchain `1020828` 的 `inputDtypes` 记录为 `c10::BFloat16; None; None; None; None; Scalar`，indexer 侧 `1020885` 为 `c10::BFloat16; None; None; Scalar; None; Scalar`；两条 `inputShapes` 都只记录 tensor `[1023,1280]`。该导出只告诉我们参数类型与形状，不包含 Scalar 的值、tensor 的 storage、stride 或 offset；它与前面保存源码的“主侧省略 dtype、indexer 显式 dtype”相容，但不能仅根据 Scalar 这个类型证明 runtime 值是 INT8，更不能证明张量相同。设备导出的结果 dtype 是 INT8，但仍不替代每次 API 参数和完整数值行为的绑定。

为避免重复查整个 trace，本次沿用已有 `query_f32_ep3_oprange_tensor.py` 的有界流式方法，把唯一目标替换成上述两条 `npu::npu_dynamic_quant` 的 host 起止时间：主侧 `1793241329115787980–1793241329117401540 ns`，indexer 侧 `1793241329123994430–1793241329125503160 ns`。从已保存 `FRAMEWORK/torch.op_range` 按记录头遍历 **165722** 条，只在固定线程和目标前后 20 ms 的时间窗口解码 **314** 条的名称与类型 5 tensor metadata，不读取 scalar 参数值或 tensor 内容。每个目标均在两个端点误差绝对值不超过 10 µs 的条件下找到唯一候选，两端均为 **+4.720 µs**；这与该采集先前三个目标观察到的近似时钟转换偏差相同，但时钟转换和生成链没有严格绑定。两个候选均**没有类型 5 字段**，故无法比较 tensor 实现、storage 指针、stride、offset 或对象生命周期。这只限定于两个候选，不证明整份 trace 的所有记录缺失 tensor 信息；同一指针即使未来找到，也仍须判断 offset、分配代际、修改及读完成时序。

复核方式：按前文固定 DB 路径以 `mode=ro` 查 `globalTid=9139690408016 AND callchainId IN (1020828,1020885)`，将 `inputDtypes/inputShapes/name` 外联 `STRING_IDS`；原始记录可在 stage 容器中按 `query_f32_ep3_oprange_tensor.py` 不改变文件本身，令其目标名、起止时间替换为本节两项，再仅比较两者类型 5 信息。原始脚本的原始 SHA 及采集路径见 `f32-ep3-oprange-tensor.json`；本轮只执行内存中的目标替换，不将替换后的脚本误记成原脚本的同一 hash。未输出地址、scalar 或请求内容。

数值和时序的可证伪条件：假设一种逐行对称量化采用 `scale=max(abs(x))/127`、最近偶数取整，并且同一可变 buffer 在第一次量化时为 `[1,2]`、第二次时变成 `[1,4]`，那么首元素分别量化为 **64** 与 **32**；指针相同也无法保证复用正确。两行数值均可精确用 BF16 表示。这只是说明保持值与读完成时序是必要条件的数学反例，不声称历史 qr 被修改、实际舍入恰为此设定或两次输入不同。现有选样的两个量化设备 task 之间还有其他设备工作，缺少可靠 alias/写入追踪时不能将它们视为不可变，也不能由此推断发生了修改。尚需同次输入 identity、版本/偏移及写入与使用时序，才能把“源码有同一个 `qr` 变量”的条件推导提升为实际可复用判断。

## EP3 两次投影前动态量化：可证明等价的条件及缺口（2026-10-30）

聚焦上节**两条投影前**的 DynamicQuant，不使用投影后 F09 的 FP16 scale 反例。已保存的 host 栈和 flow 将 EP3 layer36 中 `cv_linear.py:108` 对应到 task 3165，将 `w8a8_dynamic.py:117` 对应到 task 3171；两个设备输入均导出 BF16 `[1023,1280]`，输出均为 INT8 `[1023,1280]` 和 `FLOAT [1023]`。这些是导出的物理签名及调用位置，不包含输入值、storage、stride 或 device 对默认参数的最终解析。

在两个调用确实采用保存的 W8A8 源码的条件下，主 Q 显式调用 `npu_dynamic_quant(x)`，Indexer 的普通 Tensor 分支调用 `npu_dynamic_quant(x, dst_type=self.act_quant_type)`，该 class 的默认 `act_quant_type=torch.int8`。两者均未显式传 smooth/group；已存安装包说明和 schema 对应 `smooth_scales=None`、`group_index=None`、`quant_mode="pertoken"`、`dst_type_max=0.0`。主 Q 的 `dst_type=None` **按文档**作为 INT8 输出选择，Indexer 则在属性未被覆写时显式为 INT8；声明上其余选项一致，但文档没有证明历史 backend 将 `None` 和显式 INT8 归一为完全相同的内部参数、选择相同内核或采取相同舍入。两个输出的 `FLOAT` 是导出 dtype 标记；它不证明 scale 的内部计算精度、scale 数值或最终 tensor identity。

可以明确一个正向、与具体 rounding 无关的**条件等价命题**：设对同一设备后端和同一约定的输入域，实际解析后的两个调用都实现同一个确定函数 `F(x)=(q,s)`，包含 dtype、归约轴、scale 精度与所有舍入、饱和和异常值规则；两次读取时 `x` 的逻辑值、shape、dtype、顺序一致且没有并发写入。则按函数定义，两次返回的 `q` 和 `s` 位模式相同；无需事先知道 rounding 是哪种，即可推出这两个输出的数学复用条件成立。若只证明接口标为 INT8，或只证明两次名字相同，却没有证明同一确定函数和同一输入快照，此推论不成立。存储地址相同也不足以证明两次读到相同值：中间可能发生别名或异步修改。该命题只说明数值结果，真正共享还须允许消费者读取同一 buffer、维持各自 output dtype、事件/生命周期、无意外原地修改和其他可观察副作用；也不能仅由确定性推断节约的墙钟时间。

对照安装文档的理想有限值模型，若两个调用都按每行最后一维求 `s=max(abs(x))/127` 并使用同一确定的 `round` 与同一中间精度，那么同一行值必然得到相同 `(q,s)`。若不约束输入相等，**即使维度和 dtype 一致也不够**：设一行分别为 BF16 可精确表示的 `[1,1,0,…]` 与 `[1,2,0,…]`，理想实数公式给出的 scale 分别为 `1/127` 和 `2/127`，因此 scale 不同；若 round 在 63.5 上采用最近偶数，首元素 INT8 分别为 127 与 64。这个例子只解释输入一致性的必要性，不代表本次数据或设备舍入实测；零行、NaN/Inf、下溢、随机/非确定实现和其他精度规则不在该例证明中。

当前可将问题缩小为两条独立证据链：（1）历史调用中 None/INT8 的实际参数规范化、执行路径、结果确定性与特殊值契约；（2）两次调用之间的 `qr` 值和 view/stride 一致性、写者与流同步及两个消费者的 buffer 生命周期。源码共同使用 `qr` 及同流 flow 是支持线索，无法替代这两条链。现有安装文档、导出任务和 Python 源码还不足以断言实际可共享，且没有证明该共享是实际瓶颈或可实现 19.3 µs 收益；本次只读，不启动 NPU。

## 同次 Q 投影的稠密形状账本：条件模型，不是已测执行量

对上文 EP3/L36 的两个已按 host connection → flow → 设备任务关联的 QuantBatchMatmulV3，进一步核对既存原始聚合 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl`，其 SHA-256 为 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`：

| 已命名设备 task | 四项导出输入，均按顺序 | 输出 | 记录时长 |
|---|---|---|---:|
| 主 Q，global 3169 | INT8 ND `[1023,1280]`；INT8 FRACTAL_NZ `[1024,80,16,32]`；BF16 ND `[32768]`；FLOAT ND `[1023]` | BF16 ND `[1023,32768]` | 307.920 µs |
| Indexer Q，global 3172 | INT8 ND `[1023,1280]`；INT8 FRACTAL_NZ `[128,80,16,32]`；BF16 ND `[4096]`；FLOAT ND `[1023]` | BF16 ND `[1023,4096]` | 50.860 µs |

两条均是 **BF16 输出**，不是输出 INT32 的另一类 QuantBatchMatmulV3；都还有 BF16 `[N]` 与 FLOAT `[1023]` 输入。保存的 `cv_linear.py:143–150` 和 `w8a8_dynamic.py:124–130` 两套 `npu_quant_matmul` 源码都传入各自的 weight scale 和 per-token scale，前者以 weight scale dtype 请求输出类型，后者以原输入 dtype 请求；该源码与导出类型相容，但仍需实际参数和文件身份绑定，不能单凭形状为这两个额外设备输入逐项确认身份。每个指定 host connection 各对应一条 `CONNECTION_IDS` flow，该 flow 在 TASK 表中既有表内命名的上述记录，又有无名记录；不能把两条视图误算成两次投影，也不据此断言 task 内每条底层指令。

[昇腾 CANN 9.1 QuantMatmulV3 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnQuantMatmulV3.md) 对 INT8 权重非转置 NZ 规定四维为 `(N1,K1,16,32)`，因此**如果该合同及非转置条件适用于这两条历史 task**，输入 K=1280 时两组权重分别与 N=32768、4096 相容。它也给出转置情况下不同的解释；实际 host 记录为 `aclnnQuantMatmulWeightNz`，不是该文档所述 API 的直接版本身份证明，历史参数、对齐和完整 NZ 逻辑索引映射仍须绑定。不能仅凭物理元素数相等就认定历史映射或两权重内容相同。

同名 [CANN 9.1 `aclnnQuantMatmulWeightNz` 官方文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnQuantMatmulWeightNz.md) 进一步提供该接口的 INT8 NZ 形状规则：若非转置，权重是 `[ceil(N/32),ceil(K/16),16,32]`，两条样本分别得到 `[1024,80,16,32]` 和 `[128,80,16,32]`；按转置规则则分别是 `[40,2048,16,32]` 和 `[40,256,16,32]`。如果所保存 host 调用和设备参数确是同一次该 API、且历史 9.2 的契约与此 9.1 版本一致，形状支持非转置而不符合该版本转置布局。**API 同名和形状相容不等于历史版本、参数或对象已绑定**，也不证明完整字节索引、有效行数、传统稠密算法或真实硬容量。

在进一步假设 **1023 个物理行全部执行完整、传统直接稠密 INT8 矩阵乘**、K=1280、N 分别为上述两个输出宽度时，可得形状账本：

| 假想的直接稠密模型 | 主 Q | Indexer Q |
|---|---:|---:|
| 逻辑 MAC，`M×K×N` | 42,907,729,920 | 5,363,466,240 |
| 算术上以各自导出时长相除 | 139.347 TMAC/s | 105.455 TMAC/s |

逻辑 MAC 数为 **8:1**，而记录时长比为约 **6.054:1**。两种不同输出宽度的样本因而不能被强行解释为同一固定“每逻辑 MAC 耗时”。这些 TMAC/s 是**假想算法工作量与重 profiler 设备区间的商**，不等于芯片真实指令吞吐、用户服务 TPS、硬件利用率、理论峰值或可获得提速；有效行、零/非有限值、实现算法、fusion、内部 padding、MIX 核资源及 profiler 影响未确定。其他合法算法或跨算子重组并不必然执行此数量乘加，故这也不是允许任意重组模型的严格工作下界。

作为容量而非传输量，主/Indexer BF16 输出的逻辑元素大小分别为 `1023×32768×2 = 67,043,328` 与 `1023×4096×2 = 8,380,416` 字节；不表示这些字节一定写到 HBM、可被消除或两侧张量可共享。若未来要将此账本与理论性能界比较，需要先绑定同次 tensor 逻辑布局、有效行/数值、算子实际实现及适用的硬件容量，并保留 scale、转换和输出舍入。

复算仅用十进制除法，例如 `M,K=1023,1280; MAC=M*K*N; TMAC_per_s=MAC / Decimal(duration_us) / 10**6`；输入任务和 profiler flow 键见本报告上节。没有启动 NPU 或更改实现。

## 同 scope 第二次量化与一条投影相容任务

从同一次重 profiler 的已存提取中，选择层 2、8、14、20、24、28、32、36，16 rank 各两个 scheduler 摘要，共 **256 个 scope**。每个 scope 按完整物理 shape 唯一选出同一组两条 `DynamicQuant`：输入 BF16 ND `[M,1280]`，输出 INT8 ND `[M,1280]` 与 FLOAT ND `[M]`；另选一条 `QuantBatchMatmulV3`：输入 `[M,1280];[128,80,16,32];[4096];[M]`，输出 BF16 `[M,4096]`。前两条均报告 AI_VECTOR_CORE、block 48，后一条报告 MIX_AIC、block 24 / mix block 24；脚本逐条核验 dtype、format 和这些字段。

两次量化按设备开始时间排序，取第二条；这两条和所选 MatMul 在同设备 track、同 stream，量化与 MatMul 的导出区间有序且不相交。对每个 scope 单独计算 `第二量化时间 / (第二量化时间 + 所选 MatMul 时间)`，采用 trace duration：

| 摘要前缀 | 物理 M | scope | 第二量化时间中位数 | MatMul 时间中位数 | 逐 scope 比值中位数 |
|---|---:|---:|---:|---:|---:|
| `7ddf7ae9` | 595 | 64 | 18.280 µs | 40.499 µs | 30.73% |
| `8c18c293` | 643 | 64 | 17.920 µs | 40.730 µs | 30.59% |
| `db852bff` | 1019 | 8 | 19.450 µs | 50.669 µs | 27.74% |
| `db852bff` | 1023 | 56 | 19.230 µs | 50.880 µs | 27.49% |
| `1b826883` | 1019 | 8 | 17.130 µs | 50.659 µs | 25.55% |
| `1b826883` | 1023 | 56 | 19.410 µs | 49.540 µs | 28.18% |

各项中位数不一定来自同一 scope；全部逐 scope 比值的范围约为 24.57%–35.74%。这里的分母是**选中的两项导出任务时间之和**，不是包含间隔的墙钟时间、模块占比、端到端时间或可节省收益。两个任务虽然同流且时间有序，中间仍可能有其他活动；其他流也可能并行。相同签名不证明这条 MatMul 消费第二量化输出，更不证明两次量化读取相同输入或能共享结果。只有前文 EP3/L36 对相应候选调用有单独的 host 栈关联，不能扩展成其余 scope 的 tensor 绑定。实际有效行、权重、运行版本、关键路径、profiling 扰动与其他资源竞争均未知。

复算 `python3 agents/diagnosis/compare_second_quant_matmul_scope.py`，脚本 SHA-256 `6964587d45ef0e2b6fe1f916990428e699384afb17e21482f9ac27e089ae341e`；它检查源 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` 的 SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`、256 个 scope 唯一及签名、同流、顺序、时长和组内数量。只读本地资料，不访问设备或修改推理实现。

### 所选量化与 MatMul 记录的同流间隔

沿用上述 256 scope，按各自同轨 trace 的十进制时间计算 `MatMul 开始 − 第二量化结束`。以观察后选定的 **10 µs** 阈值分类，15/256 个间隔超过阈值，均在 EP3、track PID `2179523`、track / stream `47`；其余 241 个最大为 **1.74025 µs**。较长区间的两个摘要和层如下；表中单位为 µs，括号中为该 EP3 host scope：

| 层 | `7ddf7ae9…`，M=595 | `db852bff…`，M=1023 |
|---:|---:|---:|
| 2 | 24.860 (`2128-53`) | 23.940 (`2128-8`) |
| 8 | 28.480 (`2128-59`) | 25.180 (`2128-14`) |
| 14 | 26.300 (`2128-65`) | 40.260 (`2128-20`) |
| 20 | 22.920 (`2128-71`) | 16.360 (`2128-26`) |
| 24 | 16.680 (`2128-75`) | 17.980 (`2128-30`) |
| 28 | 23.940 (`2128-79`) | 10.580 (`2128-34`) |
| 32 | 18.100 (`2128-83`) | 25.660 (`2128-38`) |
| 36 | 25.340 (`2128-87`) | 不超过阈值 |

对每个长区间 `[A,B)`，A 是第二量化导出结束时间，B 是 MatMul 导出开始时间；只检查**该 scope 中已通过原提取器匹配、且 track PID、track TID 和 stream 与端点相同的 task**，按 `task.start < B AND task.end > A` 查相交。15 个范围均没有其他匹配 task。量化在 A 结束、MatMul 在 B 开始，端点相触不计入。以最长两例为锚点：M=595 的 L8 为 task 43454 → 43455，区间 `1793241333824419.295–1793241333824447.775 µs`；M=1023 的 L14 为 task 40401 → 40402，区间 `1793241329242541.375–1793241329242581.635 µs`。

未匹配任务、该 scope 外任务、同步与通信事件、其他流、采样遗漏及跨轨时间误差不在此查询范围。空结果不能证明 NPU 空闲、CPU 或调度造成间隔，也不能断言间隔可消除或属于端到端关键路径；同一 profiler 的同一 rank 和层并非独立运行。10 µs 是事后探索阈值，不是预先规定的性能标准。

复算 `python3 agents/diagnosis/check_second_quant_matmul_gaps.py`，脚本 SHA-256 `2ca5886bbbe0184a7910e19d3dbfb2d03bdfd817182f5fca2e77d8050f7e1ba7`；它核验上节相同的源文件 SHA、256 scope 及配对唯一性，输出 15 个区间的 scope、task、端点与同轨匹配，读取范围仅限现存提取文件。未运行 NPU、修改目标实现或采集新数据。

### 同 scope 的其他轨记录与主机 flow 起点

对上节 15 个长间隔，扩展检查该 scope 中**同设备 track PID、其他 track/stream** 的已匹配 task，以同样的导出坐标作正区间相交；15/15 均没有匹配项。此前同轨结果也为零。这是现存 scope/flow/CSV 提取范围内的结果；其他轨之间的相对时间误差没有校准，因此只能称为导出时间坐标未见相交，不能推广为硬件实际没有并行活动。scope 外、其他设备进程、未匹配或未导出任务、同步与通信记录均不在范围内。

现有提取还为 256 对量化/MatMul 各保存一个 `host_start_us` flow 起点；每对的 host PID/TID 均相同。按同一个 host 线程分别算 `MatMul flow 起点 − 量化 flow 起点`，这些是起点间距，不是 API 执行时长、enqueue/dequeue 等待或设备间隔。按上节事后阈值分组：

| 摘要与物理行数 | 设备间隔分类 | scope | 主机 flow 起点间距最小 / 中位 / 最大 |
|---|---|---:|---:|
| `7ddf7ae9…`、595 | >10 µs | 8 | 1081.340 / 2197.305 / 2672.620 µs |
| `7ddf7ae9…`、595 | ≤10 µs | 56 | 1087.430 / 1634.545 / 2440.380 µs |
| `db852bff…`、1023 | >10 µs | 7 | 1239.860 / 1910.270 / 2473.390 µs |
| `db852bff…`、1023 | ≤10 µs | 49 | 1097.170 / 1661.890 / 2736.670 µs |

在同一个 EP3 与 `db852bff…` 摘要中，L20 的主机 flow 起点间距为 1239.860 µs、设备间隔为 16.360 µs；L36 分别为 2736.670 µs 与 1.380 µs。每个差值都只在各自的 host 或 device 时间轴内部计算，没有跨时钟相减。这个例子不能用“主机 flow 起点间距越长，设备间隔必然越长”的简单单调规则描述；它不能确定等待、排队或瓶颈原因。两层的权重、输入、任务状态和未选事件未证明相同，且异步发射与采样时钟仍有不确定性。

复算 `python3 agents/diagnosis/check_quant_matmul_gap_flows.py`，脚本 SHA-256 `46d541a14b32f33ccd80f9133f50fbdbbb7ec9893e613efa0c87f9f15c41c099`；脚本先验证上节查询脚本和同一 JSONL 的固定 SHA，再按同 scope 锚定两条任务、检查同 host 线程并比较其他已选 track。它只读取本地文件，未采集新数据或操作设备。

### EP3 量化间隔与已选 scatter 的同 scope 共现

复用 [F08 已存 scatter 汇总](f08-scatter-variation.json)，按完整 `digest、EP、layer、scope` 与本节量化→MatMul 记录配对。限定 `db852bff…`、EP0–7、层 2/8/14/20/24/28/32/36，共 **64 个 scope**，每个都唯一配对到 F08 的 `[8180,2]` scatter 记录。EP7 的量化物理 M 为 1019，其余为 1023，不能将两者输入工作量视为已证相同。

沿用观察后选定的阈值：量化→MatMul 同流导出间隔 **>10 µs**，scatter 导出时长 **<50 µs**。64 个 scope 中，两项同时满足 7 个，两项均不满足 57 个，其余两类均为 0。前 7 个全部是 EP3 的 L2、8、14、20、24、28、32；同 EP3 的 L36 量化间隔为 1.380 µs、scatter 时长为 260.880 µs。其他 56 个 scope 属于 EP0/1/2/4/5/6/7，各自两项均不满足阈值。

两个阈值均为**事后**选择；这 64 个 scope 来自一次 profiler 采样，尤其 EP3 的 8 个层不是 8 个独立请求或重复实验。它只记录现存选样的共现，不能证明 scatter 使量化间隔变长、二者有共同根因或在线表现相同。七条短 scatter 的结束点在导出坐标上均早于对应量化间隔起点 596.620–1960.160 µs；scatter 在 stream 35，量化/MatMul 在 stream 47，两流相对时钟误差未证，这不证明实际先后或数据依赖。输入 tensor、slot、有效行、未匹配任务与实际状态均未绑定。

上节另有 8 个 `7ddf7ae9…` 的较长间隔，**不在**这份 F08 选定的 8180 行 scatter 表内；没有匹配不等于 scatter 时长或事件为零，不能加入上述列联表。另一个已存 4760 行表的结果见下节。复算 `python3 agents/diagnosis/check_quant_gap_scatter_cooccurrence.py`：脚本 SHA-256 `321d87a0bcf09f1f41105e90b78e949bb1bf14e671df6ed3d6fbd3ab5a00bbb4`，核验原 JSONL SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4` 与 F08 8180 行汇总 SHA-256 `1290969851878c17b1eac8638dfeb1ae71ba045bfce4fda7cbb4e22f0d091dc5`，并核验下节 4760 行输入；检查完整键唯一、目标任务和轨道，并输出两个分类表与 EP3 各层详情。只读既存文件，没有运行 NPU 或修改推理实现。

### 另一摘要中 4760 行 scatter 的有限复核

独立读取已保存的 [4760 行 scatter 汇总](f08-4760-rank-overlap.json)，SHA-256 `ea892e7faf87ad87e010b3c40c5a4f5a5ecbc711ec06b7d2e9107b3e9cbcab5f`。仅取 `7ddf7ae9…`、EP0–7 与同样 8 个 indexer 层，按完整 `digest、EP、layer、scope` 与所选量化/MatMul 配对，共 **64 个 scope**；每个 scatter 还按该表保存的 task ID 和 flow ID 在同 scope 源记录唯一核对。使用相同的**事后**阈值，量化间隔 >10 µs 且 scatter 时长 <50 µs 的有 **8 个**，两项均不满足的有 **56 个**，其余两类各 0。8 个共现均为 EP3 的 L2、8、14、20、24、28、32、36。

这不是把上述 8 个缺失项填入 8180 行表，而是另用具有独立形状范围的已存 4760 行表。该表全部 40 层中另有 EP2 L0 的短 scatter，但 L0 不属于这里的 8 个指定量化层；此处没有为它配对或填造量化间隔。两个摘要属于同一次 profiler 采集，EP3 同一次执行中的层不构成独立请求或重复实验，不能据共现证明两现象互相导致、有共同根因或属于线上瓶颈。

在导出坐标上，EP3 所选 8 条短 scatter 的结束点早于对应量化间隔起点 533.200–1616.640 µs；scatter 在 stream 35，量化/MatMul 在 stream 47，**不同流相对时钟误差未证**，因此这不是已验证的实际先后或事件依赖。任务的 tensor、有效工作与 scope 外活动仍未绑定。上面的复算脚本会同时验证两份 scatter 表和全部 128 个完整配对，输出两个各自的列联表；没有访问设备或新增采集。

### 冻结源码中 scatter 与 indexer 量化的条件顺序

只对照已保存的 `dsa_v41.py`（SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`）和 `indexer.py`（SHA-256 `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33`）。在前者 `:358–376` 所示 multistream 分支、非空 cache、非 packed 写入且 `write_cache_on_main=False` 的条件下，`:485–490` 在辅助流提交 KV norm、RoPE 和 SWA cache 写入，`:491–495` 提交主 Q matmul 后调用 `main_stream.wait_stream(aux_stream)`；此后返回 `_preprocess`，再调用 `_prepare_indexer_inputs`。若 `write_cache_on_main=True`，SWA 写在 join 后由主流提交，不能套用辅助流写入的事件关系。后者在 `:394–401` 将 `qr` 交给 `indexer.project_query`；`indexer.py:101–104,167–168` 经 `self.wq_b` 执行线性。在前文所审计的 W8A8 普通 Tensor 分支成立时，该线性会执行动态输入量化。

**条件结论：**如果所选 scatter 和第二次 DynamicQuant 分别确为上述同次调用中的辅助流写入与 indexer Wq_b 量化，实际加载的是该源码，`wait_stream` 及调用所用流保持所示语义，则源码要求 SWA 写入在相应主流 join 之前完成，indexer 量化处于 join 之后。这里的等待是流上后续工作的依赖，不能当作主机线程同步阻塞；也不能由这个顺序声称 indexer 量化读取了 scatter 写出的 tensor。scatter 写的是 SWA cache（`dsa_v41.py:227–242,296–308`），而 indexer 投影的显式输入是 `qr`；是否存在共享别名、其他间接依赖或资源影响仍需实际对象和事件证据。该条件下的次序也不能排除后续同步或先前工作对资源状态的影响，不能据此确定长间隔根因。

历史 128 个 scope 的散写与量化尚未逐次绑定到同一调用对象、执行分支、event 以及实际运行 binary。EP3/L36 有上文所列量化的 host 栈关联，但它不能替代其他 scope 的 scatter/事件绑定；源文件与安装快照哈希一致也不证明历史进程内代码身份。普通单流和 packed 写入是其他条件路径，不能直接以这里的辅助流事件覆盖。

如果已有资料无法补足，最小补证应在**同一采样的同一匿名 batch/replay、rank、layer、请求范围**内关联：所选 scatter 与两次量化/MatMul 的唯一 task/flow、实际调用分支和加载源码/二进制身份；aux/main 的 stream ID、event 对象及其分配代际、record/wait 与完成事件；scatter 的 cache storage/offset/stride 和匿名槽范围、量化输入 storage/offset/stride 及生命周期；以及同设备跨轨时间转换与相对误差界。可只记录匿名 ID、形状、地址代际摘要与槽范围，不需要提示词或 tensor 内容。仅有相同形状、同名 event 或未经校准的跨流时间不够。新采集须另行取得明确授权；本节只读现有本地源码和记录，没有启动设备。
