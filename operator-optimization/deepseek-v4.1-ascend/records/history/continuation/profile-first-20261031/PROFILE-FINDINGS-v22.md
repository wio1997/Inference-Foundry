本轮有实质进展，整体仍未完成。继续按“充分使用profile，不能单独定案时再定点查代码”执行；没有运行模型/NPU、修改目标源码或变更服务。

完整保留了4,472次ForeachAddList：1,271次直接、3,200次图内、1次精确队列桥接，共9个shape/context组合。3,200个图内合流到FFN HcPost都由已有层块结构精确定位，同一TASK stream，间隔最小/中位数20ns、最大41ns；间隔内无其他计算算子，也无原生COMM_OP区间重叠。这是保存时间线中的可见边界，不能由此证明tensor指针、所有消费者或可节省延迟。其中2,560对decode-compatible，640对仍未绑定阶段。

去除全局定义ID后，这些图内边界只有1种shape/dtype签名：合流两个BF16[3,5120]输入，产出BF16[3,5120]，交给HC的视图为[1,3,5120]；HC另收BF16[1,3,4,5120] residual及FP32 post/comb。最初680种完整定义签名不是680种shape。控制区间重叠22,854次是“区间与层间隙”的配对次数，实际涉及6,787个不同TASK键；宽MODEL_WAIT_COMPLETE不能重复计作3,200次独立等待。

profile不能单独说明缩放/alpha和后处理语义，所以本轮只取了三个固定源码文件的相关函数，以及已安装Torch的静态声明。当前runner调用torch._foreach_add([shared_output],[fused_output])[0]，未传alpha，声明默认alpha=1。若runner有效gamma不为1，非FP16路径先单独原位缩放routed，再做输出变换及普通add；FP16 shared分支采用另一缩放方式。没有把配置里的模型gamma或位置属性文本当成当前runner有效gamma。当前runner返回Tensor，与model里的legacy tuple/muls_add分支分开记录。

合流前后的shared/routed归约、输出变换、feature维裁剪和可选zero-expert贡献仍有条件。特别是Ascend final-reduce使用层的静态SP/early策略及context-aware custom op，不能将capture时output_is_reduced布尔值冻结成每次replay的真值。图中无可见中间计算只能约束这些已捕获执行，不能证明所有分支的后处理恒为identity。

另一个profile直接给出的边界是行数：320次target原生合流输出M=800，而原始target MC2只有M816和M834两个原生行数域。不能将全部raw MC2输出与最终merge张量视为同M。此次读到的_maybe_pad_hidden_states处理feature维，不解释token行数差异；尚未按序号或最近时间把某次M800强配到某次M816/M834。

数值边界也分开登记：删除中间存储，与删除BF16舍入是两件事。两个stdlib CPU有限算术例子在明确的FP32/BF16最近偶数舍入模型下，分别得到1.5与1.5078125、0与0.00390625，说明任意保留更高精度或合并scale/add并非无条件等价。这不代表冻结模型必然达到这些输入，也不是已加载HC二进制的数值验证；实际native精度/顺序仍须绑定。若仅删除外存materialization，仍可要求保留原有dtype舍入。

候选图覆盖两算子Add->HC、三算子SharedDown->Add->HC，以及允许改变生产顺序的多算子重组条件域。另有一个明确受限排除：保持现有MC2先完成、SharedDown后执行的边，却让MC2尾部等待SharedDown，会形成MC2->SharedDown->MC2环。这只排除“仅改上游epilogue且保持原生产顺序”的方案，不排除新内核或新排程。当前基线没有被判定为死锁，也没有设计或实现融合。

HC仍须保留x、全部residual通道、post/comb，以及原有积累/舍入合同；decoder单独返回ffn_pre供后层使用，不能融合后丢弃。已有Engram、auxiliary及最终状态消费者约束继续有效。普通源码调用流不等于所有alias、固定地址、ready事件和跨迭代buffer代际已闭合。

本轮还补取了原DB尚未解码的attrInfo字符串：覆盖286,950次计算执行和83类算子，121,920次命中非空候选字符串，165,030次为空或placeholder，共181个op/string组合。但3,200个FFN HcPost全部复用了前一ForeachAddList的同一属性ID和文本。这个事实要求先核对已安装exporter的字段关联，不能把input_2:1当HC参数，也不能据此断言exporter有bug。下一步先查保存schema/exporter索引，只有它无法解释时才点查精确赋值路径。

合流输出逻辑payload合计8,135,065,600字节；一写一读接口模型为16,270,131,200字节，图内部分为196,608,000字节。它们不是实际HBM/网络流量、普适必要I/O或请求收益，不能用来承诺提速。融合后必要工作、存储层和容量界需重新推导。

主台账仍为33项模型候选、7项框架候选；新增16项有限边界条件和6个F23子域，不新增重复主候选。26个整模块均未宣称关闭，Goal保持active。全局资源归属、物理拓扑、MC2内部原因、当前EngineCore成本及其他消费者仍有明确缺口；最后健康观测仍是旧v12post，本轮无新的健康或空闲声明。
