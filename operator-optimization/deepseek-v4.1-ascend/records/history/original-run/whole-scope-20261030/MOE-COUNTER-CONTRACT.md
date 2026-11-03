# MoE：实际分支、专家计数含义与容量证据缺口

**状态：源码合同增量已审阅，真实计数尚未采得；M12/M13/M23保持OPEN。** 本文用于选对下一次有限取证的字段，不实现优化、不运行NPU。

## 1. 固定enable_fused_mc2=1走哪个实现

不能因安装了cann_ops_transformer就认定在用CANN MegaMoe。冻结[AscendConfig](frozen-source-snapshot/vllm-ascend/vllm_ascend/ascend_config.py)的`_validate_user_input_ranges:670–679`将用户值0/1的`_MEGA_MOE_SUPPORTED`置false；只有用户值2时检查CANN包，再把effective enable改为1。因此保存serve明确值1、普通初始化无覆写的路径应采用自定义`dispatch_ffn_combine`。

[use_cann_megamoe](frozen-source-snapshot/vllm-ascend/vllm_ascend/ascend_forward_context.py)还同时检查能力、EP、MoE、LoRA和EP大小。`_EXTRA_CTX.use_mega_moe`为false时，[FusedMC2CommImpl.fused_experts](frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py)调用自定义op，并将预分配`expert_token_nums`交给out参数。原官方CSV的类型`DispatchFFNCombine`及形状为相容支持，仍需实际二进制/同调用身份绑定。

另一CANN路径返回fresh `expert_tokens`，不更新这个长期实例数组。此前“读统一对象属性即可取得计数”的办法不能跨两个分支使用；固定当前分支也必须先等待对应写入且防止后续层覆盖。两者都会把实际结果作为`FusedExpertsResult.expert_tokens`向上传递，该边界更适合定义采集合同。

该文件中关于长期数组用途的一段注释与实际两个分支的参数/返回代码不一致。以上结论依据实际数据流，不依据那段注释推断。

## 2. 自定义kernel的输出不是沿专家轴的累计前缀

[kernel GetCumsumForMMAIV:402起](frozen-source-snapshot/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp)读取本接收rank各专家的各来源rank计数，沿**来源rank轴**逐行累加；:825–829把最后来源rank行、长度expertPerRank写到int32的ExpertTokenNums输出。

因此，对这里的源码合同可写 `C_e = Σ_r count[r,receiver,e]`：输出是本接收rank逐专家总行数，而不是`[C0,C0+C1,...]`。不能再沿专家轴做差分，亦不能仅凭内部变量名cumsumMM将输出当group_list_type=0。Python结果类型默认group_list_type=1，与该解释一致；EPLB消费者对type0/1分别差分/直用，也支持这个区分。

这个输出在后续maxOutputSize裁剪之前写出。kernel随后按专家和来源rank计算rowStart，按容量截断搬运/处理范围。因此计数和既有容量必须一起记录：非负计数之和是请求接收量，不自动等于实际处理量。输入active mask、路由有效性、容量截断和舍弃语义仍需端到端核验；不能把zero buffer、前缀差分错误或被覆盖数组用于有效工作量。

## 3. 跨层缓存与时间版本

`get_moe_comm_method`的缓存键含通信类型、num_experts和num_local_experts。目标384专家与draft128专家可有不同对象/数组，同类配置的多层又可能共享该数组；后续调用会覆盖其内容。获取rank/layer/mode信息后、等所有推理结束再读取一个共享实例，通常只能解释最后写入，不能恢复各层分布。

需要绑定的是：请求/调度步与prefill/decode/target/draft；实际EP组、专家本地映射和kernel二进制；本次输入行数及active mask；本次输出计数的dtype/长度/内容；实际maxOutputSize；生产完成与覆盖前复制的事件。针对graph replay，Python包装可能只在capture时运行，不能把capture期零值当replay负载。正式补采须采用能观察实际replay输出的方式，并记录采集开销。

## 4. 容量与条件优化方向

若某一已绑定EP组所有rank的本轮物理行数上界为M_r，top-k为K，且每token专家ID有效且不重复，则单接收rank的安全请求量上界为 `Σ_r M_r × min(K,E_receiver)`。它不需要均衡假设，但group、行数、top-k、mask和重复专家合同必须一致；不是GEMM必要计算量下界。

原later窗口目标相容shape在16rank均有输入[3,5120]、top-k形状[3,6]、本地专家权重第一维24；与384/16专家分区相容。**若**这16rank就是同一EP组且每rank最多3个物理token，接收量上界为288行。这个条件数值不能推广到prefill、其它graph bucket、其它分组或draft；早期窗口已有816/834行等不同输入，固定小容量会有风险。

旧F23的容量/分块/通信重组可据此继续细化：按阶段/graph bucket选统一安全容量、分块发布ready专家工作、复用或拆开工作区。必须保留跨rank对称buffer/handshake一致性、静态graph地址、容量溢出语义、量化和路由/组合次序。配置上限与最大逻辑分配不等于实际触达字节；当前没有证据可给节省HBM、延迟或TPS。

也不能将物理shape上界直接当有用工作：padding/失效行、拒绝后未观察的验证行、共享专家和不同允许算法都会改变有效工作口径。待取得同调用计数、mask与容量，再分别记物理执行、有效路由和可观察输出所需工作，并与资源/时间线核对。

## 5. 下一次有限补证的验收字段

首先核实实际enable值经过validator后的分支、EP组及二进制；再针对同一固定工作负载，分别保留初始prefill和后续decode、40目标层与3个draft block的有限代表步。计数快照必须在本次op结束、同一共享数组下次覆盖前产生，并与输入行/mask、容量和请求步号绑定。没有这些字段，不能用“已成功导出一组数”关闭M12。

当前只有源码语义、已有形状与条件上界，仍未得到实际专家负载/溢出或内部阶段耗时。该缺口继续决定MoE瓶颈、资源界和重组优先级，不能以本文件成稿标整体完成。

## 6. 同次16rank接口增量

原later窗口逐文件核对SHA后，rank0显示目标相容族为BF16输入[3,5120]、INT8权重本地24专家、INT32 top-k [3,6]、BOOL mask [3]，输出为BF16[3,5120]与INT32[24]，480次；draft相容族本地8专家、top-k [3,3]，输出INT32[8]，36次。16rank的逐组次数、dtype与直方图由[接口导出](moe-interface-exports.json)保留，最终汇总另见[计数合同证据](moe-counter-contract.json)。

这把“计数应读哪个输出及其dtype/长度”从抽象接口推进到同次记录；**导出没有tensor数值**，仍不能回答专家负载、失衡、实际有效行或容量溢出。480/36与12次×40层/3块相容，不能单凭次数认定所有调用身份。
