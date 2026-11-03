# Attention 输出投影：量化方案、归约与目标缓冲区接口

## 可以确认的源码条件

- 已保存的本机量化描述与 169 既有容器内副本哈希一致（见 [checkpoint-metadata-comparison.json](checkpoint-metadata-comparison.json)），两个文件的 SHA-256 都是 `2f4020a0a12daa621cfd2ff9d09a2c73e5674af7c624ef47df5c66796c3774bf`。只读 JSON 逐键统计，`wo_a.weight` 和 `wo_b.weight` 各有 43 个以该后缀结尾的键，全部值为 `FLOAT`；例如 `layers.0.attn.wo_b.weight` 和 `layers.24.attn.wo_b.weight`。这不是在统计权重 tensor 的 dtype 或历史运行时加载实例。
- 冻结 `configs/modelslim_config.py:185–198,213–214,265–318,655–706` 中，V4.1 对运行前缀 `model.layers.N.self_attn.wo_b` 提供映射到 `layers.N.attn.wo_b` 的规则；`FLOAT` 解析为 `None`，LinearBase 因而选择 `AscendUnquantizedLinearMethod`。冻结 `model.py:799–807,438–445` 构造 `.self_attn.wo_b`，并将 quant config 传入。若模型类型、前缀、配置、代码均对应上述快照且使用相同描述，可推得该源码路径选择非量化 linear；历史 runtime 实例、加载权重、mapper 完整操作和二进制尚未绑定，不能把条件推论称作历史实测。以整模型文件名是 W8A8 直接断定每个层的所有线性都是 INT8 不成立。
- 冻结 `ops/linear.py:49–54,100–139,143–170` 的非量化方法调用 `torch.ops.vllm.unquantized_gemm(x, weight, bias)`，该自定义算子 Python 定义返回 `torch.nn.functional.linear(x, weight, bias)`。上述应用层函数签名没有目标 output；这仅说明当前封装调用未传，不证明所有设备级 GEMM 接口都不能指定输出。
- 作为对照，冻结 `quantization/method_adapters.py:209–224` 的 `AscendLinearMethod.apply(layer,x,bias)` 也没有 out，并委托 scheme；`methods/w8a8/w8a8_dynamic.py:60–65,94–134` 的 W8A8_DYNAMIC 方案在非预量化输入情况下先 `npu_dynamic_quant`，然后以权重、per-channel scale、pertoken scale 调用 `npu_quant_matmul` 返回结果。当前调用没有 out。169 **stage 容器**安装的 `torch_npu/csrc/aten/npu_native_functions_by_codegen.yaml:693–696` 声明 `npu_quant_matmul(...)->Tensor`，无 out 参数，该文件 SHA-256 为 `7bdad689a5472f854e26b5eecf0e9012da8594036c60b77bcd8c5ad684d37849`。它是 stage 安装的 API 声明，不是历史 loaded binary 的绑定，也不代表底层 ACL、自定义算子或其他 overload 不可能提供目的地址；不能推断真实历史投影走此量化方案。

## 输出和通信的分支约束

冻结 CP 源码 `_apply_wo_b` 在 `full_weight=False` 时调用 `self.wo_b(...)`，在 `full_weight=True` 时直接调用其 `quant_method.apply`（`dsa_cp.py:1657–1664,1795–1799`）。之后 `output[...] = projected_output`。该返回值若只是单 rank partial，不能要求它等价于归约后的完整结果；需先按实际分支定义 output 的值。

1. 冻结 `ops/linear.py:327,380–391` 可能选择 Ascend 自定义行并行或回退基类。`ops/linear_op.py:422–429` 在 `wo_b` 且 `oproj_tp_enable()` 为真时选择 `DSV4OProjRowParallelOp`；其 `:321–335` 仅调用量化方法得到 `output_parallel` 并返回，**本函数内没有 collective**。它的 `CustomRowParallelOp.apply` 只收 input（`:117–135`）。这一分支不能据此推断其他位置没有通信。
2. 若不选择 custom op，stage 容器的 `/vllm-workspace/vllm/vllm/model_executor/layers/linear.py:1757–1783` 回退 `RowParallelLinear.forward`：先 `quant_method.apply`，仅在 `reduce_results` 且 `tp_size>1` 时调用 `tensor_model_parallel_all_reduce`，否则返回 partial。stage 的 `distributed/communication_op.py:12–14` 委托 TP group；`parallel_state.py:721–748` 的 group `all_reduce(input_)` 没有目标 out，世界大小 1 时返回输入，否则路由到 out-of-place 自定义调用或设备 communicator。它们是 stage 容器读取的带哈希源码，与历史实际加载文件是否一致未知。
3. 冻结 `model.py:799–807,832–836,899–902` 的 sequence parallel 路径将 `wo_b.reduce_results=False`，attention 输出之后另行执行 `sp_reduce_scatter` 再 `hc_post`。若 output 是该分支的输入，它应承载归约之前的 partial，而非直接改为 reduce-scatter 后的结果；形状和调用契约也不同。另有 `OProjRowParallelOp` 的 all-to-all→linear→reduce-scatter（`linear_op.py:338–375`），但 `wo_b` 的 custom 匹配优先落在 DSV4 专用类，不能把这个通用 `o_proj` 分支当作已确认的 `wo_b` 路径。
4. `linear_op.py:194–226` 其他模型 Kimi 的特殊分支中有 `dist.reduce_scatter_tensor(output, parallel_output, ...)`，展示调用方传入 destination 的一种代码例子；其 shape、类型、进程组、同步、图捕获等与当前 attention 不一定一致，因此不能将其当成 910C 此模型可直接应用的证明。

## 候选与无法推断的部分

对于无需在 `wo_b` 后立即做归约的分支，可以研究让最后产出该分支规定值、dtype 和 shape 的 GEMM 直接写 caller 的 output；对于必须完成 all-reduce 才满足原 output 契约的分支，只能让适配该 collective 的最终结果落入 output，或在语义证明后把归约重新安排。单纯消掉 Python 赋值、提前将 partial 交给消费者，或盲目把当前接口改成 out 参数都不构成正确方案。对分块重组需额外证明每块的完整 K 累加、所有 rank contribution、存储不别名和生命周期、通信次序及事件同步、padding、BF16 舍入及 graph capture 行为；不同 API 的 shape 和 device/layout 支持也需核实。

[既有身份核查](../diagnosis/TENSORMOVE-EP8-IDENTITY.md) 将选中 EP8 L0 TensorMove 唯一 flow 关联到 `aclnnInplaceCopy` 及相应调用栈，**不绑定**权重张量、前后操作地址、此处运行时量化对象、具体图编译版本或所有通信分支。无法据此量化可节省端到端时间，也无法证明消除这个 copy 可行或会提高吞吐。后续优先检查已有启动/加载日志、profile 或已保存图中是否含实际量化方法与版本、full-weight/TP/SP 配置、producer-consumer 地址及事件；无该证据则列为未知，不新增 NPU 负载。

## 可复核资料

上述 `vllm_ascend` 文件均为现有任务的冻结快照：`linear_op.py` SHA-256 `2a28240882788db363c14a0999b1cad68b4f9f63896d0ad9475f2d39051e818b`；`modelslim_config.py` `92c924abc0af1ae37d7614ee01cbae0b3be1da86ef6abc885b2208560f3dcbb1`；`w8a8_dynamic.py` `a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910`；`method_adapters.py` `341d93cae5774414a810444a50eb752cd425330156f404d1b9a10381e6bc0138`。stage vLLM `linear.py` SHA-256 `094fdf956c35bcfcb44b924a4ff60bb1768285963e4e1ae27f3022dd7e1e852d`，`distributed/parallel_state.py` `d9b10ad56833ebc08a753e1a050ef0989c30d4134021b7b7b00150b7bd01adc6`，`distributed/communication_op.py` `3c3f8ac60db38ece891d39dc1e2f6f1947ad438685893d2e5777d072320bd1b8`。这些 stage 文件只是当前安装快照。检查仅使用本机已有描述和只读 `docker exec`；一次只读尝试导入 torch 查询运行时 schema 因容器无法创建临时文件而失败，转为读取安装的 YAML 声明，没有进行设备计算、启动新服务或修改目标实现。
