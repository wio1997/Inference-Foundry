# ZCode stage 081：Q 投影和 MoE 加法的真实封装

已读取 `/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-081-wrappers.exit-code`（0）与 `.verified.json`（model_verified=true，observed_models 为 deepseek-flash）。`hotspot-wrapper-provenance.json` 记录 baseline 和 stage 的同一冻结镜像 ID、两个文件的双方 SHA，主控侧重新计算本地文件 SHA 均匹配；未对目标服务或设备做修改。

## F05：CVLinearWrapper 会选择多条不同路径

冻结的 `frozen-source/vllm_ascend/ops/cv_linear.py`：

1. 构造时按 `custom_op` / `gather_output` 探测 `_has_communication`；从 quant_method 本身或其 `.quant_method` 内层检测是否为 `AscendW8A8DynamicLinearMethod`。
2. **仅当没有 communication 且被检测为 W8A8 dynamic**：`quantize` 调用 `torch_npu.npu_dynamic_quant(x)`；`matmul` 再调用 `torch_npu.npu_quant_matmul`，将量化输出、per-token scale、weight、weight_scale 和 bias 传入，输出 dtype 取 `weight_scale.dtype`。若 per-token scale 为二维，代码还先 squeeze、后 unsqueeze。这里是源码中两个独立算子调用；没有在 wrapper 里看到把上游 `q_norm` 与动态量化合成一次调用的实现。不能由 Python API 数目断言硬件是否单独物化、编译期有无其他融合或具体 HBM 字节数。
3. **有 communication** 时，`quantize` 直接返回 `(x, None)`，`matmul` 改为 `linear.forward(x)`，内部计算/通信未在此封装展开；其他量化类型也有 MXFP8 和 `quant_method.apply` 分支。故不能对每条执行路径都断言独立 dynamic quant。
4. `dsa_v41.py:440-495` 的多流路径将 `q_norm`、wrapper.quantize 和 wrapper.matmul 按数据顺序调用，并使相邻阶段与 KV 工作交错；`ascend_config.py:569` 的 `multistream_dsv4_dsa_overlap` 声明默认 true。默认值、冻结命令和 W8A8 配置支持相关路径作为**配置推导**，但未直接验证基线每次的 resolved 开关、`_has_communication`、实例 quant 方法及 per-token scale 形状。已有同窗 `RmsNorm/DynamicQuant/QuantBatchMatmulV3` 计数没有张量身份，不能证明每个 task 是这条具体调用。

**F05 结论**：W8A8、无通信、多流的条件分支中，已有实现表现为 norm → dynamic quant → quant matmul 的分开 API 调用，该候选值得保留；确定实际命中、数据落地和是否已被更低层合并需要 trace tensor 身份或运行配置证据。不能把旧 DSA-CP 的 norm-quant 复合接口误认为这条 V4.1 路径。

## F23：已有 `muls_add_triton` 内部融合

冻结的 `frozen-source/vllm_ascend/ops/triton/mul_add.py` 中，Triton kernel 从两个输入读取相同位置，计算 `x * scale + y` 并写入新建的 `torch.empty_like(x)` 输出。具体编译器运算指令、融合乘加与舍入细节仍未知。

`model.py:302-328` 只在 `fused_moe_out` 为 tuple、`shared_experts is not None` 等相应分支调用该内核；该分支还可能执行 MoE TP reduction，随后外层 decoder 才将结果交给 `hc_post`。所以：

- **源码层已有缩放 + 加法组合**，不能把它本身记作新增融合。
- **到 `hc_post` 的候选**须先确认实际 tensor/tuple 返回与 `shared_experts`、rank 的归约/通信顺序、输出 alias 与值，以及 `hc_post` 的 residual/post/comb 和下一层消费者。此后才能研究条件组合；当前不可声称已经可实现或一定省掉该新缓冲。
- 现有 48 次第 0 层窗口有 `DispatchFFNCombine`，表明复合 MC2 名称被采到；当前没有把某一次具体 `muls_add_kernel`、MoE 返回值以及 `HcPost` 通过 tensor identity 绑定，不能以 `ForeachAddList` 的名字代替它。

## 可优先补取的只读证据

1. 从冻结镜像确认 `AscendW8A8DynamicLinearMethod` 的实际模块和普通 `.apply` 分支及对应 `CVLinearWrapper` 运行配置：按 imports 有界定位，避免导入运行服务对象。
2. 从冻结 runner/factory 源码定位 `FusedMoEFactory` 返回值与 shared expert 流程；trace 侧检查是否存在明确 tensor/storage identity，不能仅以名称与时间相邻认定依赖。
3. 其他条件保持 UNKNOWN，禁止按本报告推导可实现的收益，也不改代码。
