# Loop-004：三个热点边界的现有实现与 trace 能证明什么

## 数据口径

`python3 agents/fusion/extract_hotspot_coverage.py` 从既有 `captured-layer-attribution/operator_metrics.csv` 过滤第 0 层、16 rank 各 3 次内部窗口，共 48 次；模块归属来自 `captured-layer-attribution/module-call-metrics.json` 的源码顺序和角色约束，不是逐 tensor 身份证明。提取结果和两个输入 SHA 在 `agents/fusion/hotspot-window-evidence.json`。各算子名称计数是在这些窗口内的聚合，包括可能的不同调用点；不能把任意两个同窗 task 直接认定为一条张量依赖。

## F02：mHC pre → 输入 RMSNorm

- **源码可证**：`model.py:858-869,888-902` 通过 `npu_hc_pre_v3` 获得 x、post/comb/pre 四类输出，再以 `input_layernorm(x)` 处理 x。`hc_pre` 已是复合接口；从当前 Python 源不能证明其与之后 norm 融为同一次底层执行，也不能证明没有通过编译器/硬件做进一步优化。
- **已有 trace 可证**：48 个第 0 层窗口共归属 `HcPre` 96、`RmsNorm` 144、`RmsNormCast` 48 个 task。多处 norm 同时存在，单靠名称数量不能把某个 `RmsNorm` 精确绑定至输入 norm，也不能仅凭计数判断中间数据是否访问 HBM。
- **状态**：候选边界存在，`HcPre` 本身已有复合；跨 pre/norm 的实际连接与物化 **UNKNOWN**。归一化输出后在 SP 分支有 all-gather，是否当前基线每次真的走该分支不能仅由本表证明。

## F05：Q norm → 动态量化 → Wq_b

- **源码可证**：`attention/dsa_v41.py:352-384,440-495` 的多流路径明确 `qr = q_norm(q_a)`、`wq_b.quantize(qr)`、`wq_b.matmul(...)`，代码还将 Q norm/量化与 KV matmul 交错；非多流路径是 `wq_b(qr)`，量化内部实现需要对应量化方法源码。`device/device_op.py:317-329` 有动态量化适配器，但当前快照并未证明上述 wrapper 具体绑定到该适配器，不能把它直接当作 Wq_b 的全部底层实现。
- **已有 trace 可证**：48 个第 0 层窗口中有 `DynamicQuant` 96 和 `QuantBatchMatmulV3` 144 个 task，也有 144 个 `RmsNorm`。同窗存在不同 Q/KV/其他投影，**名称和次数不足以把三者一一绑定成 qr→quant→Wq_b**。先前跨 40 层的相同 shape 匹配了 1920 个 matmul 样本，但 shape 与时间窗仍不等于 tensor 身份或实现域证明。
- **状态**：F05 是源码级直接消费关系，层 0 非 index source 的作用域见 `QR-consumers-and-fusion-review.md`。实际执行的是多流/单流、具体量化算子绑定、输入输出地址与 HBM 物化 **UNKNOWN**；不能把统计数据视为新的可实现融合证据。

## F23：MoE wrapper → FFN mHC post

- **源码可证**：`ops/fused_moe/moe_comm_method.py:563-611` 中 `enable_fused_mc2==1` 可走 `dispatch_ffn_combine`，或依据 `_EXTRA_CTX.use_mega_moe` 走另一条 MegaMoE 分支；前者通过一个复合调用完成 expert 路由相关工作。`model.py:302-328` 的缩放/add 仅在 tuple/shared expert 的相应条件下调用 `muls_add_triton`，并可能还需 TP 归约。`model.py:904-920` 把 MoE 最终输出、residual、ffn_post、ffn_comb 交给 `hc_post`。
- **已有 trace 可证**：48 个第 0 层窗口归属 `DispatchFFNCombine` 48、`HcPost` 96 task。该复合 task 实际出现在已归属层窗口，不能将其内部再算成新的未融合算子；这不证明 wrapper 所有 branch 的 tensor 接线、tuple/shared 路径或某次 `muls_add_triton` 是否执行。同窗出现 `ForeachAddList` 48 也**不能**仅凭名称认作 `muls_add_triton`。
- **状态**：MC2 复合执行有采样支持，后接 `hc_post` 有源码语义；两者之间的真实输出缓冲、通信完成、选用的 wrapper 分支和性能影响 **UNKNOWN**。不能因为两个 task 同时存在就声称 F23 精度或融合实现已解决。

## 值得继续的只读取证

1. 在已授权冻结镜像只读快照与双容器 SHA 比对：W8A8 动态线性 `wq_b` 的 quantize/matmul 方法与模型包装绑定；追踪 `mul_add.py` 与 MoE runner/factory 的 tuple 或 tensor 返回条件。缺少绑定时仅表示资料缺口，不能判不可融合。
2. 已有 profiler DB 若保存该调用的输入输出 tensor/storage 标识，以同一 rank、同一 graph replay、stream、连接及依赖链核验 F02/F05/F23 的来源和消费者；仅按相邻时间或名称不能证明。若数据库未记录 tensor ID，登记无法确定，不能伪造。
3. 若要分析 HBM 或省去的流量，需要对应存储层计数与 tensor 物化证据；当前没有这些就保持符号。没有新增 NPU 测试要求。

未修改目标框架、模型、operator 或设备状态。这里三项仍是条件候选，不判断提速，也不声称完整图扫描结束。
