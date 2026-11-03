本轮已按用户纠正，将主线限定为算子融合、拆分、重组及理论上限。已有框架材料只作为必要的数据依赖或数值条件，不继续独立展开HTTP、SSE、请求与框架生命周期。

完整profile现按83种算子、2082个shape/role/context组合建立台账，覆盖286950条ComputeInfo执行。87个满足形状条件的二维dense矩阵组合已填入2MNK计算量、逻辑操作数/结果字节量和条件Roofline；其余明确保留待建模，不以有表格代替分析完成。

本轮算子链：RmsNormCast → FLOAT路由MatMulV3 → MoeGatingTopKHash。3200个图层通过原始block/TASK身份绑定，其中2560有root关联，640仍phase-unbound。以下M=3,H=5120,E=384：

| 算子 | 中位设备耗时 | blockNum |
|---|---:|---:|
| RmsNormCast | 3.16μs | 3 |
| FP32路由GEMM | 17.36μs | 24 |
| TopK/Hash | 4.82μs | 3 |

root关联链的三核累计中位为25.601μs，首尾span中位33.34μs。间隔包含其它ID/control/重叠工作，不是可直接删除的空闲。block数也不是实测物理利用率。

融合结论：RMSNorm+双输出cast已经融合；后续可条件地将BF16结果在router内部widen为FP32，减少独立FP32平面；router GEMM与非线性/偏置/hash/TopK也可作为融合边界。必须保留BF16舍入后再widen的数值边界、FP32权重以及专家/shared分支的BF16消费者。不能直接删BF16平面、换成未舍入norm中间量、跨层复用同shape结果或按top-k不变性取消norm。

| 逻辑接口 | M=3 | M=800 |
|---|---:|---:|
| FP32 norm平面写入+读取，8MH | 122,880B | 32,768,000B |
| router logits写入+读取，8ME | 9,216B | 2,457,600B |

小M两项仅132,096B，约为7,864,320B路由权重的1.68%；单靠省这部分访存，不能预期大收益，应重点看启动、归约和小M tiling。大M的中间平面更大，但800→816的padding域必须保留，不能按相邻时间或数量强配同一调用。上述是逻辑接口估值，cache可改变真实HBM流量。

采用华为论文给出的910C参考die参数，BF16/FP16约376TFLOPS、INT8约752TOPS、HBM约1.6TB/s；它们用于条件模型，没有冒充当前服务器的实测容量。FLOAT router不能套用BF16峰值。模型公式为T≥max(F/P_dtype,Q_HBM/B_HBM,Q_L2/B_L2,其它pipeline/依赖下界)，每项必须确认资源与分母。M=800的FP32接口按全部穿过HBM估值20.48μs，不是实测可省时；M=3两接口同条件仅0.08256μs。[硬件原始论文](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1)

下一步优先分析DispatchFFNCombine及专家GEMM、激活、dispatch/combine通信子图，再推进投影、attention、indexer、Engram与head/sampling链。整体算子覆盖与理论上界尚未完成，继续保持进行中；没有实施优化或新增设备测试。
