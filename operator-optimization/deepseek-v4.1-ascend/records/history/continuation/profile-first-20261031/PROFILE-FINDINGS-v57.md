本轮分析共享专家 DQ→GateUp→DequantSwiGLUQuant→Down→Merge，覆盖42个捕获shape/role/context组合，每阶段4472次执行。仅补读了2个算子调用方法及3份本机SDK头文件。

| M=3图层阶段 | 中位设备耗时 |
|---|---:|
| DynamicQuant | 3.82μs |
| INT8 GateUp→INT32 | 25.681μs |
| DequantSwiGLUQuant | 4.66μs |
| INT8 Down→BF16 | 21.52μs |
| ForeachAddList | 20.561μs |

共享上投影保留INT32累加，再进行FP32反量化、clamp/SwiGLU和逐token量化；它不同于routed MC2内部FP16路径。完整行的量化尺度约束跨N-tile流式执行。可研究G1 epilogue先输出FP32激活与局部max，再完成行量化，将该中间接口从16MI字节减到8MI；仍需保留全行归约、NaN/zero及舍入条件。

本机存在GroupedMatmulSwigluQuant及NZ版V2头文件，公开算子定义也包含GEMM、反量化、SwiGLU和量化。但当前调用使用clamp=10及SwiGLU mode1，而这些参数未出现在该V2融合接口。普通SwiGLU与clipped版本有明确反例，不能把“接口存在”当成当前模型可直接替换。[原始API说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnGroupedMatmulSwigluQuantWeightNzV2.md)

优先候选是将单元素foreach合并特化为普通Add，或把同一BF16 Add融入Down epilogue。前者保留alpha=1及相同Tensor语义；同profile中更大BF16普通Add为3.88–4.35μs，支持优先验证，但不是当前shape实测提速。64个root图中merge设备时段曝光份额中位1.7103%，也不是端到端可实现收益。后者必须先保留Down的BF16舍入，再add并舍入；不能直接把未舍入累加器加到routed结果。

| 理想接口消除模型 | M=3 | M=800 |
|---|---:|---:|
| 可减少的逻辑中间读写 | 216,624B | 57,766,400B |
| 原链逻辑字节量 | 35,726,896B | 117,760,512B |
| 理想融合后逻辑字节量 | 35,510,272B | 59,994,112B |

用此前注明的参考752INT8TOPS/1.6TB/s作条件Roofline，M=800的聚合下界融合前后均为75.2967μs：计算项已高于访存项，所以省57.77MB不能直接报省36.10μs。原串行阶段的参考下界108.7222μs与理想聚合下界也不能相除当真实加速。缓存、向量计算、量化归约及本机容量尚需进一步绑定。

Down+Merge融合与routed/shared重叠还存在取舍：共享Down先完成时，routed结果可能未就绪，需要暂存、占住资源或延迟Down。若只重叠Down之前三阶段，观测span中位34.501μs；不能同时领取完整57.301μs重叠预算和“无Down暂存”的全部收益。

共记录7个条件方案：输入量化＋GateUp、G1＋激活量化、FP32 tile＋行量化拆分、受clamp条件约束的库融合复用、Down＋Add、两支量化输入共用、singleton Add特化。下一步转向HcPre/HcPost、norm/quant及其它热点算子的定量上限。整体仍未完成，没有实现或新增设备测试。
