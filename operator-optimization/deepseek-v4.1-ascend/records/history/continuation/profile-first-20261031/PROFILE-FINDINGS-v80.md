本轮补上A3局部数据通路的容量模型，并据此修正融合接口预算的适用域。未执行模型、设备测试或优化实现。

当前官方[2201架构资料](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)描述了分离的Cube/Vector核、192KiB UB、512KiB L1，以及L1到L0A/L0B的256/128字节每周期通路。[3510迁移资料](https://asc.gitcode.com/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.html)明确把UB→L1、L0C→UB直达列为新通路，因此不能将这些能力套用到A3。资料为公开master预览，未绑定本机CANN9.2二进制。

这会改变“融合后中间结果全留片上”的预算。在保留Cube矩阵计算、Vector量化/激活/合并的A3实现域内，内部GM中转仍需计价。共享专家M=3的四处数据桥合计216,576B，接近此前放松模型的216,624B全部接口预算。GM访问可以命中L2，不能直接换算为HBM流量。

F22-SPLIT外部FP32激活面8MI字节，也不能直接替代内部INT32 Cube→Vector桥的16MI；若两者都物化，应计24MI。融合仍可减少启动、缩小workspace、形成分块流水，VV可在同一UB复用，CC在符合条件时可通过L1传递。候选需要按具体计算单元和数据通路重新计价。

[GetTensorC](https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetTensorC.html)可返回UB结果，但异步模式会使用GM workspace；API名称或返回类型不足以证明硬件直连。类似地，F23若仍由Vector做共享/路由合并，就需保留Down到合并的GM桥。能否借Fixpipe/原子输出保留“Down先BF16舍入，再相加并舍入”的合同，已列为下一项具体问题。

本轮对97类、30,030次BF16/INT8矩阵执行建立了条件周期模型，并关联全部30,030份保存的PMU记录。没有发现模型超过已记录MTE1周期预算，但这不证明实际采用该通路，也不证明带宽饱和或MFU。模型明确限定为L1进入L0、必须读取未压缩权重的稠密算法；2201还存在GM直达L0通路，其他实现要重建下界。

| 条件模型示例 | 周期下界 |
|---|---:|
| Q_B权重经L1→L0B，允许初始L0B全暖 | 13,141.33 |
| 五步Markov完整W2重算，同一通路 | 105,173.33 |
| Q_B与共享Gate-Up/Down的三份独立权重联合通路 | 24,661.33 |

联合模型只允许一次初始L0容量赠予，并先累加同一资源上的必要工作，再与其他资源下界取最大值。捕获频率1.8GHz不是硬件最高频率证明，因此对应微秒值仍是带额外频率前提的参考，尚不能标为本机实际时间下界。五步Markov即使权重在L2中保持温热，在该L1重算域内也仍有局部搬运工作；memo、预计算和剪枝属于其他算法域。

已给35项模型候选建立CV/CC/VV接口索引，更新13项候选和9个模块的相关条件。现有342条分支记录、46条候选对关系保持，新增一组共享局部资源约束后共11组。实际Vector、FP32/HF32、排序、非线性、整数、内存/网络及频率上界仍有缺口；全模块最低MFU尚未定案。2,073类、285,598次条件分析与9类、1,352次关键操作数缺口保持，Goal继续进行中。
