# SWA主辅流join与较大输入交付判断

2026-11-10，loop038首个限定review。每个替换原语逐位一致；全链路未扫全；新增原语、模型请求、NPU工作和新profile均0。

介绍：接loop037的SWA局部3.07%窗口上界，沿真实主辅流Qb与缓存发布到join后的query RoPE和SparseFlashMla消费者；判断局部省时能否越过已有重叠。

到了哪里：EP0/8既有steady `kernel_details.csv`各35,664行全部保留（含重复展示的collective，未将其相加）。每rank 516条SWA三算子链与516个Qb原生任务、516个前向query RoPE一一绑定，进一步关联下一个SMLA。绑定按源顺序、几何、stream和局部时间约束；原件[逐段结果](swa-publication-join-price.json)，[输入位置/SHA](swa-publication-parent-call-pins.json)。这是source一致的native关联，尚未用runtime event ID和真实tensor地址证明每条依赖。

结果：**固定Qb到达时刻，独立SWA发布即使全免，也仅能令join提前约.65%–.66%窗口。** loop037的3.07%是原工作量，不是暴露净省。它已按冻结原件保存，本轮单独补充，未改旧Run或冒称当时已知。独立SWA-only不据3%立项；进一步看WKV/Q共同就绪、资源竞争与数学/计算共享边界。

| 已有steady窗 | EP0 | EP8 |
|---|---:|---:|
| 24行SWA→Qb matched条数 | 480 | 480 |
| 20行SWA→Qb matched条数 | 36 | 36 |
| cache比Qb晚的条数 | 516/516 | 516/516 |
| 24行cache晚于Qb中位 | 11.80µs | 11.7195µs |
| 20行cache晚于Qb中位 | 10.59µs | 10.4795µs |
| 固定Qb、SWA全免join最多提前 | 6.21354ms / .660879%窗 | 6.031321ms / .648774%窗 |
| 删半数native工作、每段加5µs | .660879%窗 | .648774%窗 |
| 删半数、每段加20µs | .442238%窗 | .422514%窗 |
| 只删25%、每段加20µs | −.327704%窗 | −.343791%窗 |

算法逐条保留负值：`max(cacheEnd,QbEnd) − max(cacheEnd − fraction*nativeSum + added,QbEnd)`。不删除两者ready到query RoPE的空隙，24行中位仅.84/.92µs、最大1.68/1.76µs；20行中位1.27/1.25µs。Qb native耗时已被覆盖的部分不再计为省时。

局部连续、形状吻合与一一映射并不证明完整runtime因果。全部被匹配cache结束都先于query RoPE，Qb与norm起点间距小于1ms；没有重复Qb/query RoPE。两个main stream池分别480和36，和目标40层及草稿3层×12吻合，但仍保留真实父事件标签缺口。trace窗口、instrumentation、rank关键路径、主机提交和跨卡先后均限制请求推断。

本价格的适用边界是**其它流到达不变**。融合若减少Vector/MTE竞争，Qb可能也变化；若增资源或同步，可能更慢。不能把.66%当作任意硬件重排的绝对界，也不能据两个短窗给六ordinary/完整请求的去留。更大的WKV/Q joint parent因此保持待确认，不将其剪掉。

下一处实际交付：`dsa_v41.multistream_preprocess`的CVLinearWrapper及W8A8方法，追Wq_a/WKV共享的输入量化和通信资格、Wq_b INT8×NZ权重→原BF16 query→partialRoPE→SMLA。只在原精度/归约/量化边界可保留、整段有净价时继续；现成RMS-DQ或FMA名字不直接准入。actual MatMulV2和QuantBatchMatmulV3的选择、原输出其它读者、真实tiling/binary与原语对拍留作具体缺口。
