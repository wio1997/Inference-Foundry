# QLI 分核、末读者与容量及净价判断

## 介绍

接续整段 QLI 6.012243% 的 first-token 有利预算，继续实际 host tiler → metadata 分核 → Cube/Vec → token/candidate topK → 最终索引交付。先找整段少算/少搬和资源复用，再判断局部实现；每替换原语逐位一致。上一轮的量化/chronological-sort 小预算停止仍有效，不剪较大 QLI/SMLA/容量范围。

## 到了哪里

补读12份当前运行树依赖；18份 QLI native 源身份复核，实际 ranges 见 `indexer-native-contract-source-review.json`。arch22 三个主 kernel 文件、候选 max/pad/pin/merge、Sort32/MrgSort/CopyOut helpers、host parser/tiler 主路径和非FD metadata分配路径已读到相应 intrinsic 接口。A5 专属与未读 API 内部另列，不以 inventory 或文件 SHA 证明全链路完成。

171 主机名核对成功；16个当前 worker 均映射 custom_transformer opapi，initial ASCEND_CUSTOM_OPP_PATH 首项也是它。当前 opapi SHA `5e7735b6…`，导出 QLI/metadata main 与 GetWorkspaceSize；QLI 包 JSON 与 .o SHA相符，原 .o 为 `f1dd0d93…`，有4个编译key。这与旧 byte-membership 对拍的 original binary 相符。源码、磁盘包、进程映射及优先顺序仍不等于每调用 device binary/tiling buffer 或加载中内存字节的完整证明；不据此冒充新设备验收。

实际固定族为 arch22、INT8 Q/K、FP16 weights/scales、TND/PA_BBND、G32/D128/top512、ratio1/2、mode1/2/3。host验证必需表/长度/metadata1024、TND批前缀、内轴连续、page0 padding stride、原key/scale dtype和候选2048×8。stride既可由动态图描述取得，也可由adapter显式传入；新布局不能绕过这些字段。

**分核与末输出**：metadata类默认 supportFd=False，只有 Ascend950 条件会置True；910C路径按完整 S1 group 分配，不调用按S2 block/FD切分，FD输出数初值0。由此不能给删除当前 LD merge payload 计收益。kernel 支持的另一路仍已读：partial topK workspace → SyncAll → ProcessLD 四路 merge/尾merge → offset → final CopyOut。**但没有LD数据不等于没有 epilogue**：ProcessDecode 无条件先在AIV上 Reset pipe、初始化LD buffers、ICachePreLoad和SyncAll，最后才检查LD enable。删除这个空 epilogue可作为联合候选，未测其独立暴露/净收益，也未通过原语逐位检查。

**PA实际消费者**：KeyNd2NzForPA按 `block_table[batch,page]×keyStride0 + page_offset×128` 连续复制到L1的原逻辑key顺序；scale按另一条64位stride寻址，不能用key stride替代。两者支持page0 padding和紧凑回退。新候选压紧、bank布局或跨层共享必须连同表、scale、原顺序和缓存代次一起交付。

**候选与数值**：mode1对8-key组amax、位型覆盖pad为−inf、最新块pin为+inf，再按原merge次序交付2048个block IDs。mode2在完整QK/head-score之后生成membership并保留候选外有限NEG_HUGE填充位置。普通token accumulator仍保留2048对而只输出512；candidate accumulator则确需独立2048。原Sort32/多路MrgSort的src1/src2次序、tie、NaN与underfull行不能由“集合一样”替代；candidate回拷用INT32 Adds(0)，不能改成浮点加零而规范化索引的NaN位型。SMLA仍需要chronological bank，既有group共享不能再收费或变成每层重算。

## 结果

| 范围 | 归类与判断 |
|---|---|
| 删除当前910C的LD归并数据 | **已停止**：所选metadata producer不生成该数据，直接信用0。 |
| 去掉空LD epilogue | **待确认**：确有Reset/init/prefetch/SyncAll调用；没有独立延迟证据。原flags、kernel-end可见性和逐位仍需验证，先留在联合范围，不能记0或抹掉整个QLI时间。 |
| top512前缀accumulator＋共享membership/QLI | **待确认**：保留全部tile打分/full sort和candidate2048；以原intrinsic接口核对前512选择、tie/NaN/填充。75%记录数减少不自动等于75%阶段时间减少。 |
| 原native workspace容量与更大bank | **待确认**：源码声明的容量显著大于显式区间；实际库实现、LibApi前缀、mixed block数、隐藏scratch、allocator共享/高水位及逐位未验。不能把差额直接当常驻可释放显存或乘八。 |
| shared membership / sparse score / SMLA megakernel | **待确认且仍值得深入**：当前原half链、membership算后位置、两个query空间、PA key/scale、末读者和共享bank代次构成明确约束。旧局部机制证据可用，旧组合失败不能改名为新通过。 |

**容量界**：host的显式user增量为每host AIC 4,326,400B，另外加平台 LibApi workspace L。kernel显式区域每个logical block：score双buffer65,536B、G32权重双buffer8192B、保留LD区域131,072B，总204,800B。仅作条件例子，当host AIC数=kernel GetBlockNum=24时，分别为99.0234375 MiB与4.6875 MiB，差94.3359375 MiB；G64区域4.875 MiB。若不访问LD，G32数据区为1.6875 MiB。**这些是源公式和区间，不是当前已观测的GetWorkspaceSize或allocator residency**；保留L和所有同步/隐藏空间，不据此立即改分配或认领性能。

**有成本净价**：重新用368个实际eager QLI窗口和原DP到达关系，仅给QLI accumulator信用，保留量化/chronological sort。若accumulator占原kernel时间f、曝光e，假设该阶段随记录数减少75%，每index source新付h：

| f / e / h | 条件P90比例 |
|---|---:|
| .25 / 1 / 0us | 1.017481% |
| .25 / 1 / 5us | 1.006756% |
| .25 / 1 / 20us | .974581% |
| .50 / .50 / 5us | 1.006756% |

要过同1%筛查，h20us时需f≥25.6246%（e1）或≥51.2491%（e.5）。这正是会改变去留的缺口；目前没有把PMU平均ratio当阶段耗时，既有PMU是不同v164 cohort、流水可重叠，不能给f赋值。表是旧cohort固定到达的条件敏感性，非预计/因果/实测收益，不向decode或six ordinary外推、不相加其它方案。

已有 native byte-mask 对拍保留原Sub/Mul/Add，局部四消费者＋转换旧1986.35us→722.58us，能支持membership形成方式有价值；但同batched-slot基线的重复C8未过预设P90/整批稳健门槛，不能认领服务收益。source/helper相遇后的旧token-merge记录已核SHA，只复用其约束；f和新成本仍未测。本轮不为不改变大QLI保留决定的参数启动模型/profile，原语准入仍空。

下一处从真实 `indexer.update_keys` 的源owned latent 继续：wk → k_norm → partial RoPE → dynamic quant → key/scale两次scatter，再追同latent的long-KV epilog与最后读者。A5 folded cache在910C守卫排除，不能因名字出现就展开或计收益。注意query量化先舍入half scale再量化；当前key是dynamic_quant产code/FP32 scale后再把scale转half，不能默认两者是同一可共享原语。大QLI剩余逐位/资源/净价缺口已具体留档，继续实际其它未扫交付。

证据：`indexer-{native-catalog,native-extra-dependencies,package-identity,native-contract-frontier,native-contract-source-review,native-contract-exploration-notes}.json`。只读源码/包/现有数据，无Torch/模型/NPU进口，模型/推理请求/新profile/远端写入/服务变化0，未操作其它controller。
