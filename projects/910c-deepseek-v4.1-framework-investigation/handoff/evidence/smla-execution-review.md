# SMLA内部交付与候选判断

## 介绍

从`dsa_v41.py:656 npu_sparse_flash_mla`沿实际绑定追至ACLNN、自带算子包、arch22 kernel及Metadata AICPU。按最终attention输出反推归一化/PV、概率、QK和选中KV的交付，形成范围后再匹配历史。

**源版本已区分**：[身份记录](smla-source-identity.json)。16个加载worker的初始`ASCEND_CUSTOM_OPP_PATH`首先指向自带`custom_transformer`，其库导出主接口及GetWorkspaceSize；adapter优先查custom，再查default vendor、transformer/common库。自带BF16对象SHA `d4aa987b…16f609`，元数据SHA与对象一致，仅6个key：66、32834、1090、33858、17474、50242。通用CANN同名BF16包有336个key、对象SHA不同。6份主要自带源码与历史冻结pin全部匹配。这是当前文件/映射/解析规则的证据，不是新增设备二进制load或每次tiling buffer观测。

## 到了哪里

| 真实范围/条件 | 原交付和读者 | 变换边界 |
|---|---|---|
| 固定BF16、TND Q、PA_BBND KV、G64/D512 | 层0/1为C0 SWA；2–19为C2/ratio2，20–39为C1/ratio1，TopK容量512 | 不按通用CANN/A5、FP16/BSND/HCA/FLASH_DECODE1估本范围 |
| 无本地query | Python不发SMLA，但cache producer可能仍运行；Metadata保留同代ready | 不跳整个Attention/远端cache服务；1840调用之外720窗口单列 |
| C0普通SWA | Cube按分页直接读ori KV，QK与PV分别消费；M块最多4个query共享窗口，Vector执行softmax/归一化 | 已有直接分页和query共享，不作为新增压缩KV收益 |
| C1/C2 CSA | G64时一query一M块；ori窗口走Cube分页读，cmp由AIV gather进`kvMergeGm_`，QK/PV各读一次 | 全头共享KV已存在；新范围是cmp gather/私有panel/后续读者，不能把原QK-softmax-PV同launch登记为新融合 |
| CSA有效长度/空KV | 先按causal可见范围、cache有效长度和容量取界，再二分首个负TopK；空行写原零输出并排空preload | 512是容量，不是已测有效K；不能用形状推出所有读者都会读512条 |
| DSpark的ori_sparse | 仅SWA、ori_mask_mode0，需ori_topk_length；draft配置额外3层ratio0。V0 gather、mask及buffer复用屏障单列 | 不外推C8 causal成本；invalid位置占位/刷-inf、尾数及图内索引更新必须保持 |
| QK→softmax→PV | 原QK FP32写GM；softmax保sink、max/sum和原tile顺序，概率CAST_ROUND到BF16，经GM交Cube；PV FP32写GM | FP32 score和BF16概率是两段，不因数学FlashAttention公式直接删掉原舍入；PV左概率已跨4个输出列块复用 |
| PV→最终attention | 首块非末tile在Vector仅复制FP32 partial，后续按原exp rescale+Add，末tile除sum并BF16 RINT | 可以研究Cube首次直接写原partial bank；保留后续读、原FP32存储、flag和ring退役，不能说全部PV免费 |
| 输出→inverse RoPE/OA | attention先有BF16输出，再原inverse RoPE、CP pack/OA等读者 | 与既存大输出producer相接时保留这次舍入；额外OA权重与shared panel空间共同计入峰值 |
| 6key中的其它条件 | HEAD_RATIO_ONE不属本G64负载；BATCH_CONSISTENCY0/1均编译。Metadata level3会启用FD，arch22入口不向SMLAType传该标志，FLASH_DECODE固定0 | level3/FD分支保留明确资格缺口；未观测本负载逐调用key，不领取新增FD/跨核归约信用 |

**地址顺序复核**：[C++原函数CPU轨迹](data/smla-pair-address-cpu-a2-result.json)编译未修改的`GetKeyGmOffset/CopyInSingleKv/CopyInKv`，仅mock传输。快pair按物理地址从小到大搬，C1/128行页的大步长可超过INT32_MAX，此时两次single回退保原逻辑顺序。C2/64行页、odd/invalid/causal尾与C1两分支共8个case。新的“逻辑排序直读”或“始终物理排序”都不能充当通用原序规则；该观察不是已测attention位差。实际ND2NZ、全kernel逐位及性能仍缺。

**原计时匹配**：[逐调用证据](smla-execution-windows-archive.json)、[摘要](smla-execution-parent-frontier.json)。2560窗口中1840次native，ratio0/2/1分别92/828/920，720无query；30组形状/dtype/DP/carrier的次数及min/median/max与独立`PROFILE-UNIT-COSTS`完全一致。KV块形状分别为C2的64与C1的128；不把两者的地址guard混用。

## 结果

### 待确认：直接选中地址读取 + 首块partial交付

完整实现方向：保持原QK/PV MMA分块和累加、softmax顺序、BF16概率/输出；让Cube按**原实际pair及fallback顺序**读选中分页KV，消去AIV私有panel写回及一次逻辑回读；独立消去首块partial的中间identity读写。改变的是交付，不取消必要计算。

以下是同C8诊断cohort、原请求/EP到达调度、50%传输暴露和归档双向charged-byte参考速率2405.30GB/s下的情景，均非预计净收益。新稀疏QK/PV速率按参考的相同比例设定，未测新Cube带宽。

| 有效K情景 | 直读单独，等参考速率 | 直读+首partial，等参考速率 | 联合方案，新稀疏速率75% |
|---:|---:|---:|---:|
| 128 | 0.406% | 0.812% | 0.677% |
| 256 | 0.812% | 1.219% | 0.948% |
| 512 | 1.625% | 2.031% | 1.490% |

联合K256/512在这个模型中要达最有利请求1%，两次新稀疏读需分别至少参考速率约78.79%/61.18%，还未扣额外地址/同步/重排成本。**K和稀疏读取速率确会改变去留**；已有shape/profile不含选中ID值，先留`CountValidCmpSparseLen`实际分布、真实页序搬运速率及单原语逐位为具名门槛，继续剩余路径。不把旧17.78ms联合估值当直读单项收益。

### 待确认：跨层选中panel或地址计划复用

实际8代(KV源/index源)分别为(2/2)、(8/8)、(14/14)各6读者，(20/20)、(20/24)、(20/28)、(20/32)、(20/36)各4读者；共38层。除了第一读者，后续30层可研究重用原序panel，继续付QK/PV两次读取。按同一参考模型，panel复用+首partial的K256/512条件信用为1.048%/1.689%，假设第一读者重定向免费。额外全panel复制、长存储、失效和放置成本都要扣。

K512时最大私有bank约409MiB/rank；同样条目的8B地址计划仅约3.20MiB。可以让QLI准备输出或第一读者发布地址/有效长度，再由相同代的SMLA消费者共用；TopK的逻辑ID、层20给后续QLI的candidate blocks、物理地址计划是不同读者合同。计划key必须包括模型/层源、cache及page-table代、ratio、位置/有效长度和行域，不能仅以TopK指针复用。是否用panel还是地址直读属于**同一KV字节的替代方案**，不相加。

跨层L2命中仍未知。DFC原GMM在该expert的`currentM<=128`时已经对权重设置L2 bypass hint；不能仅以跨层专家权重足迹大于L2就断言KV已被驱逐。原有hint不等于已证明KV驻留，C8大expert与shared、其它读写仍会争用。没有新增L2/稀疏带宽数据。

### 已停止的具体小范围 / 保留的大范围

- 原QK-softmax-PV已同launch、原PV概率跨列复用、已有shared KV/index源，不重复计新增融合。
- 仅换用已经查过的UB→L1/TSCM接口、实际仍通过GM staging的配方，没有消去目标GM交付，不准入该“已省P/KV写读”的收益。其它新接口/资源重分配需单独证据。
- 首块partial单独在上述估价模型为0.406%；不为同一小预算拆更多孤立变体，作为联合parent组件保留。它不是设备硬上界，更不据此剪整个SMLA。
- 全SMLA原native免费在固定原到达模型最高6.857%，说明大范围仍有调查量级；它也不是任意重排下的全局上界。
- 数学合并PV与OA、去掉BF16概率/输出、改softmax分段、共享不同层QK值等，尚未满足逐原语合同；不能用实数等价替代精度证据。保留原舍入的更大producer与前面Attention输出段相接。

本轮SMLA新增NPU运行、模型请求均为0；CPU地址轨迹和profile复算不是原语数值/性能验收。loop-009继续保留上述具体缺口，下一条未独立闭合入口转Engram：`model.py:1565–1591`两层lookup等待/分片→Wkv→gate→四路残差，以及更早hash/UVA producer及迭代退役。

公开逐调用归档是zlib+base64 JSON包装，含压缩/原始SHA。可用`python3 tools/decode_smla_windows.py handoff/evidence/smla-execution-windows-archive.json <新的输出文件>`恢复2560行；完整复算还需manifest注明的冻结源码及既有profile/arrival归档。CPU模拟记录只证明地址轨迹。
