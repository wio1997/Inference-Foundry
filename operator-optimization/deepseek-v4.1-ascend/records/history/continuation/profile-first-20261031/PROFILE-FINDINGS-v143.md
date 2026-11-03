本轮补齐 F02–F05/F06/F07/F08/F33：HC 输入归一化与 Q/KV、量化、RoPE 和 C1 投影的条件组合。整体目标仍未完成。

960组正Query eager前沿关联10824个原生TASK，包含完整的1920个Q_a输出归一化/量化角色，其中384个与既有Indexer源层子集逐项复核。它们均通过原始API、调用栈、CID、TASK和Compute记录绑定，没有时间邻近补配或新增原生类别关闭。C1的24组、C2的72组及192组IndexWeight仍是归一化输入的独立读者。

当前链是HC加权合并Y_BF16→input RMS5120→Z_BF16→SP all-gather→Q_a1280与SWA KV512。之后还有Q_a norm1280、native DynamicQuant的INT8/FLOAT、Q_b BF16和Query tail64 RoPE。各归约和舍入点不能因融合而自动取消；Q_a1280、SWA512、C1 latent512也不能改成一次1792/2304维norm。无残差RMS wrapper的可选bias及其BF16 Add、rstd与其它观察者仍保留适用性条件。

HC原始4H数据的平方均值不能代替合并后Y的平方均值。反例中原始4H范数相同，但通道相加或相减使Y范数不同。只有首层严格相等的四通道repeat、onehot premix、有限数值等限定域中，二者实数均值才相同；相同norm_eps也不证明native归约顺序、rstd导出或新handoff已成立。

在无bias的精确实数域，令r=(mean(Y²)+ε1)^−1/2、v=B·diag(γ1)·Y，则下一RMS(r·v)可写成γ2·v/sqrt(mean(v²)+ε2/r²)。直接去掉r却沿用ε2会改变函数；保留自适应ε2/r²仍需要输入统计量。当前BF16 Z舍入还独立阻断这个实数重写：一个合法构造中，从舍入Z做矩阵乘得到−0.07421875，直接使用未舍入值则为0。它不是实际冻结权重的结果，但足以否定无条件代数等价。

可继续研究三种不同版本：owner上融合HC与input RMS后发送Z；发送原始Y后在recipient上融合RMS与Q/KV投影；owner上归一化并投影后发送较窄carrier。前两种Y/Z carrier的形状和BF16载荷相同，不会仅因改名而减少通信量。它们选择了不同的归一化位置，不能把收益直接相加。

以M=800、Q=800、N=6400为例，原owner每卡归一化800行；rawY recipient union方案需归一化6400行，分开给QA/KV做norm则为7200行。若为统一面板而把QA从Q行扩到N行，还会额外计算N−Q行；需要新的混合行映射，不能默认免费。

在声明的逻辑接口图中，owner HC/RMS融合可消去的私有Y读写参考为4MH B；rawY recipient union norm/panel相对同一基线的参考为4MH+2QH B，两者共享其中4MH，不能相加。QA和KV在真正共享的Q行上，一次共同输入读取最多对应这份图中的2QH B。这里的B是逻辑值读写，须先关闭其它读者；不是HBM流量、额外分配量或节省时间。C1、C2 FLOAT cast及IndexWeight若仍在图外，就需要保留Z或给出兼容的新输出。

当前多流调用让Q_a完成后KV与Q norm/quant重叠，part3又同时约束Q_b和KV norm。KV norm的数据条件只需要KV，但更早释放它会与QR工作竞争Vector，必须重新计价。3072个普通QA/SWA两投影参考DAG中，给定任意假设服务后recipient组合528组较快、2544组较慢；这不是实测胜率或物理上限。C1/C2/Indexer读者与更细行流水仍是额外条件。

F02到Query RoPE是包含多个非线性、量化和舍入边界的高阶组合，不是input RMS直接连接RoPE。当前没有post-Q_b RMS512可删，完整Query512仍需要；HC的原始X、当前系数、Post残差及下一FFN的pre4也不会因归一化或carrier提前完成而退休。

本轮条件关系162→167，待核对433→428，联合组仍26。35主候选、7历史框架记录、358语义行/26模块、4472次SP关联和854节点/1346边/48提示的部分图保留。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图及3136项归属歧义仍待补齐；整模块关闭0，全模块物理MFU与最低时间尚未建立。没有新增服务器或NPU运行。
