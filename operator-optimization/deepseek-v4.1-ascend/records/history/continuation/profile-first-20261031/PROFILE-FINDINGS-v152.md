本轮补齐F03与F09/F10/F11/F17/F18/F19/F34的7项条件组合，覆盖Index query/weight/key、C2两投影/池化/latent分叉与跨迭代ring状态。整体目标仍未完成。

192处Index前沿与上一轮Post→Pre前缀按同一HC TASK、API、forward和层号关联5640个原生任务：96处仅Index query/weight、24处C1、72处C2且包含ring更新。24处有Engram正标记；本子集没有辅助mean正标记，这不证明其它读者不存在。实际张量地址、参数代次、请求/slot/状态epoch仍未绑定。

Index weight使用input RMS后的Z_BF16，Index query使用另一个QR1280归一化/量化阶段的值。Index key还经过C1或C2 latent512 norm、IndexWk、key128 norm、RoPE、native量化与scale发布；LongKV需要旧的未旋转latent读取。不能把这些路径合成一个共享H5120 GEMM或删掉全部norm。原BF16/FLOAT/FP16/INT8边界、query/key不同量化策略和v148内部精度门槛保留。

C2需要先把已舍入Z_BF16扩为FLOAT，再用两个FLOAT矩阵得到KV与score。联合矩阵和cast融合可研究，但把FLOAT权重改成BF16会改变合同。1536个逻辑接口组合证明，现有FLOAT平面上联合读取对应的4TH B已经包含在cast+joint所选12TH B接口中，不能相加。投影KV/score及未来tail仍需保留或另行满足。这里不是HBM、分配量或时延收益。

C2 featurewise softmax改变温度，后续pool norm不能一般抵消输入归一化。有限RATIO2参考可以研究one-exp，但仍需分别计算两份概率；用1−pmax可能丢失小概率贡献。pool BF16舍入、全T512 norm、bias、零输出、非有限值及全部读者保持。原始pair-pool与even-tail需求的33024个参考用例再次核对：在保留该直接算子程序的域内，所有有效当前投影行都被pool或未配对尾部需要，不能按completion数量直接裁掉当前投影。它不是所有等价算法的FLOP下界。

Z、FLOAT输入、投影KV/score、HC原始残差具有不同的最后读取点。Z可以在全部直接读者完成后退休；投影KV/score仍要满足pool和未来ring更新；HC原始残差仍服务未来attention Post。更早启动pool norm、Index或LongKV不等于这些buffer全部可复用，也不要求所有对象一律等到整个ring TASK结束。模型分别保留源range staging、状态commit、下一代old-ring读取和捕获/queued-writer条件。

另建立2304个指定全局共享池arena预算用例：仅给新增且互不别名、同时存活的raw/Y/Z、pooled、FP32 tail、FLOAT副本计价，现有HC rawX不能再次算成可释放容量。选定C=540928 B分配粒度、B=qC+r、额外D和另行证明可释放的S时，参考净空闲单位变化为S+floor((B−D)/C)−floor(B/C)，并检查该快照域B≥D。它不是所有分配/调度的可行性条件：释放先于新增分配、重算、复用、不同arena是其它允许的图。片上保留与全局KV池分开计价；原生allocator、实际生存区间、metadata/对齐和真实容量收益尚未证明。

条件关系197→204，待核对398→391，联合组仍26。F03已有26个有范围的伙伴，8个待核对，整个候选仍open。全部模块物理理论上限、最低利用率及实际融合收益仍未建立；原9类/1352次COMPUTE操作数缺口、849个PMU缺口、16个phase未绑定图、3136项归属歧义和G14运行时拷贝/同步范围保持。整模块关闭0。没有新增服务器或设备任务。
