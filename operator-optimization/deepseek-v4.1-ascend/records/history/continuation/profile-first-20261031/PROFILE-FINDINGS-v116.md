本轮补查MC2返回路径，复用已有4472次DispatchFFNCombine输入几何，没有新增推理或归属提升。所选源码按allocated local M×topK是否大于4096选择CombineV1/V2，输入形状代入得到3512次V2、960次V1；这只是源码兼容条件下的分支预测，未观测到内部flag。M3×k6选择V2不等于接收端每个expert有3行，实际receiver counts与mask值仍未记录。

V2使用GMM2的128×256分块，再把每个M块划成16行微块，由一对AIV子核分担。ceil(m/16)个微块中，子核0取前floor(q/2)个，子核1取剩余，最后不足16行的块在子核1；1..16行的小M块只有子核1工作。这不是所有新融合实现的并行度上限。已检查514个行分配参考和7565个peer区间参考：每个有效接收行恰好覆盖一次，返回段是微块与各source-rank expert计数前缀的交集。它们证明源码坐标和语义store span，不能替代编译后DMA字节范围或实际地址/lifetime证据。

进一步定点读取V2 epilogue发现，两次Cast均写repeat=128，UB max_len为8192，实际GM输入/输出由微块有效行列决定。但Cast<float,half,false>及Cast<BF16,float,false>的false表示使用外部mask，调用的-1不设置mask。只读固定安装CANN9.2的2201前端与实现，确认该bool被传给CastImpl，false跳过SetMask并进入f162f32/f322bf16r；Normal/Counter模式还受外部状态控制。因此不能把repeat128直接当作实际8192元素或固定浪费。[官方mask说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0024.html)

在明确的Normal全64-lane mask域内，repeat128的确对应每方向8192个转换元素，而当前有效微块最多16×256=4096。H5120分20个列块，令R为clipping后实际总接收行，S=sum ceil(r_e/16)，该固定源码每方向转换元素数为163840S，有效FP16→FLOAT或FLOAT→BF16值转换为5120R；比值32S/R位于2..32。若已知R和正行数expert数Z，S的精确无额外expert容量限制范围为max(Z,ceil(R/16))到Z+floor((R-Z)/16)，独立动态规划复核369个有限案例。实际r_e缺失，不能用本地M3代入，也不能把元素数比例视为整个融合算子的提速。

条件优化可以具体收敛：H5120的每个列块为256列，有效行数m下的元素数256m正好是64的倍数；若入口确为Normal全mask，可把两个Cast的repeat从128改为4m，保持false、stride、FP16→FLOAT、每行FP32 scale和BF16 RINT边界。这样不修改mask状态。若改用Counter而没有恢复，后续Muls<false>也会继承Counter count；一个纯CPU参考中，原本三行分别得到[2,3,4]，错误遗留的Counter会得到[2,6,24]。这是对盲改规则的反例，不是已测到原生bug。实际Counter或局部mask域须单独建模。

私有UB布局缩小是依赖选项：先消除所有padding Cast访问，才可考虑把max_len8192改为4096。给定FP16输入、BF16输出、FLOAT中间和每行scale、双缓冲，源码局部布局参考从131328B变为65664B；旧repeat128若保留会越过缩小后的buffer。还要保持scale cache、事件、SwiGLU/return共享UB的旧读者和FPstatus/错误观察。这个数值不是GM workspace、HBM、VRAM释放量，也没有证明占用率改善。原GM读和peer返回存储已按实际微块范围执行，因此仅裁剪UB转换不减少这些GM字节。

其他组合约束继续保留：receiver clipping选择依赖原expert和rank顺序，盲目重排即使保留R也可能保留不同assignments；应先冻结旧keep rows/offset，再讨论物理排程。C/C2、A/activation别名、跨核flag、FP16舍入、per-token scale epoch、peer bank和最终weighted unpermute不能随意删除。此前v56的细分流水/工作量切分、v97容量模型和v98 C65536证据均是已有答案；本轮没有重复计为新融合机会。

为核对Cast ABI，在原166做了一次限12秒的CPU SDK文件名查询，实际0.178秒结束；随后只读复制10份文件（9个唯一路径），acl_rt SHA控制匹配，SSH已关闭。没有Torch/NPU库加载、应用NPU上下文、图捕获、replay或模型执行。现有源码父kernel/header与完整frozen树SHA一致，但当前SDK/源码不证明历史执行加载的二进制或内部mask状态。

M12语义台账新增V2/V1两条显式guard记录，353→355条，26个暂定模块保持open；134条条件关系、26个联合组、461对待核对不变。9类/1352次原生执行、849缺PMU、16未绑定phase图和3136含糊归属不变，0个整模块关闭，物理时间上界/MFU/最低模块仍未确立。原生Pad小探针继续等待单独明确回复，本轮未运行；整体目标继续进行。
