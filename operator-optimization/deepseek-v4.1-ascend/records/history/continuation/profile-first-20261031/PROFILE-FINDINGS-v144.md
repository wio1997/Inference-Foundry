本轮补齐 F02–F09/F10/F11/F17/F18/F19：输入归一化与 Indexer、C2 双投影、池化、latent 分叉及未来 ring 状态的条件组合。整体目标仍未完成。

192组已有source-key前沿合并后，复核5064个唯一TASK：72组C2、24组C1、96组仅Index query/Weight。这是既有原生证据的组合核对，没有新增原生归属或类别关闭。当前72组都在Query/Index RoPE之后才开始C2输入cast，并在ring update之后做pool norm；66组Index量化与C2 KV TASK区间相交。这些是当前调度观察，不是不可消除的依赖或可恢复时延。

输入不能混为一谈：QA/IndexWeight读取5120维的归一化BF16 Z；IndexQ读取另外归一化过的1280维QR，经native量化后进入INT8矩阵乘；C2先把已经BF16舍入的Z扩展成FLOAT，再用两个FLOAT权重计算KV与score。QA1280+Weight32的BF16面板可以登记与输入RMS组合，但与C2连接时必须保留混合精度和不同的行映射；把FLOAT权重降成BF16不是等价实现。IndexKey的native FLOAT scale→FP16 cache，也不能用Query先FP16 scale再量化的路径替换。

C2按feature对两个原始请求partner的score做softmax，再加权KV。输入统一乘以标量会同时改变score温度和KV；构造例的输出方向比从e变成e²，后面的标量RMS无法恢复原方向。因此不能仅凭末端还有norm，就消掉前端input RMS。若一份值来自旧ring、另一份来自当前段，不同世代的缩放更不能混合。

发现一个值得继续核对的池化细化机会：在所选有限FLOAT参考域，ratio-2可用一次exp(min−max)，分别计算1/(1+e)和e/(1+e)，再恢复原始行的乘加顺序。784个参考Case与原表达式一致。不过，编译器是否已消除exp(0)、实际SFU指令与吞吐、native exp/FTZ/NaN/Inf/tie及BF16输出精度都未证明，不能声称实际提速。

较小概率不能直接写成1−较大概率：640个参考Case发生变化。一个反例中，p_max在FLOAT中已等于1，而p_min仍是约2.061×10⁻⁹；它乘以较大KV后，原BF16输出为0.2060546875，使用补数则变成0。融合或专用pair kernel必须保留这类数值边界，以及原始request pairing、旧ring读取、pool BF16舍入、全部T行初始化/norm、可选bias及key/long缓存语义。

生命周期也分开了。Z可在QA/SWA/Weight/C1/C2 cast等直接读者结束后逻辑退休；C2 FLOAT输入还需保留到两次GEMM读完；投影后的KV/score要保留到pool与任何延后ring update都读完。它们不必全部等待当前attention结束，也不能全部在pool结束时删除。实际存储别名、额外读者、capture和排队写者仍需补证，TASK结束只是一个充分的读取完成界。

若让pool norm、IndexKey或long-KV提前于ring commit执行，必须保留FP32投影tail，或通过已证明的新producer直接写入精确state；未来相同page/generation的旧状态读取仍须等待commit。6144个假设服务的DAG中，4128组当前终点较快、1792组较慢、224组相同；其中855组违反所选后续ring-read期限，包含665组表面上较快的情况。它们不能当成满足固定未来期限的收益，更不是实际跨迭代时间证据。

FLOAT输入接口也要去重：所选基线的cast输出写入4TH B、两个GEMM读取8TH B，共12TH B。两个投影在现有FLOAT plane上合并，可减少其中4TH B输入读取，剩8TH B；若新cast+joint producer能合法消去整个plane，联合参考仍是12TH B，不能再叠加前面的4TH。这里是逻辑接口计数，不是HBM流量、分配量或节省时间。C2的全部有效当前投影行、未来FP32 tail与既有ring容量不会因pool/norm融合自动消失。

条件关系167→173，待核对428→422，联合组仍26。35主候选、7历史框架记录、358语义行/26模块、4472次SP关联和854节点/1346边/48提示的部分图保留。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图、3136项归属歧义仍待补齐；整模块关闭0，全模块物理MFU与最低时间尚未建立。没有新增服务器或NPU运行。
