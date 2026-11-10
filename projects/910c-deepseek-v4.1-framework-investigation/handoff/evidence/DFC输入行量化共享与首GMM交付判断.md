# DFC输入行量化共享与首GMM交付判断

介绍：沿 BF16 FFN 输入、Top‑K 路由、动态量化和跨 EP 交付追到首个 GMM 的真实 A 操作数。先判断完整 DFC 大段，避免把已做共享或整段等待重复算成新收益。每个替换原语逐位一致。

到了哪里：9 个当前磁盘源 SHA 匹配，限定 18 处源行段；原 host DoTiling 在本机 CPU 执行 12 个形状，记录所有核/循环的原始行归属；两个代表 rank 各 516 个旧窗口父段，分开 target24 和 draft20，实际 DFC 均为 local M3。不是全链路、全部分支或 native 位验收。

结果：原 FullLoad 和 Gather 1H 都在单核/单循环内先量化输入行，再给其负责的路由项复制相同 INT8 与 scale。但原 host 按展开项切核，行可跨核重复：local M3 分成 18 个核，18 次 Compute，即每行 6 次；M24 为 48 次，而非 144 次；prefill M416 为 436 次，M816/817/818 为 836/837/838 次。只靠函数名说“已量化一次”或统一按 Top‑K 六倍删计算都不成立。当前源码导出的 21000/11000/11010 不是 active binary 的 tiling 采集证据。M0 只是 host 空形状诊断，不声称设备合法执行。

原量化经 FP32 Abs/ReduceMax、maxAbs/127 和 Div，再 FP32→half CAST_TRUNC、half→INT8 CAST_ROUND。新的共享必须保留全部 codes 和 FP32 scale，包含零、非有限与 ties 行为。shared expert 的独立 DynamicQuant 不能因同 BF16 输入/shape 直接合用。原路由排序、active mask、dummy expert、capacity 和 FP16 GMM epilogue、后续 Top‑K 原顺序归约均保留；后者此前证据继续有效。

实际交付是 initRouting 写 sender peer offsetA 的每路由 code+32B scale 记录，经全 AIV join、counts 发布/等待和 cumsum，再由 CopyGMToGMPerToken 从远端读入 UB，写本地连续 workspace gmA 与独立 scale1。每个 expert group 的 SyncAll 和 CrossCore flag 发布给 GMM1；GMM1 已按 group 就绪与搬运重叠，且按原 granularity 通知后续 SwiGLU。没有新方案时不能再认领这部分已有重叠，零工作 group 的 flag 也须完整消费。

三个可改变边界的范围应分开付成本：只合并重复量化仍需 18 个展开记录；保留三个 unique sender 记录会少写 77,280B/源 rank 的逻辑数据，但需要原始行 map 和远端 gather，Top‑K 分到不同 rank 时网络并不省六倍；进一步删 receiver gmA 则要改变 GMM 的 operand loader，现有连续 A copy 不能直接读 map。18 核改成 3 个行 owner 还会减小并行度，调用次数六倍不等于延迟六倍。延后远端读时，sender bank 要保护到所有新 GMM 读者完成，旧 dispatch copy done 不能充当末读。

两个匹配配平短窗的 target DFC 各 480 次，整个 DFC 累计占自身窗口 15.979%/17.537%；norm ready 到 FFNPost 占 21.652%/23.357%。假设真正删去整个 DFC 的 10%，每处新增 5us，条件净省为窗口 1.343%/1.496%；保存全部正负情景。此处未测 input quant/route/dispatch 的内部占比，不能把整个 DFC 时长乘 5/6。draft20 的 36 次分开，不能借 target 曝光。

原 C8 首 token 的 2560 个 DFC 调用及 24-expert populations 已有跨 rank 时钟条件证据，整个 common suffix 全免对应旧 TTFT 模型 20.090%–22.690%。假设其中一段可减半且每层付 5us，该段需占 common suffix 约 9.287%–10.489%，才到旧 1% 排序参考。这个大 parent 仍值得深入；这不是量化 phase 时长、实测请求收益、用户硬门槛或任意改调度的绝对上界。

待确认：剩余 sort/count/index producer 与 mask/dummy/zero-work 合法分支；unique-row map 的稳定专家顺序和实际 destination 重复；新 peer gather/GMM A copy、K/Fixpipe、资源、图事件与 bank 末读；每个替换原语位；匹配旧数据能否限制 phase/exposed wait，只有影响去留时才补定点原生证据。已停止：全形状每行一次/每路由重算的错误信用、把整个 DFC 当量化时间。已确认可实现：本轮新增原语 0。

环境：当前 DFC 六源入口和三份 host header 均用 docker cp stdout tar 内存只读核对，同 owned image/id，观察 running。前一 Engram 取证曾观察 exited0，是两个时点，均不代替 health 或其它采集的恢复回执。本扫描没有 NPU、模型/框架进口、推理请求、新 profile、远端写入和服务动作。

下一条待扫：原 expandedRowIdx/active mask/count publication 到远端 source-row map、receiver 连续 gmA 的完整生产与消费；保留原 flags/归约/容量，比较 unique sender 与直接 gather-to-L1 两个范围的付成本净价。证据见 dfc-input-delivery-{entry,source-review,source-excerpts,frontier,price}.json、dfc-input-host-{dependencies,cpu}.json；本机 driver.cpp 原样保留，CPU 可执行文件不入 Git。
