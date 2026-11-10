# DFC原路由行映射与GMM末读判断

介绍：接上输入量化，沿原 mask、排序、原始路由号、counts 和发布信号，走到远端搬运、首 GMM 及最终 Unpermute。分别判断保留消费者的输入共享，以及改变 GMM 读法的大范围。每个替换原语逐位一致。

到了哪里：11 当前源 SHA 匹配，其中 10 份读限定行段，mrgsort.h 仅身份核对；原 Sort/MrgSort 内部不改，不宣称已重新证明其稳定性。读取原 C8 的两个 target 层 0/20，各 227 次完整 EP16 路由，共 7264 rank-layer 调用；CPU 用原始行的非浮点 5152B 记录核对 21792 个接收端映射，比较 1,984,179,456B，0 搬运差异，含 2376 个 dummy 展开记录。三种同专家 tie 顺序只检验地址恒等式，不冒充实际 Sort 输出。另有历史五条 native 路由的 80 份实际 inverse witness，全量保留。

结果：原 `expandedRowIdx` 是 source route→sorted destination 的 inverse R。新 unique map 应为 `U[R[t]]=floor(t/6)`，不能把 sorted position 除 6 当原始输入行。FullLoad 的第一次 Sort、普通 one/multi-core 的 workspace 都有 forward route IDs；也可从原 R 派生 U，但必须付 scatter/read/发布成本。原 R 继续交最终 Unpermute，不能因共享输入而替换 tie 顺序或删除 dummy 项。

mask 非空时会原地把 inactive 的 expertIdx 写成 384，再 SyncAll；实际 host 的 expertNum 为 385，flag2 表示 counts。dummy 仍属于原 inverse/count/输入记录的域，但不产生真实 384 专家的 GMM。原 C8 保存的行仅见每源 rank 3 行全活跃或全 inactive；部分行 mask、其它 38 个 target 层、draft top3/localE8、六普通请求不能从这两层路由人口外推。原 M3 全局接收界 16×3×6=288，五路由诊断 capacity288 与模型较大 arena 的活跃域需分开，不宣称容量已释放。

counts 的零项也以 `0x800000` 哨兵发布；接收端等到非零再解码为真实零。GMM1 会消费包括无计算 group 在内的所有 flags。只删零 expert 的 publication 或 wait 会破坏原协议。原消费者按 expert、source rank、该源原排序顺序读取连续 gmA；unique 行仍需展开到这个顺序，或另改 GMM 的 operand loader。原 gmA 与 SwiGLU activation 共用 ptrA/ptrPermutedToken，少写 gmA 不等于能删除整个分配。

新 direct remote GMM 读者会延长 sender codes/map 的寿命，原 dispatch copy done 不能当末读。保留原全部依赖及全作业结束同步可作为保守候选，但 native 事件/内存可见性与新增读者仍待验。最终 CrossRankSync 在 Unpermute 前，offsetD 和原 inverseR 此后仍会被读；不能把输入 bank、返回 bank 和整个 workspace 一起退役。原 CombineV1/V2、FP16 epilogue、Top-K 归约与 BF16 舍入均保留。

沿代码到这里，才匹配历史 reference 的五条 M3/K6/E24 路由实验。当前 owner header SHA 与历史原版相同；历史诊断实际观察 `[21000,0,18,1,1]`。重新按每轮最慢 16 rank 复算全部 15 轮差值，保留异常和负数；final outputs/counts/after-IDs 的八缓冲区哈希与 control 相同，positive slots 的 5152B 行相同。这不是全部原语、inactive/padding 或新方案的逐位准入。首版 inverse witness 失败的实验继续排除，原件 SHA 保留。

免费预放置输入的各路由中位省时 1.543–6.922us，连源量化/写包也免费时为 2.505–5.640us；只跳源量化的增量中位可为负。均低于当时 14.2497us/调用的独立小范围排序参考。因此停止原样“固定 GMM/flags 的输入-only”方案，不为它补新参数或重跑。它是合成输入/权重、另一个固定接收布局内的 counterfactual 敏感性，不是物理上界、当前请求收益或新的用户门槛。

完整路由人口也表明不能只按六倍通信计价。layer0 每次最大 receiver 的 route rows 中位 28、unique source rows 中位 23；layer20 分别为 84 和 44。这是两种各自取 max 的统计，不能相除冒充配对削减率。完整 slab 几乎总比第一非空 expert 需要更多 token；统一提前搬满 slab 可能增加首 GMM 等待，实际代价未知。

待确认：改变 GMM gather-to-L1、提前 per-group 发布、较大 M、shared 资源调度或跨层共同 bank 的完整成本与曝光；保持原 K/Fixpipe、每原语位、源端代次和远端末读；其它合法分支和人口如上。已停止：上述历史固定消费者小范围，以及错误 U、删除原 dummy inverse 或原 zero-group flags 的具体配方。已确认可实现：本轮新原语 0。

下一条转向具有新函数归因的较大共同 metadata/frame 交付。已有 source/fence/reader 证据复用；先把当前 CP/base/local-token 函数自身区间与实际双 builder 调用绑定，扣掉首 producer 和必须保留的物理组装、设备权威长度与发布，估净价。各 profile 窗口分列，不用新短窗替代原完整请求。

环境：171 hostname 核对成功，同 owned container/image，11 源通过 docker cp stdout tar 内存只读核对，观察 running，不等于 health。只有本机 CPU 与旧数据分析；本扫描 NPU、模型/框架进口、请求、新 profile、远端写入及服务动作均为 0。其它流程新诊断恢复以其原 receipt 为准。
