# 当前优先级与执行方案

更新：2026-11-06（Asia/Shanghai）。固定目标：DeepSeek-V4.1-Flash / DSpark / Engram，171、CANN9.2、TP8/DP2；输入32768、输出1024、C8、前缀0.9。

本次依据公开提交 `8539c1664317317cef3a8c714e5853d56f0e1ea4` 及其来源快照 `8d4adaa34996ed7ac1fe4100955ff7683df8dc56`，校正旧排序与最新结果之间的冲突。只更新既有证据的取舍和页面，没有新profile、NPU运行、推理请求或服务操作。[本次复核](progress/evidence/priority-review-20261106.json) · [完整冻结样本与边界](progress/evidence/runtime-results-through-20261106.json) · [结构化当前分层](progress/priority-plan.json)

## 当前行动层级

|层级|具体对象|价值依据与动作|
|---|---|---|
|1：已验证组合基线|F31：CP长度去重＋Slot cache-only writer＋CPU公共计划共享|CP＋Slot原限定结果保留；CPU计划在同轮CP＋Slot上新增9.560%–10.364%。后续比较用三项组合，保留直接回退；新正常入口实际采用资格单列。|
|2：具体算法候选|F01/HC：行平方和替代无用Gram交叉项|可删明确冗余算术；现有实现捕获回放失败、完成候选比较0例。只在有具体流水修正时重开，先验全部输出，再测整个HcPre净值。|
|3：联合机制待定价|F24/F25：Engram lookup、owner传输、常量联合机制；F13：selected-KV共享或核内状态续算|先核实际删除、必要替代、代际/读者与就绪。它们尚没有已建立的请求净收益，不能因父算子热或组合更大而升级。两者不强排收益。|
|4：组合后条件清理|F29重复位置准备；H07纯文本FULL图外死embedding|仅计三项组合之后的新增可删工作；保必要投影、cache、ID/图地址及多模态分支。没有足够增量则后置。|
|其他负载/目标后备|F03 Post＋Y；F16 OA前移；F26 owner collapse；F32不同长行/容量机制|保留原数值、owner及读者合同。当前owner传输已测版本停止；其他目标需新的完整净价值依据。|
|停止不变版本|F12当前成员表示；F32当前owner传输；H02/F31 CPU联合组装；F16普通MM替换|前三者未建立完整推理增量，最后一项未过冻结数值门槛。停止具体变体，不否定全族，不为原样方案补样或扫参数。|

层级区分已验证基线、可定位的算法机制和待定价研究，不是所有未实现方案的已测收益榜。尚未证明HcPre收益一定大于Engram/注意力，也未证明新联合机制超过已验证组合。实现复杂度不降低方案价值；实际可删工作、必要成本、频率和完成端点决定价值。

## CPU计划为什么必须从后备前移

CPU计划不只是几次小整数运算。原来各组重复构造局部token交集、qsl和长度，进行clamp、锁页准备和最大长度归约；现在同一batch及输入/分区命中时复用私有CPU计划。各builder的设备qsl地址、必要H2D、原设备长度与group bank仍保留；batch owner、dtype、计数和输入值守卫保留。

|同轮对照|三次计时，秒|中位，秒|
|---|---|---:|
|CP＋Slot前参照|14.353755、14.260882、14.369703|14.353755|
|增加CPU计划|13.407464、12.855554、12.866138|12.866138|
|CP＋Slot后参照|14.416317、14.226161、14.055299|14.226161|

中位每批减少 **1.360–1.488秒，9.560%–10.364%**；全部候选样本低于两组全部参照样本。发布记录有12批各8路文本及16worker限定元数据检查，诊断在计时前移除。每臂仅3次、固定顺序、诊断历史有差异、speculative内部计数可变等限制保留；输出文本与选定元数据通过不代表全模型逐位证明。

旧CPU替身约0.4ms／26组不能代表真实运行上下文，也不足以把整个主机路径压低优先级。新的同轮完整C8结果已取代后备定位。CP＋Slot旧组合10.057%–11.198%与本次CPU计划增量来自不同对照，**不相加或相乘成已测累计提升**。

实验入口的限定结果不等于新正常入口已永久采用。真正需要采用时才检查16worker实际来源、dispatch、specialization、原bank/view、读者和启动前fallback；不因本次文档更新启动服务，也不重复已成立的计时。[实现快照](progress/evidence/implementation-snapshots-20261106.json)

## 后续如何先排除无价值想法

- HcPre：M3/K20480例子中，范数Gram块模型5,242,880 MAC可改为61,440次平方和61,437次加，但必要混合投影7,864,320 MAC、权重、workspace和输出仍保留。算术比率不能套到整个kernel；新归约/同步/流水成本全部计入。旧L0C失败实现不原样重跑。
- Engram：当前三项组合保存profile的112个rank-local周期，两次WKV投影并集中位0.553ms，约周期1%。只缓存或替换此投影缺少足够新增净值依据；联合lookup、传输或容量机制另判，不能拿投影时间和历史gather等待拼收益。
- 注意力：40次统一SMLA并集中位1.799ms，约周期3.25%，包含必要QK/softmax/PV。selected-KV共享必须扣pack写、消费者再读、驻留和代际成本；各层Q不同，不能复用整层注意力结果。旧BF16分支输出合并反例保持有效。
- 小cast、hash压缩、单次接口拼接不独立立项；只有在联合机制中能增加明确净值才加入。逻辑字节不等于实际HBM，少一个图节点不等于少一次主机提交。

上述局部周期比例是已有工作成本范围，**不是全C8收益上界或预测**。只有未知项的不同答案可能改变采用/停止决定时，才补最小证据；若两种答案都不值得做，就停止深入。

## 已完成比较的停止决定

|变体|已有证据|当前决定|
|---|---|---|
|F12当前成员表示|前／候选／后15.859／16.147／16.230s，每阶段一次计时，候选处于两参照之间|端到端未定；停止不变版追测。旧完整局部图净省1.211/0.821ms不能继续充当请求收益。|
|F32当前owner传输|三臂中位13.025／12.991／12.935s；+0.260%／−0.432%|没有已建立增量，停止该版；不同长行/容量条件保留。|
|CPU联合组装|三臂中位13.155／12.974／12.937s；+1.370%／−0.291%|未过双参照门槛，停止；不把CPU计划成功外推为所有组装方案有效。|
|OA普通MM替换|7个原生用例、63次比较中15次有限值位差异；未计时|当前逐位合同下关闭该实现，不宣称已测变慢或所有OA前移不可行。|

## 当前登记与历史来源

42个族保持稳定ID。本轮结构化登记15个具体条目，包含3项已验证改动、1项算法候选、2项联合定价机制、2项前端后备、4项其他负载后备和3项停止条目。F32当前停止变体与其未来负载后备在同条目中分开记录，故停止的已测变体共4个。条目数不代表独立收益或待开发总数，也不与51个工作单元、95条关系相加。

原11项、涉及9族的排序保留为历史来源：[固定旧页面](https://github.com/wio1997/Inference-Foundry/blob/8539c1664317317cef3a8c714e5853d56f0e1ea4/operator-optimization/deepseek-v4.1-ascend/PRIORITY-PLAN.md)，结构化副本见 `historical_items` / `historical_counts`。其中CPU计划“后备”、F12“第二顺位”等旧标签不再驱动当前开发。旧166耗时、注入观察器的v168比例、新171分支和本次固定对照不混用。

## 42族当前取舍

以下是当前具体机制的行动，不是全族最优、全分支语义闭包或全模型理论上限证明。未测频率、完整物理流量、其他合法分支和正常入口采用范围继续保留未知。

|族|当前分组|具体决定|
|---|---|---|
|[F01](progress/F01.md)|ALGORITHM_CANDIDATE|HcPre行平方和保留具体算法去冗余价值；现有实现失败。只对具体流水修正重开，不重跑旧重复collapse。|
|[F02](progress/F02.md)|DEFERRED_CURRENT_LOAD|目标路径已先SP；已有ReduceScatter/NormCast能力不重复计收益，其他条件分支后置。|
|[F03](progress/F03.md)|LONG_OR_CAPACITY|Post+Y保留长行重读消除价值；U仍有读者，不能删U。|
|[F04](progress/F04.md)|DEFERRED_CURRENT_LOAD|已有RmsNormCast；需要新的实际暴露或接口机制才重开。|
|[F05](progress/F05.md)|DEFERRED_CURRENT_LOAD|与F09共享同一qr量化包去重预算；新NormQuant窄版不独立开发。|
|[F06](progress/F06.md)|DEFERRED_CURRENT_LOAD|朴素拼接会多算QA；合法联合仍须保QA重叠，去重算术归F32。|
|[F07](progress/F07.md)|DEFERRED_CURRENT_LOAD|RoPE epilogue可行，但需扣原有/可提前的重叠，当前不独立开发。|
|[F08](progress/F08.md)|DEFERRED_CURRENT_LOAD|窄版省55296B和2启动/调用；干净窗口汇合模型0.463ms/target非请求收益。|
|[F09](progress/F09.md)|DEFERRED_CURRENT_LOAD|query旋转/量化边界保留，与F05共用输入量化去重，不重复计。|
|[F10](progress/F10.md)|DEFERRED_CURRENT_LOAD|M3仅96输出；最多约1920B接口和3后处理任务，不单独开发。|
|[F11](progress/F11.md)|DEFERRED_CURRENT_LOAD|key量化规则不能替换；paired-store小变体不独立开发，完整producer条件保留。|
|[F12](progress/F12.md)|STOP_UNCHANGED|完整C8收益未定；停止当前候选成员表示追测，旧局部净值不再授予下一轮独立优先级。|
|[F13](progress/F13.md)|JOINT_MECHANISM_PRICING|selected-KV共享/核内状态续算先定完整净值；全部SMLA约3.25%是旧成本，不是可删预算。|
|[F14](progress/F14.md)|DEFERRED_CURRENT_LOAD|逆旋转+pack原型退出；OB直写等容量变体留档，不在当前11项内。|
|[F15](progress/F15.md)|DEFERRED_CURRENT_LOAD|当前全满T24无直接裁padding收益；其他形状有条件保留。|
|[F16](progress/F16.md)|LONG_OR_CAPACITY|OA前移换窄通信保留长行条件；增加权重供给与常驻容量。 普通MM替换因数值失败停止，不能外推整个族。|
|[F17](progress/F17.md)|DEFERRED_CURRENT_LOAD|两路FP32投影的必要MAC/权重不减；朴素输出stride不兼容。|
|[F18](progress/F18.md)|DEFERRED_CURRENT_LOAD|池化+RMS可融合且需BF16边界；历史维护和其他已融合分支保留。|
|[F19](progress/F19.md)|DEFERRED_CURRENT_LOAD|latent分叉需保旋转前WK输入及ring行，与F11/F18/F34去重。|
|[F20](progress/F20.md)|DEFERRED_CURRENT_LOAD|TopK已有融合不等于算法封闭；固定ID表捷径当前不命中，需具体大净值机制。|
|[F21](progress/F21.md)|DEFERRED_CURRENT_LOAD|g12实际C8未建立加速，停止该变体；空专家权重跳过为已有能力。|
|[F22](progress/F22.md)|DEFERRED_CURRENT_LOAD|必要GEMM/权重保留；FusedMC2会关闭现有overlap，资源排程净值未知。|
|[F23](progress/F23.md)|DEFERRED_CURRENT_LOAD|局部22.342→4.717us但C8未建立加速；停止已测M3入口。|
|[F24](progress/F24.md)|JOINT_MECHANISM_PRICING|保留联合lookup/owner传输/常量机制定价；WKV投影单项约周期1%，不独立立项。|
|[F25](progress/F25.md)|JOINT_MECHANISM_PRICING|与F24按同一Engram联合预算去重；保留hidden依赖、舍入和mask/NaN，不领取整个投影或gate时间。|
|[F26](progress/F26.md)|LONG_OR_CAPACITY|无raw读者时owner collapse和单H gather，保留长行/容量条件。|
|[F27](progress/F27.md)|DEFERRED_CURRENT_LOAD|Post+aux均值保留数值和checkpoint代际；当前不独立开发。|
|[F28](progress/F28.md)|DEFERRED_CURRENT_LOAD|draft可先切行再Tile；CPU用例不等于NPU或推理收益。|
|[F29](progress/F29.md)|FRONTEND_RESERVE|只定价CP+Slot+CPUplan之后重复位置准备的新增工作；旧KV窗口裁行不成立。|
|[F30](progress/F30.md)|DEFERRED_CURRENT_LOAD|全命中特化有限局部省19.841us；混合预分类重组更慢，停止后者。|
|[F31](progress/F31.md)|QUALIFIED_BASELINE|CP+Slot+CPUplan为限定已验证组合；CPU计划新增9.560%–10.364%。联合组装当前变体停止，正常入口采用资格单列。|
|[F32](progress/F32.md)|STOP_CURRENT_WITH_CONDITIONAL_RESERVE|当前owner传输相对前后参照+0.260%/-0.432%，停止不变版；长行/容量或新机制条件留档。|
|[F33](progress/F33.md)|DEFERRED_CURRENT_LOAD|两路512维投影权重不同，不能视为重复结果；与F32预算去重。|
|[F34](progress/F34.md)|DEFERRED_CURRENT_LOAD|当前共享页仅缩逻辑ring不回收物理容量；其他池布局条件留档。|
|[F35](progress/F35.md)|DEFERRED_CURRENT_LOAD|BR01/02删死备份与大greedy协议分开；NumPy用例不是NPU正确性。|
|[H01](progress/H01.md)|DEFERRED_CURRENT_LOAD|重复等待需保错误/取消/退休语义；当前无大请求收益依据。|
|[H02](progress/H02.md)|STOP_UNCHANGED|当前CPU联合组装无双参照增量，停止不变版；其他小copy/模板仅保留有新净值的条件。|
|[H03](progress/H03.md)|DEFERRED_CURRENT_LOAD|prefix hash复用CPU成本未知；0.9前缀率不是哈希耗时占比。|
|[H04](progress/H04.md)|DEFERRED_CURRENT_LOAD|tokenization同负载CPU成本/精确复用命中未知，保留前端条件。|
|[H05](progress/H05.md)|DEFERRED_CURRENT_LOAD|分离sample-ready和资源退休可能有效；直接早完成future不成立。|
|[H06](progress/H06.md)|DEFERRED_CURRENT_LOAD|固定6×SP8最小正数公共桶24；零发送rank仍可能接收，不能删通信。|
|[H07](progress/H07.md)|FRONTEND_RESERVE|纯文本FULL图外死embedding只计三项组合后增量；多模态等原分支保留。|
