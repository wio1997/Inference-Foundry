本次继续算子范围审计，整体 PROGRESS，未完成全部目标。

原 358 条语义行已逐条保留并筛查：330 条已有明确条件分析，12 类请求语义补成算子边界约束，2 类生命周期范围补充分析，13 个旧 partial 标签逐项写出已有结论与真正缺证，另 1 条旧路径排除记录保留。这个分类不等于实际运行 flags、所有 native 参数或图覆盖均已证明。

此前未审的 91 个生命周期静态点集中于少数函数，本次沿 profile 里实际出现的 runtime dummy 算子，只读一个 hash 未变的 worker 片段，归约为 25 类输入形状、metadata、缓存、position、DP/PP、LoRA、logits、EPLB、draft、同步和清理效应。每个旧点恰好映射一个分析域，保留条件两侧的算子后果及融合/拆分/重组限制，没有研究整个框架。

关键区别包括：torch profiler 正在采集不等于 _dummy_run 的 is_profile=True；metadata 的 base predicate 实际由 force_attention 或 FULL 决定。Dummy 仍可做真实跨卡 receiver 专家工作、发布 KV/slot/状态，不能整体删除。设备和 CPU position127 的后续 reset0 需保留消费与恢复顺序；metadata executor 的 queued writer 与旧 reader 都可能阻止 bank 重用。capture finally 的状态清理也不能丢失。

6.45 秒的既有 profile CPU 查询取得 120 次 BOOL[24] 清零。其中 56 次有精确 CPU creator，分别是 valid mask16、C2 mask16、Engram 尾 mask16、SP mask8；其余64次保持 unknown。相同形状并不表示同一个字段。另有56个 MaskedFill 的 BOOL 是输入 mask，输出是 INT32，已单列为消费者，不能误算 BOOL 清零。初版宽筛记录和修正均保留，没有为了修正筛选重跑模型；当前来源也不反向宣称旧166的80个 prefix 身份已识别。

本次明确补入两个既有 F30/F31 下的条件变体：一是 dummy logits 没有真实需求时省去零索引准备；二是在所有 index 恒为0、算术逐行可分且保留全部通信/输出/精度协议时，复用重复隐藏行的 head 计算。16次 source-connected INT32[24] 初始化对应每次96B、合计1536B逻辑值写入，属于低优先级小平面，不是 HBM 或提速。当前没有正向 dummy head callback witness，不能由这个空白判断未执行，更不计实际收益。R行重复 head 的 direct 源算法工作由RKV变成KV加输出复制/原通信，仅是条件工作比较。

覆盖台账现已实际更新，而非只追加叙述：原358行旧字段完整保留，加入 M21-S19 与 M16-OP-11 后共360行；26模块、35模型候选、7框架候选不变。新的 candidate/module/branch 三份台账均有当前审计、P5指标路由和缺证条目，旧 v154 保留为不可变祖先。

已审一张12节点15边的保守组合图，涉及19个候选族并覆盖算子、层、卡、迭代。QR/QS保留各自不同量化舍入；不能因读同一hidden就合并INT8值。F21已融合dispatch、quant、双GMM、SwiGLU与return，不重复计新融合收益。跨卡重分配、同expert行调度、V2 pad优化是有不同成本/前提的选择，不能相乘或叠加理想提速。跨层HC和Engram预旋转FP32 fork、跨迭代accepted/rollback/prefetch与metadata banks置于同一资源/存活期账本。

组合下界按必要且不重复工作和真实共享池重算：每设备24Cube/48Vector；不同mode时间在无共同issue证明时取max；TP与EP同卡入口不免费，必要信息可能转发/复用，未证不重复字节时不能把cut直接相加。物理fmax/带宽未知处保留符号，1800MHz仅参考；总显存不是可用free，extended FP32/预取/旧bank和queued writer存活必须一起计。不同cohort的receiver work和PMU时间不硬配。

仍有真正未完成范围：旧900个runner静态导航关联尚未按算子效应证明覆盖，不能仅凭文件名路由宣布全审完，也不应变成900项框架研究。下一步用当前profile调用点与现有合同，一次聚合出仍会改变算子值、rows、精度、所有权、状态、RNG或就绪的具体选择；无关控制流保持原样。还需把组合图之外的23个候选族映射到共同约束模板并补未审前提，复用原有关系条款，不做笛卡尔配对。

未知 native operand、历史 epoch、私有 helper 删除和未测 host 因果成本均有具体适用边界。它们不能自动成为所有公开语义替换的前置，也不由空白推出独立。必要的新探针按已获授权直接执行，但只收会改变所选候选结论的字段。

服务health200，StartedAt未变；本次仅缓存profile查询、本地定点源码和组合分析，无NPU探针、模型推理、replay、重启或重载。查询进程终止。全部目标继续ACTIVE。
