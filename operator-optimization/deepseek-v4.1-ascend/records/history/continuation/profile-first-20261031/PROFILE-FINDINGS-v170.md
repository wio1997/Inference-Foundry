# v170：实际 dtype、公开算子边界和完整集合聚合

本轮复用现有 profile，没有新增 NPU 负载、模型重载或完整 Case。171 原服务 health200，StartedAt 未变。上一轮 v169 是实质进展。本轮同样 PROGRESS，完整目标尚未完成。

一次索引查询在4.39秒内补齐120个实际调用点组、148个dtype/shape执行分支、15,520次公开CPU调用、21,088条原生执行。复用已存调用栈与AST索引，没有重读52个源文件。首个查询因CPU时间字段为十进制TEXT、直接相减而退出；已保留错误与旧代码，并确认旧PID结束后改为int解析。成功查询不是模型或采集重试。

四项算子域得到具体补充：

1. Draft argmax：160次精确公开调用均为BF16 Slice→BF16 ArgMax→INT32索引→INT64索引Cast，没有输入FLOAT平面升级。单独替换该公开argmax边界可以保留整个已修正score bank及其其他消费者，不要求所有raw/confidence/logprob消费者死亡。这不同于已分析的整条Markov ID-only融合。给定M4,V129280，在声明的全扫描、冷外部数组、fresh输出模型中，三kernel边界最低逻辑数组访问参考为3,102,784B，组合后的必要外部接口为1,034,272B；不是HBM测量或可达提速。同Python行还有160次独立Slice，没有凭数量相等偷偷与本公开调用相配或加进预算。保持BF16比较、tie/NaN、早先bias/Add的BF16舍入和所有输入输出语义。

2. Metadata整数：960次实际INT32商调用分为5种typed regime，都展开为INT32→INT64、INT64 floorDiv、INT64→INT32。对INT32输入和正INT64除数，数学floor商必在INT32范围，解析边界和405个整数边界样例已核验。整体公开算子内部的全局Cast平面不是语义必需，计算内部仍可保持INT64；未宣称存在更快INT32原生指令。负数floor不能改成截断，INT32_MIN的abs或q*d溢出不能忽略，INT64 slot/position不能仅凭maxlen缩窄。声明数组模型40N→必要接口8N，不是物理带宽。联合q/r仍需输入值、除数和generation相同的既有条件；shape相同不证明这些条件。

3. Markov embedding：160次实际调用只选择4个ID，输出4×256 BF16；权重形状129280×256不表示必须读取全表。消除公开调用内部ID Slice的逻辑接口预算仅64B/次，不能因“两kernel”而列为高收益。冷表必要权重读量取决于不同ID数U，模型为8B+2UR+2BR；实际ID重复和缓存未知。依赖的后续循环ID和confidence embedding输出保留。

4. Engram转换分叉：profile显示同一函数line62和72各32次BF16[3,4,5120]→FLOAT。只定点读取已冻结的27行函数，确认hidden在函数内未修改；可让同一预旋转FP32值供旋转/门控和最终原始残差分支使用。旋转后的original不能直接用于残差；[1,2]交换旋转、gate×value=0即给出反例。该局部复用预算是368,640B/次转换接口，同时可能延长245,760B/次即240KiB的FP32激活存活。必须和其他候选共用内存预算，保留完整norm/dot/mask/FP32乘加/BF16最终舍入，不能与既有idealGate或整函数融合重复记账。q/k参数和rotation权重缓存是旧结论，不计新发现。

32份实际数据库的资源清点也已完成，耗时155.91秒：启动域54,336条、完整诊断负载12,188,947条TASK有COMPUTE关联；运行域另有14,088,140条TASK没有此元数据，不能记为零工作。导出资源共六组：Core、Vector、AICPU、MIX_AIC、MIX_AIV、COMM。SparseFlashMlaMetadata和QLI metadata实际导出AI_CPU/KERNEL_AICPU，不能用INT8/BF16 Cube峰值归一化。通信AivKernel的native类型为KERNEL_AIVEC，不能假设通信与Vector重算资源独立或免费重叠。同名QuantMatmul也同时出现AI_CORE与MIX_AIC，名称/dtype不足以证明完整执行模式。导出标签不是互相独立的容量池证书。

该全Case是带观察器的诊断域。新增pack/write会增加大量Vector任务并改变等待，整表不能直接用于生产模块MFU排名。InplaceScatterTranspose计数635712恰为158928次F21的四倍，只是污染警示，未把算术相等当逐native观察器身份。TASK累计时间包含等待、跨rank和潜在重叠；不是请求关键路径。

另外，一项不必要的配对门槛已移除：为计算同域聚合工作量/时间，不需要给每个eager调用逐条找到GPU计数。使用16个进程的(role,M)共128桶，源记录与native记录的键集合、每桶数量完全相等。native W1/W2验证target24个本地专家、draft8个本地专家，H5120/D2304一致；不存在额外或缺失桶。v168已有完整计数、无缺行/overflow且capacity65536未裁剪。此处在当前direct-dense源算法与返回计数stream正确性的域内，将receiver行数乘35389440 MAC，并分别汇总同一完整桶的原生时间即可；不要求逐调用配对或跨rank iteration身份。源/二进制匹配和实际物理mode仍是单独条件。

| F21域 | 调用数 | direct MAC | 累计完整native内核时间 | MAC/累计内核秒 |
| --- | ---: | ---: | ---: | ---: |
| Draft | 11088 | 2868137164800 | 2.640268027s | 1.0863 TMAC/s |
| Target小行数 | 145280 | 90863094988800 | 51.949889895s | 1.7491 TMAC/s |
| Target prefill | 2560 | 223531027660800 | 9.503194703s | 23.5217 TMAC/s |

全部158,928次调用均进入完整聚合。目标总MAC与v169相等，captured小行数Q和T与v169逐条关联结果相等。独立整数交叉乘法核验排序。共同常数每device参考容量P_ref下，eta_ref=Q/(P_ref×累计完整F21时间)，其效率排序为draft<target小行数<prefill，prefill/小行数为13.448235；未知公共P_ref在比值中消去。它说明该诊断域的完整F21 direct-MAC时间效率差异，不是实际物理MFU、整个M19/M12模块排名、集群/请求吞吐或可达提速。每个eager调用的等待原因仍可另查，但13648条未逐条关联不再阻塞此聚合结论。跨rank分组身份只在使用逐迭代联合组的其他问题中仍保留为前提。

下一步优先补当前171逻辑device到physical chip/board、执行模式和容量口径的关键桥，再继续全26模块一致理论/利用率对照。使用现有SDK和匹配可见性环境的只读查询，条件参考模式足够时先给条件结论，不把缺绝对fmax或历史私有字段当所有算子分析的门槛。当前量化计数不能硬配到旧无observer窗口时间。四项公开语义条件分析已可复用；缺实现或benchmark本身不是重新开展分析的理由。

完整35+7候选、26模块358语义行、全部条件分支和跨算子/层/卡/迭代融合、拆分、重组继续保留。all-reader、数值、状态/回滚、copy/control、实际物理容量及最低模块MFU仍开放；未伪补旧9/1352、849PMU、16phase、3136归属，也不让旧缺口支配所有新域。所有本轮CPU作业已结束，16份远端派生文件有SHA256。整体未完成。
