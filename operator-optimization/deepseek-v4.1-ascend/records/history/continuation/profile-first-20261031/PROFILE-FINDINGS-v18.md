本轮取得实质进展，整体仍未完成。先充分使用保存profile，再定点读安装版CANN9.2 profiler SDK；没有新读模型源码、运行NPU、改服务或实现优化。

原始32份DB保存了28项PMU、8,010,828条记录，先前归约只取4项。现已完整保留286,101个执行组的total cycles/time、MAC/Vector、scalar、MTE1/2/3、fixpipe和icache字段，旧4项数值全部一致，没有新增/丢失执行组。原始配置32份均为Level1、ACL_AICORE_PIPE_UTILIZATION、with_stack/record_shapes开启。

结合实际Cube24、Vector48、频率1800，所有286,101执行组在AIC/AIV两侧均能复现reported total_time：cycles/(1.8e9)*1e9/B*ceil(B/core)。这确定了本样本DB时间字段的ns尺度及周期估算关系；正周期AIC58,373组、AIV256,014组均无未匹配项。3,147,111条pipe_time=reportedtotal_time×ratio恒等式全部通过。44,325个AIC组和48,133个AIV组的pipe比例和超过1，说明各pipeline不是互斥时间，不能相加做可节省延迟。

定点SDK证据也解开了旧单位疑点：Python fallback的cube_usage在duration仍为ns时执行，随后才把输出duration除1000转us，列名与中间值属于不同阶段。SDK读频率时把1800乘1e6得到Hz。另一方面platform_version5映射CHIP_V4_1_0，满足C++device/all-export条件时会跳过Python FFTS，因此没有把参考Python代码冒充实际导出路径；观察字段的公式由上面的native数据逐项验证。

70条通用helper链已按完整caller逐条审查，覆盖14,574条原生直接来源执行。Q/KV、WoA/WoB、输入/最终/压缩/index norm、词表mask/embedding、DSpark context/combine/logits、Engram、collective及transpose已分开，actual-propose和dummy路径也保留。它们只是154,940条直接来源执行中的一个子集，不能据此宣布整模块或图内全部消费者完成。

这次补出一个实际条件分支：57次草稿调用经过_project_kv←_update_caches←_preprocess，复用已核对源码的“本地Q为空仍更新缓存”分支；各有57次KV投影/norm，36次全局M10、21次M20。此前1,167次只是多流KV子集，完整原生SWA KV为1,224次，Q_a仍1,167次。直接同M合并Q/KV的否定结论保持；F08/F29/F32等候选必须包含这个cache-only分支，不能因本地Q为空就删除全局KV/cache/collective工作，也未按sample_tokens和execute_model序号强行定阶段。

资源指标开始能按角色而不是泛算子名查看。目标直接来源Q_a/SWA KV的reported MTE2周期占比分别约0.861/0.958，MAC约0.582/0.541，支持继续研究不同local/global行域和重复全局投影；仍不支持朴素同M拼接或宣称HBM带宽已饱和。MC2 compound的reported MAC约0.033、Vector约0.007，并没有拆出内部dispatch/GMM/activation/combine或等待成本。HCCL AllReduce命名节点的reported MTE2约0.915、Vector约0.00000192，说明不能把高MTE比例直接称为算术kernel的HBM瓶颈。COMPUTE_TASK_INFO口径本身含HCCL命名kernel；286,950是该表口径执行数，不是纯算术kernel总数。现有流重叠、管线轮询、实际字节、cache分母和物理峰值仍须分开。

PMU初版派生摘要中有一个Counter字段把值求和误标成字段计数，已保留原产物并用完整原始census纠正元数据；执行关联、恒等式及资源摘要未改写。主台账仍为33项模型候选、7项框架候选；本轮关闭70helper语义归属及完整PMU字段/观察公式缺口，没有宣称MFU或提速。当前166 EngineCore/hash成本、物理拓扑/专家放置ABI、其他模块/图内消费者及数值生命周期等范围仍未完成，Goal保持active。
