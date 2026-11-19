# P 侧 Profile：选定算子在各自采集中的全时段分布

## 范围与发现

先前图只展示每框架各自P rank0的一个250ms稠密窗口，容易被误读为整个profile。本轮在确认P侧无运行容器、无NPU进程、CPU和I/O低于保护阈值后，只读已保存profile，以独立秒级桶展示**白名单事件从首次出现到最后一次结束的范围**：SG为约13.081秒，vLLM为约8.008秒。它们是独立诊断采集的trace坐标长度，不是请求时长、完整profile全事件覆盖或同一逻辑step；不能因一个区间更长就得出框架慢了多少。

每桶保留事件数及**按事件开始时刻分桶的完整事件duration之和**；事件可跨桶、多个stream可能重叠，该和可以大于桶宽，**不是墙钟、NPU利用率或TTFT贡献**。桶内少于3条的非空类别以未知值保留；空缺不说明硬件空闲或没有其他算子。

|选定P rank0事件|SG可见桶统计|vLLM可见桶统计|解释|
|---|---:|---:|---|
|KDA|272条，事件duration之和1088.749ms|272条，885.693ms|该类别全部分桶均≥3条；同次数不证明相同shape、数据、步骤或融合边界|
|lightning indexer|可见95条，1174.205ms，另1个低计数桶未展示|96条，1529.762ms|可见值不是SG全量精确和；同名但各框架记录shape不匹配|
|普通GroupedMatmul|可见686条、368.567ms，另1个低计数桶未展示|可见359条、131.416ms，另1个低计数桶未展示|不同融合和工作负载，不能按次数或事件总和推TTFT|
|融合GroupedMatmul+SwiGLU+Quant|未在SG白名单中观察到此精确名称|可见359条、308.309ms，另1个低计数桶未展示|只说明选名范围内的事件记录，不证明另一框架没有等价计算|

SG和vLLM各自固定在原有profile源的P rank0：SG读取G240冻结导出，vLLM读取G148从G138原始采集生成的分析副本；vLLM同时核验既有frozen原始组件的SHA。每框架自行确定选定事件的起点并归零，横轴彼此无跨机/跨epoch同一步映射。白名单保留KDA、稀疏attention/indexer、MoE dispatch/combine及部分通信等待等精确事件名；某个框架未匹配某类别不等于未执行相关逻辑。

## 复核与不足

源SG trace SHA `92b0b3ec44036c268c23c4eb68844313e77406a2701b7ce629ae85f349f09922`；vLLM派生trace SHA `3b3dd739f60a1259511aeece38ac05a2c5cd6e54c00115611954e964fbe2f142`。只读提取前后核对源文件SHA/大小/mtime及已归档rank收据；主机资源门要求无运行容器、无NPU进程、健康卡、CPU和I/O受限；读取时每约3秒重查容器/NPU进程、CPU、内存和I/O，卡健康只在开始时核对。没有启动/停止服务或发送推理请求。

[脱敏桶摘要](../evidence/p_timeline_061/selected_p_full_span.json) SHA `9c21d75ca1e8ee2d0a40a07b9183942545facbaeaad0b808e276810230f2fdce`；[校验器](../scripts/check_p_timeline_061.py)和[合成测试](../scripts/test_check_p_timeline_061.py)不访问设备或服务。提取器只记录允许的事件类别、相对桶、计数和非墙钟duration总和，不输出输入、请求ID、绝对时间、stream编号或主机路径。分组不小于3只是最小阈值，**并非针对所有推断攻击的匿名性保证**。

不属于该白名单的事件、CPU投递、同步点、其他rank和事件依赖不在本图中；trace生成时record_shapes关闭。要判断哪些算子导致正式TTFT，仍需匹配框架的实际逻辑step、shape、dtype/stride、数据和负载、依赖关键路径并做受控对照，不能把本图取代正式阶段histogram或客户端时延。
