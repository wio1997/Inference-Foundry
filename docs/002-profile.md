# 真实profile和算子差异

[结构化对比](../evidence/profile_comparison.json) 是本页的事实入口。每个框架都检查了P/D各8个rank的CPU和显式模型设备事件；原始大trace以SHA、大小及私有逻辑位置索引，不进Git。

SGLang G30 P是8次EXTEND与8次DRAFT，D是8次VERIFY且实际batch=1。客户端C16不能解释为设备batch16。vLLM P首段mHC局部T=960，其余T=1024；SGLang有SP分片与未分片层。D端两者也有不同的物理布局。

同一16请求输入不代表捕获到完全相同的工作窗口。CANN版本、数据简化、健康请求及异步边界都要分账。`EVENT_WAIT`、通信任务和其他stream会重叠，累计任务时长不能除以端到端墙钟作为占比。

当前重点是720次mHC pre中对应的28,080次除法：每次20轮Sinkhorn含39次行列归一化。这个数量关系支持融合候选的选择，不足以证明它就是最终关键路径。
