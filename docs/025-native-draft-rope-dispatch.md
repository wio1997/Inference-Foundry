# 完整target图通过后的草稿RoPE设备分发修正

G450的D端静态显存比例0.82，实际full token池1082496。8个rank均完成完整target verify图，bs=[4,8,12,16]；各rank图占用约8.50–8.59GB，结束可用约3.07–3.22GB。该实测说明本轮越过此前图显存不足处，不代表完整服务通过。

随后draft verify图在 `DSparkAttention.forward` 的inverse RoPE路径进入CUDA-only `fused_rope_inplace`，尝试用nvcc编译sm00而失败。D退出并清空8卡，完整日志和原始归档SHA保留；不能把这一轮的失败继续归因为显存不足。

G462在同一源码SHA门槛和频率索引AST安装过程内，对3处CUDA-only RoPE调用增加NPU排除条件：query两处、inverse output一处。已存在的第4处原生NPU排除条件保持不变。NPU进入固定源码已有的RMS/RoPE分支，沿用G334组件验证过的相同RoPE helper字节；原生fast WO_A分支保持原条件，没有将整个DSPARK_FAST_KERNEL关闭。argv、0.82预算、EP8、DSPARK5和完整图要求不变，P437保留。

G334组件覆盖8rank、64用例、64图、192次变化回放，含inverse和非连续视图；它不代替新增调用安装、完整draft图或模型质量验收。G462已实际启动，完整draft图、PD、语义、四档矩阵与profile仍待实测。

- [G450完整日志](../evidence/full_model_G450/2204/server.log)
- [8rank target图凭证、容量及原始SHA](../evidence/full_model_G450/2204/graph_and_dispatch_assessment.json)
- [退出与清卡](../evidence/full_model_G450/2204/state.json)
- [设备分发和频率索引的同一安装过程](../code/full_model_candidates/sg_pd_g462/freqs_install.py)
- [源码覆盖检查与保持的fast WO_A](../code/full_model_candidates/sg_pd_g462/native_rope_dispatch_review.json)
- [新D实际启动](../evidence/full_model_G462/D_launch_receipt.json)
