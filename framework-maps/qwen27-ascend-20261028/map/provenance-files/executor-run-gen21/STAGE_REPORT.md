# 第21代：主矩阵部署与采集时间证据闭合（阶段完成，待root验收）

原始主矩阵复现包849326161 bytes、SHA 3c3d68d562b1176c0867ad7b0205d307c4c5497fd2f0074fbc32e5452b6eb59c重新核对。明确allowlist的716项包成员按原manifest复读SHA，36项计分前freeze输入与原分析状态一致，未重跑基线/消融。原包创建时间与本轮hash观察时间和请求采集时间明确分开。

SG三覆盖模块的原始/候选SHA与四真实worker的12份load/JIT收据闭合；另2份父进程收据单列。4份weight gate绑定rank/PID，launcher/FIA/manifest来源一致。weight gate源代码是load_weights后置包装，属于启动检查。未声称新正确性/质量/greedy等价或性能采用。vLLM“未请求task覆盖”只按冻结启动契约呈现，未穷尽全运行模块。

224波逐波保存before环境UTC、client manifest创建UTC、after环境UTC、suite完成UTC与SHA，56暖机先于168正式波。1792条显式请求ID（含暖机）与既有最终地图timeline闭合。正式计分包络为2026-10-24T19:27:13.438282+00:00至2026-10-25T01:09:29.116706+00:00；不是推导的精确request-start/EOF。原模块收据无时间字段，仅保留取数前freeze事实。

更新的archive-draft仍UNALLOCATED、未发布。新增2份小CSV，合计7份均<64KB，5份旧CSV字节不变；224行大时间索引留外部。验证12草稿文件SHA、201表格行、17链接、16RESULT字段；不是完整仓CI或性能验收。规则/身份仍引用gen20观察，临发布须刷新。未更改任何unattended文件。

一次原生导出观察得到104个verified_archived条目，但pipeline submitted103/inflight4/pending97；保留计数差异，不推断完成。4原master的PID/start_ticks/command与原检查点一致，下游三个阶段仍等待。既有SDK进程已不存在、自有container停、slot resumed；未借slot、未起停NPU/服务/其它外部作业，原8卡逻辑所有权保留。原CPU审计worker与主pipeline不动。完整精确master身份、host/boot、日志、终止条件在external-jobs-receipt.json；worker个体/proc本轮未重复核查。

本代关键证据为baseline-provenance.json、frozen-extraction-receipt.json、baseline-wave-acquisition.csv、validation.json与package-receipt.json。原始trace/SSE未在本轮重新全审；模型质量、critical-path、CANN/卡组混杂等原限制仍保留。最终204地图还须显式挂载本代provenance，gen19既有合并器不会自动吸收它。

下一轮先观察native204一次；完成则闭合真实输入/导出/审计/归档SHA并运行实际最终地图合并与HTTP/browser QA。仍待导出时，处理其它消融/cache/诊断的精确部署与原采集时间或一个有依据的候选门禁，不重复已完成主矩阵证据。总体完成与最终接受由root核验，overall_complete=false。
