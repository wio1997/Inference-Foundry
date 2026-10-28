# 172 · 官方 DSpark / DeepSeek-V4.1-Flash W8A8

**官方 DSpark 置信度调度版 v15：521.07 token/s，平均首 token 805.65 ms，16/16 完整输出与冻结 vLLM 一致。** 对照 vLLM 518.42 token/s、831.25 ms；指定两轮吞吐高 0.51%。

同一172、16逻辑设备/8物理910C、DP2×TP8、EP16；相同只读W8A8权重、tokenizer和输入。32768输入/1024输出、并发8、greedy、ignore_eos，两轮各8请求；真实KV命中235904/262144=89.990234375%。吞吐按16384除以两轮总时长计算，使用实际输出token，不计未接受草稿。

|轮次|DeepInfer token/s|vLLM token/s|DeepInfer TTFT ms|vLLM TTFT ms|DeepInfer TPOT ms|vLLM TPOT ms|
|---|---:|---:|---:|---:|---:|---:|
|第1轮|514.83|509.52|833.0|885.7|14.4|14.5|
|第2轮|527.47|527.63|778.3|776.8|14.0|14.0|
|合并|521.07|518.42|805.65|831.25|—|—|

差距只有0.51%，且只有用户指定的两轮；不宣称稳定领先、任意负载性能或硬件全局最优。SSE投机输出以块到达，TPOT不是均匀逐token间隔的保证。精确输入文本→token IDs LRU预热单独计数；未包含首次新文本的全部分词成本。vLLM原有完整预热和DeepInfer规定前缀种子+C8/32输出热身均在正式轮外，真实命中率由服务器计数复核。

## 实际路线

使用检查点内mtp.0–2三层W8A8学习式草稿网络、目标37/38/39层HC特征、五位置非因果注意力、Markov头和置信度头。完整40层目标模型做贪心拒绝/修正与bonus提交。置信度策略按上游MTPVarlen的N−2轮决定预算、N−1轮分配长度，结合实测4/7/24预算图曲线和运行时中位数估计。正式负载的实测曲线选择全部5个草稿；这不意味着调度器被关闭，也不声称每一步都会缩短。短/非均匀前缀11个实际目标预测/DP在已有成本曲线前向中与相同上下文的完整目标图核对。

v14已正确但吞吐516.57未达到518.42门槛。v15将目标验证词表head按TP8分片，每rank计算16160/129280行，通过score+ID完成全局top1，保留最小ID平局规则，避免重复全词表GEMM；prefill和普通AR仍用完整head，不声称完整head权重内存节省。最终置信调度版v15达到指定两轮门槛，没有回退固定5旧版以代替调度要求。

没有提示查找、保存输出回放或vLLM Python引擎代理。固定vLLM-Ascend镜像只提供CANN/custom OPP与tokenizer。此前“提示查找63＋验证64”属于额外实验，不能代表官方DSpark适配完成，其完成结论已撤回。原始历史数据和冻结vLLM保留。

## 当前服务与复现

当前恢复运行`dspark-v15-restored-20261028-0306`，源`DeepInfer-dspark-v15`，16worker、CPU绑定、图初始化、HTTP健康已核对。服务器本机`http://127.0.0.1:8172/v1`，model=`dsv41`，greedy temperature0。重启未额外跑基准，缓存已清空，上表是已保留的正式热缓存结果。

所有源码、配置、日志、缓存与归档在`/data/tiankuan/wio/910c-deepinfer-172`；权重只读在`/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8`。本次既有privileged容器已获用户授权；仅Engram CPU卸载，主干CPU prefetch关闭。原vLLM容器停止并保留。简单执行经本机ZCode DeepSeek，未停止他人任务或修改170。

[比较数据](comparison.json) · [独立审计](audit.json) · [运行证据](runtime.json) · [逐输出SHA与一致性](output-equality.json) · [源码提交](source-freeze.json) · [409文件完整SHA](complete-source-SHA256.json) · [完整源码及原始证据归档位置/SHA](SOURCE-ARTIFACTS.json) · [复现](REPRODUCE.md) · [原始正式指标/配置/预测](raw/dspark-v15) · [冻结vLLM参考预测](raw/frozen-reference)。

完整源码提交`55c7b50cebbee87d43f352bbda874111b990d101`，官方上游未推送。GitHub保留相对e8a99a5与51737e7补丁及最终15文件，完整409文件源码归档在WIO中。恢复控制器已禁用自动基准，复现基准会产生负载，需先确认服务由本项目独占及无请求运行。

## 保留的失败与适用范围

v6异步计数和v7统一预填充未解决首轮停顿；v8将图捕获移至READY前。v9/v10动态稀疏查询NaN失败，v11第二轮OOM，v12仅准备；v13的16-query前缀出现数值不一致，故拒绝该形状。v14/v15保留已核对4/7/24预算图；大于7预算使用实际完整草稿因果图，未采样未来位置被丢弃。详情见raw各版本状态，未放宽正确性以取得性能。

本次端到端验收覆盖固定贪心C8，每DP四请求，context≥1024且prompt+max_new+6≤34816；其他批量回退原生自回归。随机采样、任意并发和通用连续批处理不在本次两轮验收范围。
