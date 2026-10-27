# 172 · 首 token 差距排查与固定场景验收

固定热身方案下，DeepInfer v21 平均 TTFT **815.25 ms**，vLLM-Ascend **831.25 ms**；合并输出 **1312.32 token/s** 对 **518.42 token/s**。16/16 正式输出均为1024 token，与冻结vLLM全文逐字节一致。

**口径：v21 包含有容量上限的“相同文本→输入token ID”缓存。** API启动时缓存为空，原有prefix seed和C8/32输出热身使正式请求命中；它不缓存模型预测，实际模型KV命中仍是235904/262144=89.990234375%。此前关闭该输入缓存的v20是1013.35ms，首次出现的新文本仍支付分词开销。815ms不能外推到新文本或任意场景；只运行用户指定两轮，没有为归档复跑。

| 配置 | 第一轮TTFT ms | 第二轮TTFT ms | 平均TTFT ms | 合并输出token/s |
|---|---:|---:|---:|---:|
| 冻结vLLM-Ascend |885.7|776.8|831.25|518.42|
| v20：绑核+FFN依赖裁剪，输入分词未缓存 |1042.4|984.3|1013.35|1211.17|
| v21：增加重复文本输入分词缓存 |811.2|819.3|815.25|1312.32|

合并吞吐按16384/(5.5512566+6.9334988)计算；TTFT按两个等量8请求轮次取均值。两轮均值仅领先16ms，第二轮819.3ms仍高于vLLM对应轮776.8ms，不能视作跨负载稳定领先。

## 找到的区别与处理

1. **查询布局和数值边界。** 原生路径曾重复计算TP8全部查询。配对诊断对比640个rank/layer：Q投影、RoPE、索引逐bit相同，38层压缩注意力也相同，只有前2层纯SWA在切换head分组时出现BF16差异。v17保留2层TP8，其余38层CP；v14/v16全CP候选因一个完整输出不一致被拒绝，不能引用其诊断速度为成绩。
2. **O投影。** v15把预填充FP32 einsum改成固定镜像同款BF16 transpose_batchmatmul；解码/修复数值路径保留。
3. **CPU绑核遗漏。** 冻结vLLM enable_cpu_binding=True，原生worker曾都能在640核间漂移。v19按40核/NPU切片绑定36主核+1ACL+1release，只操作自有16个worker/TID，不改IRQ/宿主服务或迁移内存。TTFT1118.85→1070.05ms。
4. **后半段多余FFN。** 最后长KV源在20层，此后长注意力读固定KV，只有128滑窗扩展依赖。v20在FFN层i保留尾部1+(39-i)*127个token的完整依赖范围，所有注意力/写缓存保留，纯冷前缀/解码/修复不裁剪；TTFT1070.05→1013.35ms，16条完整输出一致。
5. **请求准备。** C8/32k输入重新分词实测约120–180ms，batch/Rust/进程方案收益有限。v21增加默认关闭、启动明确启用的精确文本LRU，最多64条/1048576输入ID，每次返回独立列表；正式两轮各命中8请求/262144输入ID，约198ms客户端TTFT下降。新文本仍正常encode。

已排除：vLLM这里同样使用head all_to_all_single和SP reduce-scatter；实际comm实现未提供更快custom TP分支；enable_fused_mc2=1明确回退到同款dispatch_ffn_combine，不是新版CANN MegaMoe；无CPUquota/cpuset限制或throttling。v18把KV汇总字节减少76.5%，但TTFT无明确改善，作为INCONCLUSIVE保留。

仍有实现差别：Engram为原生CPU mmap选行/E8M0解量化/阻塞H2D，对照vLLM注册host UVA和事件/辅助流；RoPE为原生向量分解，对照预计算cos/sin+rotary_mul。当前不声称这些差异的独占耗时。注意力计算已明显下降，当前profile中FFN与TP通信较重；kernel时间重叠，不能相加当TTFT。profile中的104ms提示查找建表发生在首token发出后，也不能算首token瓶颈。

## 固定运行口径

172，16逻辑910C，DP2×TP8、EP16；固定v0.30.0rc2-a3镜像ID、同只读ModelSlim W8A8权重。AISBench固定提交7bcb6440da11bef020943b2af9ed7c1ac8362306，32768输入/1024输出/C8/两轮，greedy、ignore_eos、streaming。每轮实际KV缓存235904；所有输出仍通过完整40层目标推理。

仅Engram CPU offload，无主干CPU prefetch。原生解码是输入prompt-lookup63+64-query完整目标图验证；vLLM是DSpark5+FULL_DECODE_ONLY，两者算法不同。输出吞吐收益依赖重复上下文，不外推到任意提示词、随机采样、通用连续批处理或均匀逐token流式速度。

## 文件与复现入口

- [最终配置/实际进程命令](runtime.json)、[最终原始结果](current-result.json)、[独立输入ID审计](input-cache-independent-validation.json)、[版本矩阵](versions.json)。
- [v12之后累计补丁](accepted-after-v12.patch)、[源码链](source-chain.json)。v12之前已有补丁见上一轮记录；不要把拒绝的v14/v16补丁应用到接受链。
- 每个版本目录含原始AISBench JSON/CSV、全部完整预测和时间字段的派生compact JSONL、增量源码与控制器。compact仅移除各版本重复的长input正文，保留逐输入SHA256/字符数，原始文件SHA与wio绝对路径在raw-details-refs.json；服务器原始JSONL未改。完整日志/trace/失败现场和SHA都保存在wio，定位见[archives.json](archives.json)。
- 当前真实worker源码`src/DeepInfer-v20-tailffn`，HTTP源码`src/DeepInfer-v21-tokenizer`；同一模型文件逐字节相同，实际控制目录`serving/prefill-tail-ffn-v20`。不要误把v21当成一套已重启的新NPU进程。
- [v20实际启动控制器](prefill-tail-ffn-v20/controllers/native_service_prefill_tail_ffn_v20_job.py)保存全部环境/参数与绑核方法；[v21 API替换控制器](prefill-tokenizer-v21/controllers/native_service_prefill_tokenizer_v21_job.py)保存输入缓存开关；[两轮客户端](prefill-tokenizer-v21/controllers/native_prefill_tokenizer_v21_benchmark.py)保存热身和计数校验。
- 这些是当时顺序执行的控制器，含“目录/状态不可已存在”等防误启动断言。未来恢复应先核对设备归属和固定镜像，用runtime的实际命令、新建独立control目录，按v20方法绑自有线程，再按相同canonical热身/两轮运行。不要对当前活跃服务直接重复运行启动脚本。

当前服务保留运行。原始vLLM、v8、v9c以及上一轮20261027-ttft全部保留，未覆盖。
