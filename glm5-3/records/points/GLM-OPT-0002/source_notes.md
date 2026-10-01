# 并行配置的当前证据

安装树源码由Zcode只读定位、GPT裁决；[精确身份、locator与归约](source_evidence.json)引用现场原始Job，不复制全源码。

DP2×TP8×PCP1×EP16：EP世界为DP×PCP×TP；enable_expert_parallel时MoE内部TP=1，每rank完整拥有分配专家。256 routed experts两种配置均16/rank；不会因为DP2把每rank专家复制两份。这里只证明源码分片语义，不证明模型启动、内存或性能。

safetensors头182文件、176588张量，存储773854402152B，其中routed experts726348595200B、MTP routed experts19327352832B、其他28178454120B。这些是存储bytes；header SHA不等于weight内容SHA，加载转换/replicated buffer未知。Run0003 P启动实测每rank47.41GiB权重、2.08GiB peak activation、2.57–2.58GiB non-torch、0Graph；设备61.27–61.28GiB，KV约4.31GiB。该实测仅当前TP16/EP16/DCP16，不能直接转成DP2内存上界。研究判断：值得实际验证DP2TP8EP16，不以checkpoint/8否定，也不据此宣布一定fit。

GLM共78层、index_topk_freq4、offset3；普通层0/1/2/6/10.../38/42...执行完整indexer，其后共享buffer。PP IntermediateTensors只有hidden_states和residual。默认39/39切在共享层39，跨stage buffer未传；未证明可用。38/40使第二stage从完整indexer层38开始，是避开该依赖的候选；MTP/PP支持、调度和运行正确性还要证据。

MTP draft在相同worker运行，draft TP只能1或target TP；idle DP rank参与dummy batch，源码每32step同步状态。DP包含必要集体依赖，不能把两个DP服务当完全独立副本或预言2倍容量。Ascend具体MoE通信backend、长输入budget、KV/DCP及Graph约束按真实有效配置重新确认。

裁决INCONCLUSIVE；没有NPU执行，没有新增性能Run，没有KEEP。下一现场实验先利用当前两个已驻留完整模型验证本地完整请求，以较小代价缩小合法部署未知；随后依据该结果选择并行配置或控制层研究。
