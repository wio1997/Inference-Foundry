# 并行配置的当前证据

安装树源码由Zcode只读定位、GPT裁决；[精确身份、locator与归约](source_evidence.json)引用现场原始Job，不复制全源码。

DP2×TP8×PCP1×EP16：EP世界为DP×PCP×TP；enable_expert_parallel时MoE内部TP=1，每rank完整拥有分配专家。256 routed experts两种配置均16/rank；不会因为DP2把每rank专家复制两份。这里只证明源码分片语义，不证明模型启动、内存或性能。

safetensors头182文件、176588张量，存储773854402152B，其中routed experts726348595200B、MTP routed experts19327352832B、其他28178454120B。这些是存储bytes；header SHA不等于weight内容SHA，加载转换/replicated buffer未知。Run0003 P启动实测每rank47.41GiB权重、2.08GiB peak activation、2.57–2.58GiB non-torch、0Graph；设备61.27–61.28GiB，KV约4.31GiB。该实测仅当前TP16/EP16/DCP16，不能直接转成DP2内存上界。研究判断：值得实际验证DP2TP8EP16，不以checkpoint/8否定，也不据此宣布一定fit。

GLM共78层、index_topk_freq4、offset3；普通层0/1/2/6/10.../38/42...执行完整indexer，其后共享buffer。PP IntermediateTensors只有hidden_states和residual。默认39/39切在共享层39，跨stage buffer未传；未证明可用。38/40使第二stage从完整indexer层38开始，是避开该依赖的候选；MTP/PP支持、调度和运行正确性还要证据。

MTP draft在相同worker运行，draft TP只能1或target TP；idle DP rank参与dummy batch，源码每32step同步状态。DP包含必要集体依赖，不能把两个DP服务当完全独立副本或预言2倍容量。Ascend具体MoE通信backend、长输入budget、KV/DCP及Graph约束按真实有效配置重新确认。

裁决INCONCLUSIVE；没有NPU执行，没有新增性能Run，没有KEEP。下一现场实验先利用当前两个已驻留完整模型验证本地完整请求，以较小代价缩小合法部署未知；随后依据该结果选择并行配置或控制层研究。

2026-10-01源级MTP线索（未实施）：MTP-METADATA-SOURCE-20261001T1159Z分析CLI超时143、bridge INVALID，无Result；partial source artifact只能按GPT实际读到的源码定位使用，不冒充Zcode验证。llm_base_proposer.py SHAedcc55d15f0f868b7da4f5ee8b63f91fc4f130b9b950e2734ab0d872a30d5f61，model_runner_v1.py SHAeaa40a2053be73e4f6ca6f54eddfb506f100790b1857a8d815c6db8756a32845。后续merged MTP draft at1437–1508保持input_batch_size=num_input_tokens（MTP分支），仅前batch_size写input/hidden/positions；1757–1936 eager norm metadata实际请求batch_size、其他slot为PADDING_SLOT_ID。Run11后续draft DCP recv1.01GiB OOM与16384物理rows一致，但未证明所有ghost rows可删除或完全数值无影响。第一pass历史KV、PD消费者、多rank/Graph、ExtraCTX num/max/padded/mc2_mask一致性必须保留；未做两行猜测patch。

原生MTP depth仍候选：配置num_nextn_predict_layers1、target78层；不能把draft当78层全模型，也不能因层数少就假定额外proposal免费。K3真实每位置接受率与step统计可从原生counter读；K5后续p4/p5及target verify shape、CPU/collective/Graph成本未知。仅在保持先前三位置分布且接受前缀单调等条件下，额外接受work/step可写局部上界2*p3，不能当硬件或服务容量上界。当前Run19完整长输出期间只读其已采样日志，不并发新模型调用/调参。

## Run19 no-tools parser contract and native depth hypothesis

Run19 native serving parses tool deltas whenever configured parser emits them, then overrides model terminal reason to tool_calls. Native GLM ParserEngine accepts arbitrary names when no tool schemas exist; `_check_skip_tool_parsing` only suppresses none when tools exist, and suppression can drop marker/name/args text. Actual stock GLM47 adapter CPU replay with model tokenizer reproduces both no-tools false tools and tools-none lost text. Candidate plugin subclasses only GLM adapter/engine; each request without tools or with choice none enables native skip_tool_parsing before extraction, preserving content. Auto with real schemas continues native parse/structural-tag behavior. No model/operator computation change. Native imports/source from image remain pinned; plugin SHA75b6b8e9006ceed7effc1890fe1b984c7876b49e5ae0fb2288734a2a9aeac0ed. Live validation pending Run20. CPU source/test/error identities are in GLM-TOOL-CONTRACT-20261001T1447Z{,-v2,-v3}, actual successful stdout3081B SHA c57ffc2b707a4ecd88bf25412045d41495b6359137a7977bed7db1f2da3b915d; failed fixtures preserved.

PARTIAL-LONG19-20261001T1440Z-v2 reduction8837B SHA090ac7f69329d4ca84663348e10ecf5e6a9dd0149aac7e22549f329ca0355625 freezes native first-cohort window14:12:56–14:22:58 (601.658633s). K3 P emitted30032 native tokens/8541 drafts, accepted prefix .9301/.8568/.7293, E3=3.5162/step~70.444ms; D32974/8334, .9988/.9959/.9618, E3=3.9566/~72.193ms. Generation counters reconcile draft+accepted in that window, not E2E completed credit. D path includes synthetic repetitive tool-name generation, not a representative quality claim. If first3 prefix acceptance unchanged and additional prefix monotone, K5 expected work/step<=E3+2*p2 (ratios1.4148/1.4862); step cost changes unknown so no speedup/capacity upper bound. Earlier dynamic acceptance much lower. New native K5 configuration is an empirical cost/fit probe, not a forecast or fixed scan. DP2TP8/metadata candidates remain available.

## Native depth20 counters

NATIVE-DEPTH20-20261001T1545Z/reduction.json10447B SHA dcc249dc9655ee07f0683613a70aefcb409c9024453fd167591a51f6082aa218. Run17/20 coldbody相同、normalized channel hash不同；非precision/quality/same-output因果比较。K3 cold722drafts/2166drafttokens/1326accepted/2048committed，approx(client elapsed−first output)/draft72.159ms；K5 569/2845/1482/2048，82.074ms。K5 accepted+bonus2051超emitted3，不计信用。Work/issued seq2.837→3.605是观察量不是硬件界。Dynamic D K3 1010draftseq/1807accepted/2816committed vs K5 932/1884/2816，work2.789→3.021；K5位置.797/.565/.354/.196/.108。多active请求下draftsequence数不同physicalengine迭代，不把wall/drafts称设备step时长。Wave78.4769 vs80.0026无收益。D K5 finite21/13037有效，KV296902/3.16–3.18GiB/Graph.92；Run21双K5长输出检验负载依赖，不是固定扫描或泛化预测。
