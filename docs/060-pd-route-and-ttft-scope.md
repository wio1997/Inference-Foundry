# P/D 编排候选与 TTFT 计时边界

## 先看结论

32K cold中，SGLang P服务端请求E2E均值 **65.151秒**，客户端TTFT **65.704秒**；vLLM P服务端请求E2E **40.595秒**，客户端TTFT **41.576秒**。这种数值接近是值得调查的线索，但各项仍是**独立直方图/客户端聚合**：32个观测不证明是同一批请求，P服务结束与客户端首个流式帧也不具有已证的相同起止点。不能将两者差称为路由、网络、D侧开销，或声称P解释了TTFT差的某个百分比。

本机G4 **候选**vLLM代理源码显示：先向P发非流式单token请求，等P响应后才选择D并开启D流；候选vLLM引擎的D TTFT从该服务自己的arrival计时。**如果正式轮实际命中该代理路径**，D自身TTFT均值0.892秒无法表示客户端从发起到首帧等待的41.576秒，因为D请求可能直到P响应后才被发出。这不是对正式运行命中、物理传输完成或TTFT因果的证明。

## 原始观测：服务指标与客户端分别列出

服务P E2E、客户端首帧各自n=32；没有服务端请求ID与客户端记录逐项配对。SG服务P仅选global streaming指标，vLLM P为endpoint aggregate；计时边界、代码版本、服务epoch、两框架P/D最大运行数不同。四个cold/hot窗口也不是四次独立重复。

|输入/轮次|SG P 服务E2E均值 s|SG 客户端TTFT均值 s|vLLM P 服务E2E均值 s|vLLM 客户端TTFT均值 s|
|---|---:|---:|---:|---:|
|32K cold|65.151|65.704|40.595|41.576|
|32K hot|65.450|66.003|39.840|40.700|
|64K cold|155.142|156.178|104.804|106.174|
|64K hot|155.080|156.123|104.935|106.286|

此次进一步核对vLLM正式快照的 `vllm:e2e_request_latency_seconds`：四个P窗口中，该指标与 `vllm:time_to_first_token_seconds` 均各32个观测、增量sum完全相等，8个共同bucket阈值的累计增量也相同，但两直方图有各自bucket布局。**计数、sum或共同粗桶相同均不证明它们来自同一批事件，也不证明每条请求逐项相同。** 候选proxy把P输出限制为1个token，提供了可检验的解释，但仍未逐请求核验。

32K cold的vLLM D服务E2E均值23.566秒，而D TTFT均值0.892秒；两者是服务记录的独立指标，不等于客户端首帧到末帧或模型纯kernel时间，也不能把P和D指标相加为客户时延。

## vLLM候选：先P后D，但未证正式时命中

候选代理源码 `load_balance_proxy_server_example.py` SHA `85b8982948530758be655d3448ff10fc3979e56622221feace560272a81f1469`：

1. `build_prefill_request:819–835`复制请求，P设为非流式，`max_tokens=min_tokens=1`并填写KV参数；不改变原D请求的目标输出长度。
2. `send_request_to_service:838–853`等待P HTTP响应；`assign_instances:943–980`读取P响应携带的KV参数后选择D。
3. `handle_completions_impl:1032–1071`等待前一步返回，随后打开D流并预读首chunk。它是候选源码的控制顺序，P响应返回**不自动证明物理KV传输已经完成**。
4. 候选 `stats.py:430–473,497–499,518–555` 中，TTFT按输出处理迭代时刻减去该服务的arrival；queue取首个QUEUED→首个SCHEDULED，prefill取首个SCHEDULED→首个engine NEW_TOKEN，抢占可包含在其中。候选 `loggers.py:1133–1155` 对首token和完成请求分别更新不同指标；首输出观察不一定是客户端首个有效帧。

候选代理文件哈希与本机G4候选source清单一致；启动器代码包含代理文件SHA校验。但尚未找到正式G47/G48期间代理进程和实际运行文件的逐文件SHA、或同请求路径见证，候选 `stats.py`/`loggers.py` 亦未证明是现场import。因此只可将该顺序用于**设计假设**，不能说正式每条请求确实这样运行。候选stats和loggers SHA分别为 `af1983ddc32fa4d3284e72212c6525daf961dd84429023a02d612b68b3112cf3` 和 `c12b4fbebfe886b697127f1114eaa0bb83c54a78675919d178b619c3d8d5d655`。

## SGLang候选：P完成不等于纯前向

候选 `serving_base.py:80–108`和`req_time_stats.py:422–450,515–516`在OpenAI入口路径下使用服务自身接收与完成时间计算E2E；后端若经native `/generate`且未传received_time，则候选实现可能改用req-state初始化时间。Rust router转发的后端endpoint尚未绑定，不能认定实际采用哪一种。`prefill.py:1205–1228,1285–1301,1310–1311,1331–1335`在P发送侧poll到`KVPoll.Success`后完成并输出。这个候选P生命周期可能包括排队、P执行以及P侧转移状态等待，**不是纯P kernel时间或客户端首帧**；poll success也不等于已证纯网络wire时间。

候选 `decode.py:2970–3023,3051–3053`在非rebootstrap且成功接收的分支从handoff metadata提交P采样的token；随后进入D wait queue。若SG router在P完成前发D，D侧TTFT可包含等待P和handoff，但现有G181 Rust router核心/实装文件及实际P/D发送先后未绑定，不能宣称它确实提前或并行发送D。

**还可能有反向依赖：**候选`mooncake/conn.py:3042–3061`收到足够D目标信息后将P接收状态置`WaitingForInput`；`prefill.py:492–502,531–555`在非optimistic、非`prefill_complete`等特定条件下等待该状态才进入后续P队列。若该分支和P/D启动顺序在现场命中，D预分配/元数据进展可能影响P何时启动，后续P计算与handoff又可能影响D，不能简单画成D单向等待P。P实际生效配置、传递信息的实时事件与这些源码均未现场逐文件绑定，所以当前无法用聚合直方图确证此依赖或容量因果。

该SG候选 `serving_base.py`、`tokenizer_manager.py`、`prefill.py`、`req_time_stats.py` SHA依次为 `6bc4a06b15dfb33fab941089efdb8517b1d00b059c34a4dfa225e081cef70442`、`0ab0afe68ec80bcafb45e5ea3686e725cb69c369bff211b3e974ce2ed58ba6a8`、`519108e8d2149a13e61d837a082ceb23296abd61aa6c32e254969a467167ac02`、`21e17d68d3f18ffdb1a1203fe814f2ddcea9869021a3fa9f43210222d114a1e6`。其候选包来源链可复核，但不等于G181宿主bind文件及实际import现场验真。

## 检查与下一步

[白名单均值与直方图摘要](../evidence/p_latency_scope_060/summary_v2.json) SHA `fa4f5237092fdd9a9557e905d16b69690359148fb659c1c1dddcff20d15dcedf`；[代理候选静态关系摘要](../evidence/p_latency_scope_060/proxy_candidate_v2.json) SHA `9ea218718397efda962938db0faeaaf73eae6b4476d03f273cd10cd11aa7b980`。复算器需要本机历史vLLM快照，仅提取白名单、校验[059](059-vllm-formal-stage-snapshots.md)本轮固定SHA，**读取metrics正文到内存但不输出原文、标识或绝对时间**；候选验证器需显式提供本机代码、清单路径，不能独立证明现场命中。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_summarize_p_latency_scope_060.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_verify_proxy_candidate_060.py
```

最低限度的归因需要同一匿名attempt的client发送/首帧，router发P/收P/发D/首帧，P entry/first/finish与D entry/ready/first，并确认各进程版本及时钟误差。必要事件缺失时保持UNKNOWN；P/D可重叠，不能机械相加。本轮未发起新推理、未连接远端或修改服务。
