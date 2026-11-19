# vLLM 正式 32K/64K：P/D 阶段快照补证

## 本次更正与结论

此前只在公开归档的正式目录内寻找服务快照，未发现vLLM正式阶段数据；**不能由此断定不存在**。现在从本机历史工作区找到G4正式32K/64K各cold/hot的P/D前后快照，并复算出vLLM正式轮的阶段直方图。新的资料补齐了**正式轮两框架各自的指标展示**，但不能把两边名称相近的指标直接相减、当作同一条请求的阶段或归因为框架实现。

各轮vLLM客户端均32/32成功、C16、输入32K或64K、输出1024；所选每个P/D直方图增量也恰好32个观测。**数量相等不证明样本与32条正式请求一一对应**，也不证明P/D两侧或两个指标是同一批事件。SGLang阶段部分为33–35观测而非32，生命周期和版本亦未配平。

## vLLM正式窗口：单侧独立均值

取各自服务指标的 `after_sum − before_sum` / `after_count − before_count`，单位秒。P/D采集非原子；一个服务的TTFT、WAITING、PREFILL和另一个服务的区间可能有不同起点或重叠，**不能横向相加**。

|输入 / 轮次|P WAITING n / 均值|P PREFILL n / 均值|P 端TTFT n / 均值|D WAITING n / 均值|D PREFILL n / 均值|D 端TTFT n / 均值|
|---|---:|---:|---:|---:|---:|---:|
|32K cold|32 / 28.238|32 / 11.194|32 / 40.595|32 / 0.279|32 / 0.404|32 / 0.892|
|32K hot|32 / 27.467|32 / 11.185|32 / 39.840|32 / 0.161|32 / 0.399|32 / 0.772|
|64K cold|32 / 80.198|32 / 22.598|32 / 104.804|32 / 0.406|32 / 0.397|32 / 1.198|
|64K hot|32 / 80.248|32 / 22.591|32 / 104.935|32 / 0.406|32 / 0.388|32 / 1.177|

导出器将queue描述为请求WAITING phase，prefill描述为PREFILL phase；它们不是直接的router排队、纯设备kernel时间或网络线速。候选vLLM源码将queue与prefill从engine内部事件求时并在完成请求时观察；本次未找到正式服务实际逐文件源码SHA，更新点/重入语义依然需要现场确认。

同一选择的vLLM D `request_decode_time_seconds` 为32K cold 32个均22.673秒、64K cold 32个均24.886秒，分别是服务导出指标，不是客户端TPOT或逐token逐层拆解。P的同名decode直方图在两输入下32个观测均值0，**只表示该服务指标的记录值为0**，不是P没有完成其他计算。

## 与SGLang正式窗口并列：名称和边界不同

[SG正式快照](056-formal-stage-snapshots.md)中，32K cold P `prefill_bootstrap` 33个均26.472秒、P `prefill_forward` 33个均7.320秒；D `decode_bootstrap` 33个均26.569秒、D `kv_transfer_alloc_ms` 33个均26.565秒。64K cold分别为P bootstrap 35个均100.673秒、P forward 35个均13.725秒、D bootstrap 35个均100.568秒、D alloc 35个均100.561秒。

这些名称虽都与排队/前向相关，却**不是已证的同一计时区间**：SG bootstrap、wait-queue和D transfer阶段可能相互包含或跨端重叠；vLLM WAITING/PREFILL是各自engine exporter口径。不能由 `SG 26.569 − vLLM 0.279` 得到D侧差额，也不能将SG forward 7.320和vLLM PREFILL 11.194当同工作量kernel对照。64K两边P侧/Ｄ侧指标数值分布不同，提示应继续调查不同调度与handoff路径，但不证明等待发生位置或容量门的因果。

客户端32K cold TTFT均值SG 65.7036秒、vLLM 41.5764秒，相差24.1272秒；64K cold分别156.1784秒和106.1738秒，相差50.0046秒。这些是独立服务epoch的客户端均值，不是已配对的服务端阶段差。两框架CANN/torch_npu和max-running配置不同，正式两轮也不是独立重复A/B。

## 来源、稳定性和隐私

- 原正式快照在本机历史工作区而非公开Foundry归档；各快照SHA在本次离线提取时首次固定，并在同一执行过程前后核对文件身份、大小和mtime。**未找到这些原快照的历史预期SHA，不能称历史冻结原件。**
- 两个控制器源码SHA与各自恢复清单一致；恢复清单记录服务身份在恢复点已验证。四个原始正式audit的SHA分别对应公开归档provenance，且其脱敏条目被对应正式run manifest引用。该链支持窗口和正式客户端归属，不等于每个指标样本身份或整轮不重启已证。
- 每个vLLM P/D源直方图前后完整标签一一对应，count/sum/bucket增量非负、累计桶单调且 `+Inf=count`；`process_start_time_seconds`和`_created`在端点一致。只能说**端点未观察到重置**，不能排除区间内原地重置后追平、独立worker变化或额外/旧完成请求。
- 客户端和各服务使用不同计时范围；P/D metrics采样间隔是采集包络，不是可累加的请求时长，也不是跨机时钟校准。bucket的nearest-rank上下界不是精确p50/p90。

[仅含白名单数值的脱敏汇总](../evidence/vllm_formal_059/summary_v2.json) SHA `fa23c910c57a4221478a7dc30aedbacb3cb2304e51d5440501493668c4d1d8e6`；[本机复算器](../scripts/summarize_vllm_formal_059.py)需要授权的两组源目录与控制器文件路径作为参数，会读取历史metrics响应到内存，但不会输出原文、标签、请求/响应文本、ID或绝对时间。若不允许重读源材料，可仅校验摘要SHA和执行[合成测试](../scripts/test_summarize_vllm_formal_059.py)。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_summarize_vllm_formal_059.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

下一步仍需绑定各服务阶段计时点、同请求handoff/前缀与准入预算、观测服务epoch及调度/设备依赖；完成前不能声称已找到跨框架TTFT根因或优化收益。本次为本机历史材料复算，没有发新推理或修改服务。
