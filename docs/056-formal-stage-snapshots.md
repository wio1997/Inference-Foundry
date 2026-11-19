# 正式32K/64K阶段快照：定位秒级候选

## 结论先行

**应优先调查 SGLang P/D bootstrap 与 D 侧 `kv_transfer_alloc_ms` 命名区间，而不是先把微秒算子的 p50 差当作端到端根因。** 已存正式冷、热轮次的 P/D 前后 Prometheus 快照，存在秒级阶段直方图增量；64K同名指标比32K更长。这是在单次窗口、稳定计数序列及无未观测重置等前提下的**历史阶段线索**，不是已经证实的单请求关键路径或跨框架差额原因。

- SG 32K cold：D `decode_bootstrap` **33个观测均值26.569秒**；其中22个观测大于25.108秒。D `kv_transfer_alloc_ms` 同期33个观测均值26.565秒，22个超过该指标最后有限桶上界2.5秒。该指标自描述是“KV transfer allocation waiting time”，但G181运行版源码、资源种类和逐请求身份未绑定，不能断言是 HBM分配、full-KV容量不足或纯空等。
- SG 64K cold：D `decode_bootstrap` **35个观测均值100.568秒**，其中21个大于106.748秒。对应 `kv_transfer_alloc_ms` 35个均值100.561秒；这两个值不是独立可相加成本。
- 正式客户端每轮是32个成功请求，但stage观测数是33、34或35，且冷/热窗口间还有stage计数变化。故不能把这些观测当作那32个请求的逐条配对，不能把P、D、bootstrap、transfer等直方图加总成TTFT，更不能把其差值当成相比vLLM的24.127秒差额。
- 历史 SG 计数器前后标签一致、bucket增量非负且累计单调，`+Inf` 等于count；但快照缺process start和`_created`，无法排除重置后再次增长。G181的阶段计时源码逐文件SHA未绑定：G316或候选G156的setter语义只能作后续核验假设。

## SG 正式窗口：阶段指标并列，不是路径求和

快照对每个P/D角色的metrics响应完成采集时间形成包络；32K P cold/hot包络约267.985/269.586秒，64K P约507.739/507.878秒。它们包含客户端子进程前后的采集/审核，**不是实际负载或每个阶段时长**。各列为同一序列在前后快照中的`delta_sum / delta_count`；P、D采样不原子、各指标可能重叠，不能横向求和。

|输入 / 轮次|P prefill_bootstrap n / 均值|P prefill_forward n / 均值|D decode_bootstrap n / 均值|D KV alloc n / 均值|D KV bootstrap n / 均值|D decode_transferred n / 均值|
|---|---:|---:|---:|---:|---:|---:|
|32K cold|33 / 26.472 s|33 / 7.320 s|33 / 26.569 s|33 / 26.565 s|33 / 3.927 ms|33 / 35.050 s|
|32K hot|35 / 24.925 s|35 / 6.960 s|35 / 24.886 s|35 / 24.883 s|35 / 3.408 ms|35 / 33.481 s|
|64K cold|35 / 100.673 s|35 / 13.725 s|35 / 100.568 s|35 / 100.561 s|35 / 6.570 ms|35 / 39.491 s|
|64K hot|34 / 104.247 s|34 / 14.129 s|33 / 107.423 s|33 / 107.417 s|33 / 5.950 ms|33 / 41.033 s|

32K cold D `decode_bootstrap` 的累积桶：11/33 ≤0.0292823秒，11/33 ≤25.108184秒，26/33 ≤40.675258秒；因此在同标签、无重置和直方图差分有效前提下，22个 >25.108秒、其中7个 >40.675秒。它是独立分布，不能与P桶内观测或KV alloc桶逐条对应。64K cold D该直方图14/35 ≤106.748秒，即21个 >106.748秒。`+Inf` 桶不提供上界；直方图桶只能给范围，不能宣称精确的p50/p90。

D三项增量和在32K cold/hot满足：`decode_bootstrap` ≈ `kv_transfer_alloc` + `kv_transfer_bootstrap`，误差小于浮点显示精度；三者非stage标签相同且计数同为33/35。这是**聚合算术一致**，不是同样本/逐事件身份或计时实现证明。64K hot P、D stage计数34/33更不能配对。跨冷/热边界另有计数增长；不知是旧事件完成、探针、其他请求或重复采样，不能指定原因或直接删样。

同一快照中的服务全局streaming指标不是rank0阶段指标：32K cold P `e2e_request_latency_seconds` 32次均65.151秒；D streaming TTFT 32次均63.817秒、E2E均88.333秒。它们不等于客户端计时，也不能仅因计数32就声称恰好对应32条正式请求。

## vLLM 参考仅限独立非计分 profile

原有G138 32K诊断（16请求、非正式32请求轮）内层快照：P `request_queue_time_seconds` 16次均25.059秒、`request_prefill_time_seconds` 16次均11.069秒；D分别0.259秒和0.402秒。导出器HELP分别称WAITING/PREFILL，不能直接叫router排队或纯NPU计算。P/D两次采集时钟和样本不绑定、区间可能重叠，**不得与SG正式数字做差或求和**。

vLLM诊断前后同标签、`process_start_time_seconds`与各直方图`_created`端点不变、桶和count差非负；这只排除端点处可见的变化，不证明中途绝无重置、engine独立重启或其他完成样本。候选源码将该类时长记在完成事件，但G138现场逐文件SHA尚未绑定；样本可能从采样前开始。

## 客户端到达顺序的匿名分组

仅从已归档时间数组及基准汇总读取，**不读取输入/输出文本**：在每个run内部按客户端POST前时间排序，8条为一组，分别统计第一帧与最后一帧；每组是事后分箱，不是调度器实测的wave，也不是两框架相同请求对。时间原点为各自run首个客户端发起时刻，不能跨run对齐绝对时钟。

32K cold客户端TTFT的组内中位数：前8个SG **30.833秒**、vLLM **21.413秒**；第9–16个SG **80.168秒**、vLLM **54.664秒**。两组各自开始时间中位都在各自原点后约0.11秒以内，体现该窗口中首帧等待的明显离散；无法单凭这项客户端观察指定服务端排队或bootstrap。后续组受到C16客户端并发上限影响，也不能与前16个视为同一服务器批次。

## 重算与后续证伪

[阶段脱敏汇总](../evidence/stage_snapshots_056/summary_v7.json) SHA `ffb8f03c31b6d1608ec021a99d15cdf591a70880d70049b8a76f461f04774378`；[客户端分组汇总](../evidence/client_waves_056/summary_v2.json) SHA `7916d3e2714c2592530c12804b4abaa349ffefbb8e92af263535250677985db9`。已对来源清单SHA、单系列同标签、时窗方向、count/sum及bucket非负、桶累计与+Inf守恒做离线校验；输出不含请求、响应文本、私有路径、ID或绝对时刻。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize_stage_snapshots_056.py --check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize_client_waves_056.py --check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_summarize_stage_056.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

下一步先核实G181实际运行版阶段setter的不可变SHA、服务epoch和各series样本来源；再寻找vLLM**正式同窗**阶段指标及同请求、同batch和资源预算关联。如果这些历史信息不存在，就不能从现有离线汇总反推；任何新的埋点、服务变更、请求或A/B都应先获得独立授权，不能把本页线索宣传为已验证提速或根因。
