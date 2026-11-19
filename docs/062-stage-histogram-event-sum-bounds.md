# 阶段直方图：慢桶对事件累计和的条件边界

## 结论和适用范围

对已发布的SG与vLLM正式阶段直方图，只利用每个桶的计数、左右边界与该系列sum，可对超过指定阈值的观测，其**事件duration累计和**占该系列总sum的比例计算严格的数学区间。这是对历史聚合数据的离线算术，前提是前后series相同且窗口内无未观察到的重置，且源sum/bucket是有效测量；不是实际wall-clock、NPU利用率、请求占比、TTFT贡献或收益上限。

|历史窗口 / 指标|慢事件阈值|慢 / 全部观测|慢事件和占该系列总和的保守范围|
|---|---:|---:|---:|
|SG 32K cold D decode_bootstrap|>25.108184秒|22 / 33|[99.985212%, 99.990872%]|
|SG 64K cold D decode_bootstrap|>106.748148秒|21 / 35|[85.999332%, 91.357613%]|
|vLLM 32K cold P WAITING|>30秒|21 / 32|[79.792424%, 98.492563%]|
|vLLM 64K cold P WAITING|>60秒|24 / 32|[91.793798%, 116.897458%]|

百分比区间为向外舍入的保守闭区间；源桶的精确开闭标志见脱敏摘要。vLLM 64K上界大于100%不是声称观测到负时间：该直方图最小桶没有已知下界，**仅依桶和sum本身不能排除快桶含负值**；即使额外证明所有时间非负，得到的也仍是事件累计和的界，而非TTFT或设备占比。SG这两组的首个非空桶前已有已知正数边界，因此没有靠假设补零。

还须注意，候选SG源码在请求离开bootstrap阶段时才记录该阶段；如果窗口末还有请求尚未离开，它们不在已完成观测中，可能发生阶段完成选择偏倚。G181该源码尚未现场逐文件绑定；33/35条观测也未映射到32个客户端请求，不得删除多余样本或声称已识别其来源。

## 算法

按累计桶差分为不相交的 `(previous_upper, upper]`，第一桶下界未知、`+Inf`上界未知。令`T`为整个系列的已记录sum，`F`为阈值以下（含阈值）观测的事件和，`S`为阈值以上事件和；则`S=T−F`。直接从慢桶得到`S`的下/上界，同时以`T−F`的快桶上/下界收紧；二者分别取更强下界和更强上界。若某边界不存在，则保持未知，绝不擅自把未知负下界设为0。源summary存储的decimal字面量已有精度，不可声称恢复了源指标更多位数。

例如SG 32K cold的11个快桶观测累计和上界约0.129653秒，全部33观测的sum约876.761782秒，因此22个慢事件的和至少约876.632129秒。**这些事件可能跨请求重叠，且没有逐请求身份，不应加到另一侧或减去vLLM均值。**

[脱敏边界摘要](../evidence/hist_bounds_061/summary.json) SHA `0183eed652d94fffa5b5377d10b9db6f2fe0aa752ee616e3ea63cbec5cda2b89`；输入分别为[SG阶段摘要](../evidence/stage_snapshots_056/summary_v7.json)和[vLLM正式摘要](../evidence/vllm_formal_059/summary_v2.json)，不读取原始metrics、模型输入输出或ID。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize_hist_bounds_061.py --check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_summarize_hist_bounds_061.py
```
