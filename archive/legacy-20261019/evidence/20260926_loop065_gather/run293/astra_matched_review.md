# Astra High：Run293 A1 / Run292 B matched profile 独立审查

2026-09-26。只读原始 all8 latest-per-rank trace、Runtime报告、client与service日志；未操作NPU或修改脚本。独立提取产物：`astra_matched_metrics.json`。正式裁决由Sol负责。

## 1. 有效性与独立定位

Run293日志恰好60个completion POST；48+12请求全部输出1024，client最大in-flight=12；5cohort×8rank=40 FULL Graph报告全部pass。all8在capture命中layer2 immediate，local[12,4096]→[96,4096]、pad=0。与Run292同样覆盖。没有Run289/290式120 POST混流证据。长度/覆盖通过不替代数值语义门槛。

独立解析原始trace，未依赖Sol分析脚本的结果作定位：两次Target Graph Model45；主流Task141=共同RmsNorm，通信流Task53=hidden AllGather；A1主流Task143立即wait、Q为144..149，B的Q为143..148、Task149延后wait；两者Task150为首WKV输入DynamicQuant。首个后续AllToAll同层匹配。两种Graph的设备事件总数与此局部任务序列一致，证明模式在设备图中实际生效。

`../matched_timeline.json` 是Loop065根目录的文件。其汇总为“两个中位数之差”，不能当成“配对差的中位数”；本独立产物按rank、cycle明确配对。

## 2. 局部因果机制得到支持

单位µs，差值均B delayed−A1 immediate：

| 端点/成本 | A1中位 | B中位 | 配对差中位 | B更小 |
|---|---:|---:|---:|---:|
| 共同RmsNorm结束→首WKV启动 | 100.250 | 83.125 | **−17.000** | 16/16 |
| →下一AllToAll启动 | 604.125 | 586.000 | −14.500 | 16/16 |
| →下一AllToAll完成 | 625.625 | 615.625 | **−10.125** | 16/16 |
| hidden AllGather duration | 14.87975 | 14.88000 | −0.17975 | 8/16 |
| Q链span | 83.875 | 83.000 | −0.750 | 11/16 |
| 主流join wait | 16.62975 | 0.02000 | −16.60975 | 16/16 |
| 后续AllToAll duration | 22.08975 | 28.54975 | +3.93025 | 3/16 |

共同前驱→WKV配对范围−25.5..−12.75，→AllToAll完成−21.5..−5.75。B确实把原先暴露在Q前面的wait隐藏了，局部提前量部分保留到同层通信完成。

Q未出现H001那种大幅膨胀：q_a matmul配对中位+0.410µs，q_b −0.880µs，Rotary −0.01025µs；全Q核sum −0.79025µs。微小差值含跨run形状相同但输入/状态不同等变化，不宣称q_b被该调度加速。

后续AllToAll更长提示arrival/rendezvous或通信执行差异抵消部分局部提前量；不能把+3.93µs全归为资源竞争。不同中位数不具可加性，勿用这些中位数拼接精确因果账本。通信流在gather后长EVENT_WAIT是等待后续工作，不是当前主流join暴露时间。

若使用trace时间戳的跨rank对齐，仅作辅助描述：两个周期的“最晚共同前驱→最晚AllToAll完成”分别A1 625.75/623.00、B 616.75/614.00，即各−9µs；前驱rank spread约10–14µs。该跨rank时钟没有独立精度验证，不升级为更精确收益。16个rank-cycle相互耦合，实际上只有两个全rank周期，不能当16个统计独立重复。

## 3. 完整Target/cycle尚未证明缩短

以Task141共同前驱为起点，到该次Target Graph最后设备事件：
- 第一profile周期：B−A1 all8为−321.75..−296.25µs，配对中位−314.125。
- 第二周期：all8为+117.25..+141.50µs，配对中位+123.50。
- 混合16样本中位−89.5µs、仅8/16更快。这个数不能用来宣称局部−10µs扩大成−89.5µs收益。

从Target Graph首任务到末事件，跨run差值受首段rank-arrival/skew显著污染；首任务跨rank spread第一周期A1约20.0ms、B约12.9ms。不得把Graph总span差当模型计算或传输差。

唯一完整设备recurrence窗口“第一个Target首任务→第二个Target首任务”（含中间DSpark与下轮准备）配对仅4/8更快，中位−39.75µs、范围−14.163ms..+4.312ms，无法解析单层收益。最后profile周期没有下一Target起点，不编造第二个recurrence值。CPU extreme::cycle范围也含Host异步提交/末次profiler停止等，不能代替device completion端点。

**因此结论是局部join改善可信、全Target尾段/完整cycle收益未解析。** 这不是候选性能失败，也不是正式性能成功。

## 4. 独立capture仍不可归因处与晋级门槛

- 两次独立service/capture没有同state、同draft、同KV、同route、同acceptance轨迹。Run292各cohort284/302/313/291/300cycle；Run293为290/298/290/293/283。同序号profile周期不是同计算输入。
- cache状态、路由激活专家、prefill/parking前沿、Host提交与profiling同步都会影响远端端点。all8 FULL Graph及相同shape不消除这些混杂。
- A1是新async helper立即wait，尚无同口径A0原始同步helper成本；A1/B局部改善不能直接推广B对A0。
- Run293 warmup552.380/bench568.232 vs Run292 531.622/550.371均是带profiler轨迹，不能据此说B回退，也不能校准Product Bound。

最小晋级链：
1. 同一layer2入口fixture，A0/A0先自重放，然后A1/B/A0；比较typed gathered hidden与local Q、dtype/layout/pad，并用事件fence确保输入/临时buffer生命周期。窗口在KV写前，避免Run285整Target状态恢复歧义。
2. 实际FULL Graph各模式独立capture，在相同fixture验证图内输出与依赖；改env不能切换已捕获图。相同TP group顺序，allrank选择一致。
3. 若扩大多层以提升测量灵敏度，先证明层选择/不同ratio路径和collective顺序均安全；不直接做局部收益×层数。记录共同前驱、首消费者、后续collective、完整Target/DSpark/cycle及latest-rank端点。
4. 足够分辨的未profile内部时序与冻结重复正式E2E，再由Sol裁决。不能把同state fixture的循环吞吐称Product E2E。

Bound更新只宜记录：“固定真实shape下，hidden gather/local Q的合法并行已在FULL Graph实现，单层匹配profile约10µs提前保留到下一collective。”必要资源量未改变，完整Scheduling/Product上界仍unknown。
