# Astra High：Run292 delayed B 独立审查

2026-09-26；只读全部client/Runtime证据及latest-per-rank八份trace，未编辑运行脚本、未操作服务/NPU。

## 隔离与执行 gate

独立确认60 POST、客户端区间最大12在途、warmup48+bench12共60请求usage1024；40/40 Runtime FULL、all8 delayed layer2 [12,4096] BF16→[96,4096] pad0。rank0五cohort周期284/302/313/291/300。未见Run290那样120POST/24并发污染。
source restore/stop产物存在。一次命中日志与FULL报告不是数值正确性证明，但支持B在原产品路径实际执行。

warmup531.622、bench12 550.371 tok/s为带profiler单次诊断；不能与无profile Run291的568.814/589.044直接作性能差、不能更新Current Formal，也不能因数字较低REJECT H003。

## device overlap 已成立；收益尚未知

独立产物：astra_overlap_screen.json（16样本，all8×latest cohort cycles64/65）。
每rank/cycle：Target Model45，首个c4 QLI对应layer2；其前hidden AllGather是comm stream0 Task53，local Q链main stream1 Tasks143–148，序列为输入DynamicQuant、q_a QuantMatmul、RmsNormDynamicQuant、q_b QuantMatmul、Q RMS、Q RoPE。索引结合层序、前置HcPre/RmsNorm和首次QLI校验，不只猜最近collective。

| 观测 | 中位µs | 范围µs |
|---|---:|---:|
| hidden AllGather | 14.880 | 11.341–19.981 |
| local Q链span | 83.000 | 80.250–85.250 |
| AllGather与实际Q kernels交集 | 13.750 | 10.000–18.750 |
| gather完成到Q完成 | 67.375 | 63.250–71.000 |
| gather开始到首wkv输入量化 | 82.500 | 79.500–84.750 |
| gather开始到下一AllToAll完成 | 614.875 | 609.000–617.500 |

main Task149 wait为0–0.22µs；首wkv输入量化Task150在wait后。通信确实与Q早期DynamicQuant/q_a重叠，且所有样本在Q结束前完成；不是只有Host async。comm stream随后等待下一工作的大EVENT_WAIT不能误记成当前join暴露成本。

以上是B内部结构证据，高置信。没有同profile A1，所以**不能说join、下一collective或Target已改善**。13.75µs交集不是净wall saving；通信/Q资源膨胀、launch差异、rank rendezvous必须由matched arm测量。也不能乘43层当E2E潜力上限。

## matched A1 profile：当前最高价值下一实验

A1使用相同async helper/同buffer形状/同TP group，立即wait；与B只改变wait位置。匹配patch/config、warmup与profiling配置、latest cohort选择、cycles64/65、all8覆盖；独立capture/replay，每mode明确图身份。

共同时间原点优先layer2前置RmsNorm结束/原fork点，不能只以每arm的AllGather实际start作原点，否则可能隐藏提交/arrival延迟。分别保留：
- gather时长、Q各节点/span和交集；
- 共同前驱→首wkv消费者、→下一AllToAll end；
- 各rank arrival skew及最迟rank依赖端点；
- Target/cycle总端点和后继节点膨胀。
跨rank时钟未证明一致时，用rank内共同图起点与collective端点关联，不把绝对ts混算假精确critical path。

两cycle×8rank受同collective耦合，不是16独立轨迹。不同服务状态/routes仍是残余混杂；matched profile是screen，不是same-state因果E2E。若有净局部信号再扩必要重复，不重复全kernel扫描。

## 数值与晋级 gate

现有client长度成功未验证gathered hidden/localQ/logits。
下一数值gate固定真实layer入口，A0原helper先自重放，再比较A1/B gathered hidden与typed localQ输出，保留输入/输出指纹及padding/contiguity/group顺序。该窗口在KV写之前，可避开Run285整Target上游差异；仍需FULL Graph依赖/allocator正确性及后续产品语义验证。

A1/B局部收益不能替代原A0对照：async helper和事件本身可能额外开销。正式KEEP仅依据correctness通过、隔离、未profile、重复冻结Product E2E。若局部无净收益，说明当前松边未转化为wall saving，不表示Hardware/Scheduling极限已收敛。

Bound可更新为“固定910B3 FULL Graph上，该shape AllGather与localQ已实证可并行”；necessary bytes/FLOPs不变，attainable joint cost和Product收益等待A1/A0证据。
