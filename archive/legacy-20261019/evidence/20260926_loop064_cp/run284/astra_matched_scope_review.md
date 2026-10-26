# Astra High：Run284 matched profile 与 scope=all 晋级门槛

2026-09-26；仅只读，不运行 NPU、不安装补丁。不替 Sol 作 KEEP。

## 独立复算

输入 run284/matched_profile.json，脚本 scripts/analyze_loop064_cp_matched.py；重新按 (rank, cycle) 序号匹配16行：
- fork→Sparse start 差中位 -39.5µs，范围 -46.0 至 -31.0µs，16/16负值。
- fork→next AllToAll end 差中位 -39.875µs，范围 -64.0 至 -31.25µs，16/16负值。
- Rotary 延长差中位 +58.68975µs。
注意 JSON 汇总采用“两组中位之差”，AllToAll 该口径为 -40.875µs，Rotary 为 +58.6195µs；两者都可报告，须标清。

这是真正比单独重叠面积更强的局部证据：收益到同层下一 collective 完成尚未消失。Rotary 膨胀不等于全部收益被吃掉。置信度：局部屏幕中高；whole-cycle/E2E 低。
但16行来自两个不同服务的2个profiled cycle，rank受同一collective耦合，不能当16独立same-state重复。fork 是最近 EVENT_RECORD 的启发式匹配，AllToAll 是后续首个同名任务，未按 event id/模型层id证明；单层密集时合理，扩层后需要更强关联。
Runtime 60.988 vs60.903ms/cycle 来自不同 acceptance/position/routes 轨迹，不能说明调度优劣。约40µs仅为约61ms的0.066%，该测量分辨率未支持whole-cycle结论。

## 对照的含义

scripts/loop063_cp_fork_patch.py 中原始 off 顺序为 indexer update→QLI→main Compressor/scatter；immediate 改为 indexer update→aux main Compressor/scatter→join→QLI；overlap 只把 join 延迟到QLI后。
所以 immediate vs overlap 隔离“推迟 join”的作用，却没有排除重排/stream/event相对原始A0的成本。晋级性能必须最终与当前已接受 off/A0 比较，而不能把 immediate 换成正式baseline。

## scope=all 静态安全审查

正面事实：
- fork 位于 compress_ratio==4 内，只选择非prefill、96实际token、非draft、model.layers.<43；scope=one限layer2，all放开所有匹配c4。
- indexer cache update仍在fork前；两个分支读取hidden/metadata，main分支写main state/cache，indexer分支读indexer state/key/scale。既有 audit_disjoint 检查这两类cache交集。
- main在Sparse消费前join；next layer依赖本层attention/output，且aux为同一共享stream。因此源码没有产生可合法跨越前层join的aux队列积压，也没有把HCCL移动到aux或改变collective调用数/程序顺序。
- keepalive保留compressed_kv/cos/sin/slot直到本层函数退出；main join已排在后续消费者前。

未闭合风险：
1. 21层的每层真实alias/设备/选层inventory须逐一确认，不能用layer2 pass代替；96行不独立证明固定c12，须来自Runtime覆盖记录。
2. get_or_compute_compressor_metadata 以(cache_group_key,type)缓存，首次producer可能在aux、后续层复用同一tuple。本层join和下一fork wait理论上保证可见；但必须证实cache在各capture/substep正确reset，Graph pool不交叉复用，实际binary/workspace没有未建模全局scratch。多层温热cache的调度不是layer2冷cache的重复。
3. env mode/scope只在capture时决定分支，改env不能改变旧Graph replay；必须独立capture、明确graph身份与选择。Graph capture也会执行写cache，捕获前后要恢复状态。
4. allocator/keepalive逻辑看不到明显必然悬空，但跨stream Graph中临时storage复用仍需数值gate和device事件闭包证明；单次client输出1024 tokens并不是数值parity。
5. scope=all仍包括原A0重排收益/成本，不自动能把单层40µs线性放大21倍。后续层metadata成本、资源竞争、rank arrival不同。

## 下一步优先级：先 one-layer same-state 数值 gate

原因：已经有正的局部性能信号，当前最大可行动不确定性是“真实候选是否保语义”。再扩21层profile会增加集成风险，却仍不能回答correctness。

最小充分设计：
- 先用既有snapshot机制验证 A0→restore→A0 自重放一致；这一步只针对本候选需要的缓存/state/metadata覆盖，不重做无关实验。
- A=当前已接受off/A0；B=one-layer overlap。独立capture后从同一输入、position、cache/state/scale、metadata、RNG(若实际使用)快照跑A/B/A；graph capture造成的写必须排除。A/B/A两次A不一致则该实验无效，不归咎于B。
- 用同snapshot补一次immediate，确认三种顺序都遵守同语义；immediate只是因果控制。
- 至少保存 all8 ranks 的相关main/indexer cache写区、topk/attention(可获取时)、target logits/aux hidden、接受数/输出token、next draft；检查原有冻结数值阈值，不能临时放宽。indexer/main Compressor state和后续一个cycle读取尤为重要。包含一个可用压缩/页边界state；parking切换如现有快照可取得也应覆盖，否则标未覆盖并由后续产品回归补齐。
- Target snapshot的A/B/A只闭合Target数值；如果比较acceptance/DSpark必须同步restore这些状态，不可读取三次运行后不同轨迹的acceptance当same-state。

通过后：
1. 扩all21，逐层alias+选择清单、all8 FULL Graph replay、同state至少一次扩层回归。
2. all21 matched profiler A0/immediate/overlap screening。原analyzer硬编码仅2个aux Compressor且要求每次前有CompressorMetadata，**不适用all21**：应按cycle/layer/task关联计42aux Compressor/2cycles，并允许metadata cache命中不产生新metadata，记录真实fork/join。保留QLI计数/collective顺序、allrank critical completion、Target/cycle端点及节点膨胀，不加总kernel时间为收益。
3. 未profile的重复正式Product E2E，以acceptedA0对照、固定client协议和完整Runtime覆盖，记录useful tokens/cycle、acceptance、Target/DSpark/Host各阶段；只有收益超过配对波动且correctness稳定才可晋升。immediate/overlap本身无正式KEEP权。

粗略量级：若21层均拥有同约40µs的可消除串行间隔且没有其他损失，约0.84ms/cycle只是实验规划情景，不是可信上界、更不是TPS保证。Hardware/Resource Bound不因重排改变；Scheduling Bound可在DAG中将相关串行边替换fanout/join，但必须使用有竞争的联合时长（已见Rotary+59µs），并由Target/cycle实测校准。
