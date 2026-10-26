# Astra High：Run285 数值 gate 独立裁读

2026-09-26。只读检查 run285/analysis.json、pages/rank*.json、bench12.json、runtime/、日志与当前诊断实现。未操作 NPU/服务/源码。

## 裁读：INCONCLUSIVE，禁止晋级；不是已证明候选语义错误

- all8落盘：off0→off2 argmax96/96且acceptance/token一致；off0→off4 argmax94/96、token改变；overlap→off0 93/96且一处accepted token改变；immediate→off0 94/96且一请求acceptance count改变。
- off自身full logits max_abs 5.875/4.52734，overlap 4.82031。B某些汇总落在A经验范围内不能证明安全；三次A不是统计容差上界。Immediate logit max_abs 6.9375和count变化也是需定位的信号，不能用“没并行所以必然正确”消解。
- 未保存每个分歧位置的top2/margin、实际typed分支入口与输出，所以现有落盘不能区分上游数值变化、candidate race、遗漏恢复、持久scratch、低margin正常传播。
- 特别注意 off2在overlap之后、off4在immediate之后：观察到的A差异不自动证明“baseline固有随机性”。候选若写到snapshot未覆盖的页/全局scratch，后续off也会被污染。restore exact只验证已捕获集合，不能自证未捕获集合。
- 各rank看到同一全局gather后的logits，不是8次独立数值复现；不能以all8相同结果增加独立样本数。

因此：数值门槛未过，不能扩为正式候选；当前证据也不足直接给CP overlap REJECT。可保留“局部调度收益已测、正确性待首分叉定位”。

## post_cache数值口径必须修正

_execute_cp_fork_parity选择“存在layer2 alias”的cache视图，再比较该视图所有snapshot联合页。它不等于layer2专属语义数据。

历史Run75 manifest说明同名视图：
- kv_cache.0.0同时映射layer0/1 SWA、layer2/3 compressed与state；
- kv_cache.6.0映射layer2/3 SWA及layer4/5 compressed/state；
- kv_cache.4.0的float32视图同时映射indexer state与key；kv_cache.5.1是同storage的FP16 scale视图。
实际Run285对应关系须重新记录，历史manifest不能替代。

tensor_comparison直接a.float()，a!=b计数，finite(delta)求max。对混合类型/量化重解释、页内padding/未写区、NaN而言，约3e38不是可解释的“Compressor数值误差”；相同NaN也会被判different。不能据此断定cache严重破坏，也不能以“全是padding”忽略。

最小修正：保留原始dtype/shape/stride和层→视图→实际slot映射；分别比较
1. 实际活跃写槽/typed state区域的数值、finite mask；
2. quant key的整数/byte；
3. 未写区域的字节保持（包括NaN payload），不以float max评价；
4. 多层共享视图按物理byte/slot归属拆分。
真正state有效域尚未明晰时标unknown，不用整个物理页float max替代。

## 下一实验分两条目的，不重复全Target末端筛查

**若要裁决CP候选：最高信息价值是固定layer2 fork入口的局部因果gate。**
在原路径indexer update完成、分叉前捕获一次实际hidden_states_cache、hidden_states_local、qr/scales、Q/local RoPE、必要metadata与相关cache/state，保证所有试次使用同一不可变fixture。
先 off/off/off（在任何候选执行之前），然后 overlap/immediate及off回验；恢复完整局部写集，验证未写区。比较：
- main Compressor规范输出（scatter前）；
- QLI topk/query分支结果；
- main/indexer state实际typed写槽；
- Sparse输出（AllToAll前）。
此区间源码没有新增HCCL；各rank仍保持统一试次顺序。若分叉入口在全Target五次运行中已不同，末端token比较不能定位CP。
该gate只证明局部分支语义，后续仍需FULL Graph与完整产品验证；不要直接升all21。

**若选择下一个全项目性能/Bound实验：优先原路径owner/frontier probe。**
当前CP只证明单层约40µs局部收益，未解析whole-cycle增益；无需为它无限重复43层五次Target。ownership工作能检验当前96行全请求更新是否属compulsory，具有更大架构信息价值。按 astra_ownership_probe_checklist.md 执行，期间CP保持未晋级而非错误定罪；静态96→16不能先当安全或收益。

## 完成/退出口径纠正

本次读取bench12.json为12/12 success、每请求usage1024，runtime目录有all8 cohort报告。Sol已确认launcher exit0且EngineCore退出由cleanup产生；日志中的EngineDeadError不构成独立candidate crash根因证据，不记录“Run285服务崩溃”。这也不改变数值gate INCONCLUSIVE，更不能把客户端长度正确升级为模型语义PASS。

Run284局部matched profile证据维持原级别；Run285不给TPS或formal KEEP结论。
