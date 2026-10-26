# Astra High：Run291 隔离A1 gate及H003下一步

2026-09-26。独立只读重算server日志、warmup48/bench12与40份Runtime报告；未操作NPU/源码。

## 已验证：隔离执行 gate 通过

- 服务端恰好60条POST /v1/chat/completions；两phase请求时间区间合并最大在途12。
- warmup48与bench12全部成功，usage output_tokens均1024。
- 40/40 rank-cohort为FULL Graph、handoff_before_model_forward=true，handoff后oracle/model-runner cycles=0；generated_output_counts均1024。
- all8命中layer2 mode=immediate，[12,4096] BF16→[96,4096]，pad0。
- rank0五cohort cycles=300/281/317/297/292，wall=16.9963/15.8880/17.9059/16.8047/16.5050s。
- restore.log回到base SHA 27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e。

这支持Run290混流是覆盖异常的关键原因：移除双客户端污染后同A1恢复5/5。不能据此证明所有未来handoff漂移都已解决，也不能把A1判为更快。
warmup568.814、bench12 589.044 tok/s是单次smoke阶段数字，不是重复冻结48-request正式中位；Current Formal不更新。

通过范围：隔离、实际FULL capture/replay执行、服务输出长度合同。未做same-state gathered hidden/localQ/模型数值比较，未验证delayed并行或E2E收益。

## H003下一B profile：最低充分目标

B只把同async AllGather的wait从提交后移到local Q链之后、首个gather消费者wkv之前。控制其他scope/shape/pad/集体通信group与顺序不变，不同时启用H001 Compressor fork。

建议先B profile做机制筛查：
1. 维持Run291的隔离验收：60POST、最大12在途、all8和真实5cohort FULL。source/patch/config SHA、profiler配置、所选cohort/cycle与请求位置都保存。
2. 在all8两采样cycle中**唯一定位layer2的hidden AllGather**，用graph/task/layer关联或严格邻接链验证；不能仅取“附近首个AllGather”猜测。保存localQ输入/输出guard、实际attn_state及pad。
3. 记录 AllGather设备start/end、local wq_a→qr量化/norm→wq_b→Q RMS/RoPE start/end、wait/event、first-wkv start、下一个collective arrival/end，以及完整Target/cycle端点。
4. 看真实设备时间交集和消费者依赖：即使host async返回，若Q实际仍在HCCL完成后开始，H003重叠机制在该capture中未实现；无需把这一结构失败包装成kernel小收益。
5. 如果有并行，再与**相同profiler配置的A1 profile**作matched comparison。Run291无profile，不能将B profile耗时直接减Run291 wall/cycle。A1/B需分开capture，禁止改env重放旧Graph。

核心指标是最迟rank的“gather完成且Q完成，wkv可开始”时刻和后续critical-path端点；不是sum(kernel durations)，也不是仅通信与计算重叠面积。
不同rank时钟未经校准时，优先rank内同一图起点归一和collective同步端点，避免假跨rank全局绝对时间上界。

## 判据和后续路径

- B没有实际overlap：当前实现机制未成立；检查Work.wait/capture/HCCL进展，若明确被该实现串行化，可PIVOT而非重复formal。
- 有overlap但Q/通信变慢，最迟rank join无改善：该窗口的资源竞争吞掉收益；不能仅凭交集判KEEP。
- join改善却下一collective/Target端点无改善：局部松边尚未成为E2E收益；找被延迟的rank或后继依赖。
- B稳定改善A1：只证明延迟wait的局部作用。最终仍需A0原helper对照，排除async helper/alloc/event开销；不能以A1替代正式baseline。
- 全部结构门槛通过且有信号后，做固定layer入口gathered-hidden/localQ数值gate（A0自重放先行）与适用FULL Graph correctness；必要时保存gather输入/输出hash、typedQ误差/margin。Run291的长度成功不能代替这一项。
- 最终只由未profile、隔离、重复正式A0 vs B Product E2E裁决收益，记录acceptance/useful tokens/cycle以分开轨迹变化。

允许B profile先于完整数值晋级作为纯机制诊断，但必须标diagnostic；如果记录到错误数据/缺依赖，直接停止性能解释。

## Bound含义

A1证明该同shape异步helper可在FULL Graph执行；尚不证明attainable HCCL overlap能力。B机制与matched联合时长才可为Scheduling-aware DAG松边提供证据。Resource/Hardware necessary bytes/FLOPs并未因调度变化减少，任何通信/Q膨胀都应计入联合资源成本。H003结果不决定是否接近总体极限。
