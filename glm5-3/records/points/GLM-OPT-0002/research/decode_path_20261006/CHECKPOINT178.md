# checkpoint178 — current FULL path and RoPE whole-table materialization

2026-10-07. H6/H5 and target FULL retained; H11 remains off after valid Run275 INCONCLUSIVE. 本阶段没有新增代码级性能 KEEP。Current、完整 API、80K/600/93% 正式 SLA 未完成，继续性能研究。

Run276 原始 FAILED 保留：profile-on/off 两个真实完整 PD 请求均 exact natural23/EOS/KV，但 aggregate drafts11/12 不同，不能做 profile overhead calibration。零新增请求 terminal reconciliation 证明同 P/D workers、all16 health/idle、H6/H5、H11mode0/transition11 和 stop-profile acknowledgement200。原始 profiler data 已保留；official offline parser CPU-only exit0 导出16 ranks。wrapper 的 after path-set equality 因新增484 export/log/.complete 文件失败；原失败保留。独立 audit removed0/changed0，全部原文件保持。

Root 亲读真实事件、源码与 CSV，并独立保存 [critical_path_reduced.json](../../jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007/critical_path_reduced.json)。实际 target replay 是每rank12次，不能用 acceptance counters11或 SSE grouping 代替 step 数。排除首个长 host replay，steady 周期约55–56ms；16rank replay-start skew82–111us。rank0/13/15 period3 非 wait 硬件区间 union 为52.318/52.440/52.299ms，周期55.418/55.510/55.420ms。CPU main 同步35.393ms与设备重叠，output-thread40.525ms也重叠，均非额外可删预算。旧 Run249 eager host 晚供给不再是当前大 gap 的证据。

24 个既有 Slice 任务已沿 `hardware connection_id → CANN Node → Dequeue correlation_id → Enqueue → aten::index` 精确关联。每个 CPU index 后有两次 unsqueeze，与真实 `get_cos_and_sin_mla` 唯一表达式吻合；这属于 source/call-pattern 推断，profile 无 Python line stack。Task-ID/time 对齐 CSV 明确 input[1048576,128]→output[1048576,64] BF16 ND，而最终只需两行。一次 steady step 四次物化512MiB，三个 ranks 的 task inclusive sum约1.32–1.33ms，不能预先称实测 savings。

H12 唯一候选是 [RoPE registration 两行 `.contiguous()`](rope_contiguous_candidate/rope_contiguous.patch)，不改 lookup/negative/empty/bool/error 语义、持久 graph 输出 buffers、算子或通信。原 repeat backing 与新 dense pair 常驻有效 payload 同为256MiB；初始化峰值可增加256MiB，allocator reservation可能不同。CPU504 cases通过；诊断 observer20checks通过；workflow十个正常/故障分支通过，最多五请求/一次恢复。native-index_select 原型 CPU1516通过但因更小初始化 patch 被 PARKED，不组合。

Root 接受独立 [framework Review](POST_FULL_FRAMEWORK_REVIEW.md)、[RoPE Review](ROPE_CONTIGUOUS_CACHE_REVIEW.md) 和 [Run277 Review](RUN277_ROPE_LAYOUT_REVIEW.md)：转向已证明冗余 device preparation，保留H6/H5；停止把 host waits、旧 producer starvation 或 inclusive sums 当收益。Run277 Review 指出的末尾恢复、过早 completed、witness命名及旧描述已修正，CPU故障分支覆盖。表在 registration 建立，witness实际在 warm后，二者区分。

[Reset/事先决策表](RESET_ROPE_CONTIGUOUS.md) 限定一次 owned D reload、四小 correctness 请求、失败预算五/唯一恢复，P保护、main FULL/H6H5/H11off不变。通过独立raw reduction后，才做同 workers、无 heavy profile 的匹配 A/B/A/B natural23 完整 PD。当前 Run277 是 draft/未启动；实际 controller state 决定后续现场，不复用旧 D PID 操作。Largest realized removable gap 仍 unknown。

| Code Optimization | Type | Baseline | Patched | Gain | Correctness | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| H6 MC2 capability cache | PERFORMANCE | Run256 accepted baseline | Run256 candidate | 既有两对 D TPOT−13.62/−14.20%；不可迁移为当前 FULL 累计值 | 既有 exact PD/all16 | active research candidate retained |
| H5 on H6 单流 MoE event | PERFORMANCE | Run261 H6 | Run261 H6+H5 | 既有两对 D TPOT−6.77/−7.04%；不可与H6相加 | 既有 exact PD/all16 | active research candidate retained |
| H11 scoped MTP graph | PERFORMANCE | Run275 A1/A2 | Run275 B1/B2 | D savings−2.313/−9.623ms | Run273 PASS、Run275 valid | INCONCLUSIVE/off |
| H12 immutable RoPE layout | PERFORMANCE | matched未执行 | matched未执行 | unknown | CPU504、observer20、workflow10；NPU未验证 | 唯一候选，Run277 draft |

AISBench 跨轮 salt、分端点 cache counters 与 GitHub workload 同步已在checkpoint173完成，不重测或改写历史。正式声明93%与实测 local/external hits 分账；本研究fixture不冒充正式条件。
