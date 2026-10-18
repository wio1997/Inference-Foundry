# M005 / M006：PD prefill 准入、chunk 与两种预算

固定安装 SGLang `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的源码与 **57 项 CPU 源码模型检查**表明：准入返回值、已提交请求、raw token range、分页计费、current 预算与 total 预算必须分别记录。`OTHER` 或 `NO_TOKEN` 可以出现在请求已提交之后；负的未来预留余额不能单独证明本次请求被拒绝或发生 native OOM。

没有优化候选或生产改动。没有执行真实 allocator/cache tree、H2D、完整 scheduler/PD 网络、模型生成或性能测量。整体 Ascend PD+MTP serving 仍 E0，accepted baseline 为空。

## 源码路径与检查范围

R001/R002 收集十三个规范源码文件，均与安装 Git HEAD 一致。完整 PrefillBudget 与 PrefillAdder 类体原样提取；Req 的 range/host-hit 方法、P 侧 `_process_req`、cache lock/load-back dataclass 与 tier split 函数体也保持未改动。`sglang.srt.utils.common` 正常导入，实际 Range 类与源文件哈希已核验。

allocator 可用量、cache 可淘汰量、锁、load-back 返回值及 prefill negotiation 都是明确夹具。Req 为字段夹具，使用真实 Torch CPU prefix tensors 和 Python array；token IDs 是合成长短输入，未计算模型输出。没有完整导入 policy/budget 模块，也没有实例化真实 NPU allocator。

R003 为 49 项：23 个预算接口检查、11 个单 pass batch、9 个 transaction 边界、3 个纯 selection、3 个 continuation。R004 仅追加八个页对齐 capacity 边界，未重跑前 49 项。结果见 [admission-source-result.json](admission-source-result.json) 与 [capacity-edge-result.json](capacity-edge-result.json)。

## P 侧局部输出预留与预算分工

`disaggregation/prefill.py:455` 的 `_process_req` 在 P 侧准备请求时把本地 `sampling_params.max_new_tokens` 设为 1，注释说明用于 PrefillAdder 估计。CPU 检查直接执行该未改方法，未把它替换为零。这里只证明 P 侧局部写入，**没有验证 D 侧最终输出策略或完整 MTP 行为**。

scheduler 将 `new_token_ratio_tracker.current`、max_prefill_tokens、当前 chunk 大小及 mixed-decode 数传给 PrefillAdder；实际运行值仍未知。本夹具 page=128、chunk/input cap 通常 16384、ratio=0.6、clip=4096，仅用来冻结条件。SWA、Mamba、HiSparse 等替代预算实现不能由本固定池检查推断。

固定池主要接口为：

| 项目 | 源码行为 |
|---|---|
| live capacity | 每次查询 allocator available + cache evictable，不能在 pin/preemption 前缓存旧快照 |
| total/current | 分别减 total_offset / current_offset；未来输出预留只加到 total_offset |
| reserve | 令 q=ceil(extend/page)×page；current 增加 q+page+extra，total 再增加 max_new_tokens |
| check_prefill | 使用 `total_tokens < remaining_total` |
| can_allocate_prefill | 使用 `paged_input <= min(remaining_current,remaining_total)` |
| continuation | available_chunk_tokens 在 total 预留耗尽时保留进展 fallback；current 物理可用量仍需另验 |

同一个 `<` / `<=` 不能脱离调用者的 extra page、max_new 和 chunk 语义比较。预算 API 的数字是 token 计数，不是实测 HBM 字节、attention FLOP 或时间。

## 选择、materialization 与提交

`add_one_req` 先临时 pin prefix，live evictable 随之变化，再调用纯 `_select_prefill_admission`。selection 返回不可变的 shape 记录或停止原因；在所测路径中没有修改 range、锁或预算。随后才 negotiation、host load-back，最后 commit 设置 range、保存持久锁收据、追加 can_run_list 并预留预算。

| transaction 夹具 | 观测 |
|---|---|
| 内存等值拒绝、临时 pin 后容量不足 | 不调用 negotiation/load-back；temporary pin 被释放，未提交预算 |
| negotiation 拒绝 | 不调用 load-back，未提交预算 |
| load-back 返回 None | 未提交 range/预算，temporary pin 被释放 |
| load-back 返回零 FULL tokens | 重新选择重算路径，512 raw rows 被提交 |
| promised 128，返回 128 / 256 | raw rows 分别变为 384 / 256 |
| 更多命中使原 chunk 完成 | chunk 标志清除，并恢复 P 侧 max_new=1 的预留 |
| promised 128，仅返回 64 | fail-loud RuntimeError；预算未提交，但 host_loaded_length 和 last_node 已发生局部变化 |

最后一行是人工 partial-load 构造，不是观测到的生产错误，也没有证明回滚或 native 故障恢复。不能把“纯 selection 不修改状态”扩展成整个 admission 无副作用。所测 host-load boundary 没有真实分配或 H2D capacity 效果；下一步需要真实 cache 与 allocator 的证据。

成功 admission 保留持久锁属于尚未完成请求的生命周期，本轮未执行其最终释放。temporary pin 的收据配对已在夹具中核对，实际 cache trie/UUID/淘汰仍未运行。

## 配置 cap 不等于 raw range，更不等于设备工作

以下是单个 CPU admission pass 的请求 range 合计，内存夹具足够宽裕。它们不是图 padding 后的设备行数或 native MoE 工作量。

| 候选输入 | 提交 raw rows | quota charge |
|---|---:|---:|
| 2×512 | 1024 | 1024 |
| 3×512 | 1536 | 1536 |
| 2/3×32768 | 本 pass 先提交一个 16384 chunk | 16384 |
| 512 后接 32768 | 512+15872 | 16384 |
| 32768 后接 512 | 先提交 16384 | 16384 |
| 513 后接 512 | 1025 | 1152 |
| prefix=127、prompt=32768 | 16257，range=[127,16384) | 16384 |

无 chunk 时，首个 512 请求可以超过 input cap=256 后提交并返回 OTHER；已有一个请求后，第二个请求等于剩余 input cap 的夹具则先停止。chunk limit 小于一页时，本次无法选出正长度 chunk。

未进入本 pass 的请求和后续 continuation/调度策略没有在这里完整执行。不能把一次 pass 的停止当成丢输入，也不能仅凭预留 slack 宣称有同等比例可消除的 NPU 计算。

源码 exact-chunk-fill 依赖 gfx95，tile budget 依赖 HIP。本轮设置 non-HIP/gfx95=False，即使请求 exact-fill，也不激活这些 AMD 分支。没有为 Ascend 放开门控。

## R004：提交后返回 NO_TOKEN 的页边界

以下 capacity 为页对齐的 token 夹具；P 侧 max_new=1，固定池，无 Native 分配。

| raw extend / prefix | capacity | current reserve | total reserve | 本次提交？ | 返回 |
|---|---:|---:|---:|---|---|
| 129 / 0 | 384 | 384 | 385 | 是 | NO_TOKEN |
| 257 / 0 | 512 | 512 | 513 | 是 | NO_TOKEN |
| 385 / 0 | 640 | 640 | 641 | 是 | NO_TOKEN |
| 385 / 127 | 640 | 640 | 641 | 是 | NO_TOKEN |
| 128 / 0 | 256 | 0 | 0 | 否 | NO_TOKEN |
| 127 / 0 | 256 | 0 | 0 | 否 | NO_TOKEN |

前四行的 remaining_current=0、remaining_total=-1。检查源于 raw 准入需求与 page-ceiled 提交预留的差异；**没有据此宣称 over-allocation 或 OOM**。scheduler 源码在停止时另以 `req is adder.can_run_list[-1]` 判定 added，之后继续使用 can_run_list；本轮没有完整执行 scheduler。

三组 continuation 另外构造未来 total_offset 使 remaining_total 为 -1/0/128，但初始 remaining_current=65536；实际选择 raw rows 为 16384/16384/128。这个 fallback 不能解释成在物理 free=0 时仍可任意分配。真实 allocator 单位、预留与实际占用的关系是原生实验必检项。

## 原生验证与后续方向

[native-next-contract.json](native-next-contract.json) 保持 Ascend、PD、MTP、2/3 请求与 512/32768 输入；补充上述页边界。必须确认实际 allocator 类型、current/total/evictable 的单位与并发变化、P/D 各自 sampling/ratio、prefix/materialization 的真实占用，以及 raw range 到实际图 shape 的变化。

原生正确性须覆盖 cache receipt、chunk continuation、最终输出/接受长度；随后才在相同输入、接受分布和配置下做 AA、配对 trace 与硬件吞吐校准。不缩输入、输出或 MTP，不直接移植 HIP gate，不把保守 token quota 当已测 kernel 冗余。当前没有实现优化。

下一步转向 prefix cache 的实际 pin/evictable 生命周期，将这里的 cache 夹具逐步替换为更强实现证据。

## 可追溯性与结论

四个冻结命令在精确 ZCode 会话各执行一次；[zcode-audit.json](zcode-audit.json) 记录 deepseek-flash 配置别名，实际后端版本 UNKNOWN。R003/R004 的两个任务 CPU 容器均无设备/网络、只读 root、1 CPU/4 GiB、仅四个 metadata/tokenizer 文件只读挂载，无权重；核对 ID/标签后清理，观测服务 ID、启动时间和 restart_count 不变。没有模型生成请求或跨主机性能计时。

[source-index.json](source-index.json) 与原始结构结果保存哈希、完整类体摘要及实际 common 模块路径。Loop 为 INCONCLUSIVE：57 项检查增加了固定池源码与 CPU 状态证据，完整 scheduler、真实 cache/allocator、NPU 与 PD/MTP 端到端正确性/物理性能仍待验证，整体不升级 E0。
