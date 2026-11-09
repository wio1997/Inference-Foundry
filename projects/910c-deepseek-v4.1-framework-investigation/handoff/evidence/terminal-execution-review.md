**2026-11-10 aux接线更正**：NPU runner先对配置ID加1，setter原样保存，model再判断layer_idx+1；实际aux入口为37/38/39。此前36/37/38表述已撤回。见[完整接线](aux-config-call-chain.json)。其余bank读者与旧价格证据范围不变。

# 终端实际读者与跨迭代交付判断

**Goal ACTIVE，loop-014 EVALUATING；全链路尚未扫完。每个替换原语逐位一致。** 本轮为源码、历史 profile 复算和未部署补丁设计，没有新 NPU 测量、服务推理请求、远端文件写入或环境变更。

## 介绍

从最后一层 HC 残差交付开始，沿实际代码走到 full HC/pre_mix gather、MTP bank、collapse/RMS、target 选行/head、接受拒绝、异步输出、CPU stop 与下一轮 input IDs/positions/slot。先按全部读者确定可删范围，再给完整 parent 计价；GitHub 旧记录只在同一交付边界相遇后复用。

## 到了哪里

- [运行源码身份](terminal-execution-identity.json)：在当前冻结容器的两个实际仓库扫描 13,520 个 Python/C/C++/头文件，26 个文件命中三个 bank/getter 标识。扫描清单及 SHA 在本地 `terminal-runtime-source-inventory.json`。这是这些具体标识的引用闭合，不是审阅了 13,520 个文件，也不证明任意外部反射/hook 不存在。
- [执行前沿](terminal-execution-frontier.json)：18 份实际源码 SHA 与本地阅读件匹配；列出具体行段。新取回缺失 executor/connector 源码的 [manifest](terminal-runtime-source/manifest.json) 与同次 inventory 对上。其它硬件/模型实现、测试、GPU 新 runner 和 NPU V2 的 getter 引用，均不当成当前 NPU V1/DSpark 实例读者。
- [历史 native 引用资格](terminal-native-archive-qualification.json)：新增校验捕获 `read_compact_terminal_ffn.py` 当前文件已更新；保留 error Run `run-20261109T145803Z`，从 Git `bf09e45ca5b19ac6e56e69699a1a5e26b85e8fd6` 找回期望 SHA `0db68b7a…` 的精确脚本，不执行旧脚本、不用当前文件冒充历史件。原始 native receipts 的引用 SHA 匹配；本轮没有重测反例。
- 原 `use_eagle()` 包含 DSpark，因此 target 仍分配 BF16 `[8192,4×5120]` bank；先全 HC/pre_mix gather、bank copy，再 full-M collapse 和 RMS。V1 getter 结果只在 `method=="mtp"` 时替换 drafter 输入；DSpark 用零基 37/38/39 层入口 aux，最终 39 层输出不进入 DSpark main context。
- `ExecuteModelState` 保存完整 hidden。后续读者包括 ordered `logits_indices`，以及 `_get_prompt_logprobs_dict` 对每个 prompt chunk 的完整 hidden 切片；还有 pooling、PP/intermediate/broadcast、真正 MTP、dump/导出等独立条件。不能无条件把所有末层计算压成一个 head 行，也不能先按“将来会接受几项”删掉尚需验证的 target 行。
- greedy Triton 的 K1 与一般 K 路径已读到实际 store：首个 mismatch 输出 target argmax 后停止，全匹配加 bonus，未用后缀保留 -1；一般 K0 仍需 bonus。普通 random kernel、recovered vocab 分块归约已读；FP32 p/q、原 RNG 分配及 tie 次序都属于合同。block verifier、全部 processor/RNG 和 fallback backend 仍待扫，未标为完成。
- 下一轮校正的实际 import 是 `vllm_ascend.spec_decode.utils`，不是 upstream 同名 helper。先按旧 req-ID 映射 materialize `prev_computed/prev_drafts/valid_count`，再更新原地数组；旧草稿行 `computed=prev_computed+valid_count`，新请求或无草稿行取当前 CPU 值。把每行 load/store 直接合成独立 CTA 会破坏置换快照。
- 继承的 `_prepare_input_ids` 把上次 bonus 写到本次 `cu_num_tokens-1-draft_len`，按上次 K stride 从草稿矩阵取本次 scheduled prefix。新/prefill 仍需 CPU 来源；无 spec、无重排时已有 slice 快路。CPU optimistic seq_len 用 `prev_drafts+1-valid_count` 修正前要等上轮 count-copy event。
- `AsyncGPUModelRunnerOutput` 已在单独 stream 等生产 stream 后复制 IDs、可选 logprobs/NaN，并持有设备引用到 event 完成；WorkerProc 的另一输出线程才 `get_output` 和入消息队列。Ascend executor patch 没有覆盖这些输出方法。DSpark 直接消费 GPU IDs，不能二次认领“移除 CPU 等待”。
- Core 尝试填 batch queue 后 FIFO drain，structured output 可延迟 sample，处理 abort 后才更新 scheduler。scheduler 先按拒绝数回退，再逐 token 追加并作 EOS/length 等 stop trim；stale/preemption 不得回退当前代计数。frontend detokenizer 另有 stop-string 最早完成匹配、stop-list tie、holdback suffix 和 stream_interval/final-only；发现字符串 stop 后要回传 core abort。
- common retirement 接口已读：connector `request_finished` 在 encoder free 前；异步 send/recv 或 defer_block_free 可延迟释放；DSpark 后才 finalize `wait_for_save/clear_metadata`。provider 自己的失败/异步完成内部仍是条件缺口，不能从公共接口推定所有后端已扫。

## 结果

| 范围 | 状态 | 判断与具体缺口 |
|---|---|---|
| DSpark 无读者 MTP bank | **待确认** | 源码可实现删除 allocation/copy；[具体补丁](terminal-dspark-dead-bank.patch)保留原两次 gather、full-M collapse/RMS 和全部 BF16/FP32 舍入。AST 只变 model init/forward，未部署、未做 native/graph 验收。逻辑容量为每卡 320 MiB、16 卡合计 5 GiB；不是已测得的 allocator 释放或可多承载 token 数。剩余是隔离运行的逐位/图地址/生命周期资格，以及容量是否改变开发优先级。 |
| 末层 Post→head 输入的独立 1% 延迟方案 | **已停止（限定预算）** | [复算](terminal-cost-replay.json)绑定全部原行/shape/通信/请求。整段全免费仍为 C8 P90 TTFT ≤0.96564%、makespan ≤0.42326%；六普通请求条件插值最高 0.64242%。无需为同一 parent 再试 collapse 微融合、选行 copy 或窄 AG 小变体。 |
| target head→接受拒绝的独立 1% 方案 | **已停止（限定预算）** | 连 head GEMM、词表通信、全部 score/argmax/reject 和内部 gap 全免费，C8 P90 TTFT 0.11023–0.12445%、makespan 0.54416–0.57536%；最高个体响应 0.58204%，六普通 B1 情景最高 0.52940%。不再为同一原时序预算做 head epilogue、argmax/verify 微变体。此价格不含单独 valid-count/seed、输出 copy、下轮输入及 host plan。 |
| stock 压缩末层 FFN／改变 collapse 归约树 | **已停止（具体配方）** | [原生证据限定复用](terminal-native-evidence-reuse.json)：直接 compact 改 mHC 系数/路由权重；保留 dense 系数仍有一个 expert-combined/HCPost 元素位差，虽然 normalized readout 相同。collapse 部分树有 1/2 元素位差；删除 Post 的 BF16 舍入也有反例。不重跑原样失败配方，不用最终向量一致放过替换原语位差。 |
| 更大的末层 attention/FFN selected-row producer | **待确认** | final aux 在39层入口、早于39层body，具备从读者反推 query/output/FFN 行需求的语义空间，但必须保留所有 KV/index/cache 写入与跨卡接收。仅末 FFN+完整 readout 全免的原 join 情景 P90 预算 22.9921ms（约1.541%），不能拿 <1% 的窄尾段预算剪掉更早生产者；也不能把接收等待当计算全省。保留原语数值几何、全部 cache 读者、稀疏实现/成本为具体缺口。 |
| reject→count/seed→下一轮 packet/host 提交 | **待确认** | 已有 accepted-prefix/last-seed 共同生产机会；未来 input 行映射尚由下一次 scheduler 决定，直接写未来地址需要稳定 request-slot/代次协议。不能复用 CPU -1 placeholder 当真实 token，不能跨 CTA 无屏障读新值。下一步先联合定价 host/device 整段，再决定是否补 native；现有单独 head 或 DSpark score 价格不等于它的预算。 |

上述 1% 是历史排队参考，不是用户设定的硬性能门槛；价格是保持原 clock、请求顺序、外部工作与 next-target frontier 的条件敏感性，不是物理上界或预计加速。不同 parent 的收益不能相加。普通请求使用匹配 B1 形状、原次数及明确的 prompt 插值，未冒充普通请求的新设备测量。

没有新加入“已确认可实现”的 native 替换。bank 删除的源码设计已具体可审查，设备与完整运行资格仍归待确认；18 个 SHA、12 个模块或所有具名缺口都不构成全图完成证明。

下一处：实际 sampler processors/RNG、random block verifier 与 fallback guards；随后给 reject-count/下轮 host-device 大交付联合定价，并继续 last-layer attention/FFN producer。全图继续，不围绕低值窄尾段反复补小证据。
