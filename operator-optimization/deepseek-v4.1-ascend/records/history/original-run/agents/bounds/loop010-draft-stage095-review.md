# Loop 010：阶段 095 后草稿上限核查（可恢复）

## 当前结论

仅凭已验证冻结源码及启动 K=5，**尚未证明**运行服务所有请求的 `scheduled_spec_decode_tokens` 都不超过 5；由此也不能作为实际吞吐的无条件上界。已验证的结论仍是：假如某次 metadata 中每个请求的 draft 长度 ≤5 且运行该 sampler，则该次每序列采样结果最多 6 个槽，进一步过滤可能更少（见 `stage-091-sampler-implementation-review.md` 和 `stage-095-metadata-output-review.md`）。这不是跨实现域的普遍模型约束。

## 已做的入口审查

以下以 `metadata-path-snapshot-status.json` 记录双容器镜像及相应源码 SHA 一致的阶段 095 静态快照为准：

- `metadata-path-snapshot/vllm_scheduler.py:593,812-827`：调度映射初始化为空；普通分支的输入是 `request.spec_token_ids`，切片只能缩短这个已有列表，调度额度本身没有在此显式截到 5。因此该分支是否有 5 的上限依赖列表来源。
- 同文件 `:1046,1268-1270`：满足特定条件的预填补位分支创建 `[-1] * self.num_spec_tokens`；若已初始化为 5，它的长度恰为 5。它并不是所有情况下唯一入口。
- 同文件 `:2404-2424`：`update_draft_token_ids` 对调用方传入的 draft 列表调用 `structured_output_manager.validate_tokens` 再存入 request；这里既没有局部显式 5 截断，也不能仅凭方法名断言 validator 不增加长度。
- 同文件 `:2425-2459`：异步更新先将 draft 裁到原 placeholder 长度，然后经过 validator，再用 -1 补回原长度。要严格证明输出长度不超过原占位符长度，还需 validator 不扩张列表。不能单凭补位逻辑忽略这个前提。
- 同文件 `:272-277,1392-1396`：若 `num_speculative_tokens_per_batch_size` 存在则通过表决定当批 K，这处使用表项没有再次截断；构表合法性尚未由双容器快照证明。现有冻结启动配置显式 K=5，日志给出 num_spec_tokens=5；不足以证明运行时其他入口、动态配置及 monkeypatch 全部不存在。
- `metadata-path-snapshot/ascend_runner.py:2205-2225`：`take_draft_token_ids` 调用基类后，若该 drafter 有 `dynamic_spec.num_verify_tokens`，其每个 K 夹在 `[0,self.num_spec_tokens]` 后逐列表切片；没有此对象时直接返回，不可把它当作所有路径的统一限幅器。
- `metadata-path-snapshot/vllm_runner.py:4898-4902,4941-4955`：基类输出从 `_draft_token_ids` 转成列表；tensor 路径依 tensor 实际第二维，list 路径直接返回列表。因此若后续只证明某一 DSpark tensor 输出宽 5，还要证明实际无其他 draft 列表的输入路径。

## DSpark 待双容器核对的线索

本机 `sources/vllm-workspace/` 里的 Ascend 参考源码显示，`spec_decode/dspark_proposer.py:39-45` 依据初始化 K 分配宽 `1+K` 缓冲，`spec_decode/llm_base_proposer.py:1041-1055,1552-1605,1637-1644` 在 K=0 时返回 0 列，否则从缓冲写入并在 DSpark 的 parallel drafting 分支返回 `[:,1:]`。若相关快照与运行服务一致，且初始化 K=5、返回路径确实为该分支，则非零情况下返回宽 5。细节：这段 slice 取的是预分配缓冲的所有剩余列，不能将运行时 K 缩小自动等同于返回 tensor 宽度减小。K 超过预分配宽度的代码行为也不能靠注释中的“不会超过”证明不可发生。当前这些参考源**尚未双容器哈希核验**，只作待证线索。

## 客户端口径

本地 `benchmark-20261028/raw-results/measure1.py` 显示 8 请求、stream、输出上限 1024。`evidence/fixed-output-conditional-bound.json` 保存两次 JSONL 统计的 `output_tokens=8192`；检查第一条记录可见 `success=True, input_tokens=32768, output_tokens=1024`。这只说明客户端结果文件记录了这些数字，不能因此认定输出 token 是服务端 `EngineCoreOutput.new_token_ids` 的逐事件和，或这些事件全部落在指定的性能计时窗。先前引用的 AISBench 计数和前端源码路径本次未在当前本机资料中找到，真实 token 统计可能是服务端 usage、tokenizer 重算或其他方式，**UNKNOWN**。网络 chunk 数也不能当事件数。

## 最小下一步

已准备只读脚本 `compare_draft_constraints_readonly.py`，由主控交 ZCode 检查 baseline/stage 相同镜像及两个容器中动态 K 构表、validator、spec config、Ascend base/DSpark proposer 五份文件的绝对路径与 SHA；缺文件、路径异常、镜像或哈希不一致即停。源码快照和状态只写 `agents/bounds/`，不会操作 NPU、改容器或输出环境与凭据。验证结束后才分析动态 K/validator 约束；任何外部 benchmark/frontend 的源码须另行找到及比对实际版本，不能以本机旧参考代替。

即便证明每请求每步最多 6 槽，事件数界还要求每事件序列数、DP 归属、客户端计数映射及严格相邻事件时间下界 δ；一次 trace 的最短间隔不是硬件物理下界。没有 δ 就没有可用的数字 TPS 物理上界。
