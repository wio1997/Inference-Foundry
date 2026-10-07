# Replay coverage blocked by logical versus physical token capacity

**Applicability corrected after reading original Run249 startup evidence:** the three reproduced blockers below require SP alignment, which is disabled in this actual run. See [actual source/log adjudication](GRAPH_APPLICABILITY_CORRECTION.md). Preserve the reproduction as conditional branch evidence; aligned16/fifth-request patch design is stopped and is not a current candidate.

只读实际源码 + CPU AST 函数复现；没有 graph flag 修改、模型请求或第二个活动性能 Run。H6 MC2 predicate cache 仍是唯一 active candidate。此问题在现场 enforce_eager/NONE 下不执行，因此不是 Run249 的额外耗时事件；它解释了为何直接开启已有 FULL replay 不能消除已定位的 eager producer。

[Reproduction](reproduce_graph_alignment.py) 执行保存的实际 CUDAGraphMode、BatchDescriptor、dispatcher sizing/key/dispatch 函数，以及实际 Ascend default bound 和 TP-size filter。两套 separate modes 与 maxseq8 control 是离线函数反例，不是 NPU 参数扫描。原字节/source identity 在 [result](graph_alignment_reproduction.json)、graph_sources/alignment_source_identity.json 和 descriptor_source_snippets.json。

当前合同 DP1/TP16、native MoE SP、maxseq4、MTP K1 的 uniform query length=2。Ascend runner3070–3077将实际 decode tokens 2/4/6/8均补为physical16。真实 `enable_sp` 取 parallel_config.use_sequence_parallel_moe，不能把 flashcomm1=false 当作 SP=false。

1. Ascend platform1418–1442 默认 capture bound 是 min(maxseq × uniform_len,512)=8。platform1182–1196随后用 vLLM update_sizes_for_sequence_parallelism1705过滤非TP16尺寸。因此默认1..8没有一个合法尺寸，触发空尺寸断言。必须先处理physical capacity的边界。
2. 即使显式提供aligned sizes16/32，upstream dispatcher209–222的separate FULL key过滤仍使用logical max=8；FULL_DECODE_ONLY得0个FULL keys、dispatch(16,uniform=True)返回NONE。FULL_AND_PIECEWISE同样0个FULL keys，只返回PIECEWISE。maxseq8的control则有FULL key16并返回FULL。Resolver1390–1410会把SFA UNIFORM_BATCH支持下的FULL mixed请求降为separate mode，不能用FULL mixed避开该逻辑。没有找到Ascend patch替换此dispatcher函数。
3. 单独放宽上述过滤还不够。upstream capture warmup/capture caller6935–6962把desc.num_tokens=16、uniform=True传到Ascend `_dummy_run`3622ff。该函数3663–3676以min(maxseq,ceil(16/2))=4构建[2,2,2,2]，sum=8，在原sum==16断言失败。实际AST scalar branch和两原断言复现该失败，未调用Torch/NPU。

Descriptor144已经cap num_reqs到maxseq4，故“descriptor变8导致DCP buffer5溢出”解释已否定。后续query-start padding922–960在physical16 != padded_reqs4×2时走另一分支；真实num_reqs1–3与descriptor4还可能触发num_reqs相等断言。修复必须区分模型有效query、graph request capacity与SP physical buffer capacity，并保留虚拟padding、TND尾长度、DCP local-seq buffer和MTP的动态语义；不能只删assert、把maxseq加大或写死请求。

最小修复边界至少含aligned默认容量、separate FULL key覆盖和capture dummy的logical/physical分离，另须核查动态query-start/DCP metadata消费者。当前只证实stock source入口阻塞，没有设计/激活未验证graph patch，也没有量化replay收益。后续继续亲读消费者形成可证伪代码假说；不因此打断H6 correctness→matched A/B→自然EOS E2E。

独立Challenger已亲读上述函数及现有spec-size/LCM修正，接受三处阻塞。Sol接受其“两个容量轴必须共同修复”的结论；拒绝把dummy末尾新增长度8请求或提升maxseq8当作已正确方案。当前LCM16等于max(2,16)，runner的`lcm > max`特例不进入；编译size rounding又不能突破default max8。下一源码问题限定为真实1–4请求进入physical16时各metadata消费者，仍不构成第二个活动实验。
