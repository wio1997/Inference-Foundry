# 同机32K首轮正式cold/hot与待定位差异

固定G181 SGLang服务在原先vLLM G4使用的同一对主机上，完成充分预热后的首轮cold32和hot32。每阶段32个请求、C16、实际32768输入/1024输出全部通过；与G4正式输入文件的SHA相同。末两遍完整预热无新D编译、timing变化均小于5%；P从首遍后到两次正式阶段结束的331项缓存大小与mtime保持不变。两次正式阶段均无新D编译，cold的P/D native命中计数增量均为0。

|阶段|TTFT均值(ms)|TPOT均值(ms)|输入吞吐(token/s)|输出吞吐(token/s)|总吞吐(token/s)|
|---|---:|---:|---:|---:|---:|
|SGLang cold|65703.6|22.6|4785.7341|149.5542|4935.2882|
|SGLang hot|66002.5|22.7|4741.6241|148.1758|4889.7999|
|vLLM cold|41576.4|22.1|6658.9341|208.0917|6867.0258|
|vLLM hot|40699.7|22.7|6667.482|208.3588|6875.8408|

本轮cold输出吞吐低28.13%、TTFT高58.03%；hot输出吞吐低28.88%、TTFT高62.17%。这是同机、相同输入、独立服务epoch的首次测量差异，尚无重复A/A或匹配profile根因证明，不将其解释为架构定论。现象优先指向首token之前的计算、排队和PD路径调查；尚不能区分这些候选原因。

SG hot的P/D cached_tokens_total均无增量；vLLM hot的D prefix_cache_hits_total增加1024000。两个框架counter的请求路径更新口径尚需源码及原始请求证据核对，不能直接把这个计数差作为已证明的cache原因。两端prefix开关保持启用，不把hot标签当作真实命中。

- [同机正式验收、比较数值及边界](../evidence/matched_SG_G181/first_32K_formal/G241_SG_first_32K_acceptance.json)
- [cold原始审计](../evidence/matched_SG_G181/first_32K_formal/run/cold_32768_audit.json)
- [hot原始审计](../evidence/matched_SG_G181/first_32K_formal/run/hot_32768_audit.json)
- [P阶段结束后的独立缓存检查](../evidence/matched_SG_G181/first_32K_formal/G239_Pformal_profile_stage_2201_parsed.json)
- [D cold编译检查](../evidence/matched_SG_G181/first_32K_formal/run/cold_32768_compile_audit.json)
- [全部Run索引，含原始JSON/CSV/无损JSONL和SQLite](../records/runs.json)

SG的P/D max-running-requests均16；G4 vLLM P/D max-num-seqs为64/32，两端客户端C16相同，此配置差异保留在完整启动证据中。完整上下文263296、TP8/EP8/PP1、BF16 KV、MTP3和D目标/草稿完整图不变。64K/128K/256K同机SG矩阵、匹配P/D profile、源码机制和因果对比地图仍待完成。
