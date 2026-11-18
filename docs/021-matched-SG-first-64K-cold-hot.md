# 同机 SGLang 64K 首轮冷测与热测

G181在同一对910B1主机上，完成五轮充分独立预热后的首轮cold32/hot32。每阶段32请求、C16、实际65536输入/1024输出全部通过。正式输入SHA与vLLM G4/G48完全相同；两次正式阶段D编译缓存无变化，P在D终态确认后经独立SSH采集，371项文件大小/mtime从首遍预热后到两次正式阶段结束始终不变。此顺序依据实际采集请求先后，不比较跨主机未校准时钟。

|阶段|TTFT均值(ms)|TPOT均值(ms)|输入吞吐(token/s)|输出吞吐(token/s)|总吞吐(token/s)|
|---|---:|---:|---:|---:|---:|
|SGLang cold|156178.4|23.0|4568.4375|71.3818|4639.8194|
|SGLang hot|156123.0|23.8|4567.9428|71.3741|4639.3169|
|vLLM cold|106173.8|24.3|6493.2807|101.4575|6594.7382|
|vLLM hot|106286.0|24.3|6481.7426|101.2772|6583.0198|

相对vLLM，SG cold输出吞吐低29.6436%、TTFT高47.0969%；hot输出吞吐低29.5260%、TTFT高46.8895%。TPOT分别低5.35%、2.06%，说明本轮主要差异位于首token之前，但尚不能区分prefill、排队、PD和依赖栈等因素。两个框架本档hot的P/D原生命中计数增量均为0，不能将hot标签视为真实prefix命中或收益。

同机、同输入仍是不同服务epoch的一次首轮测量，无重复A/A；固定镜像依赖CANN9.1/9.2和P/D运行上限SG16/16、vLLM64/32的差异保留。完整上下文263296、TP8/EP8/PP1、BF16 KV、MTP3与D目标/草稿完整图保持已验收状态；未把本轮结果当作优化收益或架构定论。

- [父级验收、比较数学与边界](../evidence/matched_SG_G181/64K/formal/G295_SG_first_64K_acceptance.json)
- [冷测原始审计](../evidence/matched_SG_G181/64K/formal/run/cold_65536_audit.json)
- [热测原始审计](../evidence/matched_SG_G181/64K/formal/run/hot_65536_audit.json)
- [P 两阶段结束后的独立编译缓存](../evidence/matched_SG_G181/64K/formal/G290_P_64K_postformal_2201_parsed.json)
- [五轮预热门槛](019-matched-SG-64K-five-independent-warmups.md)
- [匹配profile的机制差异与优先候选](020-profile-mechanisms-and-priority-candidates.md)
- [全部Run与原始JSON/CSV、无损JSONL、SQLite](../records/runs.json)

128K首轮独立预热已另行启动，配置为首遍后暂停以采集P基线；128K、256K正式矩阵与最终因果地图仍待完成。不得重发64K已完成首轮来替代本轮结果。
