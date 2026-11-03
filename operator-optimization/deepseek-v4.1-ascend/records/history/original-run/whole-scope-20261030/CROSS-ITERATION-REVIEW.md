# 验证结果、下一轮输入与草稿缓冲：局部状态图

**状态：源码局部状态边已展开；实际事件、全体消费者与整体覆盖仍OPEN。** [机器可读状态边](cross-iteration-state-ledger.json)给每条边附冻结文件SHA和包含它的函数站点ID。它不是完整请求关键路径，也不将历史G1未展开的NEXT_INVOCATION边视为已消除。

## 实际源码链

`NPUModelRunner.propose_draft_token_ids` 的padded路径从拒绝结果[R,K+1]构造next_token和valid_count，然后将valid_count交给异步host复制和后续输入准备。`prepare_inputs_padded`根据cu_num_draft_tokens、valid_count、query_start_loc得到rejected_count和token_indices_to_sample；普通Triton路径已经把这些索引/拒绝计数合在一个kernel，不能再登记为新融合。

`prepare_next_token_ids_padded`当前还逐项做clone、discard行填-1、有效mask、按行sum、count-1与clamp、gather和where；CPU同时为每个请求读历史最后token并上传backup。这里的selected位置是**有效数量减1**，依赖输出有效token形成前缀的合同；不能擅自改成任意孔洞下的“最后一个有效位置”。原函数的条件与边界语义都要保留。

DSpark随后复制next_token到持久seed buffer，复制target context hidden，生成各cache group的query/context槽位，并将有效seq_len减去拒绝数再加本轮query数。draft_buffer第0列来自seed，其余列由逐位置Markov循环写入；返回[:,1:]是该持久buffer的view。外层赋给`self._draft_token_ids`后，host复制和下一轮输入都可能读取它。不能因为函数返回就允许下一轮覆盖，也不能把所有token/metadata缓冲视为一个相同寿命的“临时tensor”。

## 双权威状态与实际同步

Ascend自己的 `_copy_valid_sampled_token_count` 使用NPU side stream：wait当前stream→nonblocking D2H→record event；async spec时还保存同一device count tensor，prev_sampled_token_ids是next_token.unsqueeze(1)。不能误用上游CUDA同名函数来解释当前路径。

下一轮 `_prepare_inputs` 的async分支用上一轮device count、prev_positions、prev_num_draft_tokens和请求重排来修正num_computed_tokens；positions、seq_lens和block槽位继续由修正结果派生。CPU也可保留optimistic长度镜像；`_needs_seq_lens_cpu_sync && async_spec_decode_active`时 `_correct_optimistic_seq_lens_cpu`先event.synchronize再读host count。这是可见的同步合同。源码注释称稳态event通常已完成，**该注释不证明历史实测等待为零，也不证明这次sync处于关键路径**。

`_copy_draft_token_ids_to_cpu`在async scheduling且无structured output、无需要output_token_ids的采样消费者、PP=1时直接返回，仍先更新prev_num_spec_tokens。否则它按实际K切片复制并record独立event。故“每轮草稿都必然D2H并等待”和“所有host副本都可删除”均无依据。请求ID列表、步宽、CPU/设备长度、graph固定地址和上轮最后读都分别需要版本/生存期合同。

## WS-NEXT-01：接受结果到seed/count/index的重组

可研究两个重叠方向：在padded next-token准备中融合mask/reduce/gather/select，或让已有greedy拒绝kernel直接交付附加next_token/count，再接现有索引kernel。后者需要discard行、零有效token的backup、无spec请求、每请求K、sentinel及全部输出消费者；不能删仍供客户端/调度使用的完整输出。两方向不叠加计收益，亦与旧F31/F33的metadata方向重叠。

backup CPU读取/上传可否按需或复用已有device token，取决于“全无有效输出”行的语义和请求重排；当前无证据无条件省去。valid_count还分别供device修正与host镜像/调度消费，不能只因存在device权威状态就删除D2H。若拆开复制和状态发布，原early-return guard及静态buffer生存期也须重审。

R约4、K5时tensor很小，启动/调度可能比字节成本更重要；当前没有同调用task绑定、实际host等待或必要资源账。没有给毫秒、TPS或收益比例，也没有将准备时间与上一轮device工作重叠的部分重复相加。此方向为条件候选，未实现、未测收益。

## 尚未覆盖

状态台账仅是本次函数链的消费者子集。engine/scheduler收发与请求退出、所有继承/patch消费者、跨卡参与者和重排、KV具体物理版本、事件及stream实例、graph更新handle和异常回退仍未闭合。M04/M17/M21/M22/M24/M25保持OPEN，整体目标active。
