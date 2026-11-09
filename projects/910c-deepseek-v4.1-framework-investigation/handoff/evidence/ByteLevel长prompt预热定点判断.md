# ByteLevel长prompt预热定点判断

## 介绍

先追完整token到字节交付，再选择有首token预算的初始解码工作深入。此点只改变CPU DecodeStream初始seed，不改变模型/NPU算术。每替换原语按逐步字节/None/异常一致验收，不用拼接文本或容差替代。

## 到了哪里

实际安装tokenizers0.22.2，binary SHA `fa049ce975669d8a90fb48960f412e626fa54cf596c2f75d6820949f4888e910`，tokenizer.json SHA `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`。候选先调用原DecodeStream(ids=P)保留全量u32转换/错误；不可变ByteLevel的最后有效单token decode以ASCII结尾时，再建空seed，其它情况保留原stream。原构造复制仍存在，不能认领它的4.392ms历史预算。

上游[ByteLevel字节拼接](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/pre_tokenizers/byte_level.rs#L154)与[stream状态机](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/tokenizer/mod.rs#L1084)用于受限边界论证；tag不证明安装binary来源，故另对实际binary点证。ASCII结束保证prefix不会被未来UTF8补全改写；未完成UTF8不能盲删。完整论证见[proof](bytelevel-prompt-seed-proof.md)。

native：129280个词表ID×两种skip标志，每个先作首续token再追加ASCII/UTF8序列，另有多步随机、特殊/未知ID、非法u32/type、9类seed边界和8份实际输入重建。共259088序列/1302780步，差异0；原构造错误OverflowError/TypeError相同。盲删UTF8 seed的第二/三步确有差异，反例保留，不放宽guard。

原vLLM源AST直接执行Incremental/Base/Fast/check_stop_strings（没有导入vLLM/torch），576场景/2976次update差异0：skip/spacing、include stop、stop顺序/min_tokens、delta/cumulative、1/5token组和finished末token截断/holdback。保留原methods，只替换初始stream；这不是完整HTTP/SSE服务回归。

输入域限定：本原型guard只测试list类型prompt IDs。prompt_embeds/None等入口未来必须在调用此guard之前回退原stream；不能直接拿原型作全API替换。probe中的“request prompt IDs are a list”只指本冻结纯文本fixture，不是schema排除None的证明。本Loop没有为None声明准入，下一输入caller继续核实。

同profile输入文件SHA7863f18…一致，payload确为str。CPU直接Tokenizer.encode(add_special_tokens=True)重建8份各32768IDs，skip True/False都命中guard；IDs hash保留。尚未证明它们与server renderer逐ID相同，不能以长度或输入文件相同替代。没有捕获模型输出IDs，不把合成续token说成真实生成。

## 结果

**已确认可实现（限定上述CPU原语/原wrapper fixture）；接入与请求净收益待确认。** 9对交错配对，包含双方原构造、candidate native guard和新增空stream，8份局部中位合计101.113973→4.124833ms，净96.989140ms。CPU过程/机器负载与已有profile不同，不能把这个净差加到或替换已有84.266439ms，也不宣称模型/服务逐位或实测TTFT提升。边界论证不是有限fixture计数可以替代的全序列证明。

同负载profile给该native8整段的宽松固定schedule信用最高84.266439ms / P90 5.864800% /整批0.482047%。放松request归属和实际暴露，非紧因果上界。假定rho=.5且每seed新增.1ms，条件净41.333220ms，约P90 2.877%；若暴露为0则付新成本、负收益。以每seed .1ms计，要达到此窗P90 1%（14.368170ms）需rho≥0.180003，仍待实际暴露/队列和renderer接线。不满足则停止该请求预算，不能靠重复小格式化方案凑数。

待确认缺口：actual renderer IDs/flags和backend module/immutable tokenizer资格；真实output/stop/error到SSE wire bytes完整回归；CPU关键暴露、guard/validation归属及六ordinary父区间。此实现不替换任何NPU原语，不给全模型位一致准入，也不改变已有BF16失败配方结论。先继续输入renderer/编码的466ms实际producer/queue，再四warmup helper/startup MM；不为不改变取舍的边界重复采样。

本点3次CPU tokenizer词表加载（前两次因host/container路径与错误报告失败，无有效结果）；成功Run一次。模型权重/NPU/服务请求/新profile/远端文件写入/服务改动均0。输入从host校验后经stdin送隔离容器CPU过程；没有复制凭据或改其它环境。证据`bytelevel-prompt-seed-point.json`和`tools/probe_bytelevel_prompt_seed.py`。
