# D941：PD与SSE通过、完整质量超时及C1剩余路径

固定daily A2镜像7da963869ee5、910B1/aarch64、CANN9.1.0/torch2.10/torch_npu2.10post6，权重/config/tokenizer、TP8/EP8/PP1/CP1/DCP1、DSPARK root+5、BF16 KV/FP4 index、prefix/overlap、P8192/.88/eager、D1024/.80/full graph不变。当前[权威状态](../metadata/current_pair_P832_D941.json)仍未通过完整质量、四档SG AISBench、最终匹配P/D profile或端到端收益。

## 实际装载、图与生成

修复036的complex64 frequency gather后，八个target各1384参数、八个draft各73参数的native严格coverage全部passed，missing为空。八rank均实际完成target batch[4,8,12,16]/width6与draft batch[8,16]/width5完整捕获，证据在[实际装载与完整capture日志](../evidence/D941_readiness_G962/G957_functional_profile_2204_parsed.json)。捕获和Python路径记录不能单独代替设备重放或质量。

[父级逐raw/SHA检查](../evidence/D941_readiness_G962/G957_parent_functional_and_mechanism_assessment.json)确认同native基线的两个自然语言固定答案4/Paris，无reasoning输出；原始请求响应位于[限定功能结果](../evidence/D941_readiness_G962/raw_functional/result.json)。标准SSE实际18输入/1024输出且DONE，未使用流事件数估token。全部20个功能文件的原始SHA/公开变换SHA保留在provenance。19:22:01UTC该CPU功能进程实际完成，既有P/router保持。

[canonical guard移交](../evidence/D941_readiness_G962/G944_P832_D941_GUARD_ACTUAL_TRANSFER_JSON_parsed.json)先确认旧929实际退出，再由944取得同一DS workload flock，绑定实际P832/D941容器。当前PID2195941及fd实际核验；不是靠state文件假定锁有效。未访问GLM设备。

## 实际设备机制：C2已命中，C1仍有大位置表

仅在上述短SSE上开启两step D诊断，自动停止并解析；完整质量尚未运行，故没有将profile开销掺进本轮质量测试。八rank均有11019条实际compute、正确device编号，实际paged FP4 kernel和selected-K mapper各6次。原始1011个文件共467854108字节在SDK结束后冻结，manifest SHA93c028b03e47da4fe2f782d4bd997fff0505db90dfea2c56672f44c97d2d8b38，见[实际DB/CSV与冻结索引](../evidence/D941_readiness_G962/G957_functional_profile_2204_parsed.json)。大型trace留受控存储。

Python命中记录对应C2 FullTopK层2/8/14；设备kernel验证它们实际执行，但同一窗口各rank仍有10次旧nonpaged FP4和10次24×263435的FloorDiv。C1 DenseBlocksBackend的publish_decode/consume_decode直接调用native scoring.decode_scores，绕过FullTopK类覆盖；因此不能宣称大位置表或AI_CPU Index已全局消失，也不能从不同batch/提示长度的短profile推导端到端收益。

[当前固定镜像源码导出](../evidence/D941_readiness_G962/G960_C1_SOURCE_AND_QUALITY_JSON_parsed.json)与原归档逐字节SHA核验，复用[唯一来源索引](../metadata/C1_source_authority_G960.json)。C1原生块发布、最新块必选、块top-k与块内top-k必须保留，不能为了分页跳到Hopper专用另一套选择算法。

## 聊天编码限定核验

对旧SG格式失败的GSM8K row22，实际P832的/tokenize与同native0.32不可变镜像CPU get_tokenizer(tokenizer_mode=deepseek_v41)输出80个ID完全相同，ID SHA43619049fd6ce70edfe55e4f7f98f4e8d28d0edc3467a2455cbb0177b966edcc。[父级检查](../evidence/D941_readiness_G962/G955_parent_chat_tokenize_contract.json)确认无NPU设备、NPU initialized=false。初次使用auto模式误入无chat_template的HF backend失败已保留，随后按照实际native argv的deepseek_v41模式核验。只证明该例编码相同，不能当所有1319题或生成数学相同，也不能将旧输出格式差异归咎于该例提示ID不同。

## 同一完整质量集实际启动

确认功能CPU进程结束、SDK后台处理结束、两端队列为零，实际新pair/8卡owner/CPU客户端镜像与剩余lease/source/dataset/frozen-profile均通过门后，[完整1319题实际启动](../evidence/D941_readiness_G962/G958_MATCHED_SG941_FULL_QUALITY_START_JSON_parsed.json)，UTC19:31:05，outer PID2565945。复用官方冻结题目、严格####grader、同提示、T0/seed0/thinking=false、max2048、C16、单请求600秒和无重试，inner hard bound8070秒。新run不续跑或覆盖旧832的20行/14超时失败。

19:33:13UTC的[实际进程/原始行观察](../evidence/D941_readiness_G962/G960_C1_SOURCE_AND_QUALITY_JSON_parsed.json)仅完成2条、2正确、0传输失败，不能据此报告全量准确率。后续完整尝试已实际失败，见下节。native0.32的1270/1319是已有另一框架证据，不替代SG结果。

## 实际质量终态：14个600秒超时，已停止并排空

19:42:32UTC观察到12个实际600秒HTTP超时；随后[精确停止](../evidence/quality958_terminal_G966/G964_SG941_FAILED_QUALITY_STOP_JSON_parsed.json)本任务CPU客户端内的Python PID2565981，19:43:27UTC确认实际退出，阻止新提交。终态为22行完成记录、3个严格正确、14个传输失败、38个请求文件、8个响应文件。它不是1319题准确率，完整质量未通过。没有放宽grader、延长SG专用请求预算或覆盖旧832失败。

[全部原始请求/响应/日志与SHA](../evidence/quality958_terminal_G966/raw/sglang941_full_test/results.jsonl)保留；原脚本state仍停在full_test_running是历史字段，parent_terminal_assessment与实际PID退出为终态依据。[停止后实际核对](../evidence/quality958_terminal_G966/G965_FAILED_QUALITY_RAW_AND_DRAIN_JSON_parsed.json)确认CPU客户端仅sleep，P/D各两次running/queue计数均为零；模型和canonical guard保留，未释放NPU给另一个作业。C1大位置表是已观察到的剩余机制，不把它直接当所有超时的唯一已证根因；完整端到端收益仍未验证。

## 未完成项与下一个同机制候选

[C1候选961](../code/C1_common_prepared_G961/mechanism_plan.json)仅完成本地源码/AST准备：共享NPU decode_scores保留原DecodeScores消费者分支与完整同容量分数，以req映射替代完整int64 slot表；native C1块策略与torch排序不改，write_decode只映射选中的K。保持导入别名的function identity，并在原freqs_install之后安装，避免重复遗漏既有频率适配。它未在NPU验证、未部署，不改变当前质量运行。

后续必须验证actual msgspec返回契约、native块IDs/分数bits/物理slot/padding、空/奇数/ragged/真实长容量、变输入NPU图与8rank，再做新模型与端到端测量。本轮完整质量失败且未验收；SG四档32K/64K/128K/256K、C16/O1024的五轮独立warm、首次cold/replay及raw/usage/cache/compile链、最终与native相同16请求cohort的P8+D8 profile、归一化差异地图和收益判断均保留为未完成。

[后续C1八卡验证与D974](038-C1-native-allrank-gate-and-D974.md)：D941现已受控退休，历史失败及其原始证据保留。
