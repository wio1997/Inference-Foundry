# 原生0.32完整PD：真实生成、缓存、计数与未解决问题

当前[权威状态](../metadata/native_G630_readiness.json)仍未通过完整质量/性能验收。两台910B1使用同一原生arm64镜像d5492ecf、TP8/EP8/DP1/PCP1、DSPARK5、P eager、D target与draft FULL_DECODE_ONLY；源码/权重身份沿用008和G628。没有覆盖旧0.30补丁。

## 实际启动与探针

[P启动日志](../evidence/native_readiness_G659/startup/2203/server.log)和[D启动日志](../evidence/native_readiness_G659/startup/2204/server.log)记录各8rank装载、P逻辑容量1291720/D1812610 tokens，D8rank完成target13/draft11图捕获。图捕获不能证明实际全图重放，也不能将其0.41GiB计数与另一框架的图显存按相同口径比较。实际hostdriver26.1.1；D fused_mc2与overlap_shared_expert存在排斥，后者实际关闭。

原探针先把stockproxy健康接口写成/health，得到404且未提交模型请求；正确接口是/healthcheck，修复收集器后复用原服务/代理。随后G644取得9个有usage的原始响应，但129-ID重复输入greedy文本不同，整组测试按原断言失败，后续缓存案例未继续。[父级原始SHA核验与失败](../evidence/native_readiness_G659/G647_parent_functional_failure_assessment.json)保留该结果，不把usage计数通过升级为正确性通过。

## 异常文本与CPU复现

32768-ID输入返回32个生成token ID，却输出84609字符。绕过代理手工PD后仍复现，不能归咎于stockproxy。相同镜像内tokenizers0.23.3/transformers5.19.0的DecodeStream以完整prompt priming，纯CPU生成同样84609字符，首片84553字符；[原始CPU结果](../evidence/native_readiness_G659/CPU_and_semantic_diagnostic/functional/CPU_reproduction.json)与[复现脚本](../code/native_readiness_G659/native_CPU_detokenizer_repro_g650.py)可查。

这与上游[48853](https://github.com/vllm-project/vllm/issues/48853)的prompt中途UTF8边界问题一致，[48854](https://github.com/vllm-project/vllm/pull/48854)提供尚未合并的修复方向。本任务未部署该修复；last8-context只作诊断，不作为已验证修复。129-ID实际生成token不同属于另一项未解决现象，不能用解码器复现解释采样差异。保留原失败，不放宽greedy标准。

## 自然语言与真实缓存

后续10个有固定答案的自然语言检查全部匹配，包含13848-input的长背景算术答案437及原样重放，两次生成token ID一致。P/D独立metrics在热次均增加13824个本地prefix-hit tokens，首次D外部接收13847 tokens。[父级逐响应SHA/答案/计数器核验](../evidence/native_readiness_G659/G652_parent_semantic_cache_assessment.json)限定为语义smoke与缓存功能，不能代替完整质量集。

API未开启enable_prompt_tokens_details，stockproxy将缺少的P缓存详情填0。这个usage字段不能证明没有cachehit，也不能代表D命中；这里以两端独立原始计数器为证据，没有修改服务配置或代理源码。

## 标准流式计数契约与完整质量集

独立新文本前缀已与448个预留前缀核对无冲突；标准SSE返回DONE及精确32768输入、1024输出。D generation计数增加1024/P增加1。真实AISBench handler将server_usage.prompt_tokens/completion_tokens写入input/output字段，[源文件与原始SSE SHA契约](../evidence/native_readiness_G659/usage_contract/usage_contract.json)仅证明当前native630 epoch的协议/计数来源；该请求不是AISBench结果，不能与另一框架单请求延迟计算收益。

完整质量采用[官方GSM8K](https://github.com/openai/grade-school-math)固定commit的1319条held-out test，原始官方严格grader提取####数字并去逗号，无额外宽松匹配；zero-shot、thinking=false、T0/seed0、max2048、C16、无重试。全量原始请求/响应留受控任务目录，[准备来源](../code/quality_G655/dataset_provenance.json)及[客户端](../code/quality_G655/run_full_gsm8k.py)已归档。G656途中实际python进程存活、完成293题；G660确认客户端终止且完整1319题均有原始响应。父级核对全部请求内容、SHA、参数和逐题官方重评分，结果1270/1319（96.2851%），传输失败0、截断0，49道错题均保留，未放宽格式/答案规则。[全量父级验收](../evidence/quality_full_G660/G660_parent_full_quality_acceptance.json)限定该数学集合；SGLang必须复用同一题集、提示、预算与grader后才能比较，不能据此声称通用质量完全通过或W4A8相对FP16无损。

G646的AISBench runner修复真实CPU容器只读dataset host-bind路径，并从实际D进程env解析编译缓存物理位置，保留原先C16/1024/5独立warmups/首coldhot/no-retry/原始数据协议。仅离线准备，尚未执行；质量期间独占P/D负载锁，未提交DS正式性能数据。vLLM该质量集已全量完成；匹配的SGLang质量、双方四档矩阵、profile/因果地图及实际优化收益仍未完成。
