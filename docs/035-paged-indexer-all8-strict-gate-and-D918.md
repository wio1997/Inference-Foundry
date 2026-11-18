# 分页FP4索引器八卡严格门通过，D918实际启动

固定SGLang daily A2镜像7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec、910B1/aarch64、CANN9.1.0、torch2.10/torch_npu2.10post6。权重/config/tokenizer与832相同，TP8/EP8/PP1/CP1/DCP1、DSPARK root+5、BF16 KV、FP4 index、DeepEP、prefix/overlap及P8192/.88/eager、D1024/.80/full graph保持。没有新镜像、量化、上下文或top-k规则。当前[完整模型状态](../metadata/current_pair_P832_D918.json)仍未通过质量、正式四档性能、最终匹配profile或端到端收益。

## 实际组件覆盖

[单卡pilot](../evidence/paged_indexer_G930/G925_SINGLECARD_PAGED_ACCEPTANCE_JSON_parsed.json)12个用例/6次重放后，组件实际退出、八卡清空并释放；[随后八卡启动](../evidence/paged_indexer_G930/G924_ALL8_PAGED_GATE_STARTED_JSON_parsed.json)复用本任务原始compiler cache副本，原cache保留。复用编译产物只减少准备开销，不作为编译稳定或性能收益证据。

[八rank父级验收](../evidence/paged_indexer_G930/G927_ALL8_PAGED_PARENT_ACCEPTANCE_JSON_parsed.json)逐rank确认12个用例和12次变输入图重放，共96+96。每次FP32分数的int32 bits完全相同，512个排序物理slot完全相同，eager用例还确认所有输入未改。ratio1/2覆盖空、奇数、ragged、1/6/16/80/96 query行；实际容量263435与131717均覆盖96行短live长度及单query全可见长上下文。NPU图在96行执行live127→2048→63、单行执行127→完整capacity→63，并变更q、请求表和packed K，均与原stock eager严格相同。

输入域为原生per32 UE8M0 FP4 fake-quant query、BF16 head weights、原生68B/slot packed index K布局。该fixture的packed payload随机、K scale byte固定127；它不是所有量化参数/模型能力的穷举证明。native FP4 tile计算源码本身未改，候选仅改变分页slot访问和selected-K映射；既有原生FP4计算门另有历史覆盖。完整模型真实量化数据与质量仍需后续验收。

034中902的Triton finisher编译失败、912的基线stock TopKV2在96×131073故障全部保留，本次没有宣称131073形状通过。实际两种更大服务容量和原始严格判据均已实测通过，不能抹掉独立边界故障。

## 同一机制与受控启动

候选复用镜像已有paged FP4 tile，避免[B,Lmax]int64位置表和全容量AI_CPU Index；保留同容量-inf分数及原torch full-width top-k/sort，仅映射已选K位置。[完整候选包](../code/full_model_D918/candidate.json)含有效D argv，[源码清单](../code/full_model_D918/deployment_source_sha256.json)与provenance保存原始及公开副本SHA；未被选择的candidate_p/d及旧status文件仅为继承来源，实际入口为candidate.json。原生内核、scoring和FullTopK源码均固定SHA。

父级确认gate924进程实际结束、8卡无进程/health OK、NPU及CPU owner released，核对score/select/mapper三个函数AST与模型包一致后，[实际启动D918](../evidence/paged_indexer_G930/G927_D918_MODEL_ACTUAL_START_JSON_parsed.json)。新包仅替换索引机制及增加每PID安装/路径命中记录；P832和原生启动参数保持。

[实际装载检查](../evidence/paged_indexer_G930/G928_D918_ACTUAL_MODEL_REVIEW_JSON_parsed.json)确认新D controller/容器/镜像与八卡owner，主进程已有安装记录但尚无执行路径命中；此时模型仍装载，不能报告ready、八rank装载、target/draft图或质量成功。[canonical guard移交](../evidence/paged_indexer_G930/G929_P832_D918_GUARD_ACTUAL_TRANSFER_JSON_parsed.json)先确认无生成在飞，旧858实际退出，新929取得同一DS workload flock并绑定实际P832/D918身份。未重启P、未改路由、未访问GLM设备。

## 尚未验证

设备profile必须证明真实模型运行命中paged kernel/selected-K mapper，且大型Index/位置表工作消失；安装或图构造记录不足以证明重放路径。之后仍须两端/ready、原生target/draft严格载入、图捕获与实际生成、原始1319题/同提示/T0/seed0/thinking=false/max2048/C16/严格####grader和无重试。此前20行/14个600秒超时的失败保留，不能续跑旧run或改格式判据。

最终SG32K/64K/128K/256K、C16/O1024的五轮独立充分warm及首次cold/replay、raw/SQLite/usage/cache/compile链，最终双方P8+D8可匹配逻辑工作profile，以及正确性和端到端收益判断均未完成。八卡组件通过不能替代这些要求。

## 后续完整模型实际失败

D918在target图捕获中因新wrapper丢失既有complex64频率gather而失败，controller已实际退出并释放八卡，没有发送PD功能/质量请求。以[036失败栈、修复与D941](036-D918-complex-frequency-wrapper-failure-and-D941.md)及当前metadata为后续状态；本页组件通过不代表完整模型通过。
