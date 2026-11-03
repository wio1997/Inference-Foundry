本轮以M16 LM head与词表重建为入口，先查询全部32份保存native fullcaller/typed TASK和完整COMM。12条原生链共1080 API、1080 TASK，包含88个target base GEMM、104个draft base GEMM、520个replicated Markov GEMM、104个draft head norm、192个Transpose以及72个hcom ComputeInfo。完整COMM另有192次TP8 BF16词表AllGather：target88、draft proposal88、dummy ancestry16。层级不同，不能把其余120个collective补造成丢失的ComputeInfo或把它们相加为耗时。当前scope包含M20/其他模块角色，不替换v20 M16 certain904/optional176及全局候选归属资源账。

88个target root逐一核对了前一轮Index K、head GEMM的K×5120、COMM的K×16160和Transpose[8,K,16160]→[K,8,16160]，行数全部一致。K分别为2、12、24、14，物理本地词表16160、TP8重建129280。这里是exact root和typed-shape一致性，不是历史tensor alias/值/代际证明。draft/dummy没有借最近时间强配target root，dummy祖先也不等于可删除启动工作。

profile把代码问题收敛到normal head、gather布局和Markov owner。定点源码确认，当前head命中555→598→533→144→82→90的普通未量化F.linear路径，head_dtype为None或匹配输入类型，当前输入/权重/输出均BF16。其他FP32 head_dtype、PCP或fine-grained等分支没有从当前normal调用推成全局不存在。DSpark MarkovHead构造明确disable_tp=True，现有normal实现对复制头跳过gather；因此520个129280×256完整Markov GEMM和base的16160×5120分片GEMM是不同所有权，不能只看到同一个_apply_head函数就合并评价。

已有skip_gather=True可以返回本地、未截断的head结果，这是已存在接口，不是本轮新实现，也不能直接把该结果交给旧full-vocab sampler。正常路径先gather、按org_vocab_size截断，再按原dtype顺序做可选softcap和scale。任何局部分区/候选归约都必须保留相同的有效列、额外词表/映射、后处理和所有score消费者，不能在softcap/舍入之前比较，也不能把padding在统计里当有效词。

192个Transpose的来源也已回答：底层AllGather先按rank-major组织A[p,m,v]，之后movedim/reshape得到B[m,p*Vp+v]。M>1时，跨分片列边界的物理跳距不等于1，因此不能把rank-major buffer直接flatten成同样的二维logits。条件机会有两类：让新消费者显式读3D/分区布局，按需生成真正被请求的二维score；或让新的通信/pack接口直接写消费者布局。全score消费者、capture地址/stride和最后读者仍需保留，不能把192条Transpose全部宣布可删。

补查当前Markov后处理时，首份source-site band从1590开始，正确记录2600 TASK；保留这份pass，再只补已review过的1589 Add site，得到2080 API、4160 TASK。具体为1560 Slice、520 Add、1040 ViewCopy、520 ArgMax和520 Cast。其中440个proposal步骤、80个dummy步骤。当前Add已明确BF16+BF16→BF16，ArgMax输出INT32再转INT64；stage proposer SHA与v9已查源码一致，故复用其strided/inplace、token-remap、稳定ID bank、递归和confidence合同。ViewCopy显示全base storage的shape不代表所有逻辑字节都被物理拷贝。五步数量比与K5一致，但未按数量或时间伪造每次Python调用配对。

分区选择的数学条件也写入记录：在同一有效列、同一处理后/舍入后score与同一比较规则下，全局best等于各分片best再归约。有限数且“先出现的baseline列”规则下，可用(score,-baseline_column)作比较；这只是声明域内恒等式，没有证明当前CANN的所有NaN/tie策略。tie必须先按baseline列决定，再做可选token映射；按映射后的token ID最小值选tie会变结果。纯Python反例同时说明：绕过BF16 head输出舍入会改变winner，把base/Markov两次舍入后相加改成一次联合舍入也会改变值。这不是NPU数值测试。

F35新增三项条件变体：直接消费rank-major布局；本地处理后top1+全局ID归约；完整score只重建到指定owner并发布相同结果/状态。前两项不能独自满足F35 v28已确认的bonus强制raw scores、target raw/entropy、probability和logprobs；F26 v31的prompt reader也仍保留。需要完整分布不等于所有跨卡方案都不可能，但它需要单独的分布/owner协议，不能把local top1当作答案。

F30已有VOCAB-OWNERSHIP进一步拆成两种替代：Markov权重按base词表分片，每个依赖步骤全局选ID后再进入下一步；或在owner上执行完整Markov循环，发布同样的proposal IDs和所有必要状态/概率。前者增加逐步小collective，后者保留owner完整GEMM并承担所有非owner接口、RNG/metadata/score观察者同步。两者不能加算收益。Markov GEMM+per-head后处理+base BF16 Add+argmax可以研究联合kernel，但不能简单把两个输入维拼成一次GEMM、跳过各自输出舍入或破坏base K行批量复用。F26补norm/head联合边界条件，继续保留BF16 normalized输入和所有full-hidden/prompt/aux读者。

上述都细化既有F26/F30/F35，F35 v28本已声明与F30 ownership重叠；不新增F36，主候选仍35、框架7。新增20项共享条件。本轮5次远端CPU只读作业全部pass，状态/索引/恢复包更新；没有模型/torch导入、NPU负载、服务或新健康检查，最后健康证据仍v12post。整体26模块仍open、26项理论比较仍INCOMPLETE，没有实现或提速结论。

下一批查M17剩余的probability/entropy/rejection内容消费者，先用保存fullcaller、typed primitives和实际记录的guard。复用v28 raw/bonus/greedy和v31 prompt已答结论，只追ori_target_probs/target_probs/draft_probs的确切下一消费者及必要控制/调用块，明确哪些分布信息和状态必须保留，不整段读取全部305行或扫描SDK。
