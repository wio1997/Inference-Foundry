# 输入renderer与公共前缀交付判断

> **2026-11-10 更正：实际mode为deepseek_v41，renderer是DeepseekV4Renderer→BaseRenderer，不经过HfRenderer，自动副本池不存在。撤回本文初稿“固定服务已有独立池、调线程数即安全并行”的解释。通用pool源码/点证只属反事实能力；时间/ID原件保留。以[实际分派更正](实际tokenizer分派与prefix原语判断.md)为准。**

## 介绍

从真实CompletionRequest→OnlineRenderer→BaseRenderer→HF tokenizer→native encode_batch→每请求IDs→engine/cache/模型消费者。先整段重复编码和队列，再拆分融合/共享。每替换原语逐位一致。

## 到了哪里

8入口源＋8个当前依赖已pin，Transformers5.14.1/tokenizers0.22.2。default renderer_num_workers1，Base独立tokenizer executor和单worker MM executor，HfRenderer预建workers+1个deep-copy池。线程安全只承诺public方法，不包括raw _tokenizer或mutation。HF _encode_plus会改backend truncation/padding/encode_special_tokens，不能拿同一个可变backend裸并发。

实际C8：8任务在2.473533ms内提交，一条worker线程30599按FIFO执行；native466.209239ms，完整_tokenize_prompt487.110933ms，最后input父方法在首提交492.603961ms结束。累计queue-wait是多请求重叠等待，绝不相加成请求或batch收益。source py.detach和trace主线程在首native调用中提交其它请求，排除“整段native必然持GIL阻塞eventloop”的解释。上游tag不代表已装binary来源。

build_tok_params源码推导：max_total34816、out1024、truncateNone→HF truncation=True/max_length33793/add_special_tokens=True；没有offset/echo/embeds/MM。先text check/lowercase，后ID截断/填充/长度check及parallel fields同切片；EngineInput保留text/arrival/cache_salt，IDs继续给EngineCore/状态/prefixhash/Engram/H2D原消费者，不因CPU优化自动删H2D或模型工作。token IDs/embeds/encoder-decoder、MM分支按实际guard去向保留；固定纯文本不进入这些producer，不能让新cache接管其状态。

CPU源码重放：原TokenizeParams/Request.build_tok_params/Base方法＋实际generic TokenizersBackend的原convert_added_tokens和cached/pool包装，8份32768IDs逐项等于先前native重建hash。直接constructor避免AutoConfig模型集成，故不声称完整AutoTokenizer class resolution或实际服务IDs捕获。成功进程importTorch CPU但未importtorch_npu，无model/device/service请求或配置改动。前两AutoTokenizer尝试引入模型/NPU包或输出日志，后一次原始AddedToken dict不合constructor，均ERROR/INVALID保留；没有拿它们作准入。首次进口的设备初始化未观测，不用“无显式kernel调用”证明import无设备初始化。

## 结果

**已确认可实现（既有配置能力）**：renderer-num-workers配独立tokenizer池，属于现有调优路径，非新增融合开发；未变更服务、无实测请求收益。裸共享可变backend并发配方停止。CPU affinity/新clone内存/队列公平及engine批次改变仍需实测，不能把调线程数说成已得466ms。

**待确认，值得深入**：稳定公共前缀的pretokenization/BPE结果跨请求共享，再交付独立IDs。cached_tokenizer只缓存属性，未实现完整promptIDs cache。8实际文本全局公共前缀117286字符；本固定pretokenizer第一步按数字1–3位Isolated，存在117067字符处的ASCII三位数字边界且不是added-token proper-prefix后缀。added token的lstrip/rstrip/single_word全0、normalizer空、BPE dropoutNone，有机会证明前后段独立。边界需针对整个special→normalizer→pretokenizer→BPE→postprocessor，不拿任意字符/token cut拼接作准入。

有成本FIFO情景（原submit/其它父work/gap保留）：删除native的50%，每job新增.5ms，最后input提前229.104620ms。假定每请求下游lag不变且未知request配对可选，P90 TTFT敏感性29.271983–174.460081ms；不是因果上下界/预计或实测收益，实际engine admission/batching/CPU-GPU overlap未知。请求batch17480.95198ms；不能把最后input提前229ms当整批已提前1.31%。其它rho/h情景见JSON，不相加旧seed/metadata。

**下一处定点**：验证117067安全cut的原生逐ID等价、错误/fallback、Unicode/added-token反例，支付一次prefix构建和每次lookup/UTF8校验/suffix/ID复制。原型选择的公共前缀是已知workload元数据，在线发现/插入/字节LRU/代次仍独立缺口；点证不得冒充整cache已接入。完成此点后继续四warmup helper内部/startup MM，保留其余native/ordinary物理边界，不以缺口具名宣布扫全。

证据：input-renderer-{frontier,execution-identity,extra-dependencies,ID-replay}.json、tools/analyze_input_renderer_frontier.py、tools/replay_input_renderer_ids.py。完整旧frontend事件与clock沿上轮pin；无新profile/推理。当前只是限定source/CPU replay和价格判断。
