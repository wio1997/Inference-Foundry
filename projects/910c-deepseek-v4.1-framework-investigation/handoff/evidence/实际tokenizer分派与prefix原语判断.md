# 实际tokenizer分派与prefix原语判断

## 介绍

先从C8完整输入交付和466ms实际native编码找重复工作，再做稳定边界ID共享；随后沿显式mode核对真实类，而不是用通用HF类代替。每个替换原语逐位一致。

## 到了哪里

serve.sh SHA c2972011…明确tokenizer-mode=deepseek_v41、未给renderer-num-workers。tokenizer registry→DeepseekV41Tokenizer.from_pretrained→TokenizersBackend.from_pretrained→V41 factory→cached wrapper。V41 wrapper只覆写apply_chat_template、num_special_tokens_to_add、get_added_vocab、__reduce__，不覆写scalar __call__/encode/decode。renderer registry却指向DeepseekV4Renderer，其父类是BaseRenderer；没有HfRenderer的maybe_make_thread_pool。chat格式化不在固定completion请求路径；其模块只有stdlib进口，不能记chat内部已扫。

**更正**：撤回初稿“实际服务已有独立tokenizer副本池，调renderer线程数即安全并行”的解释。池只在通用HfRenderer存在；此前pool回放只证明可选机制，不能证明本mode已生效。实际unpooled CachedDSV41TokenizersBackend＋原renderer方法重放8份IDs，逐hash等于原32768 native IDs。实际wrapper/无pool路径已核，但使用直接配置的backend constructor，未执行完整from_pretrained bootstrap，也未捕获在线服务对象/wire。21进程initial-exec环境未设VLLM_USE_FASTOKENS，当前source默认0且profile确为native tokenizers调用；不是加载中Python flag/object内存证明。

已有trace：8任务2.473533ms提交、worker30599串行，native466.209239ms、父方法487.110933ms、最后input492.603961ms。原始事件父子已去重。累计queue-wait不能加作收益；不把后台工作当eventloop阻塞。

## 结果

**已确认可实现（限定给定prefix的CPU原语）**：固定pipeline有117067字符/29385 IDs的安全三位数字cut，不跨AddedToken；原语1256cases/269698 IDs零差异，包含24fallback、cap1/17/33793、Unicode/special/错误及list ownership控制。已知prefix来自固定workload公共前缀117286字符；它是可行性点证，不是在线prefix发现实现。证明/输入域见[边界论证](shared-prefix-tokenizer-proof.md)。

第一点局部配对old8中位759.538242ms、warm79.447431ms，独立构建prefix129.943227ms，cold合计209.390658ms。深入actual消费者后可从首次完整Encoding和已有IDs直接提取，不再重编码prefix：8次字符位置提取完全等价；增量插入中位.182108ms/max.261111ms。第二组独立CPU配对old8 781.565971ms，保留首个完整miss＋7hits为166.072773ms。两组CPUcohort不同，各自9对原件保存，不能互相做前后对、不能拿615ms局部净差替换旧profile466ms或当TTFT省时。

**待确认，值得开发：完整在线prefix cache交付**。缺一般性prefix发现/插入、字节LRU、代次/实际backend fingerprint失效、全bootstrap与服务wire、真实CPU资源/关键暴露及六ordinary资格。既有首次Encoding可以作为producer，cache只存不可变ID tuple和prefix key，每请求独立list；原text/arrival/cache_salt、validation和所有EngineCore/Engram/H2D消费者保留。不认领新的H2D/模型工作删除。

**有成本请求敏感性**：旧FIFO保留submit/其它父work/gap。冷miss保留第一job全部native，只对后7hits假设删50%，每job付.5ms另首次插入.182108ms，则最后input提前199.150528ms；first input延迟.682108ms。若下游lag不变、未知request配对枚举，P90敏感性−.682108到144.505989ms；既非因果界、预计值也非实测收益。warm8hit同条件最后input229.104620ms。真实engine admission/batching/CPU-GPU overlap未知，不能把最后input提前改名为整批收益，也不能叠加并行线程/旧seed/metadata预算。

**已停止**：本mode仅增线程、依赖不存在自动副本池的配方；裸共享可变HF backend会竞争truncation/padding/split-special state。显式独立backend/调度重写仍待确认，与prefix共享用同一466ms预算，不相加。任意commonprefix拼接、错误UTF8-byte cut也停止；实际BPE/字符offset反例已留证。

下一段继续kernel_warmup/workspace/helper内部与实际MM预算；Loop027只记录启动前置mode守卫，未展开helper，不当已扫。全链路未扫完，原native/ordinary/cache物理与逐位门槛继续保留。

证据：input-renderer-{frontier,extra-dependencies,ID-replay}.json、actual-tokenizer-{dispatch-frontier,execution-identity,runtime-flags,unpooled-replay}.json、shared-prefix-tokenizer-point.json、prefix-encoding-handoff-point.json。成功原语过程无Torch/NPU进口，成功scalar replay只importTorch CPU；被拒AutoTokenizer尝试含模型/NPU包进口，未观测其设备初始化，不当无NPU初始化证明。所有进程未请求模型/设备计算、未发推理/新profile/改服务。已核owned cache路径不存在；源/库只读，无远端文件写入。Task失败/更正原件均保留。
