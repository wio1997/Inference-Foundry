# 两保存短诊断的采样源码契约

同一2758字节短诊断请求：两栈在所审查的源码路径均选argmax；数字温度/top_k归一化不同不构成已证随机分歧。

已观测：保存payload SHA256 6da29d412ee3d16de364f6596e3c4d41f402971c6b76612f4cba34910324d7dc；512/C1/out32，显式temperature=0、top_p=1、seed=20261637。

已观测：vLLM历史日志默认temperature=1/top_k20/top_p=.95；SG历史server_args为sampling_defaults=model。两栈实际输出文本不一致。

源码与CPU重放：vLLM：top_k20→0、temperature0→0；SamplingParams为GREEDY，原Sampler.sample与Ascend greedy_sample走argmax。

源码与CPU重放：SGLang：top_k-1→1、temperature0→1；is_all_greedy为真，原Sampler.forward走argmax，TPsync仅stub。

源码与CPU重放：22文件SHA核对，18份Python同git HEAD；19源码事实、11CPU敏感性检查，源码部分重放不当作历史sampler遥测。

## 参数与证据

[18行采样参数](assets/sampling-parameters.csv)；[19条源码事实](assets/sampling-source-links.csv)。源码路径指向仓外gen38证据包内installed-source，不能当作仓内代码路径。

证据包SHA256 4b99915a4c684faa54d8ed239b192650a8d98cbdc2ba7f4e772ecba11c126785；审计JSON SHA256 a26681f1f46c1361e0ebc677d6a0ab1352f4692483bdae53e86da9f27745d505。文件与来源绑定见[SAMPLING_CONTRACT_INDEX.json](SAMPLING_CONTRACT_INDEX.json)。

## 适用边界与未知

证据是保存请求/日志观测、当前同容器源码及CPU部分重放；未直接观测历史sampler实参或运行时类绑定。

当前配置SHA不代表历史已加载generation_config身份；源码与git HEAD一致也不证明历史每步运行路径。

CPU重放使用捕获构造器、argmax/shape哨兵和预处理/TPsync stub，未运行完整Pydantic/msgspec验证、数值张量或真实通信。

仅审查显式temperature恰为0的短诊断；不是所有正小温度、原矩阵或一般服务行为的证明。

两栈按审查路径均到argmax；不证明logits相等、greedy输出等价或实际历史每步选择相同。

vLLM采样ID、首个不同采样token及输出不一致的根因仍未知。不得把do_sample/top_k默认值认定为随机分歧根因。

不增加性能数据或采用结论；跨CANN/卡组冻结比较不构成纯框架因果，HCCL等待与去重保留。

原生层端点不是独占kernel或全stream关键路径；实际204、质量/greedy等价和root最终验收仍待办。

采集aarch64 / 910B1；CPU重放x86_64。11项为CPU敏感性检查，不是性能试验。未分配R、未发布、未采用优化；最终由root核验。
