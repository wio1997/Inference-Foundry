# Flash vLLM 性能路径候选

本包仅供 A 线（14P/15D）独立资格检查，不等待 SGLang 适配。固定 rc1 镜像与官方 Flash 路径；R127 已通过 bulk FP32/IEEE KDA 的局部数值门禁，尚无整模型、PD、新DP拓扑、Graph、MTP、缓存或性能资格。

只替换8份源码。原生模型参数加载、native recurrent/MTP rollback、缓存组/块所有权、异构TP地址生成均保留。必要的 per-head ReLU 与 R080 数学一致，去掉其CPU审计。KDA使用镜像自带Triton chunk和FP32/IEEE中间值，移除Python逐token发核参考。

严格PD SDMA绑定使用MF1.2枚举，不允许RDMA回退；只替换MooncakeV2的PD引擎，Store的原生Mooncake引擎保持独立。资格检查audit1逐字节对比；超过64MiB的大packet仅拆校验视图，SDK仍收原始向量且只调一次。性能audit0禁用这些CPU拷贝和HTTP校验；两种模式仍须整模型门禁。

正式基线仍为空，实际80K/600未发。后续采用官方P DP2/TP8、D DP16/TP1、D目标FULL_DECODE_ONLY和MTP D5（draft eager）、原生align/whole-segment混合缓存，先真实PD/缓存/Graph/D5和80K600，再按独立74400前缀自然缓存的C16/32/64四变体合同测量。
