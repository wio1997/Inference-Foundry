# 当前开发状态（2026-11-12）

## 2026-11-12更新：共同draft图能力已实测，匹配原生结果待出

固定SDK的GLM多缓存组MTP实现支持draft图。此前双方采用镜像示例的eager draft配置，不能作为共同适用优化全部开启的最终参考。新参考83只将`speculative_config.enforce_eager`改为`false`；卡组、5draft、target FULL、EP8、DSA-CP/SP、static/super、async、prefix及冻结AISBench负载均保留。

|并发|旧eager参考81 token/s|新draft图参考83 token/s|原生至少领先30%的新门槛|
|---|---:|---:|---:|
|C4|177.40808|214.40089|278.72115|
|C8|257.49225|299.82196|389.76855|

新参考每档32成功、0失败，原始时间戳独立重算通过；8rank各10次capture内MTP forward，请求阶段各612次已有draft图成功重放，关闭及资源释放已核实。每档只有一批，未标定自然波动，不将差值直接宣布为稳定收益。draft图是双方共同能力，不计作DeepInfer独有优化。

源码`b1b5b48`新增`--mtp-draft-graph`，映射到同一SDK的`enforce_eager=false`；双方使用相同字节的只读重放计数器，不替换模型计算或分派。原生84已启动，但本更新时尚无结果。旧82对旧81的+11.58%/+7.08%保留为历史限定结果，不能拿来与83混算，也不能代表当前目标完成。

下一候选针对原生调度预算与真实缓存命中的衔接，仍待NPU验证；不会把现成prefix cache或CANN算子改名为新技术。目标仍为同等共有优化条件下至少30%的原生优势。

[新参考完整参数和脱敏核验](validation/shared-features-20261111/reference83.json) · [配置及历史](docs/shared-features-20261111.md)


以下保留上一轮已核验结果，当前匹配参考以83为准。

# 当前开发状态（2026-11-12）

**8die限定功能、正式计分与正常退出已核验；相对vLLM至少30%的目标仍未完成。** 这是实验适配分支，通用模型质量和生产部署未验收。

|同一8die卡组、同一冻结负载|vLLM参考 token/s|DeepInfer token/s|本批原生相对变化|
|---|---:|---:|---:|
|C4|177.40808|197.95191|+11.58%|
|C8|257.49225|275.71708|+7.08%|

双方每档32成功、0失败、1024入/256出、2次排除预热。原始SQLite独立重算通过；每档只有一批，不声明稳定收益。64条完整文本56条与参考相同，差异的质量影响尚未核实。

实测源码`6b00f78ee354a97085903f20d61a729af640a4b8`。DeepInfer原生引擎、C++ scheduler、Worker完成80请求（工程/预热/计分72，计分后独立诊断8）；8rank均exit0，设备占用已释放。此前g79的重复关闭错误仍保留为失败，不能用新结果覆盖旧失败。

活动Ascend缓存解析器在8rank恢复640/640调度与哈希粒度，保留不参与prefix的9-token私有scratch组；记录37次真实prefix命中、23680命中token。原生真实logprob取行优化保留SDK采样、实际概率与rank；重复drain已正常退出。

共有5draft/eager draft、FULL target、DSA-CP/SP MoE、AIV、shared-expert overlap、static/super、async及prefix/align均按同一配置实测生效。fused MC2与shared-expert overlap互斥，选择后者；不声称全局最优已穷尽。

下一步依据当前路径已有profile选择一项干预；不更换负载、不关闭参考共有能力、不把prefix缓存或现成CANN算子改名为独有技术。C4/C8达到当前参考+30%仍需约14.17%/17.63%的整批耗时下降。

[完整配置和历史边界](docs/shared-features-20261111.md) · [脱敏实测摘要](validation/shared-features-20261111/summary.json) · [源码](DeepInfer/deepinfer/model_executor/models/glm5_next/full_model/)
