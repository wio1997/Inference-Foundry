## 2026-11-12：同8die正式对照，C4领先15.26%，C8领先21.79%

原生94（源码2a8d34a）与同组参考95均已完成。32请求/档、1024输入/256输出，C4为242.81672对210.67674 token/s，C8为367.21170对301.51731 token/s。双方各64个请求成功，原始计时重算通过，作业及资源已释放。

**30%未达到，整体Goal继续。** C4还需减少原生整批耗时11.34%，C8需6.32%。完整文本54/64相同，通用质量与稳定收益未验证；原生C4 TPOT和C8 TTFT较参考更慢，不能宣称所有延迟改善。

[对照结果、限制与下一决策](docs/performance-decision-20261112.md) · [原生命令与源码](validation/shared-features-20261111/native94.json) · [参考命令与环境](validation/shared-features-20261111/reference95.json)


下一候选96（8cd3bfc）已完成目标ARM64编译和导入，同一8die的实际Worker已加载新扩展。只合并被640-token状态checkpoint提前截断的独立预填充片段，保留共同开关、状态及批次上限；性能未验证。[决策和否决条件](docs/checkpoint-prefill-candidate-20261112.md) · [源码补丁及实际命令](validation/shared-features-20261111/native96-candidate.json)

---

## 2026-11-12：执行器缺陷已复现并最小修复，当前资源阻塞

原生86在图捕获前退出，未到工程请求和AISBench；没有Python异常，资源已释放。8个真实exec命令的退出事件位于容器stop/kill之后；客户端为何退出尚未确定，不能声称已定位到OOM或模型数值错误。

现有管理器把docker exec客户端结束直接当成模型结束。8秒CPU检查确认：终止已知客户端后，实际容器任务仍存活并返回退出码0。控制包88改为容器内命令写日志与原子退出回执，管理器核对映射后的模型PID及启动标识；客户端返回不再冒充NPU作业退出。0和7退出码回执检查通过，但NPU全生命周期仍未验证。本次只修既有链路，没有增加新调度服务。

88随后在原子占卡前被其他16个vLLM Worker阻止，0模型、0AIS；当前没有本任务存活NPU作业，不抢占其他进程。设备窗口恢复后继续同一88包。此前参考83的214.40089/299.82196 token/s及+30%门槛278.72115/389.76855保留。

本次同时保存待验收的前缀调度源码`0dc887e`，默认关闭`--native-prefix-admission`。它先持有真实SDK缓存块引用，再将命中长度交给原生C++调度器；首个正token计算批次由正常SDK分配器接管引用。冷请求、取消、重复释放及错误位置等6项元数据/引用检查通过，未做客户端模型或精度仿真；NPU状态和性能未验证。现成prefix cache属于共同能力，不能将本衔接直接称为新缓存算法或已证实独有收益。

控制88仍冻结在源码`3de3dad`，对应上一公开快照`1c81dc63`；当前新增prefix源码不是已经测出的性能结果。目标≥30%保持未完成。


---

## 2026-11-12：原生84工程失败，输出行契约修复86正在验证

原生84在三并发工程请求中触发`SDK did not produce the expected real device draft tensor`，没有进入AISBench；8rank实际图捕获已完成，资源已退出释放。旧失败保留，不写成性能或正确性通过。参考83的图模式结果仍有效。

固定SDK源码显示，draft图可以返回补齐后的请求行，普通SDK消费端按`[:len(req_ids), :K]`取真实请求前缀。原生桥此前强制整张draft tensor行数等于当前请求数，拒绝该契约。新源码`3de3dad`仅对当前runnable已经捕获、实际缓冲区地址和形状匹配的输出取有效行，随后保留原有独占快照与异步复制；不改draft token、采样、模型计算或共有图开关。未知补齐、行不足和宽度错误仍拒绝。

同时补上驱动端元数据拒绝时的同阶段失败回执，避免其他rank把后续RPC误当成本步回执。新候选86已启动；本更新时未完成硬件验收。元数据边界检查通过不等于NPU状态或性能通过。

固定源码指纹：core GPUModelRunner `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51`；Ascend proposer `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`。这不是换SDK、补零token或关闭参考图功能。

当前参考83为214.40089/299.82196 token/s，原生+30%目标仍为278.72115/389.76855；没有新的合格原生性能数据。前缀调度候选留待本项通过后试验。


---

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
