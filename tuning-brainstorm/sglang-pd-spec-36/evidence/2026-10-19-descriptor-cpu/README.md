# M035 / PD-CHUNK-001：连续区间元数据的CPU实证

北京时间2026-10-19，安装源码ce0a8fea5f4245839d174b2b8c55692be1810a7f。Task仍ACTIVE、整体PD＋MTP证据仍E0。本文只报告本机CPU fixture的地址描述符等价与局部成本，没有NPU或端到端加速结论。与M035关联同一个PD-CHUNK-001候选，不重复计算收益。

## 当前做法与候选

Ascend `send_kvcache` 已经用 `group_concurrent_contiguous` 合并源/目标同时连续的页。现有helper随后为每段调用tolist，物化全部源/目标页号；发送器只读取每段第一个页号与长度。候选保持相同break条件、连续区间和后续发送，只物化每段的源起点、目标起点、长度。

隔离AST中只替换group赋值与内部set_transfer_blocks两个节点；其余原方法不变。真实生产源码没有修改，当前没有部署补丁。完整候选见[candidate-method.py](candidate-method.py)，实际执行脚本见[scripts/probe_ascend_descriptor_cpu.py](scripts/probe_ascend_descriptor_cpu.py)。原文件快照保留在sources/installed，哈希保存在receipt与result中。

两者都扫描O(N)配对页号；旧中间表示装箱O(N)个页值，候选装箱O(R)个区间值。最终仍生成O(层数×R)个有序描述符，载荷字节与发送调用数相同。R为源和目标同时连续的区间数，不能只根据源连续就合并。

## 正确性、噪声与实际结果

151个CPU用例全部通过：原样安装方法AST、候选和独立逐页合并oracle生成完全相同的地址、长度和顺序。覆盖空/单页、不同断点、碎片、随机重复/乱序、int32/int64、非连续只读视图、两侧非空错长异常及HiSparse拒绝。空侧先返回的现有行为保留。fixture限定PP=1、非hybrid、custom memory pool=false；真实envelope校验为no-op，地址没有解引用。

固定78层，每Case预热，5组A/A估噪声，再9组A/B按AB/BA交错。每样本迭代次数按Case冻结，A/B不变；原始每组样本、顺序、分位范围、配对比值及bootstrap区间均在[result.json](result.json)。tracemalloc单独测group分配，没有混入计时样本。

| 页索引N / 连续区间R | 原方法中位时间 | range候选中位时间 | 配对几何平均比值A/B（95% bootstrap） | 判定 |
| --- | --- | --- | --- | --- |
| 8 / 1 | 26.68 µs | 18.84 µs | 1.416（1.410–1.423） | 局部差异与A/A分离 |
| 4096 / 1 | 97.58 µs | 21.42 µs | 4.565（4.501–4.627） | 局部差异与A/A分离 |
| 32768 / 1 | 584.30 µs | 32.23 µs | 18.170（17.796–18.589） | 局部差异与A/A分离 |
| 4096 / 4096 | 46.75 ms | 29.98 ms | 1.342（1.048–1.610） | INCONCLUSIVE，波动覆盖判定余量 |

碎片Case的A/A几何比值区间为1.020–1.060，A/B下界未越过这段噪声范围；不能仅凭中位数较低接受收益。两次候选wall样本约56.8ms和111.1ms，其他多在29–33ms，下一轮要区分CPU工作、调度停顿与GC，不能删掉离群点后报告。

连续N=32768时group的tracemalloc峰值从2,622,177字节降到197,046字节，符合少物化页号的预测；这不是HBM下降。对于碎片R=N，区间数、描述符构造与后续消费仍大，没有因此逼近物理上界。

主控接受限定CPU语义及三个连续Case局部成本证据，碎片时延保持INCONCLUSIVE。全体生产/PD优化仍INCONCLUSIVE，未修改最佳已验证服务实现。

## 关键限制与下一轮

本轮 `_transfer_data` 是no-op，只测send方法的Python/NumPy准备。后续源码审计发现真实Mooncake基类 `_transfer_data` 还会将tuple列表zip转置，再创建源地址、目标地址、长度三个list。它尚未计入上表，不能把上表当整个CPU传输链，更不能当传输耗时。

下一Loop会将真实 `_transfer_data` 纳入并用no-op原生引擎隔离网络；保存thread CPU与wall、GC事件；对比range＋tuple＋转置与直接构造三列，检验冗余元组是否值得消除。保留当前range候选作锚点，并检查小Case回退。真实memfabric绑定、PP/DCP、线程池、错误/取消生命周期和PD端到端仍需各自验证。

必要工作是读取配对页映射、保持有序地址/长度、满足实际native API输入契约；具体HBM/链路带宽与MTP总有效token成本未测。N/R的CPU fixture不是模型输入token数或真实碎片分布。若实际服务中这段被计算/传输覆盖，局部CPU下降可能没有TTFT/吞吐收益；高R情况下也可能由描述符数量或native提交主导。

## 模型映射与执行来源的补充

只读观察进程使用NEXTN、3 steps、topk=1、4 draft tokens；源码spec_registry保留NEXTN→EAGLE别名。GLM DSA target/NextN类实际在glm4_moe.py:1461/1465，分别继承DeepseekV2ForCausalLM及DeepseekV3ForCausalLMNextN；源码映射已闭合。没有通过导入或修改该现有进程来观察已加载类身份，这些参数也未作为本任务合同。

一次补充源码采集把文件误写为deepseek_v3_nextn.py，内层exit1、外层ZCode exit0；主控按内层失败记录，未接受。改为inventory中实际的deepseek_nextn.py后另建Run，exit0。失败stderr、两次冻结spec和receipt均保留。新的execute_frozen controller在执行前后验证输入SHA，失败不自动重跑。后续采集与CPU实验的inputs_unchanged均为true。

真实执行：`python3 scripts/execute_frozen.py specs/L004-R001.json`，内层为`.venv-cpu/bin/python scripts/probe_ascend_descriptor_cpu.py`；脚本/输入/候选/结果哈希和退出码见[receipts/L004-R001-frozen-controller.json](receipts/L004-R001-frozen-controller.json)及[receipts/L004-R001-descriptor-cpu.json](receipts/L004-R001-descriptor-cpu.json)。CPU环境Python3.12、NumPy2.3.3，具体平台和其余依赖均在原始结果及requirements中。

44/48的资源归属信息仍待补充；没有因此停止CPU独立研究，也没有将别的服务改为本任务。Task控制工具会把CPU correctness Loop自动升为E1；主控已按运行域纠正：整体Ascend PD＋MTP仍E0，CPU局部证据单独记录，不能借工具自动晋级跳过服务器基线。
