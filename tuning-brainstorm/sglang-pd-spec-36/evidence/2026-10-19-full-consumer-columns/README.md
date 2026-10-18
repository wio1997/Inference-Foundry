# M035 / PD-CHUNK-001：补齐consumer后继续消除冗余元数据

北京时间2026-10-19。Task继续ACTIVE，整体Ascend PD＋MTP证据仍E0。L005和L006验证的是本机CPU fixture；没有调用native传输、NPU或模型，没有把其他服务当作本任务基线。生产源码未修改。

## L005：完整Python消费链与噪声反证

保留前轮原方法A和range方法B，加入原样提取的MooncakeKVManager._transfer_data。它实际执行zip转置与三个list构造，原生batch_transfer_sync仍为no-op。151组地址/长度/顺序检查扩展为固定78层，并验证空描述符不调用engine、非空调用一次、三个Python list及非零状态7透传。L004原3/78层用例另外只用于转录一致性检查，未冒充同层数性能比较。

L005-R001保留所有5组A/A、9组交错A/B及wall/thread CPU样本；另做带包装器和GC callback的分段诊断。三个连续Case的局部差异继续与A/A分离。碎片N=R=4096时B的consumer诊断CPU中位约25.16ms，占其约51.93ms诊断总段的很大部分；这些诊断带观察开销，不能与干净计时直接相减推算严格收益。

碎片A/A出现171.08ms wall、103.73ms thread CPU的样本，差约67.35ms。它说明存在非当前线程执行的时间，原因尚未定位，不能全部归因于GC或删除该样本。GC数量、分段观察和原始wall/thread样本均保留。

为减少初始A/A与后续A/B分块的漂移影响，L005-R002只补测这一未决Case：13个AA/AB交织块，块顺序及AB/BA内顺序交替。A/B工作量、方法和native-no-op合同不变。原方法A wall中位71.77ms、range B53.81ms，配对几何比1.310（95%bootstrap1.275–1.343）；同轮A/A区间0.971–1.014。Thread CPU也分离。它加强了这个本机Case的证据，没有解释或改写R001异常。

完整干净样本及分段观察见[L005公共结果视图](L005-profile-public.json)、[交织补测原始样本](L005-noise-followup.json)。5568条详细GC时间戳留在本地Task原始结果中，入口、完整文件哈希与事件序列哈希都在公共视图中；[主控派生分析](L005-controller-analysis.json)明确标出GC聚合受观察器影响。公共视图只移除了详细GC事件数组，没有移除任何干净计时样本。

## L006：直接形成native参数列

分段证据支持继续检查冗余tuple和转置。候选C沿用B的range分组，直接形成源地址、目标地址、长度三个list，保持同一个native函数、相同参数内容/顺序、调用次数与返回码。空列仍返回0；PP!=1、hybrid或custom-pool模式明确转回原安装sender。回退仅做参数转发分派检查，不声称这些模式的完整行为已测。

候选通过151组固定78层oracle检查、状态7透传，以及三种回退分派、空层数和HiSparse拒绝检查。只验证Python调用期间的参数内容与类型，未验证真实native绑定的转换/复制/所有权，也未验证PP/DCP/异步取消或PD端到端。

先独立执行新的B/B基线控制Run，再执行B/C配对Run。两者都保存3轮预热、5组A/A、9组交错对照、wall/thread CPU及GC collection计数；每Case固定迭代数相同。主因果证据是第二Run内部B/C配对，Task工具另存独立baseline/after标量比较；没有用两个不同Run的比值相乘。

| N页索引 / R连续区间 | B：range＋tuple/转置，wall中位 | C：直接三列，wall中位 | 配对几何比B/C（95%bootstrap） |
| --- | --- | --- | --- |
| 8 / 1 | 21.05 µs | 16.29 µs | 1.297（1.275–1.322） |
| 4096 / 1 | 22.90 µs | 18.34 µs | 1.253（1.234–1.272） |
| 32768 / 1 | 33.39 µs | 28.64 µs | 1.171（1.152–1.191） |
| 4096 / 4096 | 54.84 ms | 21.04 ms | 2.802（2.586–3.128） |

所有四个已测Case的差异均超过本轮A/A波动，没有观察到这些小输入的性能回退。碎片Case的thread CPU中位从54.52ms降到20.84ms；同一批样本中GC collection从每调用约459次降到3次，与少创建受追踪中间对象的预测一致。不能把全部节省都精确归因于GC，分配、遍历与转置也一起改变了。

[候选源码](candidates/L006/direct_native_columns.py)、[独立B/B基线](L006-baseline.json)、[B/C原始配对样本](L006-paired-candidate.json)以及[Loop判定](task/L006-direct-native-columns-cpu/decision.json)均保存。保留的是有条件的CPU候选；没有部署到服务器，也没有更新任何“最佳服务实现”。

## 成本边界与下一步

两轮改动属于M035与同一个PD-CHUNK-001演进链，不重复算成多个独立加速。输入扫描仍O(N)；直接三列仍需O(层数×R)的整数运算和有序列表条目。碎片Case生成319488条描述符、958464个标量槽；native转换/拷贝、实际PD传输、MTP有效token成本和可达硬件速率都未测。没有据此宣称物理最优，也没有把本机Mac数据当Ascend主机/NPU数据。

继续向量化、缓冲复用或native列接口可能还有空间；其优先级要由实际PD中的N/R分布、暴露CPU时间与native协议确定。当前转向M017/M018的NPU长度镜像和staging消费者审计，补齐运行合同后再排序。附带[消费者静态种子](next-audit-consumer-seeds.json)和[观察到的模型维度](observed-model-shapes.json)仅是下一轮输入，未把对应机制标完成。

44/48资源归属信息仍待补充，本任务没有PD＋MTP服务基线。Task主证据等级保持E0；CPU fixture的基线/profile/候选证据单独记域，不能由工具自动升级整个serving Task。

## 执行与证据来源

安装提交ce0a8fea5f4245839d174b2b8c55692be1810a7f。脚本、冻结spec、原方法快照、逐Run SHA256/退出码在本目录。实际执行命令是 `python3 scripts/execute_frozen.py specs/L005-R001.json`、`L005-R002.json`、`L006-R000.json`、`L006-R001.json`、`L006-R002.json`，均通过本任务ZCode入口。内层和外层退出码独立核对，冻结输入哈希保持不变。

L005编写会话在产物已写出后持续静态检查，由主控只终止该会话进程，退出143；这不算实验成功。主控核验其只编辑目标脚本、没有运行目标，完成审核后另起冻结执行会话，真实实验exit0。编写边界、交接、审核及实际模型usage别名见相关JSON；DeepSeek底层发布版本仍UNKNOWN。没有发布provider配置、headers、prompt、密码或密钥。

本检查点保留了AST探针所需的源码、L004冻结输入和候选相对路径；CPU探针只需Python3.12与requirements-probe.txt中的NumPy。复现实验要使用新的输出目录，不能覆盖历史Run。候选已物化，materialize步骤仅作来源记录，无需重复执行。
