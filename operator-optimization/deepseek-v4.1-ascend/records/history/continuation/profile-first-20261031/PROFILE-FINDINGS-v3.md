# 投影、共享专家与通信边界：Profile 分析 v3

本轮先查profile，再按明确的问题检查3个冻结源码文件的相关函数。没有修改目标代码、运行设备或新增采集。32候选/26模块仍是覆盖台账，不是独立机会数量或完成比例。

## F06/F32：不能忽略局部与全局行域

所选layer36的8个rank记录中，Q输入行为7×1023+1019=8180，而各rank的KV输入均为8180行。按这些当前物理行域，直接将两份权重拼起来、共用同一个输入矩阵且不处理行选择的变体不适用；这不排除对子行共同投影、重映射、前移投影或重新分区的条件候选。

EP3的host栈把Q/KV投影定位到`multistream_preprocess`两处调用，gather定位到模型层`sp_all_gather`。选定通信记录的count=5237760、rankSize=8、导出dtype=BFP16，元数据的communicator key关联到TP成员0–7；group name并未直接匹配。没有跨库证明同一物理collective，也没有tensor/storage身份。若按等长gather解释，count/5120=1023、总行数=8184；这个相容的算术不能独自证明4行的padding身份或可删除性。

## Profile 无法回答的消费者问题，已定点补源码

问题是：投影前移后，除了SWA KV，还有哪些路径必须保留全局原始hidden值？已检查冻结`model.py`、`dsa_v41.py`、`context_parallel/dsa_v41_cp.py`，SHA与既存版本一致。范围和问题记录在`PROJECTION-CODE-QUESTION-GATE.json`，摘录和角色表在`projection-source-contract.json`。

在这些显式Python路径及其既存multistream局部合同下：

- 模型先对HC输入归一化结果作SP gather，再交给attention；残差另行保留。CP给Q传局部子视图，给缓存侧传全局视图。
- Q和Indexer权重使用局部query hidden。SWA KV使用全局cache hidden。
- `is_kv_source`时，cache hidden还进入压缩路径：C1传给compressor；C2转FP32后分别进入compressor.wkv和wgate，再写index keys与long-KV持久缓存。
- 空query、非空cache时，`_update_caches`仍做全局KV及条件压缩工作。不能因为本rank没有Q就删掉cache分支。
- sparse index选择、CP输出恢复和graph稳定输出还依赖正确的行数/shape/metadata/地址契约。改变值的表示不等于可以删除这些接口要求。
- 没有attention metadata的启动分支会把输出置零并提前返回；其外层collective及初始化顺序单列，不把正常推理的payload模型套给它。

配置有43项compress_ratios，但冻结topology builder明确只取前40项作为target。target中KV-source是2、8、14、20：前三层走C2，20层走C1。其余36层在已审阅路径没有上述额外全局原始hidden消费者。因此F32要按角色分开：36层进入只保留局部Q及通信后SWA投影值的条件考察域；4个source层若要真正替换hidden gather，还必须保留或等价重组F17/F19压缩消费链。

这没有证明历史实例绑定、全部primitive/alias效果、行所有权或数值等价。Wkv从全局M改为局部M可能改变后端计算/舍入；必须保持每行原数值契约、row order、padding、metadata和输出地址，才能接受变换。

**条件接口账**：若等长TP8/BF16等前提均成立，1023×5120元素对应每rank逻辑输入10,475,520字节；改为1023×512的SWA投影值则是1,047,552字节。10是特征宽度比，不是实测链路字节比、不可超越的下界或10倍提速。4个KV-source层不能仅套这个payload替换模型；资源交叠与联合图也须重算，不能与F06/F17/F19收益叠加。

## F10：补齐了跨流的转换候选

同一EP3/layer36/scope `2128-42`中，全stream查询找出四个同形候选：BF16投影、BF16→FLOAT、FLOAT乘法、FLOAT→FLOAT16。投影有提交栈/flow关联，后面转换/乘法仍是shape、stream及时间关联，不能据此证明实际tensor边或标量值。

已有局部数值合同继续适用：保留BF16投影舍入；有限BF16值在满足2^-6、舍入和非FTZ条件时可讨论缩放epilogue等价；把权重缩放直接移到query scale有FP16边界反例。packed路径另有FP32消费者，不能套非packed结论。此处没有将条件CPU数学模型升级为NPU实测等价。

## F16：没有两投影间的大空洞证据

完整遍历16rank、1280个已存target scope，全部各有唯一的匹配两投影候选，同流且行数相容。行数4760/5144/8184分别有320/320/640组。导出间隔最小20ns、中位20ns、最大41ns，即0.020–0.041微秒。EP3的两任务与保存的host/connection记录逐项匹配。

这排除用“所选两任务之间存在大段可消除空洞”解释本批记录，不能把CPU下发API时间加成设备间隔。它不排除中间读写、布局或流水重组；BF16中间接口、collective所有权及所有消费者仍须保留。kernel区间内部、导出时钟和请求关键路径也没有由这个间隔展开。

## F22：复合激活已存在，另有长间隔待追踪

1280个scope全部有相容up→DequantSwigluQuant→down三任务链，均同流。up到激活的导出间隔为20–41ns；激活到down中位1.180微秒，最大1.7652毫秒。已有DequantSwigluQuant不重复计新增融合。

最长样本为EP3/L4/scope `2128-55`，activation task43276至down task43277；该段尚不能归为CPU下发、通信等待、设备空闲或可节省时间。后续先查已有TASK/EVENT及全部可用流记录，保存覆盖不足，不先扩写算子代码。逐组量化scale、clamp、半区gate/up布局和tile就绪仍是进一步融合的必要合同。

## 当前状态与下一步

v3台账中，16项还只有本轮family入口，需继续纳入细粒度scope/stack证据；既往局部研究可能已存在，先复用并核对，避免重复查库。这不是说其余16项已完成。所有候选、完整分支/消费者/跨迭代及资源闭包仍开放，0个整模块关闭。

下一步优先核对F17/F18/F19已存compressor profile及状态消费者，使F32的source层联合重组具有完整依赖；并定位F22最长间隔的已有TASK/EVENT材料。之后继续其余候选和模块，不以本报告成稿结束整体任务。
