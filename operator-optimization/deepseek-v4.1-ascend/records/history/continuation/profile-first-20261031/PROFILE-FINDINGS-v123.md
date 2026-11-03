本轮把专家放置与 token/目标 EP 分组传输放到同一份真实路由数据里比较。v17 已有按 assignment 总量的放置分析，v122 已有当前 linear 放置下的重复引用计数；本轮新增的问题是：改变专家 owner 后，分组 packet 数、远端返回和接收端专家工作是否仍能独立领取收益。没有部署映射、修改算子或运行模型。

从原166实际主机105的授权归档中只读1392份tensor和1392份metadata，逐份SHA核对，共35,632,142B；同时验证了每份快照的 expert_map 与 ascend_expert_map，全部符合其记录的连续linear分区。导出的仅是原 active source-row index 与原TopK专家ID，压缩后4,021,974B，展开7,327,280B，共456,400条target记录行和260条draft记录行。没有导出激活、scale或权重。CPU归档读取约3.57秒墙钟耗时，不是NPU或内部MC2计时；没有Torch/NPU/model/replay，SSH已明确关闭，Pad探针授权没有再次使用。

分析域为每rank仍放E个专家、EP16、无副本的固定双射。每个role/layer只使用第一组formal归档训练映射，随后保持映射固定，评估40组target和4组draft后续formal归档；这些来自同一观测run，不能当成独立生产泛化验证。43个layer域、5种场景得到215套映射与435个cohort/场景比较。五种场景是原linear、按训练assignment总量均衡、按训练共选pair亲和度聚类，以及后两者限制在原TP8编号半区内的版本。聚类和rank标签选择均为明确的greedy参考，没有证明全局最优。

每套映射保存了owner、完整physical-ID双射及每rank原logical-expert列表，保证恰好E个slot且不漏专家。v19源码已经存在expert_map_path/log2phy静态自定义映射能力，因此本轮没有把静态映射说成完全不存在的接口；但checkpoint加载映射、执行映射、权重及全部FP/packed scales、图地址和各rank generation的配对仍是实际使用前的证据缺口。只改ID而不配对权重会改变函数。

40组后续target样本中，linear映射的远端token/目标EP分组总数D为1,582,709。计数均衡映射在40/40组中都降低最繁忙rank的接收assignment行数，但D增加为1,697,370，40/40组都增加。相反，共选聚类映射把D降到850,221，40/40组都减少，但最繁忙rank行数40/40组都上升。各场景在这40组样本中的最高receiver行数分别为linear14,852、count-balanced7,849、共选聚类43,901；这是各场景的样本最大值，不能拿它们的比值当同一次调用的速度比。

限制在原TP8编号半区的计数均衡，D为1,687,139；半区内共选聚类为934,895。固定owner半区确实保持跨半区assignment数A，在本组后续target中均为975,386，但不保持token/目标EP分组数D：linear跨半区D为844,073，半区计数均衡为892,695，半区共选聚类为498,875。这里的半区只是已有编号分区，不能由此证明物理网络拓扑、实际链路流量或耗时。

draft不能套用target结论。4组后续draft样本中linear远端D为390，计数均衡为395，共选聚类为158；共选聚类在其中3组增加receiver-max行数、1组相同。半区共选聚类D为274，但3组增加receiver-max、1组降低。小样本、录制方法和固定输入域均保留，不能据此选择全局最优生产方案。

单独改变packet格式时，同一mapping下每assignment返回不变；改变专家owner则会改变远端assignment数A，返回是否跨rank也随之变化。因此组合必须在同一映射pi上重算。沿用v122选定的INT8 H5120、原stride5152、新stride5216与每cohort新增D-count内容15360B参考，定义旧linear逻辑内容为A_linear*(5152+10240)，联合候选为D_pi*5216+15360+A_pi*10240。准确差额为(A_linear-A_pi)*(5152+10240)+(A_pi*5152-D_pi*5216-15360)。不能再加一份linear映射下独立算出的packet收益。

这40组target的旧linear dispatch+return参考内容为28,327,698,464B。只在linear上应用选定packet格式，参考节省1,225,803,840B；与计数均衡共同使用时，准确联合节省为518,071,904B，而独立相加会误报1,060,970,912B。共选聚类下准确联合参考节省为3,826,160,448B；半区共选为4,595,923,584B，但其receiver-max工作明显增加。四个非baseline场景在40/40组target中都不能独立相加。这些仍是选定record格式、量化内容共用和协议合法前提下的逻辑内容参考，不是实测wire/HBM字节或时间；新增grouping、packing、D-count、descriptor/offset、检查、展开及银行切换开销尚未计时。

Pair共选得分也不能代替完整token超边。一个K6 token若目的地占用为[3,1,1,1]，同目的地专家pair得分为3，需4个packet；若为[2,2,2]，pair得分同为3，却只需3个packet。所以训练时可用pair作为启发式，评估必须用完整原TopK。当前统计只分析声明的无副本assignment域；总行数/16与最热单专家计数给出的receiver行数边界，是assignment记账边界，不是任意算法下的MAC或耗时下界。

所有435个所选计数场景的receiver行数不超过43,901，低于归档capacity参数65536；这只说明记录输入在这些计数模型里不截断，并未证明新的native物理写范围或workspace自动缩小。稳态slot数一样也不能证明迁移峰值内存一样：old/new权重和map、全部scale、原量化bank、新packet stage、capture bank可能共存，旧dispatch/remote pull/GMM/return读者与queued writer都要退役。没有把这些映射与原v97 capacity/private-state预算当成可独立释放的显存。

逻辑层面，若physical函数F[pi(e)]与原logical函数F[e]严格配对，保留每个TopK slot及其概率，返回原logical assignment并按原顺序组合，专家重标号是等价的。但实际physical sort、expert row顺序、quantizer分区、GMM shape/tiling、rounding、capacity选择和combine顺序可能改变；不能从逻辑双射推出native bit identity。v122还没有证明所有拟共用quantized code/scale一致，本轮也没有补出这些字节，相关合法性门槛仍保留。

局部条件图在v122的13节点上补入logical TopK、固定map generation、配对weight/scale bank和ID remap，形成17节点DAG。原A counts、GMM/return、inverse/probability边与packet/tile readiness保留；没有加入每次调用必须等待整层权重复制的虚构屏障，也没有增加global observed catalog节点。理论仍为SYMBOLIC_CEILING：实际必要物理工作cut、相应容量上界与同调用内部时延尚未绑定，数值时间下界/吞吐上界/MFU及最低模块仍未建立。

独立二进制/bitmask路径复核了全部记录与435个比较；六专家、三owner、每owner两slot的36,000个小规模组合中，29,784个体现独立相加错误。核验通过后将证据补入现有REL73-07/HG73-MOE，不新增根候选或伪造pair关闭。全局仍为35模型候选、7历史框架记录、356语义行/26模块、135条件关系/26联合组/460待核对pair；9类/1352次原生待补字段、849缺PMU、16 phase未绑定图、3136含糊归属不变，整模块关闭0。整体目标继续，不因本轮报告或核验通过而标记完成。
