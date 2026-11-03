# v171：本机设备映射、周期下界与跨卡数据下界

本轮上一目标回合v170为实质PROGRESS。本轮补的是会影响算子容量和跨卡结论的硬件证据；没有重跑完整Case、加载模型或运行应用kernel，也没有重启服务。171 health200且StartedAt未变。原始框架和算子实现未改。

当前CANN9.2接口对16个user device完成了user→logic→driver physical的正向、直接和逆向查询，均成功且往返一致。NPU-smi记录8个NPU卡号，每卡ChipID0/1两个计算设备；同卡对经SIO，其他设备对标为HCCS_SW。16个设备均报告arch2201、24Cube/48Vector核、非虚拟设备。16个实际profile的PID、native replay deviceID、源EP/DP/TP rank、captured parallel_group_info与设备记录完成关联。两个TP8组分别为global0..7和8..15，各占4张卡；EP/MC2组含全部16rank。每个TP组56个非self有向endpoint pair中有8个SIO、48个HCCS_SW。

应用当前映射到先前采集区间保留“期间未发生driver重映射”的环境前提；同一容器身份、可见性、CANN版本和采集核心数一致。这里不回填旧166历史。静态组成员不等于某次native通信的全局调用序号。

几种名称不能混用：driver physical deviceID有16个，ACL属性PHY_CHIP_ID和SMP_ID各只有8组，MAINBOARD_ID全为20。容量表按“一个暴露的计算设备”计数，不能因chip/card名词再翻倍。AICPU每设备查询报6，但SMP成对，未证明16个独立AICPU池。L2查询报192MiB，不能直接合并成自由共享缓存；globalmemory是API暴露总量，不是可供优化新用的free。UB属性204全部返回rc0/value0，此字段不能当物理零UB，也不能据此宣判某种融合不可能。

频率字段已定案：本机16个DCMI enum9均返回1800MHz，enum7空闲时800MHz，HBM字段1600MHz，Vector current查询返回不支持。尽管安装头文件把enum9命名为DCMI_FREQ_AICORE_MAX，官方A3文档将其定义为TDP条件下持续运行的额定频率。因此1800不能直接升为最高频率上界，HBM频率也不是带宽。保留真实fmax未知，停止为同一个字段反复查询；该缺口不再阻止其他算子域的条件分析。[A3 DCMI接口定义](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3DCMI/dcmia3_049.html)

现有A3_2201指令表与本机架构/24Cube核现在匹配。在声明的direct-dense INT8 Cube源算法域内，必要corecycles≥Q_MAC/8192，整设备等效cycles≥Q_MAC/(24×8192)。每个receiver assignment的35389440MAC至少对应180个整设备等效cycles。该推导不依赖实际时钟上界，不把最快实测或可行调度倒数当上界。应用到当前计数仍保留返回计数stream/source算法合同；不是loaded binary中每条指令的证明。

| F21域 | 必要累计整设备等效cycles | 1800MHz参考计算时间 | 参考计算时间/累计完整kernel时间 |
| --- | ---: | ---: | ---: |
| Draft | 14588100 | 0.0081045s | 0.30696% |
| Target小行数 | 462153600 | 0.256752s | 0.49423% |
| Target prefill | 1136937600 | 0.631632s | 6.64652% |

时间来自v170同域完整聚合，跨rank/调用累计；不是请求walltime。这些百分比仅是额定时钟参考下、有效INT8 MAC周期下界占完整F21累计时间的比例，不能称真实物理MFU或剩余可达提速空间。其他必要访存、Vector和通信约束可能远强于该Cube下界。6个Cube模式已形成明确的每cycle容量/额定参考表；不同资源下界取max，未证明共同issue关系时不把多个模式/资源时间简单相加。

跨卡下界也已按实际放置重算。对一个4卡TP8完整raw AllGather，设每rank初始单独持有S字节，所有rank都需要全部raw数据，且没有预存远端副本、压缩或移除需求。每卡已有2S，总数据8S，至少需从卡外进入6S；4卡总入口必要payload至少24S。允许卡内转发/组播后，按48个跨卡rank pair计算的48S只是直接投递参考，不是通用wire下界。更一般地，总必要数据G、卡b初有C_b，则入口至少G−C_b，总入口至少(k−1)G。它是必要cut证明，没有将一个可行执行时间倒数成上界。

若入口容量上界为B_b，则覆盖全部所需跨界传输的时间窗至少max_b((G−C_b)/B_b)。这不强迫完整producer先完成才通信，也不把GEMM和通信下界串行相加。实际链路带宽、共享端口和协议仍未确定；HCCS_SW标签不表示独立全带宽。TP和EP若同时占用同一卡入口，就必须共用预算。

条件padding示例进一步说明字节比例不能直接解释时延：若一个BF16/H5120的AllGather输入确实由20行补到24行、padding到边界仍为已知常量，8rank有效行数为[3,3,3,3,3,3,2,0]。4卡已有有效行[6,6,6,2]，各卡需要远端[14,14,14,18]行。总入口下界从完整24行时的720KiB降为600KiB，最大单卡入口却仍为180KiB。相同卡入口容量下，此瓶颈cut未变；这不排除其他瓶颈改善。BOOL mask shape本身不能证明native AllGather的输入就是该tensor，已将此额外条件保留。65个几何边界例和精确有理数周期核算通过，不称通用形式化验证。

F26/F32/F33/F31复用同一拓扑cut，并在消费者融合、初始副本、投影后payload、row ownership或压缩改变时重新推导G/C_b。已经存在的HC局部collapse路径不算新优化。新算术、局部常量重建、存活期和通信需在同一候选图中计价，避免重复收益或免费重叠。

取证封装有三项错误已如实记录：第一次info.json.*也读了16个辅助marker；第一次ACL输出在JSON后追加运行时footer导致严格JSON解析失败，初次raw未保存，第二次只读metadata查询已先存raw再解析；另一次主机脚本字符串拼接SyntaxError发生在执行任何查询前。均已修正且没有因此重跑负载或重载模型。当前完整API结果、unsupported字段、头文件/动态库SHA及11份远端派生文件已保留。显式setDevice/context/allocation/model/kernel调用为0；不对SDK内部初始化行为作额外假设。

下一步转向统一模块诊断与固定Case分支审计，先按MODULE_DIAGNOSIS确定主指标和适用域，复用无observer窗口、当前资源/typed调用点和已完成的公开算子条件分析。只补影响候选判断或最低利用率结论的字段，必要时合并最小探针，不按无关候选配对重复取证。完整35+7候选、26模块358语义行、全部条件分支和跨算子/层/卡/迭代范围保持。实际时频/带宽上限、物理模式、all-reader/数值/状态/copy/control及最低模块结论仍有缺口；整体PROGRESS，未完成。
