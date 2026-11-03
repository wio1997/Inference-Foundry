本轮补齐F03与F05/F06/F07/F08/F33的5条有范围的条件关系：跨层Post→Pre与下一层input RMS、Q/KV、量化、RoPE及C1/SWA组合。整体目标仍未完成。

936处目标模型FFN Post→下一层attention Pre与既有投影前沿使用同一HC TASK、API、forward和层号，关联13368个原生任务。投影前沿来自16份正Query eager profile，是Post/Pre原32份数据库的子集；不是所有初始层、draft或空Query分支。48处有正Engram更新，72处有辅助mean读取，另有24处C1、72处C2和192处IndexWeight归一化输入读者。

这些组合需要保留明确的版本：Post产生X0，辅助mean读取旧X0；Engram若更新hidden则产生X1，下一Pre的系数、collapse和未来attention Post残差应使用更新后的值。查表准备可前移，hidden相关更新不能当成恒等映射跨过。前一个FFN Pre的premix与当前attention Pre系数也不同。相同HC控制记录并不证明实际指针、参数代次或新tile发布已经可用。

在保留原始4H残差发布的参考图中，转发更新后残差到Pre只对应一份8MH B逻辑API输入端口，不能再计一份原始残差写回消除；未来Post及其它读者仍可能需要它。选定私有Y读写全部内部化时，HC/input RMS参考接口为4MH B，联合参考为12MH B。2048个接口组合显式区分X0/X1和旧辅助读者，共用Y norm读取只计一次。它们不是原生kernel读取遍数、HBM流量或节省时延。

input RMS5120、Q_a norm1280、SWA norm512与C1 norm512仍是不同统计和参数域。HC的4H统计不能代替collapse后Y统计；Post/Y/Z的BF16舍入、native DynamicQuant的INT8/FLOAT、Q_b BF16与Query tail64 RoPE均保留。F03到Query RoPE是跨多个边界的高阶组合，没有新的直接线性折叠。C1/SWA共享输入不等于把QA1280、KV512与C1512改成一次大norm。

owner上发送normalized Z、发送raw Y后在recipient归一化、或owner先投影再发送较窄carrier是不同图。Y/Z同为H宽BF16不自动减少通信；M=800、Q=800、N=6400时，recipient union归一化6400行，相比owner800行增加5600行；把QA扩成N行还增加N−Q行矩阵工作。原始Post→Pre公共前缀不能在多个方案中重复计入收益。所有C1其它投影、C2 FLOAT、IndexWeight和未来原始残差读者仍需保留或给出兼容输出。

必要工作与容量上界仍只在声明的直接稠密图、行域、精度和共用资源池成立时才能形成T≥max_r(W_required_r/C_upper_r)。参考全行同时保留raw4H、Y、Z及所选QA/KV/C1输出，在H5120时为66048B/行，尚未加入其它保留量；它只是该明确保留策略的载荷，不是所有合法实现的存储下界。H分块、重算、部分输出和表示重组仍可选，实际预算与通信进展合同未建立。沿用v148精度门槛，profile HF32标签不能直接选择内部矩阵指令容量。

条件关系192→197，待核对403→398，联合组保持26；F03已有19个有范围的伙伴，15个待核对，未关闭整个候选。原9类/1352次COMPUTE操作数缺口、849个PMU缺口、16个phase未绑定图、3136项归属歧义，以及G14全部16342条拷贝的原生操作数和223941条其它运行时记录范围保持。整模块关闭0，全模块物理理论上限、最低利用率与实际收益仍未建立。没有新增服务器或设备任务。
