本轮展开Engram掩码为假的算子需求，补读已缓存的非旋转Triton gate函数体，复用已有旋转路径及WKV profile，没有运行服务器或NPU。v62已经指出“gate为0也不能直接返回hidden”；本轮没有把这条旧警告当作新发现，而是进一步拆清key/value需求与组合代价。

普通旋转路径先做hidden旋转、key/hidden RMS、加权dot和gate，再把token_mask为假的gate填为+0；非旋转融合体同样在全部RMS/dot之后选择gate=0，随后仍加载共享value，计算hidden+gate*value并写回hidden dtype。后者仅在rotation=None、NPU和特定能力guard成立时调用，stride不符抛错，零token不发kernel。它使用(tokens,HC)程序网格，H5120补到BLOCK8192、num_warps4、enable_fp_fusion=False都是源码设置，不证明当前910C的执行分支、物理占用率、实际指令或8192元素必要工作。已有208次WKV记录继续归于原来的旋转普通路径范围，没有替换成这个融合体。

在有效地址/shape、没有其他key/KV/FPstatus/错误/hook读者、并保留数值与状态合同的域内，false-mask行的key投影、key/hidden norm、weighted dot和gate旋转不再影响输出；真正留下的是 roundBF16(addFP32(float(hidden),mulFP32(+0,float(value))))。因此可以研究key/value拆分，不能把整行、lookup或历史写入一起删除。0*Inf、0*NaN仍会使输出NaN；hidden=-0、value为正的有限值时，原残差变为+0，直接复制hidden会改变位。对全部65280种有限BF16 hidden及8种value类别，522240项主机标量参考通过。该结果验证此标量域，不是原生NPU精度或NaN payload测试。

对有限BF16 h/v及分离的IEEE RNE运算，直接复制h的精确标量条件是h不是-0，或v的signbit为1。一种更简单的充分整行合同是所有hidden有限且不是-0，所有投影后的value有限。当前没有这样的实际负载证书；输入有限也不能保证投影不溢出。证书建立、扫描、版本检查、存储和就绪时间都要计价；若先算完value才验证有限性，这份投影工作已经发生，不能再算作节省。保留NaN/Inf/有符号零等原行为的value摘要或分类接口是另一个可能的表示域，本轮没有把完整value GEMM称为所有算法的必要工作。

令T为本地输入描述行数、A为真掩码行数、B为另有充分证书可整行绕过的假掩码行数，0≤B≤T-A。模型D6144、H5120、HC=4下，选定常规稠密点积域的需求参考为 MAC=D*H*(4A+T-B)，原基线为D*H*5T；私有BF16 KV值数同样从5HT变为H*(4A+T-B)。B=0时所有行仍需value，只有key按A行计算。26个形状/条件模型和3876个独立小维度输出坐标集合核对通过。它们是明确算法域的计算/接口参考，不是全算法下界，不是HBM流量或数值物理上限。

拆分代价不能遗漏。若先算所有R=T-B行value，再算A行key，逻辑输入接口会引用R+A行；若先算A行完整KV，再算其余R-A行value，每行输入只引用一次，但两个调用都请求value权重块。当0<A<R时，后者每调用权重内容合计6块，而原来是5块。以profile中T3形状、假设A1/B0为例，MAC参考降到7/15；完整活跃行+非活跃value拆法的权重调用内容却是377487360B，对比基线314572800B，增加20%。value优先拆法则读4行逻辑feature，对比基线3行。唯一冷权重内容仍为314572800B；这些数量都不能直接当成实际HBM或提速。仅A=0时四块key权重整体无需求，value块仍为62914560B。新的融合分块可以减少接口重复，但还要支付tiling、mask、精度、调度和资源代价。

与已有方案的组合也明确收紧：v84若已经提前算了全WKV，之后发现false mask不能追溯节省已发生计算；应比较等mask就绪后按需算与提前全算的工作/重叠权衡。value-only缓存项不能算完整KV命中，后来true-mask读者仍缺key，必须保留内容、表/权重/量化/布局/精度generation及有效字段标签。value仍需求时，它依赖相同6144维lookup特征，不能同时领取F24整行UVA/lookup删除收益。即使gate lookup可删，v114的DEAD/pad/history cache写入、commit/rollback、其他层读者及集体通信参与仍须单独审查。

本轮复用5个WKV regime、208次执行，保留32次phase未绑定及TP副本口径。真实A/B、mask字段身份、负载证书、加载kernel、改变GEMM行数/列数后的舍入、量化输出耦合、所有读者和mask-ready时间都未知；80次待定BOOL[24]没有被指定为Engram mask。现有TASK时间不能当作这项改写的可删除延迟。必要工作与同精度/层级能力上界仍须分别证明，再组合时间下界；本轮没有获得新的数值物理时间下界、MFU或提速。

M14新增一条显式false-mask语义行，355→356条、26个暂定模块，原非旋转/其他分支条目保留并补入函数体证据。F24/F25既有REL73-20与HG84联合合同细化，134条条件关系、26个联合组、461对待核对不变。9类/1352次关键原生字段、849缺PMU、16个未绑定phase图、3136项含糊归属和0整模块关闭保持；整体目标继续。v111单独的44B Pad探针仍未执行。
