本轮补齐 F01–F09、F01–F10：HC 分支与 IndexQuery、IndexWeight、FP16 product 的融合和调度机会。仅使用保存的profile、直接算子源码和条件模型；整体目标仍未完成。

192组eager indexer前沿各新增关联9类生产者，共1728个原生TASK：QR1280 norm、Q_B与IndexQuery各自的DynamicQuant、INT8 IndexQuery GEMM、BF16 Weight GEMM及FLOAT转换/缩放/FP16转换、IndexQuery RoPE。与已有640组图前沿的IndexQ TASK键无重复，共832组已保存前沿得到条件关联；旧512组root绑定、128组phase未绑定状态保持。192组Weight与IndexQuant均在同一辅助流，176组Weight/RoPE TASK区间相交，72组C2中有66组IndexQuant/C2KV TASK区间相交。这些是记录到的时间区间，不能直接证明有效硬件重叠或改变调度后的收益。

输入与数值边界必须分清：HC当前系数读原始4H/FLOAT-HF32状态；Q_a与IndexWeight读归一化后的BF16 hidden5120；IndexQuery读Q_a后再归一化的QR1280，并走INT8激活与权重计算。IndexQuery和Weight不能直接合成一个同输入投影。Q_a与Weight可研究共享输入分块或1312列联合输出，但必须保持Q_a的1280维norm与Weight的32维非归一化结果，核对新M/舍入及packed输出stride，或使用两个输出缓冲区。Q_B与IndexQuery对同一QR表达式各有量化调用，描述符一致仍不能代替实际值、量化方法、世代和读者证明。

IndexQuery融合可沿BF16 GEMM→RoPE BF16→每head128维amax→舍入后的FP16 scale→INT8量化展开。不能在RoPE前确定amax，也不能只拿部分head元素就最终量化。例如有限参考旋转后，原tail最大值1变为约0.707，另一未旋转值0.5的正确INT8值与使用旧amax的结果不同。amax依赖完整128维head，允许在新合同下按head/行流水，不要求所有token共同完成；原GEMM/RoPE舍入、tile归属、FP16 scale每16个head的32B写入归属及发布语义仍需保留。

Weight路径保留BF16投影→FLOAT乘2^-6→FP16，再与FP16 QueryScale相乘。v64/v106的有限反例仍适用：缩放前先转FP16会过早溢出；省略product之前的Weight FP16舍入也会改变值。本轮将这些已知限制接到HC分支，未把它们重新算成新发现。

当前Weight在IndexQuery之后才启动，辅助流执行Weight时主流做RoPE，随后主流等待Weight，再继续其它准备。提前启动Weight或延迟到第一位真正读者再join有条件价值，但辅助流中的Weight与IndexQuant依然串行，提前的Cube工作还会与Q_a、IndexQuery、HC系数竞争。若将Weight×QueryScale搬进辅助流量化，而仍让主流在等待量化完成后才做Weight FP16转换，就产生“量化等FP16值、FP16转换等量化”的依赖环。可将原FP16舍入移入更早的辅助流/融合计算，也可保留QLI内的乘法；不能只删一个wait。

compact alpha=FP16(FP16(WeightBF16/64)×QueryScaleFP16)有两种不同交付方向。新的alpha-only QLI输入可在关闭所有旧plane读者后只保留compact product。另一个兼容方向是在所核对arch22、QueryScale只通过该乘法使用、有限值及原生乘1语义成立的条件下，把alpha作为原Weight输入，QueryScale改为常驻FP16全1 plane。后者保留QLI入口、乘法与broadcast工作，但需要额外constant bank、初始化、读取与生命周期合同；native FTZ/非有限值、实际加载kernel和其它架构仍未验证。

| 仅QLI前scalar接口的声明值读写 | 每query行的逻辑参考 |
|---|---:|
| 原Weight FP16与QueryScale FP16各写入、读取 | 256B |
| alpha-only新输入合同 | 128B |
| alpha加常驻unit-scale的原入口方向 | 192B，另有64B/行的unit bank内容 |
| 保留原两plane后另写读alpha | 在原接口上增加128B |

这些参考不包括Weight上游FLOAT临时、INT8 query和QLI内部M16 product broadcast；后者的声明写读参考仍为2048B/query行，不能因改成compact alpha就自动删除。所有数字都是所选接口的逻辑内容，不是HBM、wire字节或提速。query行M与SP sender行也有区别：当前有sender834、query830的记录，前移交付需处理真实行映射、pad/trim和参数副本。

12288组独立加权DAG比较了当前图、仅HC提前Y、提前Weight、延迟join及辅助流compact product。模型保留同一辅助流的串行关系，并把QLI/attention/output共同后续服务放在最终HC系数join之前。给定假设服务中，compact相对单独FP16路径有5174组较快、2384组较慢、4730组相同；这些不是实际负载胜率或硬件最优时间。当前4H系数、post/comb、下一FFN的pre4和Post原始X残差仍然存在，不能把各项独立接口或launch收益直接相加。

本地分析曾出现路径变量遮蔽和binary16参考函数处理整数溢出的错误，均已更正并保留记录；完整校验随后通过，没有运行模型或原生算子。新增REL139-01/02，条件关系148→150、待核对447→445；HG93加入F09、F10，联合组仍26，全部子集和native联合合法性保持开放。35主候选、7历史框架记录、358语义行/26模块、4472次固定SP关联和854节点/1346边/48提示的部分图保留。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图、3136项归属歧义未补齐，整模块关闭0，全部模块物理MFU和最低时间仍未建立。
