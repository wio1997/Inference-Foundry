本轮按保存profile→必要定点源码推进M20及下一ID交界。覆盖4464原生API、7520 TASK、28条链、104条完整COMM；104组seed到五步发布的源码顺序模式全部匹配，每步11TASK，共520步。16组dummy B4、32组proposal B2、56组proposal B4。5824个seed/step TASK之外保留552个base index/head/reconstruction和1144个下一ID TASK，没有按相同计数或最近时间绑定target请求/root。全体127920原生API补查确切scheduler/confidence/取回/backup位置后仍0个ComputeInfo-linked命中；不据此认定配置关闭或没有CPU、memcpy、H2D/D2H和同步。

Markov与confidence的输入关系已经说清：T[:,0]是已知seed，步骤i读Embed(T[:,i])并把预测写入T[:,i+1]；early-return分支返回T[:,1:]。第一列不是“多算后丢弃的第一个预测”。confidence再次读取的正是T[:,:D]，因而可条件地保留/流送循环中的embedding，不能错移成当前新预测。confidence使用raw last_hidden_states，将同dtype的hidden与Markov embedding拼接再float，经过FP32的5376→1投影和sigmoid；它不使用base LM-head的RMSNorm输出。可研究直接双源读取、查表复用和pipeline，但必须保留原dtype/单次dot/row关系。

实际DynamicSpecScheduler在spec_decode/utils.py。它clamp概率、FP32 cumprod，再周期性以threshold计数的.item()更新共享budget，然后从整批survival[:,min_k:]选globalTopK，按request计数得到INT32长度。不是每请求独立固定k或阈值截断。周期counter、B0、ceil/奇数调整和“本次更新立即用于本次分配”的顺序都有影响。旧注释的Qwen3-8B/A3成本观察不证明910C/CANN9.2最优预算。device预算重组需要新的选择/发布ABI，不能只把Pythonint换成Tensor。

长度在take_draft_token_ids取回阶段被逐个int(k)、clamp，再通过zip对已生成列表tokens[:k]裁剪。这证明它是输出取回时的prefix，不是当前Markov生成时的提前退出。批量异步回传长度、与draft IDs按request/generation一起发布，是H02条件变体；不能把源码中的B个标量消费说成测得B次同步。prefix上界剪枝也只能在最终extra预算L已固定、有足够已评估候选、未来native survival上界严格低于边界且尾部无其他观察者时考虑。周期更新不能拿旧L套用；非因果draft decoder仍需要其未来query/KV行。这里只给条件证明，没有实现或测速。

下一ID辅助函数88组，各13个ComputeInfo TASK，输入K1或6。它先生成CPU请求状态中的backup并拷到GPU，克隆sampled IDs、把discard行填-1，然后使用字面谓词(x!=-1)&(x<V)计数，再取max(count-1,0)列，count0回退backup。并非寻找任意位置最后一个有效token：[7,-1,8]得到count2且选-1；[-2,-1]得到count1且选-2。单核可用虚拟discard读原矩阵，精确计算同样两项输出，省私有clone/mask/index平面；不能擅自增加x>=0判断或把count-1改成max-true-index。默认方案保持backup构造/发布，另将sampler直接发布摘要与device backup代际复用列为独立条件。

count还有异步消费者：eventNone会提前返回；否则copy stream等待default，非阻塞D2H并记event，async模式保留GPU count引用，随后更新prev_sampled IDs。融合/复用不能只保持指针，还要保持CPU拷贝、GPU修正和next-ID状态的ready/last-reader代际。producer的R+1只有在相同discard/ID有效性/prefix不变量下才可能代替这里的计数。

本轮细化F30/F31并关联F26/F35，同时更新H02/H06，记录11个条件变体、35项约束、分支台账与3张关系图。8次远端CPU作业7pass、首版tail跨度guard失败1次保留并已修正；主审通过。持久化在新helper v2中增加框架台账细化，原v35已校验helper保持字节不变。没有导入模型/torch、启动NPU、改服务或做新健康检查，最后健康仍v12post。35模型候选、7框架候选、0整模块关闭；26项理论比较INCOMPLETE、3136candidate资源行、849PMU缺失和16个phase-unbound图保持。下一批M21完整metadata/输入发布，继续用profile找未审字段，再定点补代码。整体Goal active。
