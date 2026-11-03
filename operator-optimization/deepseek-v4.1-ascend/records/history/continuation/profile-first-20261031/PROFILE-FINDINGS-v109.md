最小探针已获得用户明确授权并完成；没有加载模型、replay图、发送推理或修改既有任务。

开始前检查原166（实际host105）的16个设备AICore均为0、8张物理卡均无运行进程，原容器running/PID2032500。第一次Python提交在创建/data/tiankuan/wio输出目录时因只读挂载退出；尚未加载NPU库或导入Torch。保留该失败后，只把输出限制改为自有可写/tmp/wio-operand-probe-v109-*目录，其余探针代码保持一致，完成了一次实际NPU捕获尝试。未启动、重启或停止原容器。

成功探针耗时约6.55秒（进程墙钟，不是算子延迟），自有应用张量8240B，另有驱动/上下文/分配器开销。安装头文件SHA与历史控制一致；当前磁盘libascendcl.so解析到cann-9.2.0并记录SHA。这些仅是新受控诊断的环境证据，不证明历史模型进程的加载ABI。

自有图共4个任务、1条流：两次受控提交aclrtMemsetAsync(count=64B)和aclrtMemsetD32Async(N=16)在dump中均呈现MEMCPY_ASYNC；随后为ZerosLike内核和NOTIFY_RECORD。3个DEFAULT任务的GetParams均返回107000，精确安装头文件定义为ACL_ERROR_RT_PARAM_INVALID，没有返回填充范围。该返回码未给出根因，不能据此认定所有DEFAULT任务永久不支持，更不能把受控MEMCPY路径等同于原profile的ConstantPadNd-owned MemSet。

BOOL[24] ZerosLike的GetParams成功，报告argsSize=64、opInfoSize=136、isHostArgs=0、numBlocks=1。dump渲染8个uint64 word，其中已知输出地址出现两次、数值24出现三次、已知输入地址未出现。它证明可以取得这一个内核的通用包信息；重复地址/数值并不解释每个字段的ABI，也不识别原模型中BOOL[24]的语义字段或全部读者。没有直接解引用opaque参数或读取张量内容。

图未replay。dump中的ts/dur不用作设备实测时间；argsSize64也不能当作实际填充64字节，和两条受控fill恰好都写64B无关。没有计算物理带宽、MFU或提速。

退出后8张卡仍无NPU运行进程，原容器PID不变，SSH控制连接已关闭。两份小型原始输出及失败/资源/源码记录均已保存。本轮把“接口能力未知”推进到上述受控实测，但原7类MemSet/1272次和2类BOOL[24]/80次执行仍缺字段；MC2同call计数及其他关键证据未获得。不能因此批准删除/融合这些算子或缩小缓冲。

不再重复等价的小图探针。后续必须绑定原生provider/ABI和同call/图/更新/存储epoch；完整模型诊断会加载或replay模型，超出本次最小授权，需要另外准备具体有限计划。其他工作继续围绕算子融合与理论界，已有profile和定点源码分析不受影响。

35项模型候选、353条语义记录/26暂定模块、132条条件关系/25组/463对待核对以及9类/1352次执行缺口保持不变。整模块关闭数仍为0；全模块最低MFU及整体物理理论上限尚未建立。
