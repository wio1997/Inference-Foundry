# v159：检查点与内置 profiler 字段证据增量

本轮封存暂停前完成的只读取证。原完整目标、35+7候选、26模块358语义行、跨算子/层/卡/迭代范围全部保留；这些清单仍不是穷尽性证明。整模块闭合0，整体任务未完成。171迁移及本次普通启动是单独运行，不能填补原历史profile字段。

实际配置检查点的142个分片header已全部缓存并独立核对：143324个索引声明均在指定分片中存在；物理header有143330个条目，差值来自6个Engram同名参数在基础与extra分片各出现一次。索引指向extra不自动证明实际loader胜出者；本次未读权重payload，未证明这6组值相同，不判定损坏或原生失败。初次本地聚合错误地要求跨文件名称唯一；已保留失败记录并纠正为“索引声明”和“物理条目”两层计数，没有重新读取已缓存header。

全部40主层及3MTP的86个WO_A/B参数均已核对。WO_A为BF16[8192,4096]，WO_B为BF16[5120,8192]；描述中的FLOAT是量化类别，不能当FP32存储。全局检查点包含所有组，并不证明各rank已驻留所有组。精确镜像中的线性层切分、reshape及RL再加载路径仅建立源合同，不是实际对象/回调执行证明。

这也补充了初始化域：候选实现可以在加载时从全局检查点填充所选复制权重，不必仅因“原rank只持有分片”而强制每轮AllGather。对于已声明的TP8逻辑内容模型，方案B额外56MiB/层/rank，方案C额外126MiB；若选40主层，分别2240/5040MiB。初始I/O/H2D、实际分配/格式、缓存、workspace、代次和退役成本仍要计价，MTP不能自动套用同一TP域。稳定权重代次之后，参考图B的直接输出值交付量为A的2/3；C仍要求原分组BF16 partial及SUM语义，不能替换成一次完整WO_B点积。以上均非实际wire/HBM量、时间下界、显存可行性或可达加速证明。

精确镜像声明torch_npu 2.10.0.post7、Git提交1d88b8608fd1e44545b5804026c7a57c44240bb4；8份公开源文件已校验Git blob，3份安装/缓存Python文件逐字节相符。该提交的正常CPU RecordFunction路径需要record_shapes&&(with_stack||with_modules)&&profile_memory才写Torch输入细节。record_op_args不在此门内，但另有CANN配置转发，不可泛称无效。见[固定提交的原始实现](https://github.com/Ascend/pytorch/blob/1d88b8608fd1e44545b5804026c7a57c44240bb4/torch_npu/csrc/profiler/npu_profiler.cpp)。声明版本和Python相符仍不等于已安装编译对象、实际callback或绕行覆盖已实测。

内置序列化器提供TensorImpl、storage基址、dtype、元素字节数、shape、strided布局的stride、device类型和序号；不是t.data_ptr。它不提供storage_offset/nbytes、NPU实际格式、分配/参数更新代次、全部读者和tensor内容。Scalar字段是类型标签，ScalarList在此路径不展开实际值。相同已发字段可对应偏移0/2的相邻不交叠view，也可对应偏移0/1的交叠view；因此空白或相同/不同指针标签都不能替代alias证明。参数角色还须结合schema、输入类型/顺序和有效tensor过滤，不能把压缩列表直接当全部原始参数。

后续按共享证据问题推进：复用内置字段，定点补offset/范围/代次/值/原生桥和读者；核对实际9.2 provider ABI后再准备有界验证，不另写一套重复tensor logger。1272个eager与80个captured缺口保持分路，历史9/1352、849PMU、16phase、3136归属及G14 copy/control、完整分支与物理容量/最低MFU均保留。此次新增配对、独立性证明、原生字段回填和整模块闭合均为0。
