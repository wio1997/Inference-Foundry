# v161：171 联合探针与共享证据增量

本次单独授权的联合最小探针只执行一次，输出正确；临时探针容器已退出，原主服务恢复并返回 health 200，恢复快照中的全部 NPU 进程属于原主容器。单次授权已消耗。171 的全部资料位于 `/data/tiankuan/wio` 下。普通模型启动及单次回答“2”通过，完整 C8×32768→1024、前缀90%负载尚未在171实测。首次启动与恢复后的PID属于两个运行时段，原始169/166-105记录的主机与时间身份保持不变。

此前 v160 的有界 CPU 读证和此次 v161 结果合并做一次权威更新，减少重复检查点工作。6组 Engram 同名参数在基础/extra分片的当前payload逐字节相同，共读取1,258,618,880参数payload字节，相关文件前后stat稳定。这解决了这6组初始数值内容的磁盘副本歧义，后续无需为选数值而研究一般loader顺序；实际加载布局、驻留、更新代次和历史值仍未证明。

首次普通启动时，检查21个自有进程，20个进程各映射两份CANN9.2前端库，共40条记录。容器overlay路径的stat与kernel VMA inode不一致，已沿Docker精确22个层根、仅两个相对路径找到真实backing文件，device/inode及完整SHA相符。它证明该时段映射的ELF backing身份，不证明私有COW代码、运行时调用选择或实际device.o指令字节。4个SDK声明及AArch64类型/布局CPU编译检查通过；可执行文件只依赖libc，没有新NPU上下文。复用已成功的v127c 23例CPU结果，不重跑初次失败构建或整套用例。

联合探针在逻辑设备0创建BOOL[21] owner、offset=1的BOOL[20] view及两个BOOL[24]输出，应用NPU张量共69B，运行时/分配器/workspace开销另计；无模型、无建图、无replay。两段CPU profiler均打开shape/stack：关闭profile_memory的对照没有TLV5/6/7，开启后有6个Tensor详情事件和5个Scalar类型事件。本例8字段的地址等于owner storage基址，实际view数据指针比它多1B；storage_offset、nbytes和实际标量/列表值仍不在该内置记录中。因此复用内置字段，并只补影响alias和融合判断的缺项。

真实拦截记录1条MemSet HostArgs：152B参数包、48个launched blocks、placeholder [8,24]，launch_rc=0；任务/stream getter未调用，rc=-1和task_id=0是logger哨兵。首word落在已记录的2560B NPU分配范围内，偏移1024B；公共输出24B对应的分配器记录为512B。应用字节、storage范围、分配块大小和真实写量不可互换。分配器stream token与原生opaque stream handle不直接混为同一命名空间；CPU profiler原始syscnt与CLOCK_REALTIME也没有按最近时间硬拼接。

inline96B记录与旧v128自有记录完全相同，可在“实际执行device.o匹配已pin对象、[8,24]正确提供第二参数、首word角色未被运行时改变”的前提下复用原标量证明：48个启动块中block0到达一个store位点，source-matched请求32B。本轮不重跑147个合成记录/7056个block trace。该32B不是实测HBM事务或GPU耗时；未把首地址差异当workspace角色、无alias、全读者退役或helper可删除证明。零relocation槽也不是NULL/no-data证明。

本次25份诊断文件共108031B，独立原始TLV/参数包解析和SHA检查通过。期间本地控制模板替换和离线Counter解析错误已记录并修正，没有因此重提交原生探针。主服务的成功恢复是本次授权的组成部分。

旧Loop004拒绝跨硬件用例的登记后，保留其冻结合同并记录PIVOTED，建立171派生Loop005。已发生执行按回溯登记标明，不伪称预注册；没有重新执行设备操作。

覆盖仍保持全部G00–G14、35+7候选、26模块358语义行；清单不是穷尽性证明。新增配对、独立性证明、原始native字段回填和整模块闭合均为0。9/1352原生任务、849 PMU、16 phase、3136归属、G14 copy/control、全部未闭合分支与物理容量/最低MFU仍保留。下一步冻结一次可审查的171固定负载诊断采集方案，优先复用内置profile字段并解决显式原生关联与graph创建/更新缺口。整体任务未完成。
