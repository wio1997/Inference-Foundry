本轮补齐剩余960次eager target通信与HC/RMS/Post的关联，使原始v66清单中的4472次attention SP均有相应源码角色前沿。这个完成范围仅指该固定profile清单，不能代表全部模块、条件分支、实际融合合法性或整体目标完成。

这些960次通信没有唯一COMPUTE_TASK节点，但有HcclAllGather的精确CID和调用栈。在同一globalTid上找到唯一的forward CPU父区间，再按40层HC889、RMS896、AG898、RS901、Post902、FFNPre905、FFNPost919的源码调用序列配对。24个forward各有40组，均处在正面的host execute_model记录内。随后通过CONNECTION_IDS关联原生TASK或COMM，未按设备时间最近补造节点。CPU区间包含关系也不等同于设备区间、正式客户端step或跨卡同次迭代。

新关联包含960次Q_a、960次SWA KV、24次C1 BF16投影、各72次C2 KV与score FLOAT投影，并保留48处Engram正面调用、72处辅助状态读者和24组初始HC。C1的compressor71与SWA的multistream475是不同源码角色；C2在hidden.float()后分别执行530/531两次投影。相同形状不能证明C1与SWA权重相同或输出可CSE；本轮eager证据也不能替代旧的80处captured graph角色歧义。

每组320次通信对应sender M800/816/834，SWA KV报告的行数分别6400/6528/6668，而均匀gather行数为6400/6528/6672。最后一组中，280次Q_a为834行、40次为830行。统一按owner的834行投影，会在后40次增加4行Q_a计算；不能机械领取8倍工作减少。真正的SP/CP row map、receiver trim、padding错误行为及新M下数值等价性仍须满足。这里的N是算子报告行数，不直接等于客户端有效token数。

前移投影的接口内容必须覆盖全部活跃分支。以每行原BF16 hidden为基准，SWA512平面是10%；若还要传Q_a1280则为35%。保留第二份C1 BF16平面时，分别为20%或45%。C2必须保留两份FLOAT512：与SWA合计50%，再加Q_a为75%。v88已有不含Q_a的50%公式，本轮新增的是72组大M原生投影的正面角色关联，以及Q_a路由/padding一并计价。其它Indexer、raw输入、rstd、C1/C2和cache读者尚未闭合时，这些不是完整合法载荷；若仍需传原hidden，新平面甚至成为额外内容。

不同dtype可以研究多次typed collective或保位packed ABI，但不能把FLOAT直接转BF16以取得表面节省。新布局、对齐、打包、队列、ready事件、接收者行序、pool/state/RoPE/cache世代和原始kernel舍入都需计价。原始COMM内部记录数实际为196×935、207×19、210×4、193×2，不能固定假设每次196条。保留了HC、RMS、QA、SWA、AG/RS及Post的native区间分布；这些包含原调用的等待/传输及竞争，不是普适带宽、互斥计算时间、请求关键路径或优化前后benchmark。载荷比例不能直接变成延迟比例。

本轮没有新增中央pair，而是补强现有HC/SP、QKV/SP、C2/SP与C1/SP关系：142组条件关系、26个联合组、453组待核对仍保留。35主候选、7历史框架记录、358语义行/26模块和854节点/1346边/48提示的部分图不变。9类/1352次原生操作数字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义均未被本轮关联消除，整模块关闭0。全程未执行设备任务或修改目标模型，整体继续。
