本轮按 profile 补齐 M21 的 C2 metadata 和初始化写入：39 条原未归属链、4280 TASK，其中 C2 3256、block/slot fill936、QSL fill88。208 次 FP32 gather 来自两处 C2 RoPE 表查询。各主语句执行数104（88OTHER+16dummy）与32个skip零化TASK按31×104+32对齐；这是计数核对，不是请求/回调配对。source copy_ 的效果不因缺独立 ComputeInfo 而被忽略。

C2 metadata 的五行公式已确认：ring0=max(seq_len-query_len,0)，ring1=按实际请求/实际 token/skip 截断的used，ring2/3均start，ring4为used>0时首个block。完成行由有效 token 边界与 position 奇数决定，源位置为position−1，否则0。两个 FP32 表按源位置查询。现有 packed 分支已经有融合 builder；其直接 kernel 读普通QSL/seq/position/firstblock/FP32表、写同一五行和mask/source/rope ABI，没有读取 packed cache 对象。因此可条件扩展该既有机制到当前逐算子路径，无需重新发明metadata producer。保留 INT32/INT64/FP32、索引/整数值域、负位置 remainder、cos/sin 共享 stride、bank别名与frontier；没有编译/运行新kernel或声称native验证。

skip_ring_update 不等于整个回调可删：used/complete为0，但ring0和start仍动态，source位置0仍发布实际表第0行的cos/sin，不能默认1/0。复用skip plan还必须证明输出bank没有被中间非skip写入覆盖，单有相同输入key不足。_publish_task本就按c2:compressor去重，并在device metadata与captured COMPRESSOR分支选择异步Task或同步执行；不把已有去重算新增收益。

直接消费者复用v24同SHA源码：pool读ring metadata/旧FP32ring，indexer.update_keys和long latent RoPE均读sourcecos/sin，之后写各自cache。任何lazy lookup、坐标压缩或producer pipeline都需保留这些读者、旧读先于ring更新的WAR、固定bank和首读ready/末读reuse。

初始化效果也已分清：QSL容量尾填−1、slot padded区间填−1、block table inactive行填0；encoder分支另分配零tensor。PAD_SLOT_ID已核为−1。v38 fused slot kernel会先写−1 tail，后续slot fill可在同bank、同代际、完整区间覆盖且无中间写入时视为重复发布；这不证明所有936 fill TASK可删。Dirty-range可研究只重置新变脏区间，但必须先有正确sentinel初始化，跟踪全部writer/alias、batch缩小/增长和固定capture完整读者。比如active end从4缩为2，至少仍需清[2,4)，不能只因bucket不变省全部写。

登记7项条件细化、26项约束，更新F31/H06并关联F18/F19/F34/H02。选中的186条native链现无未归属项，仍有108条部分字段链需要有限审计，且QLI eager/captured读者、M22-M25及全模块/分支/跨层跨卡跨迭代审计尚未完成。0未归属只表示本selector导航完整，整体不完成。未实现/未测速仅是候选证据级别，不单独算未分析缺口。

3次远端CPU点读均pass，复用旧consumer/profile，加主审4runs全部pass。35模型候选、7框架候选、0整模块关闭。26理论比较INCOMPLETE、3136candidate资源行、849缺PMU、16phase-unbound图保持。无model/torch导入、新NPU/服务/健康工作。Goal active。
