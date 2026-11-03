本轮补齐F29–F32在draft SWA路径上的条件组合：上下文KV预计算与SP gather前移投影如何共同选择计算、通信和发布方式。没有运行服务器、模型、Torch/CANN/NPU、capture/replay或优化实现。

先复用固定profile中的104个context→Tile顺序前沿，精确定位312次draft SP gather和88次主上下文feature gather。初始“每次SP后都有query KV”的假设被第一组dummy记录否定，原脚本与错误已保留。补查得到264次proposal窗口各有一个BF16[M,5120]×[512,5120]的KV投影；48次dummy窗口则有来自dsa_v41.py(698)的正面ZerosLike记录，对应None-metadata清零分支。它们是BF16[20,5120]，不能拿来绑定原80次BOOL24字段。没有因缺少投影记录就猜测分支；每条dummy结论有源码栈/CID关联证据。

源码进一步限定了可共享内容：上下文来自三份target辅助状态经main projection和完整5120维norm；当前query来自本层mHC pre/input norm，两者数值不同。选中draft层的context和query投影表达式都使用该层self_attn.wkv/kv_norm，因此可以研究按行打包、共享weight tiles；不能把两份KV值当成公共子表达式。draft role明确是SWA-only；目标C1/C2的额外投影、FLOAT或quant/index平面仍属其他域。context原RoPE模块与query inplace partial RoPE也须保留各自位置、dtype、舍入及读者，不能因共同512维就合并数值边界。

88次主上下文feature gather仍保留，count=640×Nc、P=8、BF16。这里只把已经全局可用的normed context送到新的KV计算owner，不删除上游gather，也不假装获得完整main weight的免费复制。仍需绑定实际communicator成员/顺序、每层参数副本、row map和全量读者；相同group label或rankSize本身不足以证明这些条件。

在明确的新ABI模型中，每层保留Nc-s行context在各卡重复计算，另有s行按owner分发，query在SP owner投影后交换512维BF16平面。每sender的carrier内容为1024×(m_q+ceil(s/8))字节；原query hidden gather内容为10240×m_q字节。不超过原carrier内容的条件是s<=72m_q。检查了5类正面profile几何的全部19637个s选项，得到2460个计算行数/接口内容的非支配选项；这不是硬件性能Pareto曲线或最优s。

小例Nc24、query20、m_q3：全部context分发时，每层carrier为30KiB→6KiB；保留一次main feature gather并计入3个不同层的carrier后，是122880→49152字节。每卡每层的dense投影行数为44→6。query实际20行却需按8×3=24行发送，新增计算不能凭空忽略padding；也不能把变化概括成无条件8倍工作减少。

长例Nc6400、query10、m_q2：全部context分发时，carrier为20KiB→802KiB；加上保留的main gather和3层carrier，8253440→10655744字节，接口内容反而增加。与此同时dense投影行数6410→802。只分发144行context时，可维持原20KiB carrier，投影行数变为6276。此处是计算与通信的取舍，不能单凭小维度平面宣布整体更快；GEMM小M效率、weight工作集、Vector/Cube/link竞争、打包与通知成本尚未测定。

实际cache接收者是关键条件。若原始context已在所有相关卡可用，一行需由R个卡消费、其中k个卡各自计算，则新接口需要R-k份512维endpoint值交付；k=R可不新增KV传送。9330组接收者/计算者参考用例覆盖了这个计数关系。接收者local方案、全复制cache方案和部分分发方案属于不同图，profile没有记录足以选择它们的真实cache需求集合。endpoint内容也不是网络wire下界。

联合打包还须保留每个重叠slot上的context→query写顺序。两次原调用分别把context值11和query值29写入同一slot7，原最终值为29；按rank排列打包后，一个合法串行顺序可能把context放到最后，得到11。这个有限反例不是native race或实际slot的发现，而是否定“同一批行可以任意scatter”的推断。可选方案必须保持role-aware写入、真实RAW/WAW范围、旧读者退役和noncausal query-KV就绪，不凭图增加全局barrier。

按层推迟context投影直到query输入ready，还会把context bank的最后读者从早期precompute延长到后面的draft层，影响跨迭代复用与队列容量。既有“3份context投影合成一个1536输出、分别做512 norm”和“按层C/Q row packing”是耦合的调度选择，不能同时领取两套完整launch/IO收益。所有数值都是指定表示的逻辑内容或当前/候选dense工作计数，不是HBM必要字节、HCCL wire流量、实测延迟、物理MFU或理论最低时间。

中央新增REL130-01，条件关系136→137、待核对459→458；HG85补入F32成员，联合组仍26且全部子集/运行时合法性未关闭。主候选35、历史框架记录7、语义358行/26模块、部分图854节点/1346边/48提示不变。原9类/1352次操作数字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义仍保留，整模块关闭0。前置核验通过，整体目标继续。
