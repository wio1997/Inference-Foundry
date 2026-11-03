本轮检查目标采样F35到下一轮Engram hash/lookup F24的跨迭代边界。复用旧profile中88组req-index/cache-write/hash和176次UVA lookup，未新增运行或执行归属。一次hash已覆盖两个Engram层、request搜索已每token一次，均是当前基线。旧F24模板的CPU资源标签不能用于所选native任务的资源计数：这些hash/lookup是加速器任务，lookup输入来自host-mapped存储；CPU、NPU、UVA和通信容量仍须分开。

定点核对默认构造器得到新的整数范围证明。令M=2^63-1，压缩词表大小为V，且1<=V<=2^31；默认multiplier由B=floor(floor(M/V)/2)、u∈[0,B)、m=2u+1产生。映射值和pad均在[0,V-1]，所以乘积不超过M；DEAD在乘前被替换为pad，非负63位乘积的XOR仍在[0,M]。正常cache初始为0，指定writer只写有效mapped code或DEAD，维持该不变量。因此，这个明确域内不需要额外的INT64溢出回绕假设，正除数的有符号/无符号取余一致。实际注册buffer、外部覆盖、无效raw ID和越界状态不在证明内，INT32最终表索引布局与整数硬件吞吐也未由此证明。

对一个确定位置、request、参数和已解析历史，可预备6个INT64历史前缀H，共48B：每层对最近1/2/3项，按源码的DEAD/负位置阻断规则作乘积XOR。当前token有效时，各层各ngram的R=(mapped_current*m0) XOR H；当前token被阻断时，R是全pad的参数常量。之后保持原来的prime取余、offset和INT32写入。最终48个hash ID共192B。12,288组源码整数参考、589,824个ID比较，以及独立12,288组因式分离参考和4,001个范围用例通过；这些均是CPU参考，没有原生编译或设备验证。

提前计算有严格条件：H必须与最终hash调用的in_batch/known-window/fallback解析结果相同，且在选择前不能依赖尚未确定的current ID、dead predicate或当前cache写入。否则只能准备逐候选的H，或保留后算。已有hash ID是取余后的INT32值，不能直接当H；若保留取余前R/H，需要新的输出、bank和生命周期接口。dead proposal把整段都替换为pad，不能通过简单XOR移除proposal项恢复alive recovery的历史。未知window里的-1会走fallback，含义也不同于已解析的DEAD。还保留了单个最长accumulator不能通用移位更新的反例。

采样后只应为真正有效的accepted/recovered/bonus输出选择历史与owner，物理输出中的-1占位和未接受proposal不能直接索引token_map。候选前缀H、hash或只读lookup预备均保持私有，不能提前提交canonical历史。原有slot-cache程序可能先写proposal、再按rollback覆盖；新的逻辑committed suffix接口不能据此删掉原物理状态协议。token-map/multiplier/prime/offset/pad、request/位置、窗口/slot-cache快照、DP/TP owner、COW页和lookup表/scale epoch均要匹配，原始ID的embedding、用户输出、penalty、logprob和其他读者继续保留。

理论工作量没有按选择后阶段单独报收益。已有H时，终态只需2次当前项乘法、6次XOR和48次取余，但H的构建、写入、筛选和保活都要计费。若独立准备D+1个接受前缀H，再选择一个，参考总量相对一次晚算完整hash增加6D次乘法及4D+4次非平凡XOR；48次取余仍在。D=5时H载荷288B/request，另有标签/映射和生命周期开销。已有当前alive hash可研究保留取余前状态，但要新增6次XOR及48B/保留位置的写入；dead/current与最后bonus位置需单独处理。没有可用slack或资源容量证据时，不能声称这些额外工作能被隐藏。

跨卡也存在选择：sampler owner计算后发布192B hash包，或保留实际4/8B raw ID fanout后由接收端计算。若原始ID仍被其他读者需要，192B是额外通信，不能替代掉它。48倍/24倍只是包载荷参考比。TP8旧布局中，一个token两层各3个local head的冷读逻辑载荷为BF16 3072B、INT8_GROUP32 1728B、MXFP8_GROUP32 1584B；应按真正不同的(layer,head,hashID,table/scaleepoch)冷key及预取浪费计费，而不是套HBM峰值或假设预取命中。整数、UVA、通信的处理能力上界和关键路径证据均未补齐。

新增REL114-01（F24/F35）与三方组HG114（F24/F31/F35）。当前134条条件关系、26个联合组、461对待核对；35项模型候选、353条语义记录/26暂定模块及全部原生/归属缺口不变。9类/1352次原生执行仍待补证，0个整模块关闭，全模块MFU与最低模块仍未确立。原生Pad小探针等待单独明确回复，本轮没有远程操作、模型、NPU计算、捕获、replay或目标实现修改。整体继续进行。
