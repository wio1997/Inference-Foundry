本轮继续采用 profile 定位 → 明确未定问题 → 定点代码补证。没有修改模型、启动推理、导入 torch、检查当前服务器健康或改变服务。

保存配置的事实已经分清：DP2、TP8、EP，主模型有效日志为 PP1、DCP1、warmup1、capture sizes[8,16,24]、copy_inputs=False；早期 parser 的 warmup0、[1,2,4,8,16,24] 不代替有效配置。dspark 草稿为5步、嵌套 enforce_eager=True，构造器另将 use_cuda_graph=False，因此草稿外层 ACLGraphWrapper 不选；主模型和嵌套编译体需要各自取证。

算法开关 method=dspark 与自适应预算 dynamic_spec_config.method 分开。后者源码默认 None；缓存配置和 live object 未快照，不能仅凭默认值宣称实际关闭。sample_from_anchor 的源码默认也不是实际 HF 字段，query bank20/24两种构造域保留。metadata 与 draft group 的 cache kind、逻辑 blocksize、backend/spec 分组、每组独立 bank 和 dtype/capacity 已分析。

PCP 作为跨卡替代方案分析了全局/局部 metadata、空 query、恢复 row map、RoPE、C2 状态所有权与输出权重切换。当前 v35 完整草稿调用栈是直接 CP 输入→base forward，支持该观察域使用 legacy CP；它不测得主模型或所有阶段的 PCP 整数。工厂里的 target/draft 注释也不是本次配置证明。enable_pcp、decode-sharding 等精确谓词读取遇到堡垒机鉴权失败，没有产物，仍明确保留缺口。

4项条件细化与20条约束写入既有候选，没有新造候选计数：
1. F32：将所有必要的逐行投影移到 PCP gather/restore 前。逐行仿射映射与 concat/index_select 可交换，682个含重复行/物理 PAD 的整数示例核对通过；这只证明行映射恒等式。SWA KV 也是全局 hidden 消费者，仅搬 compressed KV/score 不能消除原 gather；C2 相邻池化、历史和 ring 更新须保留正确恢复顺序。native GEMM/量化舍入没有因此得到证明。
2. F16：比较 PCP 输出投影的权重/activation 调度与分片重组，保留 warmup aliases、pending handles、等待与 finally switch-back。不把可能缓存的权重 gather 当作每轮必发通信。
3. F31：坐标与 RoPE table 索引 producer 可研究直接写入各自固定 bank；已有 local/global 分离和表指针 dedup 属于基线，不能重复计收益。
4. F31：在所有读者绕过且 bank 专有的 eager 空 query 路径，可条件省去 SMLA plan 的 value zero_。保留 task/frontier、input-ready、reuse/release 和全局 cache/output 通信；下一次正 query 前必须完整覆盖 bank。19次 draft zero callback 与19个 no-query pattern没有按计数配对；QLI 和主模型 logical-zero graph 不借用此结论。

原 S13 的63个静态 guard 中，27个具体位置已与 metadata 选择、cache update、preprocess、attention、forward 的数据流对照；36个余项按原 ID/owner/condition 明确列出，下一步复用 M07-M11/M19 证据核对。18个稳定语义范围、268个导航 site 和8项原始 mandatory gate 保留，8项仍未证明完整关闭。实际主图/QLI body、参数 bank、所有读者与跨代复用、16个未绑定图、同分母资源上界、完整进程/回传仍未分析完。

本轮8项登记为4 pass、4 error：远端保存配置读取成功；远端谓词读取鉴权失败；本地缺冻结文件、长 loader 读取守卫和持久化前字段名校验三次错误保留，字段名修正发生在共享状态更新前；定点源码、数学/guard 分析与持久化审计通过。失败没有被标记为成功，也没有自动重试登录。全部作业已终止，35模型候选、7框架候选、0整模块关闭，26理论比较 INCOMPLETE、3136资源行、849缺 PMU、16未绑定图保持。Goal active，未因报告或条件变体成稿标记完成。
