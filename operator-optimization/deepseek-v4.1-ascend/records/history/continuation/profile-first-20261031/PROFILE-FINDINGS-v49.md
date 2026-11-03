本轮从profile已定位的采样模块继续补齐明确代码缺口：只读两个PyTorch接受函数及其直接依赖的两个恢复函数，四个函数均与历史166源码SHA一致。4个函数、10个if位置及20个分支臂完成局部源码解释；这不是全M17覆盖，也不是运行频次。

发现block接受公式的后端差异：PyTorch先逐项min(r,1)再cumprod，Triton每步min(prefix*r,1)。合法两词表反例的概率比分别为1/2、2，uniform均为0.9；Triton累计值为0.5、1，PyTorch为0.5、0.5，最后接受位置分别1与-1，因而发出的token数量不同。6174个精确有理数例中214个最后接受位置不同。这是源码公式的非等价证据，不是当前负载实测故障，也没有执行Torch/Triton或设备测试。

若无draft_probs、q=1且概率有限归一化、有效段没有placeholder，两式在实数域可相同；仍须保留F64/F32随机数、placeholder、归约顺序及native差异，不能扩大为两后端无条件等价。

恢复路径也有独立选择条件。根调用未传use_block_verify，默认False；启用block验证不会自动启用blockwise恢复。普通恢复使用max(P-D,0)，单独的blockwise路径使用排他前缀p_i后的max(p_i*P-D,0)。原有request共享指数随机数、零/Inf处理、greedy原输出保留和全部随机数draw顺序均已写入契约。

这些证据细化了现有F35的先决策再恢复、逐行entropy摘要，以及H02的结构buffer复用条件。没有新增主候选，没有实现优化或声称时延收益。当前profile只提供840条采样API、1144条TASK、14条链及72次greedy Triton任务，不能证明非贪心、block或fallback分支发生。

原profile访问入口仍未恢复；本轮没有新连服务器或启动模型、设备、服务/进程检查。其余采样条件、全部score/raw/processor消费者、跨层/卡/迭代关系和资源分母仍需继续。35模型候选、7框架候选、0整模块关闭；整体目标保持进行中。
