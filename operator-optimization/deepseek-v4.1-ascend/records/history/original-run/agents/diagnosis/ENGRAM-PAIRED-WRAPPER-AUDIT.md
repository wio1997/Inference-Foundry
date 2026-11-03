# Engram 配对输入的弱引用对象关系

只读分析既存 169 诊断日志，不访问 NPU、模型权重或用户输入，不更改目标实现。`engram_paired_wrapper_audit.py` 在 stage 容器内读取诊断日志及 progress/status，验证日志 SHA-256 为 `b32f03b3aca61939fbf76e56a6c7faaa0d1d8a9c4d225515f03e72082c44cfbf`，按同进程、同调用号及 run/window/status/PID 关联，生成仅含汇总的 `engram-paired-wrapper-audit.json`。分析范围是 826 条 begin 和唯一正常 end 配对，不是全量调用。此前独立审计的采集完整性边界继续有效。

这 826 条的 tensor 弱引用 token 全部非空且互不相同；storage wrapper token 全部非空，按同 process 分为 658 个 token，其中 168 个恰好各出现两次（另 490 个仅出现一次）。168 对每对均为同线程、相邻调用号，相同 shape 与 num_tokens，不同 tensor token，输入 offset 依次为 0、24；先一调用的 Python 正常返回记录在后一调用 begin 之前。只按同 process 比较，不能跨进程比较 token。

该结论以保存的 observer 源码 `diagnostic-20261029/binary-binding-draft/second-binding-run/engram-audited-bootstrap/audited_observer.py`（SHA-256 `b87308a636201eebd97b48b605382409a05d8e661eea471bd811d819ab710d48`）对应本次运行实现为条件：`identity` 同类型仅在已有弱引用 `ref() is obj` 时返回旧 token；每次分配新 token 递增，弱引用失效或者条目被淘汰后不会复用旧 token，同一对象被淘汰后可被赋新 token。所以同进程同 storage-wrapper token 支持两次观察时为同一 Python wrapper 对象；不同 token **不证明**不同物理存储。该 token 既非裸地址，也不是物理内存不变的生命周期证明；未记录两次间的数据是否被修改、设备读操作是否结束、graph 或跨流事件。若安装记录不能独立证明具体 observer 文件身份，这个语义仍有运行时绑定缺口。

这些记录支持将两个跨 slot 查表放入共同 producer 的条件候选，但它们未保存 layer、table、实际 lookup 后端、物理存储或下游事件，不能只据此把特定两层和设备 task 绑定；offset 顺序相容于同一 `gathered[:,0]`、`gathered[:,1]`，却不单独证明真实 producer。`embed_end: returned` 是 Python 函数返回，**不是 NPU 异步任务完成**；即使同一线程先返回后再次调用，也不能推断设备读取顺序或原始存储可以提早复用。冻结源码 `model.py:1199–1226,1565–1589` 的条件路径含各层分开的 lookup、通信与 ready 事件；最终联合前仍须保留 table 身份、逐层 event、跨流依赖和存储可见性。联合读取与单层免复制候选会重叠，不能将二者按两份收益叠加。现有 15,725,568 字节是全部 826 条调用的中间结果逻辑大小，不应把共用 wrapper 误当作一条调用、删除一半或当作实际 HBM 字节。
