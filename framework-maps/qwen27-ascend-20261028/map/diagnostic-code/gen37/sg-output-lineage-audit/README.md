# SG 输出谱系 CPU 审计组件（未发布，待 root 验收）

此组件复用第35代三个原字节脚本，以显式完整请求 ID、RID、rank/PID 和 step 关联保存的 scheduler commit、图回放 invocation 与原先已接受的 native compact。它是已有 cleanSG gen67、aarch64 / 910B1、CANN9.0.0、物理4–7卡TP4、512/C1/out32 插桩诊断的 CPU 复核工具。没有启动服务、SDK或NPU；没有新性能数字或优化采用。

`executor-run-gen35/` 中三个脚本保持历史字节与相对布局；不要直接在原证据树执行它们，因为它们会写派生结果。新增 `check_cpu.py` 先校验组件 SHA，再复制确切输入到全新的输出树，用 `-I -B -S` 的独立 Python 进程运行。只依赖标准库，不隐式加载 sitecustomize、torch 或 torch_npu。测试失败或输入不匹配会保留日志并返回非零。

只跑14个合成身份/负向用例：

```bash
python3 -I -B -S check_cpu.py --output PRIVATE_PATH_cb51f3b855e6
```

完整复核保存输入（输出目录必须不存在）：

```bash
python3 -I -B -S check_cpu.py --evidence-root /path/to/evidence-root --output PRIVATE_PATH_b6f936feadba
```

证据根可用第35代仓外 review 包的解压根：包名及全文件 SHA 见 `pinned-inputs.json`。该清单绑定30份保存输入，包括29份唯一谱系输入和一份相同commit的detokenizer源码，不把大profile、完整tokenizer、原始日志或上游源码树复制进本代码组件。完整模式在执行前逐文件核对 SHA/大小，复核后再确认源输入未改；缺失或变更不自动降级为通过。生成的源码契约和两份台账须与原件全 SHA 相同，谱系 JSON 仅允许 verified_ns 不同。

14例覆盖多token增量、累计跳号、重复step、错误零增量、混入其他RID、前缀ID匹配、不同PID、缺失/冲突端点和缺少原生层。CPU反例与已保存数据重放不是重新采集、实际204完成或aarch64设备性能测试。

32是逻辑输出token；132是四rank的commit副本行，包含每rank一个零增量tail。四rank不相加成墙钟。native端点关联复用原接受的compact，不代表独占kernel、全stream关键路径或可恢复收益；未知及跨边界原生行不被重新归属。源码AST契约只证明所保存源码的step/token传递结构，不能代替全部模型数值或线程时序证据。

当前tokenizer的32个累积内容前缀与SSE相等，不证明历史实际加载tokenizer的不可变SHA、逐SSE消息token传输身份、vLLM生成ID、首个分歧采样token或跨框架根因。全性能矩阵质量/greedy等价仍未知；跨CANN/卡组的冻结部署混杂、HCCL等待及别名去重边界保留。代码、证据和性能接受分别有范围；root负责整体最终核验。未分R、未发布、优化未采用。
