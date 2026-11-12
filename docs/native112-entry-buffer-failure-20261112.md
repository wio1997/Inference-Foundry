# 原生112：入口覆盖通信缓冲的失败与修正

2026-11-12。原生112没有产生正式性能结果。8个实际Worker、加载的扩展身份均已核实，但在模型权重加载前全部触发环境契约检查：预期1280MiB，实际400MiB。8rank退出码均为1，0个正式AISBench请求；管理器完成容器停止及占用释放。关闭阶段的Gloo断连是连带错误。

根因是此前修正不完整：外层启动入口已设置1280MiB，SDK校验也要求1280MiB，但 `examples/glm5_full_model_offline.py` 的MTP初始化仍无条件写入400MiB。配置在进入SDK前被再次覆盖。这是明确的实现遗漏，不是模型精度问题或MC2算子不支持。

源码 `c4ec8e000d04966574970802ae01db76e57a01b5` 相对 `5fd2cd7` 只改这一处：开启prefill MC2时设置1280MiB，原decode-only路径保持400MiB。权重、模型计算、算子、共享配置、AISBench负载和参考111均不变。

已执行实际入口AST中的环境设置语句，在初始值16/400/1280MiB、prefill开/关的6种组合下验证最终值；旧源码的1280→400覆盖也已复现。检查不导入torch、不运行模型；它证明入口配置修正，不能替代NPU验收。

原生113使用这项修正与同组参考111配对，当前未验收。此前112失败完整保留，参考111已通过的成绩不重跑、不与旧原生混算。参考C4/C8为215.04058/374.02446 token/s，双档30%所需门槛279.55276/486.23180；当前没有新的原生收益结论。

[八rank失败、退出及采集SHA](../validation/shared-features-20261111/native112-entry-failure.json) · [入口回归](../validation/shared-features-20261111/native-entry-buffer-regression.json) · [参考111参数和结果](shared-prefill-mc2-reference111-20261112.md)

这是共有能力接入修复，不作为DeepInfer专有加速。原6/12小时预算和完整Goal保持不变；CPU/NPU逐位精度搜索、参数网格及新调度系统不在本次修复范围内。
