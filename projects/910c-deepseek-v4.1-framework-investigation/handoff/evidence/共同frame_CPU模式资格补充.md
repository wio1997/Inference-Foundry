# 共同frame CPU模式资格补充

介绍：给上一段“原 CPU helper 结果复用”补输入模式门槛，防止把普通 CPU Tensor 的原型核验当成所有服务输入模式已通过。

到了哪里：仍用原 owned image、Torch2.10.0+cpu、后端 autoload 关闭、-S 显式 site 路径；第二个私有 CPU 目录 1.78s 完成。NPU/模型/服务/profile 操作 0，前后无 TorchNPU/vllm/NPU 运行库。

结果：v1 的 1728 用例/44928 次字节核验及六组配对 CPU 成本是普通 CPU Tensor 的原件。普通 Tensor 在 inference 调用中仍通过；若 Tensor 本身在 inference mode 中创建，直接 `_version` 会报 `Inference tensors do not track version counter.`。这是实际反例，原件保留。

guard 变体仅把版本读取替换为 `None if tensor.is_inference() else tensor._version`，其它 CPU frame/首快照/原 helper/optional-out fallback 不变。三种创建/调用模式共 78 次原函数字节比较，0 差异；测试代码生成的 guard 源和 SHA 保留。该变体仍必须依赖封闭调用方的不可变输入租约，不能据此容忍 NumPy 写入，也不能将 v1 普通模式的完整配对收益直接当成 v2 的稳定收益。v2 三个单轮成本只作敏感性，未用于请求预测。

已确认可实现，限定 CPU 原型：带模式 guard 的封闭 CPU 结果复用。已停止：把 v1 `_version` 读取和 mutation 检测泛化到所有 Tensor/NumPy 写入。待确认：框架实际输入模式与集成、共同 device projection、原生 packet/图/地址/末读以及请求净收益。新框架替换准入 0、隔离 CPU 原型通过 1，不是完整框架逐位验收。

下一条仍为 loop046 Run105846Z：一次原主流 CP device projection 至 base 首缓存、全部 native closure、稳定 QSL 与事件 owner。此次模式门槛不改写已结束 loop045 的原实验或判定。证据：common-frame-cpu-modes.json、common-frame-cpu-mode-guarded.py、common-frame-cpu-mode-frontier.json。
