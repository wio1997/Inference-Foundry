# AISBench base API 源码归档更正

## 错误与修正

本修正前检出的仓库 commit 为 `a51cca7b9c5e0cfe415ed240dcfc70f87952bacf`。其 [base API 路径](../code/aisbench/base_api.py)当时内容是 157 项来源记录组成的 JSON 数组，SHA-256 为 `e6e224dec62fd7a6d7448daedf1aa547b42165aa637cb3be391bd0da33cd1c9e`，不是对应的 AISBench Python 模块。该错误字节在检验过的历史 commit `238dd46ca7583a28e677f4a1521e06d050be17ce` 也已存在；历史 Git blob 为 `2745f076b980ee3c4a90df07564b1c0634c62e86`。此更正不改写 Git 历史，也不推断错误产生的具体原因。

一个 JSON 数组恰好可能被 Python `ast.parse` 当作合法的列表表达式，因此“语法解析成功”并不能证明其为预期源文件。现从保留的 G27/G1 D 侧来源副本按**原始字节**恢复该文件，共 966 行、41088 字节，SHA-256 `203a416444b972193274a13e1593328a288b6879a3d62e4f201bbb9adfa81755`，未重写其内容。

## 来源及证据边界

- [来源元数据](../evidence/aisbench_base_api_correction.json)记录原错误对象、恢复源字节摘要和来源收据标识；[provenance](../provenance/artifacts.json)记录恢复源 SHA 与公开副本 SHA 相同，[SHA256SUMS](../SHA256SUMS)核验公开文件。原始包含环境细节的只读收据留在受控存储，不放入本仓库。
- 归档中已有 [g45 客户端 usage 合同](../evidence/matched_SG_G316/128K/formal/client_usage_contract_g45.json)记录该源码同一 SHA；一次已有 G576 静态文件只读观察也记录了同一 SHA。恢复时核对的是本地保存的原始源码字节和这些**历史记录**，没有重新访问服务或镜像。
- 此前严格隔离的实际 import 尝试没有成功。**源码静态 SHA 和 AST 结构不证明运行时导入成功，也不证明服务器实际返回 usage、请求计数正确或未来测量有效。**此前性能和质量结论不因这次归档修正自动变更。

## 回归检查

[语义结构检查](../scripts/aisbench_source_guard.py)要求模块顶层存在唯一 `BaseAPIModel`、其 `async generate` 方法直接包含 `for … in range(self.retry)`，循环内 `try` 存在流式及非流式 infer 调用、异常处理和成功后 `break`。这是静态结构检查，不导入 AISBench、不请求服务，也不是动态重试行为证明。

完整档案检查已调用该结构检查；[回归测试](../scripts/test_aisbench_source_guard.py)创建合成 JSON 文件，先验证它可以被 `ast.parse` 解析，再验证结构检查将其拒绝；另验证真实源码通过，以及错误类名、重试上限、错误方法和不完整 try 等结构被拒绝。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_aisbench_source_guard.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_reproduce_029.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_analyze_trace_030.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```
