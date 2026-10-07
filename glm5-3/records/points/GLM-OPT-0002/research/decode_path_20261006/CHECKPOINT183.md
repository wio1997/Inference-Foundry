# checkpoint183 — MTP / 已有结果 patch 发布与设备调优暂停

用户指令：先把 MTP 代码与当前调试中有结果的代码分别打成 patch，上传 GitHub 并做好说明；两台机器暂时不做代码调优测试。本 checkpoint 的工作范围改为本地交付，不继承 checkpoint182 的设备后续动作。

## 交付范围与事实

[包说明](../../../../../patches/20261007/README.md)、[manifest](../../../../../patches/20261007/manifest.json)、[local verification](../../../../../patches/20261007/verification.json)。13 个独立 patch，加 H11/H13 原字节溯源替代版共 15 文件、33 个受影响文件实例；原文件 before/after 字节位于 fixtures.zip。SHA256SUMS 与 zip 独立校验值随包提供。

- H6/H5：保留已验证 research performance 身份和既有 PD 收益，不相加、不宣称当前 FULL 完整栈对 stock 的累计收益。H6 发布版是无实验 selector 的静态缓存；本次没有重编译/安装，原共用 rebuilt binary 因果证据保留。
- H11/MTP：proposer 字节与最终设备验证版本完全相同；主版 runner 在干净基线重放原实测单外层 capture hunk，`_dummy_run` 整函数 AST 与实测一致；原 d017→4c90 增量另附。默认 opt-in OFF；限域 all16 capture/replay/短与自然 EOS/PD 正确性通过，但 Run275 没有重复性能收益。发布版干净组合本次没有新的设备验证。
- H12：发布已验证 contiguous 版本，不混入旧 row-index 试验。H13：保留基线 CRLF 形成最小 diff，整个 AST 等于原候选；原 LF 字节另附。Run283 正确性与 Run284 INCONCLUSIVE 事实保留；不包含诊断 controller/bank selector/observer。
- H4/H8/H9：保留正确性及未重复性能收益结论。H10：实际 async=true，不适用；没有候选设备正确性/性能证据。
- F1/F2：KV 失败 logical IDs 和空 token/text SSE error/finish 终态的限域功能修复；不计性能、不冒称完整 API 验收。Run245/246 的 KV 容量处理是 1→2GiB 配置修正，不杜撰独立代码修复 patch。
- B1/B2：80,000 输入/600 输出/93% prefix/93% declared KV 条件与跨轮 salt、分端点 HBM/external 计数分账。main 74910eab 的上一 cache 同步只有 README；本次实际补齐 5 个 adapter helper 和直接依赖 `runtime/phase_runner.py`。B2 配置历史 patch 已在 main，不重复应用。

本地校验执行 `git apply --check`→apply→逐字 SHA/AST→reverse check→reverse→原字节验证，全 15 个 patch 通过。没有导入或执行模型代码，没有 native build，没有网络/服务检查或模型请求；0 NPU Run、0 服务器动作。GitHub main 采用独立基线的紧凑发布提交，研究分支保存暂停入口与完整证据；不把研究 raw/整条 ancestry 合并到 main。

## 当前状态与下一步

**PAUSED_BY_USER：两台机器的代码调优与测试暂停。** 未触碰服务/源码/进程。最后已有现场证据是 Run284 terminal guard 的 P249/D283 all16 healthy/idle、H6/H5/mainFULL ON、H11/H12/H13 OFF；最后只读 health 记录为 `2026-10-07T14:47:39.758Z`（北京时间22:47:39），不是本阶段的新核验，不推断之后现场仍相同。

交付后等待用户明确恢复设备调优/测试；不自动执行 Run、scan、重启或 patch 安装。此前 initial target critical-path 剩余归因仍开放，原源码/trace/裁决保留。最终 Current=None、完整 API/正式 80K600/93% SLA/稳定性/完整 stack E2E 仍未验收；此处暂停来自用户新的设备指令，不能解释成 SLA 未完成导致清空 H6/H5。

本阶段没有新增代码级性能 KEEP。
