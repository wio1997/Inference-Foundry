# 169 重采集：sidecar 调用嵌套

已保存的 23 个 metadata 文件与 early summary 的哈希一致；2,757 个 begin 和 2,757 个 end 按 PID、TID、call_id 唯一成对，label 一致、outcome 均成功，同线程栈无交叉。16 个 rank 的 trace 目录 PID 和 EP 唯一。

每 rank 有 80 个 layer scope 在 execute_model 内但不在 DSpark 内（conditional target，层 0–39 各 2），另有 39 个在 DSpark 内（draft，层 0–2 各 13）；16 rank 共 1,280 / 624，待判 unknown 0。对于嵌套 layer、多重 model/runner 祖先或无法归属的场景，复算程序保留 unknown；本次未见。

1,280 个 conditional target 都可取得最近 runner 的 schedule digest，四个 digest 各覆盖 8 rank、每层各 8 条；624 个 draft scope 没有 runner 祖先，因此没有可绑定的 parent digest。该结果不支持靠时间邻近将 draft 归入某个请求或 batch。输出 JSON 按 role、layer、digest 给出匹配数量和 CPU sidecar wall 时间的最小/中位/最大值，并包含单条可定位的 call_id。

已经在 scheduler 证据中关联到 8,180 调度 token 的 digest `db852bff987d2b98d3dc0734e3731ec5528e37dcf6e874860d1ebb4fd12eefc8`，层 0 有 EP0–7 八条，可从 call_id 736-6、753-6、1113-6、2128-6、2165-6、2711-6、3556-6、3612-6 定向检查 trace flow。其 CPU sidecar wall 为 37.477–43.063 ms，不能解释成 NPU 运行时间、设备利用率或单层瓶颈。

现有 scope-summary-v2 的 16 rank 各 144 个 trace scope 都各自唯一匹配 sidecar label、数字 PID 和 TID，未见同线程交叉。但 trace-to-device flow 尚未闭合，sidecar 是 hook 包裹的 CPU 时间，包含埋点开销；不能由此判定设备在 scope 范围内执行了什么。图重放和缺失调用也不能由此排除。

复现脚本 [summarize_recapture_sidecar_hierarchy.py](summarize_recapture_sidecar_hierarchy.py) SHA-256 `7ff9fc9e27aa82a7c4f69169b6908dc39348d83e493a57e2602a50b5a88d4721`；对包含 runtime 的固定诊断目录以 `--root` 指定，纯 CPU 读取。受限输出 [sidecar-hierarchy-20261029.json](../../diagnostic-20261029/sidecar-hierarchy-20261029.json) SHA-256 `e4ffaf3cacaeebddb6beb249d9f48d76a3800dc5f02528cdc2ae46e6c3a8a023`；[scope-summary-v2.json](../../diagnostic-20261029/scope-summary-v2.json) SHA-256 `3cf4a030e5e665a485bb9dccd77d9d8c77308bf425cc1d01cb4e16ba78d7ec6b`。脚本还重新计算每个关联 runner digest 并核对原摘要的每个文件哈希。
