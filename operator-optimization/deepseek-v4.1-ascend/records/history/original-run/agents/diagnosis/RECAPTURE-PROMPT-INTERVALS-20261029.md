# 169 重采集：prompt 调度区间复算

23 个已保存 metadata 文件均与 early summary 中 SHA-256 一致；其中 453 个成功 scheduler 事件、8 个匿名 HMAC。每个 HMAC 恰有一个 new descriptor，后续记录来自同一 scheduler PID。初始值 3×29,568 + 5×29,440 = **235,904**。

以 32,768 为 prompt 末端截取记录中调度区间，8/8 从各初始值连续到达末端，无缺口、重叠、重复或其后重新进入 prompt 的记录。六条在首个记录到达，两条分别是 [29,568,31,220) + [31,220,32,768) 与 [29,440,30,964) + [30,964,32,768)。共 **26,240** token 的 prompt 区间；每条区间、状态与数字在输出 JSON 中。它是观测到的调度描述算术，不能称为硬件完成次数或实际浮点计算量。尚不能仅凭 HMAC 将八条各自映射回客户端请求。

客户端 measure1 指标增量：queries 262,144，hits 235,904，prompt 262,144，preemptions 0。首个 descriptor 合计恰好等于 hits 的聚合数，但它们不是同一字段；指标没有提供每个匿名请求的细分，且缺少外部 ID 到内部 HMAC 的映射，因此不证明每条初始值就是本地 prefix 命中数。上述 preemptions 为测量窗口的聚合计数，不单独证明每个 HMAC 的历史身份。

## 语义依据与边界

- 当前本地保存的 hook [sitecustomize.py](../../diagnostic-20261029/instrumentation/sitecustomize.py) SHA-256 `6e9bbadd7205a08f95de0c63a022645accbfe9fbae8a74e37d7859e3b4c8ef08`：91–125 从 scheduler output 的 num_scheduled_tokens、scheduled_new_reqs 和 scheduled_cached_reqs 字段读取数字；136–146 读取 scheduler 自身状态；其 199–211 先调用原 schedule，再采样返回值及 scheduler 状态。这个字段映射是 hook 语义，不能独立证明历史源码版本。
- 保存的 scheduler [vllm_scheduler.py](../bounds/metadata-path-snapshot/vllm_scheduler.py) SHA-256 `c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884`：912–925、983–1016 中初始查找所得 computed 可以是 local 与 external 之和，KVTransfer / streaming 还可沿用已有 computed；1249–1265 决定 new/resumed 并设置请求 computed；1324–1343 调用 NewRequestData.from_request 构造输出，其中 v2 会将 resumed 合并进 new；1346–1353、1613–1666 对 cached 保存更新前 req.num_computed_tokens；1454–1456 调用更新后才返回；1520–1534 以 scheduled 值增加请求 computed，且源码 1522–1529 明示推测 token 拒绝时可调整。
- 以上 scheduler 是之前保存的源快照，**尚未证明与本次 10 月 29 日采集的实际加载代码逐字节相同**。快照未包含 NewRequestData.from_request 的定义，单凭当前快照尚不能独立验证 new descriptor 字段由请求如何复制；八个初始值、与 post 的算术相等是直接记录的事实，区间解释仍以版本及构造语义为条件。
- [profile_client.py](../../diagnostic-20261029/profile_client.py) SHA-256 `e3528495c48ae02f114dc3943263ba67c6494f795b67db4bfef9079408d9c2a3`：76–79 先种前缀再采 before，94–101 采 after 并计算测量窗口服务指标差；服务聚合不等于逐请求归属。spec count 为 null 意味 hook 未记录可数 list/tuple，不能当零。

复现脚本 [recheck_recapture_prompt_intervals.py](recheck_recapture_prompt_intervals.py) SHA-256 `f4e5726b20d526e401ec14444094eaae98f842104a8f1d1841a86366c61582a2`；离线使用 `python3 recheck_recapture_prompt_intervals.py --root <diagnostic-20261029目录>`，只读取受 SHA 验证的早期数据，不调用 NPU。受限输出 [recapture-prompt-intervals-early.json](../../diagnostic-20261029/recapture-prompt-intervals-early.json) SHA-256 `caf464ff24246da79f831df1fb357b75dbeb7c4cd75d0636dafce40ae9209799`；early summary SHA-256 `2e322b98e2f972e35d74cb2d5fc98dd7071a0ff48f38f8b47a62d91a2b48538a`，progress SHA-256 `609fdeb1f65d348eb2bdc8347dda1fe609803af56cef6205b15427d0df5f8e82`。
