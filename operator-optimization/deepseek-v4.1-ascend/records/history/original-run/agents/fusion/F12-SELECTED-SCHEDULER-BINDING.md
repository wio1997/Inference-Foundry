# F12 的 32 个选定窗口：与 scheduler descriptor 的关联及边界

对 [32 个选定事件](../diagnosis/INDEXER-ATTENTION-EVENT-IDENTITY.md)，按 layer `scope_id` 唯一连接已核验的 sidecar 层级；32/32 可归为 conditional target，形成 EP3、EP8 各两个批次，每组涵盖层 2、8、14、20、24、28、32、36。四组分别的 runner 祖先 schedule digest 在已采集数据内各匹配到 1 条成功 scheduler 返回和 8 条 runner 入口。摘要是内容哈希、不是全局唯一 step ID；这里说明已采集范围内的身份关联，不是跨时段唯一性保证，也不是设备 tensor 内容绑定。

四组 scheduler output 的匿名、无请求顺序 descriptor（每项是 scheduled Q、descriptor C、是否记录 spec 数；其中 C 来自 new/cached descriptor，不保证等于设备 C）如下：

| scope | total Q | (Q, C, spec count) | prepare 物理 R |
|---|---:|---|---:|
| EP3 digest `7ddf7ae9…` | 4760 | (1548,31220,null), (6,32768,5), (6,32768,5), (3200,29568,null) | 595 |
| EP3 digest `db852bff…` | 8180 | (3328,29440,null), (1652,29568,null), (3200,29568,null) | 1023 |
| EP8 digest `1b826883…` | 8180 | (1524,29440,null), (3328,29440,null), (3328,29440,null) | 1023 |
| EP8 digest `8c18c293…` | 5144 | (1804,30964,null), (6,32768,5), (6,32768,5), (3328,29440,null) | 643 |

记录中四组的 descriptor Q 与对应 C 之和都等于 scheduler post 值。该相等是描述符算术，不是设备执行或推测拒绝后值的核验；已记录的 `null` 表示 hook 未采到可数 list/tuple，不能当 0。两组含有两条 `spec_token_count=5` 的 descriptor，因此不能把它们整个批次简单称为纯 prefill。两个 total Q=8180 的组分解不同且摘要不同，不能只因相同 total/shape 将它们视作相同 batch。请求按 HMAC 排序仅用于生成 digest；输出主动重新排序和去标识，不能把表中顺序作为 runner 实际排列，也不能把各行对应到某个具体 prepare 行。

R 与总 Q 在前三种数值上存在算术一致性：4760/8=595、5144/8=643、ceil(8180/8)=1023。但即使知道 8 rank，一致性也**不证明**具体 CP 构造路径、补齐后位置、某请求 slice 边界或每个物理行是否有效；同一个 R/总量可对应多种排列和切片。设备 `positions`、`seq_lens`、`query_start_loc` 的逐行数值、张量身份和修改，特定 CP/backend 分支及二进制版本均未与 32 个任务闭合。故不能套用 F12 源码条件证明为这 32 次运行确证 `p=L-Q+i`，更不能据此删过滤。已有历史中查到的 scheduler descriptor 比先前只列“未发现同次长度”的广义说法更细；仍缺的是 device 侧同一请求、同一 metadata 和 slice 的值及绑定。

复算脚本 [f12_selected_schedule_bind.py](f12_selected_schedule_bind.py) SHA-256 `4288cb2e80d07c7a5206a783744c8af14f21dbe5ee0a24d3e6d72491cbfcf23f`；匿名结果 [f12-selected-scheduler-binding.json](f12-selected-scheduler-binding.json) SHA-256 `d4798d9751c643d0ef70249dc0d83f41c5a1d89ffbf0821cf1da0dd1883f8016`。脚本重新校验选定任务与层级输入 hash、23 份 sidecar 的 early-summary hash、各摘要 canonical 内容和每组匹配数量；只读取历史文件，不输出请求 HMAC、原文或 KV 内容，不启动 NPU、不修改目标实现。结果限早期采集窗口，不证明未采到的数据不存在。下一步若找到同次 GPU metadata 的值及 slice/rank 映射和相应执行分支，可以在设备侧逐行验证 F12 不变量；scheduler 数字或物理 shape 单独无法填补该缺口。
