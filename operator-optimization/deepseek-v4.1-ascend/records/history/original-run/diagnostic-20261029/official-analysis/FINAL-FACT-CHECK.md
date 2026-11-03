# 交付前事实核查

检查范围：`publication-draft/README.md` 的首次 `<details>` 之前、`official-capture.json`、`official-operators.json`、`official-operator-shapes.json` 与现有三个复算脚本。仅使用本地现存证据，没有访问设备或重新采样；后续 EngineCore CPU 采样不在本次核查范围。

## 必要文字修正

1. **环境差异不能仅说 profiler 和调试入口。** 正文“仅增加官方 profiler 与调试入口”比记录更窄：隔离诊断还使用 bridge 网络、private IPC、独立 cache/tmp 和 custom profiling scopes。建议改为“沿用原镜像、冻结服务脚本及模型参数；隔离副本开启 profiler 与调试入口，并使用独立 cache/tmp、bridge/private IPC 和自定义采集范围”。这不影响现有指标算术，也不声称各项差异一定导致性能变化。
2. **旧诊断停止线的时间范围需明确。** 正文“已有服务不重启、不改配置”容易与本轮临时切换本人服务并恢复的记录冲突。建议改为“该旧探针未通过修改运行中服务取得；后续服务操作以本轮记录为准”，避免对整个报告作与已发生操作不符的表述。

## 复核结果和边界

- `verify_official_capture.py`、`verify_official_operators.py`、`verify_summary.py` 各运行一次，均 exit 0。正文各阶段窗口、输出吞吐、前缀命中、TPOT、AICore 与 HBM 显示、两窗各类并集中位和比例、已公开的 shape 统计与脚本输出一致；复算仅核对公开数据算术，不能独立证明原始 CSV 真实性、shape 身份、硬件上界或端到端收益。
- 公开 `official-capture.json` 中每阶段请求、窗口、计数、各 engine 指标及芯片采样与独立匿名提取 `workload-framework.json` 一致；前端两个 trace 的摘要 hash 和 RESTORED 状态相符。`HBM` 的 JSON 字段与正文均使用原始显示 `MB`，不声称物理 MiB。
- 正文对 profiler 单轮扰动、不同工作量、帧间隔和跨流并集的限制充分；Indexer 2048→512 的 37.5%/75% 是限定的逻辑工作量、回拷和重复次数，不是实测算子或整模型收益。目标工程中的 trace、原始输入、已保存服务证据不在这些复算 JSON 中。
- 扫描 `publication-draft` 内 Markdown、JSON、Python、CSV 的常见密码/私钥/令牌、JumpServer 地址、私人 IP/路径及 prompt/text/response 等字段，没有发现明显命中。该规则扫描不是对所有编码形式的安全证明，也不证明公开前不需要其他权限审查。
- 对 `publication-draft` 所有 Markdown 的相对文件链接逐项检查，目标均存在；正文 `REFERENCES.md#官方采集入口与版本限制` 对应标题存在。未逐一验证外部链接在线可用性。
