# Zcode 操作指南（DeepSeek Extreme P0）

Zcode 是外部命令，适合执行边界清楚的机械任务：环境检查、启停服务、现成控制脚本、benchmark 和日志提取。主 Agent 负责实验设计、正确性与性能裁决，并独立验收 Zcode 的实际执行。

## 1. 命令提示写法

最可靠的 headless 形式是让 prompt **只包含一条完整命令**，末尾停在命令本身：

    timeout 1500s zcode --prompt "bash scripts/run_loop081_example.sh" --cwd /data/wio/Inference_Foundry --json

不要在命令后放句号、括号里的补充解释或示例参数。2026-09-29 两次 prose prompt 即使说明“句号不是参数”，Zcode 仍把句号作为脚本参数，受控脚本以 64 退出；纯命令 prompt 成功启动了控制脚本。零参数命令尤其应使用纯命令 prompt。

在调用前确认命令、cwd、输出目录和现有进程；已有 live 输出目录时不要盲目重跑。测试连通性的 1 不等于一次实验委派。

### 本项目已验证的调用原文

以下两条在 2026-09-29 的 Run665 实际调用中把命令正确交给 Zcode；prompt 必须恰好以 .sh 或选项结束，不要追加句号：

    cd /data/wio/Inference_Foundry
    timeout 180s zcode --prompt "bash scripts/run_loop081_metadata_graph_run665.sh --preflight-only" --cwd /data/wio/Inference_Foundry --json

    cd /data/wio/Inference_Foundry
    timeout 1500s zcode --prompt "bash scripts/run_loop081_metadata_graph_run665.sh" --cwd /data/wio/Inference_Foundry --json

这两条只是调用格式示例。再次运行必须换新的 Run 编号和独立输出目录，先验收当前服务、源码及 controller 状态。纯命令 prompt 仍需按下面的规则核验内层脚本结果。

Run667 的正式 OFF/ON E2E 也使用同一形式实际启动，纯命令 prompt 为 `bash scripts/run_loop081_metadata_graph_run667.sh`，外层命令为：

    timeout 3600s zcode --prompt "bash scripts/run_loop081_metadata_graph_run667.sh" --cwd /data/wio/Inference_Foundry --json

Run667 controller 完成 2 个服务 arm、每 arm warmup48 + 3×48 正式测量，并写出 `live/cleanup_status.txt` 全 0。Zcode CLI 在 controller 完成后仍可能保持运行；以 controller、服务/NPU 和 source SHA 的独立检查判定内层完成，不能仅凭 CLI 沉默重跑。此命令是已用格式示例，已有 `run667/live` 时不能再次运行。

## 2. 启停服务

只让 Zcode 运行已经过 主 Agent 审核的 guarded controller。控制脚本应自带：

- 独占锁和服务/NPU preflight；
- source SHA pin、可逆安装与恢复；
- 有界启动、health、client 超时；
- 失败时停服、确认 8 卡 idle、恢复 source；
- 每次 Run 独立输出目录及 cleanup_status。

不要让 Zcode 根据自然语言自行拼接服务参数或裁决性能。服务启动成功以实际 PID、health 和 controller 产物为准。

## 3. 返回值与超时

Zcode CLI 外层 exit 0 **不代表内层命令 exit 0**。核验其报告的实际命令、cwd、内层退出码、signal/timeout、stdout/stderr，再直接读取 controller 的 cleanup_status、admission、rank/runtime 文件和 source before/after SHA。

2026-09-29 的 Run663/664 内层 guarded script 以 1 结束，Run665 内层以 0 结束；三次 controller 均完成 stop/restore，但 Zcode CLI 长时间未返回。此时先独立证明脚本已结束和服务/NPU/source 已恢复；只终止本次调用自己启动的 Zcode PID，记录 CLI 被终止。不得把 Zcode 沉默当作脚本仍在运行，也不得因 CLI 沉默自动重复实验。

如果 Zcode 的模型/runner 身份无法从输出和日志确认，写 unknown；不要猜。重要结果以 raw evidence 和 主 Agent 复核为准。任何 timeout 默认为 inconclusive，除非独立证据能证明具体动作已完整完成。

## 4. 失败后的继续规则

- 参数错误且未创建 live Run：修正 prompt，核对无服务/无 patch 后可重试。
- 受控脚本已运行：先检查 cleanup 和证据，再决定是否需要新 Run；不原地重复。
- 校验器自身有 bug：纠正证据类别和 TaskCtl，修好校验器后再跑；不能把假阳性当候选 correctness fail。
- Zcode 调用不稳定时，改用更小的纯命令任务和更严的验收；不要因一次调用失败就长期跳过 Zcode 的机械分工。

项目分工与证据门槛仍以 AGENTS.md、HANDOFF.md 为准。
