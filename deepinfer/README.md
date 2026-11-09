# DeepInfer：910C 原版框架适配基线

此目录保存 2026-11-09 验收的完整 DeepInfer 快照，原版参照提交 `553741557a57df44a46fb3ab72b6aafeaf945337`。继续使用原版 LLMEngine、Scheduler、Worker、ModelRunner 与上层 Sampler/DFlash 策略，适配 NPU 算子、图资源、通信及缓存接口。

## 目录

| 路径 | 内容 |
|---|---|
| [DeepInfer/](DeepInfer/) | 完整验收源码、已物化 vendor 文件及匹配的 aarch64 编译依赖 |
| [SOURCE.json](SOURCE.json) | 原版提交、验收快照与编译依赖 SHA-256 |
| [validation/](validation/) | 组件结果、普通服务联合验收及独立审计摘录 |
| [changes-from-fd7.patch](changes-from-fd7.patch) | 相对此前可用基线的 24 文件实际补丁 |
| [changed-files.json](changed-files.json) | 本轮改动文件与前后 SHA-256 |
| [verify_snapshot.py](verify_snapshot.py) | 只用 Python 标准库验证完整源码和发布清单 |

## 已通过的三项适配

| 项目 | 结果 |
|---|---|
| YOCO 窗口 | 1/64/128/129/256 × mask/compact；共 10 组、80 个 CP 切片 |
| 严格跨批一致性 | DP0/1 × prompt 33/257/769/1025；8/8 组 token、logprob、top-token、top-logprob 精确一致 |
| 原版确定性生成 | 固定 seq_seed、temperature=1、top_p=0.7、nextn=5；5 步随机草稿图和 EOS 通过 |

最终联合验收为 **1 次候选加载、16 个自由采样请求 + 1 个公共 EOS 请求，共 101 个输出 token**。此前定位/失败的执行量另存于 `validation/acceptance.json`，这些数字不是整个开发过程的总量或理论最少次数。原版调度器 22 个源码文件逐字节一致；Engine 30 个方法、Sampler 7 个方法与 5 个辅助函数保持原样。

源码清单 SHA-256：

```text
36d2a7d7117bfb3144e0e5acd0cd4bfb3d34202f6b0b4044aae36653ec8b0046
```

```bash
python3 deepinfer/verify_snapshot.py
```

## 验收环境与当前入口

DeepSeek-V4.1-Flash-1017 W8A8；Ascend 910C，16 rank，DP2 / CP8 / EP16；CANN 9.2.0、Torch 2.10.0+cpu、torch_npu 2.10.0.post7。原版确定性配置为：

```text
profile=original-deterministic
YOCO=compact, keep=140
BI=1, MTP_VARLEN=0, nextn=5
ZOS=1, enforce_eager=0
```

[验收实际参数](validation/runtime.json) 另记录 max_tokens=16384、max_model_len=16368、prefix_tokens=12288。`DeepInfer/ascend910c/serve.py` 支持 `plan/start/inspect/stop`，`plan` 仅生成计划：

```bash
# 在匹配的 172 运行环境中、从 DeepInfer 源码根目录执行
python3 ascend910c/serve.py plan --service my-deepinfer --profile original-deterministic
```

`launch.json` 保存原验证环境的项目路径、容器身份、模型挂载与依赖路径；换机器需先匹配这些运行条件。编译依赖是 CPython 3.12 / Linux aarch64，模型权重使用独立挂载。此快照不是通用一键安装包。

`DeepInfer/README.md`、`DeepInfer/README-910C.md` 等原有文件为保证验收源码身份按原字节保留，其中旧版 bridge/native140 等描述属于历史入口；本基线以本页、`original-engine` / `original-deterministic` 配置及本目录验收记录为准。源码和 vendor 已完整物化，作为本仓库普通文件保存。

## 性能与许可证

当前源码尚未进行新的 vLLM Ascend 性能对比；功能通过不记作加速。已有研究和历史性能记录保留在仓库的 `projects/` 下，按各自冻结版本和负载解释。

DeepInfer 及其各 vendor 目录保留原许可证；匹配的 libstdc++ 运行库许可证见 [third_party_licenses/](third_party_licenses/)。
