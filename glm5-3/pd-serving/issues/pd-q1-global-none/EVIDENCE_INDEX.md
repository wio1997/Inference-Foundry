# 600 输出复现：证据索引与离线复算

此问题来自 **2P2D**，与 [4P2D 性能基线](../../baselines/4p2d/README.md) 分开保存。结论、发现过程、纠错和未证明事项见 [问题报告](README.md)。归档补齐日期：2026-10-08（Asia/Shanghai）；此次仅做本地复核和 GitHub 发布，没有执行服务器操作或新增性能测试。

## 发布范围与入口

原问题摘要、私有包哈希和匿名数据校验器已随提交 `e54a649dcc892376f5183ed19a0f25b3bacb1a77` 保存到 `glm5-3-autonomous-20261001` 分支。本次补齐原始包复算脚本、校验清单、复算结果和证据定位。GitHub 默认分支 `main` 与该工作分支不同，查看时须选择该工作分支。

| 材料 | GitHub 中的入口 | 能证明什么 |
|---|---|---|
| 结论与发现过程 | [README](README.md) | 工作负载、触发机制、观测关联、纠错、结论边界 |
| 完整原始包身份 | [PROVENANCE.json](PROVENANCE.json) | 原包文件名、大小和 SHA256；原包不在公开仓库 |
| 原包各文件校验清单 | [offline/SHA256SUMS](offline/SHA256SUMS) | 封包时每个文件的 SHA256 |
| 原始日志复算脚本 | [offline/verify_evidence.py](offline/verify_evidence.py) | 736×16 记录、首步 q1、KV 完成顺序、DP 模式传播、间隔统计，以及筛选行属于完整日志 |
| 日志解析器 | [offline/analyze_cg_dispatch_diag.py](offline/analyze_cg_dispatch_diag.py) | 从 CG_DISPATCH_DIAG 文本解析和对齐 DP/step |
| 已复核的聚合结果 | [offline/evidence_recheck.json](offline/evidence_recheck.json) | 关键断言与统计值；本次离线执行再次通过 |
| 匿名衍生数据校验器 | [verify.py](verify.py) | 需另行提供私有匿名数据文件，不能只靠 clone 仓库运行 |

**公开仓库没有完整现场日志、请求输入输出、运行环境和完整运行源码。** 这些保留在原始私有证据包中；本页列出的原包内部路径用于持包者复查，不是 GitHub 下载链接。仅有公开汇总不能替代原始包独立审计。

## 冻结实验身份

| 拓扑 | 实际输入 token/请求 | 输出 token/请求 | 请求数 | 并发上限 | 到达速率 | MTP | TPOT P50 | 输出 token/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2P2D，D DP16×TP2 | 81,932–81,934 | 600 | 48/48 成功 | 16 | 0.93 请求/s | 5 | 22.1 ms | 447.3294 |

原正式 614 输出轮的 21.9 ms / 约 455 token/s 是另一轮；不得混用。vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`，vllm-ascend `2ed6cbbaf481d84cfd3f2d01d47bcf79ee064c1f`；现场存在工作区修改，精确身份以原包源码及哈希为准。

## 原包内证据定位

| 证据项 | 原包内部路径 / 位置 | 判据与边界 |
|---|---|---|
| 发现过程和纠错原文 | `EVIDENCE_INDEX_20261008.txt`、`DIAGNOSIS_20261008.md` | 保存历史解释与已撤回统计 |
| 两台 D 筛选日志 | `D170_repro600_events.log`、`D171_repro600_events.log` | 11,776 条记录、736 完整全局步 |
| 完整 D 日志 | `raw_archive/D170_diagnostics.tar.gz`、`raw_archive/D171_diagnostics.tar.gz` | 校验筛选行确实存在于对应原始日志 |
| 请求、结果、配置及前后 metrics | `raw_archive/benchmark600.tar.gz` | 工作负载、成功率、token 和缓存口径 |
| uniform 判定 | `model_runner_v1.diag.py:2928` 起 | q1 不满足 MTP5 的 uniform query length 6 |
| DP 模式传播 | `model_runner_v1.diag.py:5264` 起 | 同步取 mode 的 min；任一 NONE 使全部 NONE |
| 图键、后端支持与捕获 | `raw_archive/runtime_source.tar.gz` | `cudagraph_dispatcher.py:37,166,235`；`sfa_v1.py:287–294`；`compiler_interface.py:71` |
| 修改前后与诊断补丁 | `model_runner_v1.original.py` / `.diag.py`、`envs.original.py` / `.diag.py`、`apply_graph_diag_patch.py` | 核对诊断记录点和改动内容 |
| 只改 FULL 的反证 | `full_candidate_runtime_evidence.txt` 及两台 D 完整归档 | 后端降回 FULL_DECODE_ONLY，3 请求首步仍 NONE |
| 独立 DP0 交叉核对 | `independent_step_timing_check.json` | 不依赖 16 DP 中位数聚合的核算 |
| 单次 KV 长尾 | D171 归档中 `repro600_20261007/D171.log:10537,10538,10568,10734` | 同请求 TP0 3278.451 ms，TP1 122511.935 ms，均 ret=0；内部等待原因未知 |

源码行号仅对应冻结包，不能套用到其他安装版本。历史大型 profiling 数据库不包含在此包中；已有查询结果不等于全量备份。

## 一条手算与整体核对

在 `D170_repro600_events.log` 中，DP1 的 step217（第 1944 行）为 q1 / uniform=False / local NONE / final NONE；step218（第 1956 行）恢复 q6 / FULL。两条 `ts_ns` 相减得到 **288.071946 ms**。可用原日志请求 ID 核对同一请求，不在公开页复制其身份。

| 起点→终点 | 间隔数 | 16 DP 间隔中位数的整体中位数 |
|---|---:|---:|
| FULL→FULL | 643 | 76.890978 ms |
| FULL→NONE | 46 | 80.297 ms（四舍五入） |
| NONE→FULL | 46 | 262.9554015 ms |

48 个请求首次出现都为 q1/local NONE，合并为 46 个全局 NONE 步。48/48 在节点合并日志中先出现 KV 完成记录；46 个 >150 ms 的间隔全部紧随 NONE 步。间隔包含起点执行及下一步准备/同步，不是纯 kernel 时间，也不是单 token 延迟。

早期按**终点**模式分类的“NONE 约 80 ms”和约 0.06 ms/token 已撤回；当前按**起点**模式分类。3.135 ms/token 是反事实加权估计，不能当作已实测修复收益或从 TPOT P50 直接扣除。baseline 与 trace 缓存命中不同，193→447 token/s 不能解释为补丁收益。

已证实触发机制及其与长间隔的对应关系；没有修复 A/B 证明全部 TPOT 超标由此引起。首次 q1 的完整生产链、旧 static 空洞、独立 KV 长尾仍有未解问题。

## 离线复算

需要持有私有原始包 `glm53-pd-evidence-20261008.tar.gz`（10,693,746 字节）。SHA256：

```text
be012c46cc118cc04a6fbbcc83088b40f1ee0e6bdebfb847be612b42fdc85d74
```

在新目录解压，以免覆盖工作目录。使用 Python 3.10+ 标准库，无须 NPU、vLLM 或服务器连接：

```sh
shasum -a 256 glm53-pd-evidence-20261008.tar.gz
mkdir evidence-replay
tar -xzf glm53-pd-evidence-20261008.tar.gz -C evidence-replay
cd evidence-replay/glm53-pd-readonly-review-20261007
shasum -a 256 -c SHA256SUMS
python3 verify_evidence.py
```

预期 `verified: true`、`records: 11776`、`complete_global_steps: 736`、`global_none_steps: 46`、`raw_log_membership_verified: true`。脚本会写出聚合结果和归档成员清单，原始包保持不变。公开 `offline/` 下两个 Python 脚本与冻结包字节一致，可按校验清单核对。

哈希只能证明保存后内容一致，不是第三方时间戳、数字签名或因果证明。
