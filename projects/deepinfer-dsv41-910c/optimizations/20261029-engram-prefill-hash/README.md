# 910C Prefill Engram 设备 Hash

本候选在已验证的 [prefill Engram UVA 取行](../20261029-engram-prefill-uva/README.md)基础上，将恢复前缀后的 Engram hash 从 CPU 计算改为复用现有 910C 设备 hash。CPU 继续维护完整历史并构造原始 token 窗口；CPU 表、UVA 取行、学习式 DSpark、共享专家、精确 FFN 及 bounded replay 均保留。该实现不是完整的设备端历史管理或官方 ZOS。

## 适用条件

固定部署为 172、16 个逻辑设备、DP2 × TP8 / EP16、原 ModelSlim W8A8 权重，Engram 表在 CPU 共享内存，主干不使用 CPU prefetch。候选仅在已恢复前缀的序列并行 / 上下文并行 prefill 且 `prefill_engram_uva=true`、`engram_uva=true`、`engram_device_hash=true`、`engram_prefetch=false` 时执行；其他条件回退原 CPU hash。

job 字段 `prefill_device_hash=true` 启用候选；`prefill_device_hash_check=true` 用于逐层核对 CPU hash，`prefill_engram_uva_check=true` 用于逐位核对原 CPU 选行。两种诊断均增加同步及错误投票，不用于性能测量，且分别要求对应候选开启。当前运行模式默认关闭候选和诊断。

## 源码准备

父版本为服务器的 `src/DeepInfer-official-engram-prefill-uva-audit-20261029`，其完整补丁及父版本要求见 [prefill UVA 文档](../20261029-engram-prefill-uva/README.md)。[清单](manifest.json)记录本候选三个变更文件的父子 SHA256。以下命令应在本仓库的本目录执行：

```bash
set -e
PATCH_DIR="$(pwd)"
PARENT=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-engram-prefill-uva-audit-20261029
CANDIDATE=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-engram-prefill-hash-copy
if [ -e "$CANDIDATE" ]; then echo '目标目录已存在' >&2; exit 1; fi
python3 "$PATCH_DIR/verify.py" "$PARENT" parent
cp -a "$PARENT" "$CANDIDATE"
cd "$CANDIDATE"
patch --dry-run -p1 < "$PATCH_DIR/candidate.patch"
patch -p1 < "$PATCH_DIR/candidate.patch"
python3 "$PATCH_DIR/verify.py" "$CANDIDATE" candidate
```

父文件校验失败、目标目录已存在或补丁预检失败时应停止并核对版本。其余模型、设备映射和服务启动条件使用项目现有配置。

## 单因素结果

真实模型短程验证中，16 个 rank 对两个 Engram 层的设备 hash 与 CPU hash，以及后续设备直读行与 CPU 选行逐位比较；8 条请求各 32 token 与参考一致，实际 KV 命中 235904 / 262144。

关闭逐位诊断后，在同服务固定 32K 输入、1K 输出、并发 8、实际 KV 命中 235904 / 262144，固定开启 prefill UVA、DSpark 异步草稿预发和 256 token bounded replay。顺序为 CPU 第 1 轮、设备第 1 轮、设备第 2 轮、CPU 第 2 轮。

| Prefill hash | 轮次 | 平均 TTFT（ms） | 平均 TPOT（ms） | 输出吞吐（token/s） |
|---|---:|---:|---:|---:|
| CPU | 1 | 620.9 | 12.4 | 606.753 |
| 设备 | 1 | 609.7 | 12.3 | 608.525 |
| 设备 | 2 | 629.9 | 12.2 | 604.080 |
| CPU | 2 | 625.7 | 12.3 | 603.236 |

两轮平均 TTFT 为 623.30→619.80 ms，平均 TPOT 为 12.35→12.25 ms，合并吞吐为 604.990→606.295 token/s。设备 hash 第二轮 TTFT 高 4.2 ms；两轮数据不足以证明 TTFT 或吞吐的稳定提升，也不能将 TPOT 差异直接归因于 prefill hash。

四轮各 8 条完整 1024 token 序列与保留参考逐 token 一致，每轮分词缓存 8 / 8 命中；16 rank 实际 hash、UVA、DSpark 和 bounded replay 路径均经核验。短程逐位审计 SHA256 `407392c5ddbc18c78ba5e0bc795b624506c51631300eb0f95c6f871b1050d3ba`；性能和完整输出审计 SHA256 `a1cab98fb0f6c87c841dd721e1f337ba6ab574e22524e5f42a63794dcc49efce`；rank 与分词审计 SHA256 `020f9554fcf8e7b507474176b47a4cad2fe02b72b126c81b0e210fb8f0a4dab6`。审计文件位于服务器 `diagnostics/official-engram-prefill-hash-ipc-20261029/` 和 `fair-comparison-20261028/completion/official-engram-prefill-hash-ab-20261029/`。

请求时标显示，两轮从 IPC 发布到收到 worker 首 token 分别为 CPU 558.471/559.179 ms、设备 550.959/567.268 ms；该区间包含缓存恢复、prefill、首轮生成与 IPC，不能作为单算子独占耗时或因果归因。时标审计 SHA256 `8254f5344122711e05fdfc41a59ca6174375967d4cc35b401cee77c039ab73df`。

结果仅覆盖固定负载，设备缓存张量未逐项比较；其他长度、EOS 和缓存状态尚未全面验证。候选和原始数据保留，不替换已有路径。
