# 910C Prefill Engram 设备直读取行

该候选将恢复前缀后的序列并行 prefill Engram 取行，从 CPU 选取和解码改为复用现有 CPU 表 UVA 设备直读。每个 TP rank 读取其持有的 hash head，并通过 TP all-to-all 将各 head 数据交换到对应的序列分片，再执行原有 Engram 投影与门控。INT8 和 E8M0 原始表仍在 CPU 共享内存中；模型主干不使用 CPU prefetch。

该实现借助现有 910C UVA 路径适配 DeepInfer 的 Engram 获取机制，不是原样启用官方 Ascend Engram 后端。学习式 DSpark、设备 hash、共享专家、精确 FFN 和两种 bounded replay 候选均保留。

## 使用条件

- 服务器 172，16 个逻辑设备，DP2 × TP8、EP16，原 ModelSlim W8A8 权重。
- 已创建现有 Engram UVA runtime；job 使用 `engram_uva=true`、`engram_prefetch=false`。
- 前缀已恢复且模型进入 sequence parallel / context parallel prefill。其他情况沿用 CPU 原路径。
- `prefill_engram_uva=true` 启用候选；`prefill_engram_uva_check=true` 额外启用诊断时的原路径逐位比较和跨 rank 错误投票，不用于性能测量。检查开关要求候选同时启用。

本次保留独立候选，服务控制字段当前默认关闭。已测范围为固定 32K/1K/C8、greedy、ignore EOS，不能据此推断其他缓存状态、长度或请求形态的正确性。

## 源码准备

父版本为服务器项目下的 `src/DeepInfer-official-bounded-replay-140-20261029`。补丁同时包含运行路径和默认关闭的逐位诊断；[清单](full-manifest.json)列出四个变更文件的父子 SHA256。以下命令应在本仓库的本目录执行：

```bash
set -e
PATCH_DIR="$(pwd)"
PARENT=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-bounded-replay-140-20261029
CANDIDATE=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-engram-prefill-uva-copy
if [ -e "$CANDIDATE" ]; then echo '目标目录已存在' >&2; exit 1; fi
python3 "$PATCH_DIR/verify.py" "$PARENT" parent
cp -a "$PARENT" "$CANDIDATE"
cd "$CANDIDATE"
patch --dry-run -p1 < "$PATCH_DIR/full.patch"
patch -p1 < "$PATCH_DIR/full.patch"
python3 "$PATCH_DIR/verify.py" "$CANDIDATE" candidate
```

父文件校验失败、目标目录已存在或补丁预检失败时应停止并核对版本。现有服务通过不可变 job 模式传递字段；模型、设备映射和 UVA runtime 的其他启动条件使用项目现有配置。

## 正确性与单因素结果

16 个 rank 在真实模型短程检查中分别对两个 Engram 层的候选数据和原 CPU 选行结果逐位比较；8 条请求各 32 token 与参考一致。该检查同时启用 DSpark 异步草稿预发和 256 token bounded replay，实际 KV 命中 235904 / 262144。

随后关闭逐位检查，在同服务固定 32K 输入、1K 输出、并发 8，实际 KV 命中 235904 / 262144；两组均启用上述 DSpark 和 bounded replay。顺序为 CPU 第 1 轮、UVA 第 1 轮、UVA 第 2 轮、CPU 第 2 轮。

| Prefill 取行 | 轮次 | 平均 TTFT（ms） | 平均 TPOT（ms） | 输出吞吐（token/s） |
|---|---:|---:|---:|---:|
| CPU 选行 | 1 | 615.0 | 12.3 | 609.078 |
| UVA 设备直读 | 1 | 588.4 | 12.3 | 609.418 |
| UVA 设备直读 | 2 | 589.8 | 12.3 | 604.303 |
| CPU 选行 | 2 | 605.5 | 12.2 | 604.578 |

两轮平均 TTFT 为 610.25→589.10 ms，降低约 3.47%；TPOT 为 12.25→12.30 ms，合并吞吐为 606.819→606.850 token/s。第二轮 UVA 的 TPOT 高 0.1 ms、吞吐低约 0.275 token/s；仅两轮不足以判定这些差异的成因或稳定性。

四轮各 8 条完整 1024 token 序列与保留参考逐 token 一致，16 rank 取行模式、有界重放和 DSpark 路径经核验；每轮分词缓存 8 / 8 命中。短程逐位审计 SHA256 `ac28157d7b4720a8a1f7c2c5cd04aee12b26bf46398207b77b5f0991e98f28e4`；完整输出和性能审计 SHA256 `1af3bdd9773ecc4e3f0ec3234455d623e9acfebe4632acea61d79a31ed85b006`；rank 与分词审计 SHA256 `5094de1de3bf3a38555d4d73adc699e536397b508cb47b8aee1686b951df6e7d`。文件分别位于服务器 `diagnostics/official-engram-prefill-uva-audit-ipc-20261029/` 和 `fair-comparison-20261028/completion/official-engram-prefill-uva-ab-20261029/`。

请求时标显示两轮从 IPC 发布到收到 worker 首 token 的均值分别从 561.612 降至 534.944 ms、从 551.105 降至 534.435 ms。该区间包含缓存恢复、prefill、首轮生成和 IPC，不代表单个算子的独占耗时。时标审计 SHA256 `14ca3a40c6db3c0265dd52d89cbdc58b3f9bb56ee50746a317da9ef99239f683`。

验证未逐项比较设备缓存张量，也未覆盖所有批处理、EOS 和缓存状态；最终仍需组合其他有效优化并进行同期 vLLM Ascend 对照。
