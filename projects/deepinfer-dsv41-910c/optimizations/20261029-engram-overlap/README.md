# 910C Engram 消费端查表重叠候选

本候选在已有 910C W8A8 适配上，将 4 × 6 DSpark target 图的 Engram UVA 查表和 TP all-gather 安排到独立侧流；对应层消费前通过外部事件等待。该方案参考 DeepInfer 按层获取 Engram 的思路。固定官方 Ascend Engram 后端仅支持设备表，本候选使用 CPU 表和 UVA，并非原样启用该后端。

候选保留设备 hash、CPU 共享 Engram 表、DSpark、共享专家、精确 FFN 及有界重放；其余图形状沿用原查表方式。设备 hash 和输入准备仍在目标图之前执行。为避免后台通信与模型 collective 共用同一组，候选额外建立 TP HCCL 通信组。图捕获、重复 replay 和 profile 过程均包含事件同步；两种对照使用同一张带事件的图，按 job 字段 `engram_overlap` 选择主流或侧流查表。

## 实现与运行边界

父版本为 `src/DeepInfer-official-bounded-replay-140-20261029`，对应 [140 token 候选](../20261029-bounded-replay-140/README.md)。[源码清单](manifest.json)记录六个变更文件的父子 SHA256。补丁应应用于父版本的独立副本：

```bash
set -e
PATCH_DIR="$(pwd)"
PARENT=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-bounded-replay-140-20261029
CANDIDATE=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-engram-overlap-copy
if [ -e "$CANDIDATE" ]; then echo '目标目录已存在' >&2; exit 1; fi
python3 "$PATCH_DIR/verify.py" "$PARENT" parent
cp -a "$PARENT" "$CANDIDATE"
cd "$CANDIDATE"
patch --dry-run -p1 < "$PATCH_DIR/current.patch"
patch -p1 < "$PATCH_DIR/current.patch"
python3 "$PATCH_DIR/verify.py" "$CANDIDATE" candidate
```

服务启动时须为 API 和 worker 设置 `DEEPINFER_910C_ENGRAM_OVERLAP=1` 以准备图和通信组；job 模式设置 `engram_overlap=true` 才在侧流执行。启用要求 `engram_uva=true` 和 `engram_device_hash=true`。该选项属于受限实验路径，不能推广为任意批次、EOS 或采样策略均已验证。由于本次测试未证明收益，现已恢复冻结父版本服务，未将该路径作为默认部署。

## 同服务单因素结果

固定 32K 输入、1K 输出、并发 8、每轮实际 KV 命中 235904 / 262144，256 token 有界重放与 DSpark 异步草稿预发在两组均开启。测试顺序为主流第 1 轮、侧流第 1 轮、侧流第 2 轮、主流第 2 轮。

| 查表方式 | 轮次 | 平均 TTFT（ms） | 平均 TPOT（ms） | 输出吞吐（token/s） |
|---|---:|---:|---:|---:|
| 主流 | 1 | 602.2 | 12.4 | 604.939 |
| 侧流 | 1 | 600.0 | 12.5 | 603.852 |
| 侧流 | 2 | 639.9 | 12.3 | 600.227 |
| 主流 | 2 | 598.7 | 12.3 | 601.667 |

两轮合并吞吐主流 603.299、侧流 602.034 token/s；平均 TTFT 分别为 600.45 和 619.95 ms，平均 TPOT 分别为 12.35 和 12.40 ms。侧流第二轮 TTFT 高 41.2 ms；两轮不能确定该差异的原因，也未证明该候选有性能收益。

四轮各 8 条完整 1024 token 输出与保留参考逐 token 一致，分词缓存每轮 8 / 8 命中；16 个 rank 的查表分支、有界重放和 DSpark 异步预发均经核验。性能和输出审计 SHA256 `26bd10a16547b98eab1ab48482ef03c02ca7e8121a5001b149278f9b77cf302f`；rank 与分词审计 SHA256 `924feb0bc070b5c6d31fef6c40fbc760e781adcb3af50e5c1973b44b33046fbc`。审计文件位于服务器 `fair-comparison-20261028/completion/official-engram-overlap-ab-20261029/`。

两组均包含新增事件图与通信组，因此本表不能衡量这些共同开销相对于父版本的影响；侧流派发计数也不能证明设备计算与查表实际重叠。验证仅覆盖固定负载，未逐项比较设备缓存张量，也未完成其他输入及模型质量验证。

已有请求时标对第二轮的服务端耗时进行分账：主流与侧流从 IPC 发布至收到 worker 首 token 分别为 540.723 和 582.875 ms；worker 自报首 token elapsed 分别为 505.059 和 547.161 ms。分词缓存均命中，批次形成与 IPC 发布阶段的差异较小。该 worker 区间包含缓存恢复、prefill、首轮生成和 IPC，不构成对单一算子或侧流影响的因果证明；两轮测量仍可能受到波动影响。分账审计 SHA256 为 `b5fc63ffdf96bc64c62e13ce0e15f69d55b7bfdcc65b117db522097193dc5f4e`。
