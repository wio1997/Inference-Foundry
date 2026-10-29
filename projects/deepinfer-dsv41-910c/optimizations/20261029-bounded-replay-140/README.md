# 910C 有界重放：140 token 候选

本候选在已有 256 token 有界重放路径上，增加与 DeepInfer 官方默认保留长度相同的 140 token 选项。140 token 的起点通常不与 128 token 页边界对齐；实现将 SWA 数据按请求重排到可复用的紧凑页中，压缩 KV 和因果索引继续使用绝对位置。两种长度对应的近似计算范围不同，不能视为完全等效。

## 适用范围

适用设备、模型、缓存和请求约束与 [256 token 候选](../20261029-bounded-replay/README.md)相同，包括 172 上的 16 逻辑设备、DP2 × TP8、EP16、W8A8、前缀恢复、每组四请求的页对齐终点和 ignore EOS。所有 EP rank 共同判断资格；不满足条件时回退。学习式 DSpark、Engram CPU 表与 UVA、设备 hash、精确 FFN 尾裁剪均保留。主干权重不使用 CPU 预取。

该局部实现不表示覆盖官方实现的全部请求形态。SWA 重排每个 consumer 层执行，候选不把近似结果写入前缀快照。现有服务以不可变 job 字段 `yoco_bounded=true` 启用有界重放，再以 `yoco_bounded_140=true` 选择 140 token；后者为 `false` 时选择 256 token。默认两项均关闭。

## 源码准备

父版本必须为服务器项目目录下已校验的 `src/DeepInfer-official-bounded-replay-20261029`；[清单](bounded-replay-140-manifest.json)列出七个受影响文件的父版本及候选 SHA256。以下操作创建独立副本，不修改父版本：

```bash
set -e
REPLAY_DIR="$(pwd)"
PARENT=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-bounded-replay-20261029
CANDIDATE=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-bounded-replay-140-copy
if [ -e "$CANDIDATE" ]; then echo '目标目录已存在' >&2; exit 1; fi
python3 "$REPLAY_DIR/verify.py" "$PARENT" parent
cp -a "$PARENT" "$CANDIDATE"
cd "$CANDIDATE"
patch --dry-run -p1 < "$REPLAY_DIR/bounded-replay-140.patch"
patch -p1 < "$REPLAY_DIR/bounded-replay-140.patch"
python3 "$REPLAY_DIR/verify.py" "$CANDIDATE" candidate
```

应在本仓库的本目录执行上述命令。父文件校验失败、目标目录已存在或补丁预检失败时应停止并核对版本。服务启动与设备映射使用项目现有配置，部署到其他场景前须完成适用性验证。

## 同服务单因素结果

固定 32K 输入、1K 输出、并发 8，每轮实际 KV 命中 235904 / 262144；异步草稿预发两组均关闭。顺序为 256 第 1 轮、140 第 1 轮、140 第 2 轮、256 第 2 轮。

| 路径 | 轮次 | 平均 TTFT（ms） | 平均 TPOT（ms） | 输出吞吐（token/s） |
|---|---:|---:|---:|---:|
| 256 token | 1 | 617.0 | 12.7 | 591.720 |
| 140 token | 1 | 608.4 | 12.7 | 592.614 |
| 140 token | 2 | 607.5 | 12.5 | 590.035 |
| 256 token | 2 | 599.5 | 12.5 | 591.975 |

256 与 140 的两轮合并吞吐分别为 591.847 和 591.321 token/s，平均 TTFT 分别为 608.25 和 607.95 ms，平均 TPOT 均为 12.6 ms。两轮结果未证明 140 token 有额外性能收益；第二轮 TTFT 高于 256 token，应保留该差异。该候选保留供后续验证，不据此替换已验证的 256 token 路径。

四轮各 8 条完整 1024 token 输出与保留参考逐 token 一致，分词缓存每轮 8 / 8 命中。16 个 rank 均执行 19 层 SWA 消费；140 token 每层执行重排，256 token 未执行重排。性能和完整输出审计 SHA256 为 `784a452251ed5b7afc4a59d9c8cfa91cdcde6af51bd87f5de6d2aa0c11bc9566`，rank 与分词审计为 `7a2ec55d0ab1641e20c3f8453e495f68f62bf8c5733eb6e847dc163844b3ebac`，文件位于服务器 `fair-comparison-20261028/completion/official-bounded-replay-140-ab-20261029/`。

验证仅覆盖上述固定负载；设备缓存张量未逐项比较，其他批处理形态、EOS、padding、缓存状态及模型质量尚未全面验证。两轮测量不足以排除波动。
