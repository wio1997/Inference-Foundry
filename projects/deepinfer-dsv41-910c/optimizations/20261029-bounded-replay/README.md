# 910C 有界重放适配

该候选在 DeepInfer 910C W8A8 运行路径中接入 decoder SWA bounded replay。实现保留 256 个 token，以适配 128 token 页边界及 DSpark 回退范围；官方默认配置保留 140 个 token，二者的截断范围不同。

## 适用条件

- 172，16 个逻辑设备，DP2 × TP8、EP16；原始 ModelSlim W8A8 权重。
- 已恢复有效前缀，四个请求均进入最后一个 prefill 分块，结束位置相同且按 128 token 页对齐。
- 使用 sequence parallel、context parallel、精确 FFN 尾裁剪及学习式 DSpark；请求忽略 EOS。
- 所有 EP rank 对执行资格达成一致。不满足条件时使用原有精确路径。

第 20 层写入共享压缩 KV 后，候选在 consumer 层保留最后 256 行；SWA 页表和 SWA 使用长度按页重定址，位置、压缩 KV 和其索引仍使用绝对坐标。该重放为近似计算，不能用于生成前缀快照。Engram 表继续保留在 CPU 共享内存，主干权重不使用 CPU 预取。

## 源码准备

补丁的父版本是服务器项目目录下的 `src/DeepInfer-official-zos-target-device-20261029`，并非仓库主文档中较早的冻结部署版本。父版本必须已存在；[源码清单](bounded-replay-manifest.json)列出七个变更文件的父版本和候选 SHA256。将补丁应用到父版本的独立副本，并核对所有候选 SHA256：

```bash
# 在本仓库的本目录执行。
set -e
REPLAY_DIR="$(pwd)"
PARENT=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-zos-target-device-20261029
CANDIDATE=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-bounded-replay-copy
if [ -e "$CANDIDATE" ]; then echo '目标目录已存在' >&2; exit 1; fi
python3 "$REPLAY_DIR/verify.py" "$PARENT" parent
cp -a "$PARENT" "$CANDIDATE"
cd "$CANDIDATE"
patch --dry-run -p1 < "$REPLAY_DIR/bounded-replay.patch"
patch -p1 < "$REPLAY_DIR/bounded-replay.patch"
python3 "$REPLAY_DIR/verify.py" "$CANDIDATE" candidate
```

若父文件哈希不一致、目标目录已存在或补丁预检失败，应停止操作并重新核对版本。

现有项目服务通过不可变 job 配置字段 `yoco_bounded` 控制该路径，默认值为 `false`。启用时须保留 `tail_ffn=true`，并保持 `yoco_exact_compact=false`、`yoco_meta_fuse=false`。修改控制字段不会改变正在运行的请求；服务启动、模型路径和现有设备映射仍以项目主部署文档及服务器配置为准。

## 验证结果与限制

固定 32K 输入、1K 输出、并发 8、实际 KV 命中 235904 / 262144，同服务两轮 ABBA 对照如下：

| 配置 | 平均 TTFT（ms） | 平均 TPOT（ms） | 两轮合并吞吐（token/s） |
|---|---:|---:|---:|
| 原路径 | 781.25 | 12.70 | 581.96 |
| 256 token 重放 | 601.90 | 12.65 | 591.82 |

四轮共 32 条完整输出与保留参考逐 token 一致；四轮正式请求的分词缓存均为 8 / 8 命中。16 个 rank 的实际执行及关闭路径已核验。审计文件位于 `fair-comparison-20261028/completion/official-bounded-replay-ab-20261029/`，性能和完整输出审计 SHA256 为 `f422a6d3c0e90a0757d1a92dc29ca1d1f854749e09d9490e83a8684a0f109fd4`，rank 和分词审计 SHA256 为 `92e92ac45a87c999a08d174c605a42daeebe1d18a2302d690c724ca582814647`。

该结果只覆盖上述固定负载。设备缓存张量未逐项比较；其他输入、并发、EOS、采样方式及模型质量尚未全面验证。候选默认关闭，正式使用前应依据目标工作负载完成正确性和性能验收。
