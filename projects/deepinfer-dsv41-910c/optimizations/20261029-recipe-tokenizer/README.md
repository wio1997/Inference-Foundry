# DeepSeek Recipe 原始文本分词接入

## 部署范围

在 910C W8A8、DP2×TP8/EP16 服务的 `/v1/completions` 原始文本入口接入 DeepSeek Recipe 0.2.0 的 `Tokenizer.encode`。原有 tokenizer 继续负责输出解码；两种输入分词器使用独立的有界缓存。切换配置仅在服务空闲时进行，不改变模型权重、异步 DSpark、bounded replay、Engram 或通信路径。

该接口针对已经渲染的原始字符串。Recipe 的聊天请求转换、特殊 token 边界标记和分段并行编码尚未接入，不能将本结果视为完整 Recipe 聊天服务的性能。

## 依赖和源码

- 服务工作目录：`/data/tiankuan/wio/910c-deepinfer-172`。
- 模型目录：`/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8`；`tokenizer.json` SHA256 为 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`。
- Python 包：官方 [DeepSeek Recipe 0.2.0](https://pypi.org/project/deepseek-recipe/0.2.0/) aarch64 wheel，文件名为 `deepseek_recipe-0.2.0-cp310-abi3-manylinux_2_28_aarch64.whl`，SHA256 为 `7c31fa6ed6ff8319670fa6efdf682e724f4e78f76218f6d08dfc4c0d6f47a071`。
- 隔离安装目录：`dependencies/deepseek-recipe-0.2.0-20261029/site`。安装文件清单 SHA256 为 `0c48d1a2eff72018852d52a7c056eb33e77050ed3f6bd712c7298a4d6d862352`。安装路径加入前端进程的 `PYTHONPATH`，不修改镜像或模型目录。
- 父源码：`src/DeepInfer-yoco-preplan-20261029`。候选源码：`src/DeepInfer-recipe-raw-20261029`。

在独立副本应用补丁前后分别运行 `python3 verify.py <源码目录> base` 和 `python3 verify.py <源码目录> candidate`；补丁使用 `patch -p1 < candidate.patch`。`manifest.json` 列出两个受影响文件及父子 SHA。

当前服务通过作业配置 `recipe_tokenizer=true` 使用 Recipe，`zos_fast`、`yoco_bounded`、`prefill_engram_uva` 均为 true。切换为 false 可恢复原始输入分词器。代码缺省使用原分词器；启用 Recipe 前必须使隔离包在前端可导入，并设置 `DEEPINFER_910C_RECIPE_AVAILABLE=1`。配置切换应先核验任务队列和 HTTP 连接均为空，以免单个批次混用设置。

172 已保存重启控制器 `tools/controller-recipe-raw-validated-20261029.py`，SHA256 `6e30db064f0a744b280e09774d428a92cf8e2445574d36e9668b67c2eaf3282b`。该文件已建立并校验，但尚未通过独立重启验证；现行服务运行的是已验证的候选控制器。重启前必须检查设备和端口空闲、核对源码与依赖清单，并保留当前服务恢复方案。

## 两轮单因素对照

同一服务、同一模型和计算路径，仅切换输入分词器。输入 32768 token、输出 1024 token、并发 8，正式请求分词缓存均为 0/8 命中，KV 实际命中均为 235904/262144。按原分词器、Recipe、Recipe、原分词器各测两轮：

| 输入分词器 | 轮次 | 平均 TTFT (ms) | 平均 TPOT (ms) | 输出吞吐 (token/s) | 前端平均分词 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 原分词器 | 1 | 798.7 | 12.4 | 598.061 | 109.48 |
| Recipe | 1 | 657.7 | 12.4 | 605.170 | 65.35 |
| Recipe | 2 | 657.9 | 12.2 | 602.137 | 64.10 |
| 原分词器 | 2 | 714.0 | 12.3 | 598.003 | 107.78 |

两轮平均 TTFT 为 756.35 → 657.80 ms，合并吞吐为 598.032 → 603.650 token/s，平均 TPOT 为 12.35 → 12.30 ms。TTFT 差异还包含排队和 worker 波动，不能全部归因于分词。输出吞吐按整个基准时间计算，首 token 时间变化也会影响该值。

四轮共 32 条完整 1024 token 序列与既有参考一致；每轮 16 rank 的异步 DSpark、bounded replay 和 Engram UVA 实际路径均经核验。16 条冻结长输入及 15 个特殊字符样例的原始文本编码一致；另对模型全部 1283 个 added vocabulary 条目、随机 Unicode 等共 6934 个生成样例逐 token 对照，均一致。上述检查不是对所有可能文本或设备缓存状态的证明。

原始 CPU 初筛、逐轮数据和审计摘要见 `results.json`。结果仅适用于上述固定请求、greedy、ignore EOS 和冷分词条件；两轮样本不足以判断所有负载的稳定性，本地锁也无法完全排除外部客户端干扰。
