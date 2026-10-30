# 可选解码 token 黑名单的适配状态

核对时间：2026 年 10 月 30 日。该功能控制生成结果，不属于已证明的性能优化。

## 官方接口与当前路径

DeepInfer 固定参考提交 `553741557a57df44a46fb3ab72b6aafeaf945337` 的 [Sampler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/sampler.py) 在设置 `DEEPINFER_DECODE_TOKEN_BLACKLIST` 后，对指定 token 的 logits 执行屏蔽。[评测入口](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/eval/run_eval.py) 的 `--decode-blacklist` 默认值是 `0,2,129262..129264`，允许通过空字符串关闭。这是该评测入口的设置，不应描述为所有 DeepInfer 服务默认启用的模型规则。

当前 910C 保留版通过 `dspark910c.py` 的 `target_tokens` 及 `greedy910c.py` 的词表并行 argmax 选择目标 token；所核对路径没有调用上述 Sampler 或其黑名单处理。已核验 DP0、DP1 的代表 worker 未设置该环境变量。两份实际文件 SHA256 分别为 `bbbdcfc2e4be931885c679b6b8c496ccdffb6fd9ce2bccf163f2f11e262e5b88`、`dabcc4ac57f1ebf42b063bf326a791137da552c02af54d62219c6b1bf781adb4`。此处仅说明当前路径的静态接口与运行配置，不代表其他生成控制均已审查。

## 已有输出中的出现情况

对[七题长文配对检查](../20261030-yoco-longbench/README.md)的完整输出，按官方评测入口的五个 token ID 统计：三种模式各 56 条并发输出，EOS 前均没有出现这些 ID；EOS 后至少出现一次的输出数在精确、140、256 路径分别为 8、10、8，出现的均为 token ID 0。所有输出都出现了 EOS。这里只统计出现次数，不将 EOS 后的差异解释为该黑名单造成的差异，也不据此判断此前非有限值异常的原因。

独立统计文件位于 `/data/tiankuan/wio/910c-deepinfer-172/evidence/longbench-optional-blacklist-inventory-20261030.json`，SHA256 `3616de3ed6600bda6b04a7ccd31220c132bdfc9611ff25cabe6c24543b602c79`。七题样本有限，不能证明其他输入的 EOS 前不会出现被屏蔽 token。

## 后续适配边界

当前基线和既有比较继续使用原生成设置。若后续接入该功能，应在独立候选中保持默认关闭，统一处理首 token、词表分片选择和 DSpark 目标验证，并保留未开启时的原路径；开启时再对照官方规则、输出及性能。不能仅在草稿或单个 TP 分片屏蔽，也不能借此跳过任何现有检查。功能是否需要启用，应由具体评测配置决定。
