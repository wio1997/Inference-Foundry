# 四个短样本精度审计：首轮报告

截至 2026-11-02，本报告复用两个已完成案例，并执行两个新短案例；四案例首轮检查已完成，但不能将其视为整模型精度验收。测试环境为 aarch64、Ascend910_9382、四个协同 die，模型配置和权重索引哈希、依赖与镜像身份见[首页](../README.md)。参考路径为已安装的 vLLM-Ascend，适配路径为本分支 DeepInfer。两者参考实现和执行形态不同；无经批准的全模型数值容差。

- **算术**：保留案例的生成 token 相同，停止标记为 154827。
- **多轮回忆**：两条路径最终都输出 42，但生成序列在第三个生成 token 首次不同；参考路径为 738，适配路径为 3229。相同提示下第一生成步的 top-five logprob 也已有差异。答案一致不代表精度通过。
- **数值替换对照（17）**：保留三轮对话结构，将 42 换为 17，参考与适配均使用 41 token 提示、生成上限 64、上下文上限 128。四 rank 均自然命中停止标记 154827，未到 cap 或 tokenizer EOS 154820，最终答案均为 17；适配输出 12 token、参考输出 13 token，首次不同为生成 token 下标 3。共同前缀的 top-five logprob 在第一步已有差异。详见 [recall17-comparison-round363.json](recall17-comparison-round363.json)。该对照支持“更换记忆数值后仍观察到序列差异”，不证明特定算子有缺陷。
- **算术替换对照（8+5）**：26 token 单轮提示、生成上限 6、上下文上限 64；四 rank 的参考与适配输出均为 `[154842,99366,154827]`，最终答案 13，均自然命中停止标记 154827，未触及 cap 或 tokenizer EOS 154820。生成 token 相同不代表概率相同：首步 token 154842 的 logprob 在适配路径约为 `-0.2601974`，参考路径约为 `-0.2324302`。完整 top-five 观测见 [arithmetic85-comparison-round364.json](arithmetic85-comparison-round364.json)。
- **共享前缀的逐层捕获**：对同一 41 token 提示和共同生成前缀 `[785,1196]` 后的位置 42，比较层 0–43 的物化 BF16 残差流，形状均为 `[1,4,4096]`。四 rank 结果一致。层 0 是最早的逐位差异：最大绝对差 `0.000732421875`，RMSE `0.00011916575022041798`，相对于参考 RMS 约 `0.0127665`。层 43 的对应相对 RMSE 约 `0.568905`。完整逐层数据见 [layer-comparison-round362.json](layer-comparison-round362.json)。这些数值是观测量，不是判定阈值。

参考路径执行批量 prefill，适配路径采用 tokenwise 正确性模式；层 0 输出包含 mHC、KDA 和 MLP 等步骤与历史状态。当前结果尚不能把差异归因于某个算子、量化项或单一位置。应继续比较层 0 的输入、注意力前后与 MLP 中间量，并验证对齐的计算语义。

两次本轮捕获中，参考组从经理启动到结束约 569.5 秒，适配组约 504.0 秒，含装载、启动与执行；现有记录不足以准确拆分 setup 与 NPU 计算时间。这不是性能基准。八张物理卡的工作前提见[离线准备边界](eight-card-readiness.zh-CN.md)；当前授权不包括八卡执行。

17 对照的四 rank 适配组从经理启动到结束约 596.2 秒、参考组约 445.2 秒，均包括装载、启动和执行；无法从现有计时单独拆出 setup 和 NPU 计算时间。这不是性能测试。

8+5 对照的四 rank 适配组从经理启动到结束约 344.4 秒、参考组约 523.1 秒，均含装载、启动及执行；没有独立仪表可准确拆分 setup 与实际 NPU 计算时间。这不是性能测试。

**首轮结论与下一步**：两个算术案例的生成 token 一致，两个多轮记忆案例最终答案一致但推理 token 有差异；四例不能代表整模型质量。已观测的最早残差流逐位差异位于层 0 输出，但具体来源和容差尚未确认。下一步在相同输入与对齐的 prefill/state 语义下比较层 0 输入、KDA、mHC 和 MLP 中间量；不得以本报告替代精度、八卡性能或生产验收。

## 层 0 边界后续观察

在同一 41-token 提示和共同生成前缀 `[785,1196]` 上，继续比较位置 0、40、41、42 的层 0 边界；每个位置采集 embedding、归一化注意力输入、注意力输出、注意力后残差、MLP 输入/输出与层输出，四 rank 指标一致。位置 0 无此前 token 的状态历史：embedding 和归一化注意力输入逐位相同；最早逐位差异出现在层 0 注意力输出，最大绝对差 `0.00006103515625`，RMSE 相对于参考 RMS 约 `0.00613807`。其后残差、MLP 输入/输出和层输出也不同。完整观察见 [layer0-boundaries-round365.json](layer0-boundaries-round365.json)。

这将最早观测差异缩小至注意力输出边界，尚不能确定 KDA 内部哪个步骤或执行方式造成差异：参考采用批量 prefill，适配采用逐 token 路径。下一步需在相同输入与状态下分解投影、卷积、递推、门控和输出投影，并控制两条执行路径；不能把这组观测当作容差验收或根因证明。

## 层 0 投影形状的有界实验

在四个 910C die 上，复用位置 0 的相同归一化输入及当前真实层 0 BF16 权重，将输入复制成 41 行，比较全量融合投影与按四 rank 切片的投影。全量投影的 41 行与单行结果并非逐位相同，最大绝对差为 `3.0517578125e-05`；四个 rank 的切片投影在 41 行与单行之间逐位相同，41 行切片也与全量投影对应切片逐位相同。原始指标见 [projection-shapes-round366.json](projection-shapes-round366.json)。

该实验仅说明特定输入、权重和 GEMM 形状可以产生数值差异；输入其余行是复制值，且没有直接调用参考模块内部投影，不能证明先前注意力输出差异或生成差异的根因，也不构成精度验收。下一步应捕获实际参考与适配路径的投影、卷积、递推及门控中间值，并对齐输入和状态。

## 实际 KDA 内部边界对照

沿用相同首位置归一化输入及检查点，参考完整模型采集批量 prefill 中间值，适配层单独从全零状态执行单 token decode；四个 rank 均采集投影、卷积、递推和门控边界。四个 rank 最早逐位差异出现在融合输入投影，最大绝对差在 `4.77e-7` 至 `3.05e-5` 之间；`f_b` 和 `g_b` 门投影均逐位相同。递推核心输出的相对 RMSE 约 `0.00073` 至 `0.00521`，门控结果也存在差异。指标及各 rank 结果见 [kda-internals-round368.json](kda-internals-round368.json)。

参考路径采用 chunk prefill，适配路径采用单 token recurrent decode，输入投影和卷积结果也并非全部逐位相同；当前不能将递推输出差异归因于某一个算子或判定阈值合格。下一步应使用对齐的捕获操作数隔离投影误差与两种递推路径的影响；独立整模型质量仍未验收。

## 对齐操作数的递推比较

使用参考首位置捕获的卷积输出、门控、beta 和全零状态，在四个 die 上直接调用适配侧安装版 decode core，与参考 chunk prefill 的对应输出比较。对齐后相对 RMSE 仍约为 `0.00073` 至 `0.00521`，与对齐前几乎相同；逐 rank 指标见 [kda-aligned-core-round369.json](kda-aligned-core-round369.json)。

该结果显示当前首位置核心输出差异在对齐上述操作数后仍存在，支持继续检查两条递推路径；它不证明具体内核机制、整模型输出因果关系或精度合格。后续须验证有界修正的状态与分段续算，并在 NPU 上回归。

## 因果递推备选路径

四个 die 上使用相同首 token 捕获操作数及零状态，比较安装版 decode、显式归一化 native recurrent 与数学 recurrence。相对参考 chunk prefill，显式归一化 native 和数学 recurrence 的输出相对 RMSE 约 `0.000615`–`0.00513`，仍有差异；这两种路径的状态彼此很接近，而相对安装版 decode 的状态相对 RMSE 约 `0.00160`–`0.00229`。逐 rank 指标见 [kda-causal-alternatives-round370.json](kda-causal-alternatives-round370.json)。

这是单 token 检查，不证明替换后状态和多 token 续算正确。历史测试还观察到安装版 chunk 的未来位置敏感性，ABI 与根因尚未确认；不能仅为接近该参考而切换内核。任何候选替换需要先做非零状态、分段续算和整模型硬件回归；整体精度仍未验收。

## 非零状态与分段续算

四 die 各用首位置捕获操作数生成非零状态，再将位置 40、41、42 的捕获操作数直接接续，比较显式归一化 native recurrent 的三 token 一次运行与逐 token 运行。四 rank 的输出与最终状态均逐位相同；其相对数学 recurrence 的状态相对 RMSE 为 `1.82e-8`–`3.51e-8`。安装版 decode 相对数学 recurrence 的状态相对 RMSE 为 `0.00122`–`0.00156`。逐 rank 结果见 [kda-state-continuation-round371.json](kda-state-continuation-round371.json)。

这是拼接的操作数序列：位置 0 后直接接位置 40–42，不代表原提示的完整历史，也没有规定质量容差。该有限结果支持继续对已有 native backend 做完整模型回归；不能据此宣称整体质量或生产验收通过。

## 现有 recurrent backend 的整模型短回归

固定多轮案例、相同检查点和四 rank 执行，将实验中的 KDA backend 由 `ascendc_installed_decode` 切换为已有 `ascendc`；未修改模型源码。生成上限 3 token，四 rank 一致，生成 `[785,1196,738]`，与已保存参考的前三 token 一致；原 installed-decode 基线为 `[785,1196,3229]`。位置 0 注意力输出相对参考的 RMSE/RMS 由约 `0.006138` 变为 `0.005685`。原始范围内指标见 [recurrent-fullmodel-round372.json](recurrent-fullmodel-round372.json)。

本次达到长度上限，未获得完整回答；单案例、三 token 和首层指标无法证明整体精度改善，也不构成 backend 切换或生产验收。保留原 backend 作为回退，后续需完成同配置完整输出和状态回归。

## recurrent backend 完整短案例

固定三消息记忆案例采用 `ascendc` 后端、64 token 上限和真实模板，四 rank 生成一致并自然停止，最终答案为 `42`。生成的全部 45 个 token 与同提示、同检查点的已保存独立参考逐位一致；先前 `ascendc_installed_decode` 的 44 token 输出在下标 2 首次不同。停止 ID `154827` 对应 `<|user|>`，属于配置的停止集合，不是 tokenizer EOS `154820`；未达到长度上限。结果和 top-five 概率见 [recurrent-complete-round373.json](recurrent-complete-round373.json)。

该结果仅覆盖一个短案例和当前实验配置，不能推断整模型精度或生产可用性；路径间浮点及概率仍需进一步核对，状态恢复和其他负载也未因此自动验收。当前未修改模型默认后端，保留 installed-decode 回退。

## recurrent backend 算术回归

固定 `8+5` 案例使用 `ascendc` 后端、真实模板和 6 token 上限；四 rank 生成一致，完整 token `[154842,99366,154827]` 与保留的独立参考及先前 installed-decode 基线均一致，最终答案 `13`。停止 ID `154827` 是配置的 `<|user|>`，不是 tokenizer EOS `154820`；未达到上限。指标见 [recurrent-arithmetic-round374.json](recurrent-arithmetic-round374.json)。

该结果只覆盖一个短算术案例，不能代表整模型精度或生产验收。候选后端仍需整模型归档恢复、状态和续算验证，原 backend 保留为回退。

## recurrent backend 整模型归档恢复

使用已有 `ascendc` 递推后端、45 层短上下文实验工厂和四 rank：各 rank 的 22 个提示位置在直接执行与分片执行之间 logits 逐位相同；生成、续算 logits 及全部 45 层最终状态一致。archive-v2 序列化、重新打开并恢复后，续算 logits、生成 token 和全部 45 层状态也均逐位一致。完整逐 rank 结果见 [recurrent-restore-round375.json](recurrent-restore-round375.json)。

本测试生成四个 token 后达到上限，未自然停止，不能视为完整回答；它验证该配置下的状态/恢复一致性，不代表跨请求、长上下文、整体质量或生产验收。原 installed-decode 后端仍可回退。

## recurrent backend 第二个记忆案例

固定三消息记忆案例采用公开 runner 的 `ascendc` 后端、41 token 提示和 64 token 上限。四 rank 均返回最终答案 `17` 并在配置停止 token `154827` 自然停止，没有达到上限；该停止 token 不等于 tokenizer EOS `154820`。生成 38 token，与同提示和检查点的保留独立参考生成序列从下标 0 起不同；参考同样回答 `17`，生成 13 token。配置、逐 rank 序列与依赖见 [recurrent-recall17-round377.json](recurrent-recall17-round377.json)。

该案例证明有限配置下的答案一致和 rank 一致，不能证明完整生成一致、整体质量或生产验收。独立质量验收仍待完成，原后端保持回退。
