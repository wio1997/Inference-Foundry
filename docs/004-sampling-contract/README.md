# 004：修复 overlap 调度丢失的采样惩罚

## 原理

`SamplingBatchInfo.update_penalties()`把最小生成长度等约束保存为`acc_additive_penalties`，`copy_for_forward()`随后清除live orchestrator。两个DSPARK消费者仍读取不存在的`acc_linear_penalties`，因而可能把需要约束的请求判断成无需调整，并漏掉已经准备好的EOS屏蔽。普通请求配置了`min_tokens=1024`，不能据此推定实际执行的约束也相同；`ignore_eos=True`也不表示可以忽略最小长度屏蔽。

## 源码变更

修正两个消费者共12处字段/局部名称引用，没有增加算子或更改采样公式、模型数学、权重、并行、精度及请求。当前实际DP4/static路径本来不构造目标verify epilogue，因此不能把这个修复解释为“丢失既有图内accept融合”。未测量惩罚处理的独占成本。

- [目标verify消费者](../../compat/runtime/g130/dspark_verify.py)
- [logit调整消费者](../../compat/runtime/g130/dflash_utils.py)
- [独立补丁](../../patches/sglang-additive-penalty-contract.patch)与[实际源码SHA/环境清单](../../patches/g130-contract-manifest.json)

## 实测范围

实际producer、copy与consumer方法组成的24个CPU用例通过：B=1/2/4、verify宽1/6、FP32/BF16、最小长度前/后，检查正确屏蔽、广播、原live路径和无惩罚路径。采用小型request/orchestrator DTO，不是完整框架导入。原实验torch为2.12.0+cpu；[CPU结果](../../evidence/g130/CPU_regression.json)。

实际910C模型实例执行两条32输入/12输出探针：相同EOS=1强偏置，min0时12个输出均为EOS，min12时12个输出均非EOS。原始[控制响应](../../evidence/g130/model_probe/min0.json)与[约束响应](../../evidence/g130/model_probe/min12.json)逐字节保留。它证明此项约束生效，不是整模型质量认证，也不是本仓历史G43加补丁后重新部署的验收。

## 波动与性能边界

本条是接口正确性修复，未建立独立端到端性能收益或噪声标定。后续实验还有其他适配源码，不能把其吞吐变化归因于这两个改名。历史归档的吞吐及原始数据保留原值，但在该缺陷修正前，不能把SG与native差异完全解释成算子速度。未逐条恢复全部历史正式输出ID，不宣称每条历史请求都出现了提前EOS。

## 结论与取用

此补丁补回当前字段契约；不删除`min_tokens`来制造更快的结果。不覆盖任意scaling penalty、speculation block内最小长度边界、全部采样功能或整模型质量。完整32K/64K/128K/256K性能对齐目标仍未完成。

便携CPU runner只修改路径和输出位置，函数/类AST与原实验相同，**尚未重新执行**。在安装CPU PyTorch的环境可运行：

```bash
python3 benchmark/g130/CPU_regression.py
```

输出写入`.work/sampling-contract/CPU_regression.json`。[实际producer与native约束来源](../../benchmark/g130/source)及[原始消费者](../../benchmark/g130/original)随附。

对已准备好的历史归档环境，可先检查挂载命令：

```bash
python3 scripts/launch.py --generation g43 --model-path /data/models/DeepSeek-V4.1-Flash-1017-w8a8 --sampling-contract-fix
```

默认dry run；新增选项仅加入两个只读源码挂载，不改变模型参数。历史默认命令保持用于归档重放。执行仍须按[复现说明](../REPRODUCE.md)准备依赖、独占资源及共享锁；G43加该选项尚未单独完成NPU验收。完整后续优化环境不由本条代替。
