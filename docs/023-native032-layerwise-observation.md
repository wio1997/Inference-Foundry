# 原生0.32单节点：相同前缀的逐层观察

固定910B1/aarch64、原生arm64镜像`d5492ecf`、同一1017 W4A8权重；P1146保留P1121的NoSP/eager/TP8EP8/prefix/async参数，只通过固定版本实际支持的`--worker-extension-cls`加载观察模块。[完整配置及观察代码](../code/native_layer_probe_G1146/candidate.json)、[实际就绪和8worker安装](../evidence/native_layer_probe_G1164/G1148_observer_ready_2203_parsed.json)保留，113个目标源码文件仍匹配。镜像源码未覆盖，但运行时有观察包装，不能用它当无观察器的性能baseline。

## 真实输入与响应

选择既有math1提示的精确30个token，位置0至29，问题为14×6+7；先关闭观察，再开启观察，均max_tokens1、T0/seed0、无重试、串行标准completions。两份请求字节相同，均返回91，usage均30输入/1输出。[父级逐请求SHA和边界核查](../evidence/native_layer_probe_G1164/G1164_parent_native_layer_observation.json)、[未采样请求](../evidence/native_layer_probe_G1164/unarmed/native_request.json)和[采样响应](../evidence/native_layer_probe_G1164/armed/native_response.json)可查。这是首token诊断，不是全量质量得分。

## 40层、8rank实际覆盖

原模型方法每次仅调用一次、原样返回其结果；观察器仅在ARM文件存在且前缀位置匹配时复制真实30行的attention归一化输入和投影输出至CPU。输入在模型入口由8rank逐一核对，不取padding尾行。全部40层×8rank共320份safetensors，合计196756480字节；每层在本框架内8rank的input/output原始BF16字节SHA均相同。[全部rank/层/张量SHA与冻结证明](../evidence/native_layer_probe_G1164/G1150_layer_snapshots_2203_parsed.json)和[rank0元数据](../evidence/native_layer_probe_G1164/rank0_metadata/rank0_layer00.json)保留。

全部原始向量继续保留在受控服务器；只将一份rank0副本用于远端CPU客户端比较，公开库不包含向量数据。请求实际退出后两端队列为0，ARM已改名关闭。复制会同步设备，采样延迟不能与正式AISBench结果计算收益。相同rank字节链只支持本前缀、层边界和版本，不能外推其他负载或所有算子。

SGLang部署、响应、独立CPU比较和适配候选归其独立分支；此处只维护原生证据。完整质量、四档矩阵和P8+D8原生profile仍引用各自原epoch归档，最终跨框架地图与收益未完成。
