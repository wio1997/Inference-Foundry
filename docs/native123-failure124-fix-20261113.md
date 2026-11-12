> 最新裁决：[124已运行完成但性能不采用](native124-decision-and-stage-review-20261113.md)。下文运行中及待验证说明为历史；默认保留114。

# 123工程失败与124布局修正

2026-11-13，北京时间。**123没有进入性能测试；默认保留114。124是同一元数据方案的一处布局修正，已经启动，尚未NPU验收。双档30%目标仍未完成。**

## 123发生了什么

源码a4df69bc805a871c3587e007b5224e7b76eaf129在初始化末端安装新计划时抛出ValueError：`native plan only supports NoPE SFA plus one pooled-indexer cache group`。8个rank均有同一异常，8份Worker退出码均为1。rank0随后出现的Gloo连接关闭发生在异常清理期间，不能当成首因。与121的HCCL初始化故障不同。

结果为0工程请求、0正式AISBench样本，尚未进入新metadata图的捕获/重放。62个回收文件逐个哈希通过，无省略文件。容器停止、占用登记释放，另一次只读核验确认8个原Worker PID均不再存活、无本作业占用。容器137来自清理结果，不能代替8个Worker实际退出码1；OOMKilled为false。

## 已查清的实现缺口

固定SDK按`(backend class, physical group ID)`建立注意力组。该GLM配置的计算关系为：

| metadata后端 | 物理缓存组 | 职责 |
|---|---|---|
| SFA | 主组 | 主注意力缓存地址与长度 |
| Indexer KPool | 同一个主组 | 已压缩索引缓存的地址与长度 |
| Indexer KPool Tail | 独立尾部组 | 未完成压缩池的请求私有尾部缓存映射 |

因此日志中的物理组`[0,1]`对应三个metadata后端。123原型只允许两个后端、一个cache-only builder，不能覆盖该固定模型的配置。这是本次原型实现错误，前面的4项主机检查没有覆盖真实混合缓存布局。

证据是同镜像已停止容器内的模型构造、缓存组装、multi-group proposer和builder源码，以及实际模型配置；没有再加载模型获取这些源码。失败时没有记录每个对象的全部条件值，因此不声称已经动态重放了所有失败谓词；源码足以证明原校验与该布局不相容。

## 124具体改动

源码8162ea6c4e8246b5f930f74af05d356c8ba307b8。相对123只改一份实现及对应检查：明确接收SFA、Indexer、Tail三个后端，要求Indexer与SFA同物理组、Tail独立；保留NoPE、非PCP及固定SDK哈希等限制，并在失败/统计中记录实际布局。

原SDK循环本来就能构造两个cache-only后端，124继续使用这一循环。没有改注意力数学、模型权重、浮点精度、缓存内容、采样或MTP接受规则。首次原NPU metadata比较仍覆盖返回的嵌套tensor及标量；Tail的slot mapping和block table也必须参加比较。没有把尾部缓存省略来获得速度。

6项主机契约检查通过，其中新增两项覆盖“三后端/两物理组”及尾部缺失/别名错误。主机检查不是NPU正确性证据；图捕获、动态重放与收益仍待实测。

## 明确记录预算修订

原N3一次加载已由123消耗。依据这次新发现，允许**同一方案一次局部工程修正确认**；不是新方向，也不是参数搜索。不把实现未接通当作该优化性能已被否决。

124仍在与114/117相同8die上运行，模型参数、共有开关、4工程请求、C4/C8各2预热及各32正式请求保持不变，总72请求、1024输入/256输出，无额外模型预热。Worker限25分钟，manager限27.5分钟并负责清理，原02:40阶段复盘点不变。

若124仍失败，保存失败，不继续第三次加载；若机制生效但成绩不足，按结果否决；若数值过线，再核对输出、实际机制贡献及已有波动限制，不仅凭一次分数宣布整体完成。

## 保留结果与复现

114相对固定强参考117：C4 306.94205 / 228.57797 token/s，+34.28%；C8 501.22023 / 415.61018 token/s，+20.60%。当前没有新增性能成绩。C8目标540.29324 token/s，原批次还需净省1.18198秒。

- [123失败记录](../validation/shared-features-20261111/native123-failure.json)
- [123实际环境、源码与启动命令](../validation/shared-features-20261111/native123-failed-reproduction.json)
- [124准备记录](../validation/shared-features-20261111/native124-preparation.json)
- [124未应用完整补丁](../validation/shared-features-20261111/native124-layout-correction.patch)
- [保留114复现记录](../validation/shared-features-20261111/native114.json)
- [固定117参考复现记录](../validation/shared-features-20261111/reference117.json)

默认发布源码保持114；候选补丁须在DeepInfer源码目录父级应用。所有失败、有限检查和未验证项保留，整体Goal不标完成。
