# 98/99：共同启动准备后的正式公平对比

2026-11-12。本批未同时达到两个并发档30%；完整目标未完成。 **共同准备是两侧相同的工程能力，不计DeepInfer独有收益。** 本报告不把原生98除以旧参考95。

| 并发 | vLLM99 token/s | DeepInfer98 token/s | 领先幅度 | 完整文本相同 |
|---|---:|---:|---:|---:|
| C4 | 223.31462 | 281.38870 | +26.01% | 29/32 |
| C8 | 352.49706 | 412.10826 | +16.91% | 27/32 |

两侧每档32请求、1024输入/256输出；128个正式请求均成功。原生8Worker exit0，双方服务结束并释放资源。正式计时从原始时间戳独立重算，与AISBench官方结果一致；各档仅一批，尚未标定自然波动。

## 这次确认了什么

共同准备代码在服务就绪前编译有界的已有内核模板，不改算子数学、不增加模型forward或kernel执行，不用评测prompt定制预热。原生与参考使用字节相同的helper和固定SDK源文件指纹；配置、实际卡组、镜像ID、容器CPU/内存限制和负载哈希已核对。

原生准备耗时范围39.97–40.95秒/每rank，参考40.12–41.01秒/每rank；并行rank时间不能相加成启动延迟。两个受观察内核的服务期迟到编译次数：原生0、参考0。其它内核不在此观察范围内。每rank详细模板与事件见JSON记录。

## 延迟、剩余差额和限制

| 并发 | TTFT中位数ms（原生 / 参考） | TPOT中位数ms（原生 / 参考） |
|---|---:|---:|
| C4 | 931.27 / 1403.72 | 9.906 / 10.795 |
| C8 | 1206.02 / 1272.64 | 15.055 / 16.080 |

| 并发 | 原生计分秒数 | 领先30%所需≤秒数 | 差额秒数（正值需减少） |
|---|---:|---:|---:|
| C4 | 29.11275 | 28.21821 | +0.89455 |
| C8 | 19.87827 | 17.87685 | +2.00142 |

完整文本分叉保留，未将其自动判为错误或质量通过。工程用例和请求生命周期通过只证明对应范围。原生仍向C++输出提供真实采样token概率，参考HTTP不请求概率；没有人为增加参考负载。两档按C4→C8执行且不清缓存，因此该顺序与复用状态也是结果条件；不推广为冷启动C8或通用负载。

## 版本、参数与复现

原生源码`58e20789507080e1f0f9284d268dc1f00f5de4dd`；实际扩展SHA-256 `fe6e265ccd8b1c85f4de1b61602dbb6e0c4647153e156c54c0c59b466641da61`，构建来源为`8cd3bfc7ec4df7e8edc4b41c44e26d5208f35882`。新增helper不改C++。实测卡组die为2,3,4,5,6,7,8,9，双方完全相同。公共分支HEAD还有历史材料，不能直接冒充实测源码版本。

基础源码按[原生96复现配方](../validation/shared-features-20261111/native96-candidate.json)构造，再应用[共同准备补丁](../patches/shared-kernel-preparation.patch)，按[安装步骤](shared-kernel-preparation-20261112.md)在两侧注册同一模块。新增helper SHA-256为`22ed4b3071952e09795771da67f78164d157917c7448d2ab8d5039d44c69c328`。

固定910C/aarch64、驱动25.5、CANN9.2、Python3.12.13、torch2.10.0+cpu、torch-npu2.10.0.post7、Triton3.6；基础镜像本地ID `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`（非已核验仓库digest）。vLLM0.30.0/Ascend0.30.0rc2固定版本，GLM-5.3-Flash W8A8/BF16只读权重。TP8/DP1/PP1/EP8，context2048、max sequences8、max step1792、每rank4GiB缓存、MTP5；共同适用的graph、async、prefix、AIV/EP、shared-expert overlap、DSA-CP/FlashComm/SP、static/super配置见两侧记录。互斥配置不同时打开，不声称所有参数组合已全局最优。

每侧容器32CPU/512GiB、共享内存8GiB；AISBench3.1固定版本与冻结数据集，每档2个排除预热，温度0、top_p1、seed0、ignore_eos。下列JSON保存实际容器及rank/服务argv、共同helper指纹、计分命令和配置/数据集哈希。C8命令为manager直接记录；C4仅将同一已保存循环的并发项替换为4，并与实际结果回执核对。

[原生98完整参数与结果](../validation/shared-features-20261111/native98-common-preparation.json) · [参考99完整参数与结果](../validation/shared-features-20261111/reference99.json) · [源码清单](../validation/shared-features-20261111/native98-source-manifest.json) · [原始时间戳分析](../validation/shared-features-20261111/matched98-99-request-cost.json)

## 下一项决定

用本组真实差额筛选一个执行机制：先核对与固定参考的实际不同，再确认可缩短的关键路径足够。已有双批异步不重复实现，共同编译准备与参考已有的checkpoint合批能力不包装成独有算法。若缺少足够收益空间，明确否决该方向；不继续扩展参数矩阵，也不把旧CPU/NPU逐位数值搜索重新列为性能前置门槛。
