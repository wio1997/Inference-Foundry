# 双侧共同启动准备：移出首批同步编译，尚未验证性能

## 改什么，为什么

诊断97的首批出现两个同步Triton编译调用，rank0合计5.135秒。新的有限准备模块把已有内核的必要编译模板放到服务就绪之前；不改算子数学，不执行额外模型forward或kernel，不读取评测prompt或答案。两边用完全相同的代码。这是共同基线修复，不作为DeepInfer独有技术收益。

槽位映射首次慢编译对应tile256；indexer对应packed cache、REQ_POW2=4、OUTER_POOL=2048。使用固定镜像内真实Triton编译器，两个模板的缓存键都精确复现了97的缓存文件键。这个检查没有生成二进制或执行NPU，不能代替模型验证。

## 实现与边界

先在既有`NPUWorker.compile_or_warm_up_model`执行期间捕获indexer的dtype和标量调用契约；不保留tensor数据。原有启动流程结束后，以Triton `.warmup`的compile-only接口准备6个槽位模板和当前seed下12个indexer模板。配置上限8请求、2048步内token；扩展配置不在本次结论内。对源文件哈希漂移、未观察到seed、模板超预算均明确失败。

每rank在ready前保存准备回执，避免非主Worker的`os._exit`丢失诊断；两个内核后续真正的cache miss另存回执。启动准备耗时单列，不删除任何慢请求。AISBench仍为C4/C8各32请求、1024输入/256输出、2个排除预热。镜像、权重、共同开关和命令沿用[正式环境记录](performance-decision-20261112.md)，仅增加以下共享安装步骤。

## 如何接入和回退

原生侧在创建Worker前调用：

```python
from deepinfer.model_executor.models.glm5_next.full_model.shared_kernel_preparation import install_shared_kernel_preparation
install_shared_kernel_preparation(output_dir / 'kernel-preparation')
```

参考侧把同一文件放入已有GLM plugin的模块搜索目录，在原有注册结束处导入并调用同一函数，输出到自己的统计目录。必须核对两份文件SHA相同；保留原有KDA/MTP/图等注册。此helper不启动服务或分配设备，操作者仍须沿原有资源协议启动。回退时移除双方安装调用，并重新启动两侧进程，不在运行中热切换。

源码提交`58e20789507080e1f0f9284d268dc1f00f5de4dd`，基线`8cd3bfc7ec4df7e8edc4b41c44e26d5208f35882`；C++未改，实际复用基线构建的扩展，不能声称扩展由新提交重编译。公开提交编号与实测源码编号分别记录。

## 当前结果及未验证项

真实编译器签名核对通过，原生98的8个实际Worker已核对扩展身份；**准备函数的硬件运行、迟到编译是否消失、98/99公平性能结果尚未验证**。参考99须使用原生98的实际同一卡组，不能拿98除以旧95。两次无NPU二进制编译失败与部署失败均保留在[记录](../validation/shared-features-20261111/shared-kernel-preparation.json)。

没有新增性能或自然波动结论。此前96/95的C4 +9.48%、C8单批+36.80%继续保留原范围，完整双档独有30%目标未完成。

[源码](../DeepInfer/deepinfer/model_executor/models/glm5_next/full_model/shared_kernel_preparation.py) · [仅此改动的补丁](../patches/shared-kernel-preparation.patch) · [签名与失败记录](../validation/shared-features-20261111/shared-kernel-preparation.json)
