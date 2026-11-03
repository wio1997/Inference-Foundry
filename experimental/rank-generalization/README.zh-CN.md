# Rank 接缝泛化试验

补丁只修改模型内部六个 rank 组件；现有公开运行入口、参考与回滚模式仍固定四 rank。补丁尚未集成到默认源路径。它将分支和集合检查绑定到显式 world size，并保留 16 行对齐及实际 process group 检查。支持候选为 1、2、4、8、16；除四 rank 外均未进行硬件验证，不可据此宣称可运行。

在仓库根目录执行以下只读检查，实际应用前须使用隔离工作树并重新固定源清单：

```bash
git apply --check --directory=DeepInfer/deepinfer/model_executor/models/glm5_next experimental/rank-generalization/rank-seams-round408.patch
```

四 rank 同运行时直接路径与分片路径在 45 层短请求、全部状态和 archive-v2 恢复上匹配；生成只运行四 token，达到上限，不能当作完整答复或独立精度验证。[原始结果摘要](../../validation/rank-seams-round408.json) 给出参数、哈希与检查范围。仍需实现安全参数化启动器、完整约束以及对应硬件验证；其他卡未分配。
