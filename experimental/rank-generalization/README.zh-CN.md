# Rank 接缝泛化记录

实验补丁已集成到模型 rank 组件，普通文本运行器支持显式 world size；参考和回滚仍限定四 rank。保留 [原补丁](rank-seams-round408.patch) 作为变更记录，它不应在已集成源码上重复应用。

四 rank 同运行时 45 层 direct/ranked、状态和 archive-v2 恢复检查通过；参数化普通命令另行通过四 rank 固定 8+5 检查。2、8、16 rank、显存和性能均未实测。当前目标 helper 和授权仅覆盖四 die；其他配置需要真实资源、独立授权、匹配的 helper、源码清单及硬件验证。结果见 [模型接缝](../../validation/rank-seams-round408.json) 和 [普通命令](../../validation/rank-portable-round408.json)。
