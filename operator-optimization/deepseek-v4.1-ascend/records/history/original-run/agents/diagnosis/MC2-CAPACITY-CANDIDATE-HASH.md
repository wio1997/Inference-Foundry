# 旧采样 MC2 未命名属性的容量候选：仅一致性线索

## 已记录的事实

此前 `mc2-group-attr.json` 从旧采样中唯一匹配了 EP0、EP8 各一条第一层 `DispatchFFNCombine` 设备任务，并保存每个属性值的 UTF-8 SHA-256 与长度，未保存原字符串。复算 `python3 agents/diagnosis/mc2_candidate_capacity_hash.py`：两条记录的 `attr_1` 均为五字节，SHA-256 等于字符串 `65536` 的 SHA-256；此判断依赖 SHA-256 碰撞难度。输入哈希、源码哈希及各条结果见 `mc2-candidate-capacity-hash.json`，旧 DB 的身份和原查询限制见 `mc2-group-attr.json` 与 `query_mc2_group_attr.py`。

冻结参考 tiling 源码 `dispatch_ffn_combine_tiling.cpp` 定义 `ATTR_MAX_OUTPUT_SIZE_INDEX = 1`，此前冻结参考配置的默认值也是 65536。三者数值一致，为后续寻找属性契约的线索。

## 未证明事项和影响

profiler 导出的 `attr_1` 与参考 tiling ABI 属性索引 1 一一对应的契约仍未找到；参考源码并未与历史实际加载编译对象绑定；也未读到同次实际 tiling 字节、动态专家计数或 effective M。故该结果**不能作为当次真实 maxOutputSize 的实测值**，不能说明专家负载、内部等待或根因，也不能外推到新采样的第 14/39 层。两条记录来自同一旧 profile，非独立实验。

下一步优先定位同版本 profiler 的 attr 序列编码契约和这两条任务实际执行二进制的可验证映射；若已有资料无法建立，继续标为 unknown，不能依照名称、序号或默认值补齐。只对既有文件做 CPU 复算，未操作 NPU 或修改目标实现。
