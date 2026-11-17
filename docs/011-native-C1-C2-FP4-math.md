# 原生 C1/C2 压缩器、RoPE 与 FP4 路径验证

固定 A2 镜像中的原生 Indexer FP4 在16/96/8192行、RoPE 在96×8×512、C1/C2 压缩器在16/96/8192行，均完成八张物理卡的数值检查、图捕获和三次变更输入重放。输入隐藏维5120、KV维512；C2 保留 FP32 的 wkv/wgate 投影和 softmax pair pooling，最后以 BF16 做原生 finish。它们是合成参数的原生组件，未验证完整检查点质量或模型 PD。

104 个实际案例中80通过、24失败；失败为压缩 KV 的原生 FP4 路径在16/96/8192行的八卡试验。错误位于 float8_e4m3fn scale 转换，返回 561103；不能据此改成 INT8 Indexer 或改变每16元素 scale 和 E2M1 舍入语义。

G170 用 log2/exp2 的算术形式替代 E4M3 cast，在全有限正编码与舍入边界的精确比较门槛失败；保留该候选和原始失败，未进入实际 FP4 输入验证。G174 后继候选改用 IEEE 位字段构造精确二次幂和显式 ties-to-even：全部八卡每卡509个正有限编码及中点两侧FP32边界案例零差异；16/96/8192行BF16压缩KV与原生CPU结果逐元素一致，24次图捕获、72次变更输入重放通过，终态确认全部八卡进程清理。镜像、CANN和驱动均未更换。

- [八卡原生组件汇总](../evidence/native_math_G166/summary.json)
- [原生 rank0 明细](../evidence/native_math_G166/evidence/rank0.json)
- [首个替代 scale 候选失败](../evidence/native_math_G170/evidence/rank0.json)
- [原生执行 probe](../code/native_math_G166/probe.py)
- [失败替代实现](../code/native_math_G170/fp4_compat.py)
- [精确舍入与全形状结果 rank0](../evidence/native_math_G174/evidence/rank0.json)
- [实际通过的 scale 兼容实现](../code/native_math_G174/fp4_compat.py)

本轮未测端到端性能、同臂波动或收益；图重放数值通过也不代表已经完成完整服务或最终框架对齐。
