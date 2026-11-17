# C1/C2 sparse attention 内核：八卡数值与图组件
+
+固定 SGLang A2 daily、CANN9.1、910B1，每张物理卡独立执行，使用原有 Ascend Triton 编译器。没有换驱动、CANN 或导入9.2二进制。实际模型要求 C1/C2，固定原生 target 缺少 libopapi metadata 符号，builtin 仅通过 C4/C128 调度控制组；完整模型尚不可服务。
+
+## 原生组合路径的证据与新内核
+
+G110 验证原生 SFA 分开计算 SWA 与 compressed partition 再用 LSE 合并的可能性。metadata 构造成功，但执行明确拒绝 return_softmax_lse=true。因此不把无归一化系数的简单输出相加作为替代。
+
+G114 直接实现 shared-KV sparse attention：输入原生 BF16 latent、合法 SWA physical IDs、合法 compressed physical IDs，-1 填充，attention sink 只进入一次分母，在线稳定 softmax。首版在编译 PlanMemory 阶段 UB 溢出：3961344 bits 要求超过1572864 bits 可用，未执行数值测试。G120保留同一512维数学，以16-key和128-value tile降低局部存储。每个输出分块重算score四次是已知优化候选，当前不宣称最终性能。
+
+- [原生 LSE 拒绝](../evidence/SFA_operator_G110/evidence/rank0.json)
+- [G114 编译失败和完整 IR](../evidence/SFA_operator_G114/evidence/rank0.json)
+- [G120 实际通过的完整内核](../code/SFA_operator_G120/lowratio_sparse_attention.py)
+- [独立 FP64 oracle 与改变图输入的探针](../code/SFA_operator_G120/probe.py)
+
+## 实测通过的范围
+
+G120 全8卡分别通过 C1/C2 × decode/verify/prefill 六个fixture，合计48 cases、48 captures、144 changed-input replays。输入含两个请求、窗口/页/奇偶pair边界以及512候选，Q/KV为非恒定BF16，有限FP32 sink，参考在目标服务器CPU用FP64 softmax与matmul独立计算。图重放同时修改Q、两路KV、sink和候选ID。最大绝对误差0.00031238794326782227，全部输出finite，全部8卡每个fixture的重放输出SHA相同。
+
+本阶段形状为decode2、verify12、prefill20行，不冒充完整C16或8192 chunk。G124单独扩展到C16 decode16、verify96、prefill8192；大prefill只对32行做FP64数值检查，同时检查全部输出finite和完整图变化，不能把32行sample写成8192行全量数值验收。
+
+- [G120 rank0 逐case证据](../evidence/SFA_operator_G120/evidence/rank0.json)
+- [G120 controller正常终态和全部8卡清空](../evidence/SFA_operator_G120/state.json)
+- [G124完整服务形状的有界探针](../code/SFA_operator_G124/probe.py)
+
+## 未完成的接入
+
+该内核只接收已经选好的合法候选。原生C1/C2 compressor、INT8/FP4 indexer计算与选取、page physical ID映射、KV owner复用、奇偶pair pending state、DSPARK rollback/accept、PD完整buffer注册/传输及runtime graph metadata更新都还需要接入、单独数值和完整模型验收。固定KV字节预算候选仍未部署到完整模型。没有DeepSeek正式AISBench性能、准确率或最终框架对齐结论。
+