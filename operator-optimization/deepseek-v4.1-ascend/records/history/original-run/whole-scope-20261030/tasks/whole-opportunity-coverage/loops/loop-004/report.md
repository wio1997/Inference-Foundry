# Loop 报告：loop-004

- 结论：`PIVOTED`
- 决定时间：`2026-11-02T13:55:34Z`
- 模式：`spec`
- 冻结目标是否满足：`no`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

Migration changes the active case/hardware to171. Loop004 preserves its frozen169/166 case domain and all scoped profile/cut evidence;full-scope completeness is not achieved. Continue in a derived171 evidence Loop,not by relabeling171 probe as166.

## 证据

- `/Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/partial-persistence-correction-v161.json`
- `/Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/joint-runtime-summary-v161.json`

## 下一步

继续全模块、全分支和跨算子/层/卡/迭代的原整体目标；复用已证明的共享算子cut，不展开383旧关系行的笛卡尔排查。v160/v161取证合并封存为一次权威更新：当前6组Engram重复payload逐字节相同，停止为其数值初始内容追查一般loader；实际布局、驻留和更新代次仍须按算子需要取证。171普通启动及单次回答2已通过；完整C8×32768→1024/前缀90%尚未实测，旧169/166-105身份不回填。实际CANN9.2映射backing ELF与SDK ABI已核对；一次单独授权的69B eager Pad联合探针完成且输出正确，原服务已恢复health200。授权已消耗，不得重跑或扩展至全模型诊断。内置字段门与storage基址语义已在本例实测；直接复用它们，定点补offset/extent/epochs/scalar values/native桥和读者。152B HostArgs含[8,24] relocation，word0位于记录的2560B私有分配+1024；不据此定角色、NULL、不交叠或可删除。下步先把现成profile采集入口与实际版本逐项对齐，冻结可审查的171全Case诊断方案和原生桥缺口，准备工作不触发新设备采集；不再重查已匹配的SDK或重跑旧控制模型。保留9/1352原生参数、849PMU、16phase、3136归属、G14 copy/control、分支穷尽性、实际数值/读者/物理容量和最低MFU全部缺口；整模块闭合0、整体未完成。
