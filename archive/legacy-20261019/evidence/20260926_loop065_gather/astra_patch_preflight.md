# Astra High：Loop065 H003 补丁预检

2026-09-26；只读审查 scripts/loop065_hidden_gather_patch.py、run_loop065_hidden_gather_diag.sh / profile.sh、当前borrowed源码及Run288启动记录。未操作NPU/服务/源码。

## 结论

未发现应立即中断Run288的明确语义错误。当前Run是执行/Graph smoke，不是数值或性能KEEP gate。H003准确含义：Target decode的hidden AllGather与独立local Q链并行，join在第一个gather结果消费者wkv之前。

## 对照是否成立

immediate/delayed调用同一 all_gather_async(hidden_states_local,self.tp_group)，输出均由同helper分配，默认async_op=True、同TP device_group。
immediate立即Work.wait，清空handle并unpad；delayed保留handle，沿原有代码在Q RMS/RoPE之后、hidden_states_cache/wkv之前wait并unpad。两者设备工作量相同，除wait移动及同一unpad视图的时机，无另改operator。
在冻结pad=0时差异就是wait位置。若pad>0，两条路径也都仅裁一次，但需确认裁后不少于96有效行。
off原始helper仍是另一个A0，必须保留最终A0对照；immediate不是当前正式baseline。

## 命中与边界

- ops/dsa.py:164把forward_context.flash_comm_v1_enabled传need_gather_q_kv。
- 确切层名model.layers.2.self_attn.attn已在既有CP实验实际命中；本patch另要求非prefill、actual96、非draft，因此固定产品Target下具备命中条件。
- input必须[12,4096] BF16；建议验收实际all8输出[96,4096]、TP8、pad=0和CP local geometry。
- Guard当前在提交collective之后才检查；若未来shape不一致会先入HCCL再单rank报错，可能挂起。今后应将input/TP/pad前置检查；当前已冻结shape，不据此中断正在启动的Run288。
- get_forward_context.is_draft_model不是唯一屏障；exact target layer名也排除了mtp.*，所以未见误作用DSpark的路径。

重要纠偏：_maybe_all_gather_o_proj_full_weight调用位于delayed wait之前；full_gather_wo_a_enabled在forward中依据attn_state排除DecodeOnly/SpecDecoding。其安全依据是实际decode状态，不是“910B3天然关闭A5分支”。需核实运行实际attn_state；若意外状态使该额外collective启用，两种wait会跨过另一collective，因果控制范围扩大，应停止将其解释为纯Q重叠。

## Graph / HCCL / 生命周期风险

- dsa_forward是PrivateUse1自定义op，现有prefill已调用同async helper；这支持实现可行性，但不能证明decode FULL Graph中的Work.wait会形成期望device依赖。
- Graph capture后Python分支固定；跨服务mode启动符合独立capture要求，不可同Graph改env切模式。
- input是localQ只读源；gather输出保持引用至wkv/forward，原代码wait在首次读取前。没有KV/state写集变化，也没有新数学依赖。
- HCCL仍同group、同payload和调用顺序；不能保证有后台进展。Work.wait可能阻塞host/插入event，Graph或HCCL资源也可能将Q排在collective后；必须由设备timeline判断。
- 单次EXTREME_HIDDEN_GATHER日志只证明一次Python命中，不能独立证明所有FULL replay都选中该分支。应联合all8日志、实际capture/replay、HCCL与Q相对时序。
- 脚本使用serve.sh，max_num_seqs16仍可能handoff覆盖缺口；按真实Runtime cohort计数，不用48+12客户端自动推40报告。diag无profiling，profile脚本配置cycle64/65；每个cohort会独立采集，应确认选择哪个cohort导出。

## 最小验收 / 下一步

1. all8命中相同mode/layer/input/output/pad；完成实际FULL Graph报告，无collective hang或错误，client输出合同完整。只记smoke通过。
2. 固定layer输入，对原A0/async immediate/delayed比较gathered hidden及localQ输出（先A0自重放），符合冻结数值标准。不要重演Run285仅以全Target末端差异裁断。
3. matched profile取all8 AllGather start/end、Q链、wait、first-wkv、下一collective与Target/cycle端点。B须实际同时执行并使最迟rank join更早；Q膨胀/transport竞争计入，不加总kernel时间。
4. 若B仅优于immediate但不优于A0，不构成产品收益。最终晋级仍需未profile重复正式A0对照E2E。

补丁SHA在Run288记录为225ce19679706437970bcc42f5715b786dc71d20f5ba0a33a7c7680c8ee87129，base 27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e。无需依据预检推断任何TPS。
