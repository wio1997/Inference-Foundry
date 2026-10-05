# glm5-3 HANDOFF

2026-10-05T08:20:28.892574+00:00 checkpoint145 Run210 terminal；已发布父144 915a63854d478ee53f708ecdb4fff810e89f15b8/treee65ff4326d3380a511179ab83b1f922447cc9d99。GLM-OPT-0002 ACTIVE。

持续自主推进；两台910C内功能完整GLM框架、动态可变请求及PD/并行容量。GPT研究/代码/裁决；真实zcode --prompt→Result→bridge执行监控归约，唯一GPUcontroller。研究代码raw在SERVER、Mac仅SSH；本任务必要启动/清理/重启已授权。无旧队列/Browser/subagent/automation/额外审批。Current=None；无性能KEEP，稳定达标容量/全可行域上界unknown。

## 当前资源

Run210 controller595813/start293294823 已于08:05:22Z完成，无GPU controller。D0/166 HOST2326304/start292799754/epoch dbf6d3e8c7ffeabe9b073496bd8f1bcdfa5c95622dd68ae6057c05c54f88d807/private204：TP8PP2DCP8 K3，Graph[4,8,12,16,20,24,28,32]/max32，8192/t1024/c1/serial1。D1/167 HOST2864155/start292303441/epoch1072299010005f607bb1d40f6cb025e3696d0445ec3e527de8583ec30a2ac4e7/private200：TP4PP4DCP4 K1，Graph[4,8,16,32]/max32，8192/t1024/c1/serial5。32 native rank 保持原身份。旧public204已正常退休；新public210 HOST603477/start293296365，private210/runtime_bundle V12/config210/restored/service_config.json，placement=shape_split_idle_spill/threshold32768，state125相同epochs，STORE204D0、200D1及新210D032+previous16，SDKinit0 active，observer210。bundle70只改shape_split_placement.py与native_engines_service_config.py；原runtime不改，完整文件索引及base144 commit固定。

## 新证据与裁决

真实 lease idle spill 已发生：late0 从 D0 借给 D1，其余短请求 D0、五条大输入 D1。功能59输出/5请求与动态14208输出/16请求均通过完整 native IDs、wire、计数、SDK 和 STORE 审计；总计14267/21。两端 native32、物理 epoch、K/Graph、D0 serial1 与 D1 serial5 保持一致，未重载模型或写 native 策略。仅旧 public204 正常关闭并 finalize0，新 public210 启动 init0，仍在运行。相同晚到窗口118.335s/120.066有限TPS，对比209104.022s/136.586；TTFTP90及输出间隔P50/P90三个参考SLA仍失败。借出 late0 输出间隔35.597ms，比20932.729ms慢，最大HTTPchunk间隔.0733s比.2756s短。单次配对、异步、MTP接受率、cold salts及新public冷memo混杂，不支持性能KEEP或孤立因果收益。REJECT 本轮配置与负载的QoS；功能机制可用不等于容量收益，Current=None，稳定达标容量与全可行域上界unknown。

Run210 audit v2 VALID87396B SHA178a51c1a2ef8e4d5cbf00953c7676ac463ac69829c97f188fe8f867f70d346a。v1 INVALID 为旧kind断言，不重放Run。Graph2 CPU1–4 均 INVALID/raw保留：路径前缀、remote归档路径、builder上下文问题；CPU5纠正脚本并完成有效归约，0模型/inference。

## 下一问题

CPU5 原生完整CLI与VllmConfig、Ascend V1 update_pass_config、实际SFA DCP builder support=UNIFORM_BATCH、原生CompilationConfig resolver及dispatcher核验 VALID；SDKinit-final0、native32身份和全部vllm计数前后相同，0模型/NPUtensor/inference。请求capture[2,4,8,16,32]最终仍保留2，noSP；K1宽2单请求 resident200 key pad4，而candidate key2。实际GPU capture fit与当前nativebatch/padding收益仍unknown。接下来只替换D1 native epoch，在唯一controller下验证size2真实capture/完整API/原STORE失效owner合同，再用相同210 late16窗口辨认收益；D0与其STORE保留。

## 恢复入口与历史

现场 /data/tiankuan/wio/glm52-pd/deploy，容器glm52-single；仓库 /data/tiankuan/wio/Inference-Foundry，branch glm5-3-autonomous-20261001。读现场 HANDOFF.md、SSH_AND_OPERATIONS.md、PD_START_AND_TEST_GUIDE.md。Mac166socket /tmp/glm52-166.sock；167通过166 HOST ssh root@172.16.10.167。只按当前boot/start/argv/HOSTNSpid/NPU所有权操作，controller锁 .controller.lock/.formal-test.lock。完整native计划/根/rank身份在204/restored；新public210/restored，源177和spec不可改。脚本glm5-3/scripts/zcode_bridge.py；Zcode cwd deploy/private/zcode-relay-work。GitHub权威，精确父commit非force CAS，保留其他dirty；live memo/observerlatest不捕获为immutable，原始trace仅SERVER。

历史204–209/CPU35及复用条件见 [checkpoint144 HANDOFF](https://github.com/wio1997/Inference-Foundry/blob/915a63854d478ee53f708ecdb4fff810e89f15b8/glm5-3/HANDOFF.md) 与 [checkpoint144 point](https://github.com/wio1997/Inference-Foundry/blob/915a63854d478ee53f708ecdb4fff810e89f15b8/glm5-3/records/points/GLM-OPT-0002/point.md)。Graph收益未隔离，旧profile105不是当前硬件界；大chunk2048/4096分别拉长HTTPchunk间隔，4096新增TTFTP75失败；保留1024。完整native工具/推理/采样/logprobs/Responses/STORE/previous/retrieve/cancel-drain/native错误/epoch绑定持续保留，不以手工清单缩小功能。
