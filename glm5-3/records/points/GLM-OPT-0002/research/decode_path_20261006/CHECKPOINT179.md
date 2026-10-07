# checkpoint179 — actual MTP caller, preserved recovery, DeepSeek graph reference

2026-10-07. 本阶段没有新增代码级性能 KEEP。H6/H5/mainFULL保留；H11off，H12仍无设备正确性/性能结论。Current、完整API、80K/600/93%正式SLA未完成，继续研究。

Run277保持FAILED。原候选old布局short8、terminal short8、唯一恢复short8共三真实PD请求，独立raw还原均exactIDs/usage/length和D external queries/hits2334。原候选first请求all16native transfer已查。错误是布局observer必须凑两cached+两uncached；Root亲读actual MTP proposer2510–2546、SFA283–349及现有rank0period3 index57038/58759之后copy_，确认MTP普通builder.build与target均use_cache=True。候选并未跑到dense布局，不能claim数值正确或性能收益。

恢复中的H11 immutable witness filename复用问题在恢复写入前修复：48个原candidate文件原样归档epochs/candidate/witnesses，记录SHA，重建恢复目录；无源码/模式/新增请求。恢复完成后独立零请求closure检查controller退出/锁空闲、同guard记录workers、all16healthidle、H6H5、mainFULLactualcapture及H11mode0/transition1。D恢复root2835838/start311746121，P249root1916718不变；这些是本checkpoint启动前历史身份，后续以当前Run279现场为准。

Run278保持PARKED_NOT_STARTED：没有spec/controller/性能请求；不得从FAILED277准入。Run279只修observer和epoch路径，不改production f5ae2c69两行contiguous。首四shape2调用采集、严格全cached和持久输出指针；恢复使用独立epoch H9/H11 witness/mode路径。CPUobserver20、workflow10、actual launch/epoch fixture通过；独立RUN277_FAILURE_CALLER_REVIEW纠正先前“draft临时输出”推断，未见新source blocker。原Review/失败raw保留。

用户要求参考DeepSeek MTP图已落实至[源代码对照](DEEPSEEK_MTP_GRAPH_REFERENCE.md)：primary V2 commit2bb3f447三文件原样保存/hash。普通DeepSeekMTP共享图路径存在，GLM原源码被_is_glm_model单独禁图；旧V1 DSpark eager不代表普通MTP不支持。H11已经capture merged model/logits/sample，不把重复扩大既有capture边界当新patch。V2 K1只首轮capture，其draft-prefill命名不是P端prefill；MLA graph参数更新不能套作SFA已no-op的成本。GLM额外pointer contract/outputclone是可研究差异，尚无可删大gap证明，保留异步消费者安全。

唯一Run279于11:44:17Z由controller2099946/start311906533启动，spec85e34995/74pins；81个installed-source gate、原失败恢复closure/同workers/H6H5/FULL/H11off预检通过。四固定PDcorrectness请求，max5含唯一恢复；无profile/scan/性能比较。当前仍初始化、没有device结果。实际state/active_epoch/terminal evidence决定当前状态，不复用旧PID操作。

下一步：独立raw设备正确性通过才freeze新same-worker matched comparison；不重放Run278/277/279。有效patch进入研究stack，最终产品Current另由完整API/正式SLA/稳定性/完整stackE2E裁决。
