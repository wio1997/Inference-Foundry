# 真实 worker 调用家族与有效配置判断

介绍：继续三层body交付，追真实V1 worker→继承profile/capture→target/draft dummy→实际消费者。本轮是固定源码、原C8和历史配置回执的离线核对；无新远端操作，不能重申当前服务身份。**Goal ACTIVE，全链路未扫全，每个替换原语逐位一致。**

到了哪里：22份局部源范围（15匹配旧identity，selector另匹配历史SHA）；原C8的3632个活跃波参与窗全部对上CP分支、SP行数、head行数和每层sender。源码家族包含真实metadata和无metadata dummy，不能用相同M混为一图。此前“shared已重叠”的无条件表述已撤回，详见下文。

结果：三层body联合范围仍有上轮整批3.569201%的原调度全免费有利包络；有成本情景保持不变，**不是预计/实测收益**。本轮排除了两个仅翻flag的配方，定位真实capture模板和DP伴跑的大范围；没有新增“已确认可实现”的设备准入。

|具体范围/配方|状态|结果与缺口|
|---|---|---|
|只翻DSpark eager flag，复用现有dummy图为真实body|已停止|spec enforce_eager、DSpark override、dispatcher、dummy metadata与真实分支均未由一个flag解决；dummy图没有query/cache/SMLA。|
|只按M3共用真实Q0和metadata=None的图|已停止|Q0但有metadata仍做cache预处理/TP恢复/O投影；None直接zero输出，任务序列不同。|
|只翻shared multistream开关，保持现行DFC策略|已停止|`enable_fused_mc2=1`会把shared开关改回False；历史candidate样本0，无可引用的service speedup。|
|同原语body graph/提交与有资源控制的shared/routed重写|待确认，仍值得深入|真实metadata模板、独立dispatcher/all-rank capture决定、storage与event代次、M1/2/3三组权重逐位、控核共驻和净成本仍需实现；不能直接绕过selector当受支持方案。|
|只删原C8的draft dummy伴跑，独立冲首token1%|已停止同小预算|原C8仅一波有idle DP；连同真实peer全部三body parent免费，也仅P90 0.647747%、整批0.051980%。不再为该C8小方案补native参数。|
|更大的inactive-DP target40+draft计算/路由交付|待确认，下一段|保留跨DP真实expert服务，查dummy输出读者、mask和接收几何。六普通可能有不同dummy频率，不能被C8一波预算剪掉；也不能跳过EP成员。|

## 真实调用及合法条件

1. `NPUWorker.init_device`按`use_v2_model_runner`选择runner。本页沿实际V1固定源和原V1证据；不把另一MRV2实现自动并入。`determine_available_memory`即使显式指定KV容量也会调用profile_run。V1 profile可先按MC2 capacity限长，再调用继承GPUModelRunner.profile_run→V1 `_dummy_run(is_profile=True)`。
2. profile在完整cache绑定前，不强制构建attention metadata；DSpark `dummy_run(is_profile=True)`直接调用`precompute_and_store_context_kv`却不传context slot list，模型因此提前返回，再执行无attention metadata的body。不能拿这个profile分支当真实query/cache峰值或模板。
3. cache初始化先建target backend/metadata owner，再最终绑定KV与C2 ring，最后调用DSpark的backend初始化。KV backing可进sleep pool，persistent metadata在外；这不是按shape可重新分配地址的许可。独立kernel_warmup、workspace和部分清理callee不在选取的snapshot内，留具体未读边，不把caller已读记为callee完成。
4. `NPUWorker.compile_or_warm_up_model`先compile尺寸降序dummy、kernel_warmup，再capture。V1 capture委托GPUModelRunner.capture_model：descriptor家族→`_warmup_and_capture`→V1 `_dummy_run`；warmup后同步辅助流，target先执行，再`drafter.dummy_run`。`profile_cpp`跳过drafter、MM encoder-only早退、混合dummy明确拒绝、FULL才构建metadata、LoRA/EPLB等条件不能任意合并。
5. **target FULL不等于draft FULL。** 已保存配置为target FULL_DECODE_ONLY/24桶、spec enforce_eager=True；DSpark构造又强制`use_cuda_graph=False`，dummy也重设NONE。其父类虽可经AscendDflash/AscendEagle的MRO到upstream dispatcher，但upstream仍按spec eager与mixed-mode设NONE。不能把普通词表TP8误解成fine-grained lmhead开关已开。
6. DSpark普通dummy传`set_ascend_forward_context(None, ..., draft_attn_metadatas=[])`。`dsa_v41.forward:697–699`直接zero/return；真实metadata本卡Q0则仍进入preprocess与`_project_output`。dummy仍执行三层HC/FFN和三次EP，context槽设-1不表示其所有投影都被跳过；normal dummy和profile dummy的context路径也不同。
7. 运行期不只有warmup dummy：DP engine没执行本地ready请求、但其它engine仍运行且未sleep时，经executor collective RPC→`NPUWorker.execute_dummy_batch`→target uniform-decode dummy→draft dummy。另有`_input_fits_in_drafter`失败→`_skip_drafting`→synthetic B1草稿伴跑，并发布全宽zero draft占位。全空engine直接继续，sleep分支不执行dummy；不能把任意零本地请求都当可绕过分布式协议。

## 有效shared配置更正

我上一轮从可选helper的提交顺序，错误推成冻结配置已在重叠。冻结`ascend_config.py:843–848`的逻辑是`enable_fused_mc2==1 && multistream_overlap_shared_expert`时直接置False；源码SHA为`d5ea2cd…`，与原flag-only C8启动排除回执完全一致。saved EP0正值快照也记录False，作为另一保存cohort的佐证，不拿它补原C8时序。

False时`_prepare_shared_for_router_overlap`返回None；routed先提交，随后shared在当前流执行。源码中的提前量化/Gate-Up和aux stream是**可选实现**，并非当前已生效的重叠。旧flag-only实验没有candidate性能样本，其停止只约束不改backend的flag配方。此前4AIC/8AIV原语资格、整shared预算与本轮全body价格没有因此失效，但共驻仍须显式重写/证明，不准宣称已支持或免费获得重叠。

## 形状、分支与原C8价格

body输入是5×DPmaxB，SP为ceil(输入/8)；head在当前fine-grained lmhead size0下仍为5×ownB，普通词表TP仍为8。原先把两者相等的新增断言在56个不对称rank窗失败，194027Z ERROR保留；修正195530Z复算通过。例：一侧B1、另一侧B2，body SP_M2，而前者head M5。

3632个参与窗中，3136真实Q>0、488真实Q0但metadata存在、8个无metadata dummy。三层分支一致；逐层sender与原F21计数全部吻合。10个正own-B/DPmaxB组合加4个synthetic-B1 dummy组合，只是由source推导的设计集合，**不是14种图已穷尽/准入**，与旧ownB0枚举恰好同数不代表同语义。实际graph键还要包含num_reqs/QSL、local Q、metadata缺省、输入/输出行域、mask、资源代次和caller模式。

8个idle窗都在原第3波DP1；其synthetic有效发送分布为[1,1,1,1,1,0,0,0]，不是全False。将该波**连同真实DP0整个三body到head输入parent**一起免费，按上轮时钟/原调度/客户端deadline有利模型重算：P90 9.663440ms/0.647747%，整批9.663440ms/0.051980%。这是dummy-only配方的更大有利预算，不是整inactive target40路径的上界，也不是新净收益。后者和六普通缺匹配的实际caller/收发/时间，继续扫描；不因这里小而放弃大范围。

## 代次与异常边界

`synchronize_input_prep`同时保护dummy/real pinned源；全局reserved MC2 mask在每次forward context改写，然后按SP offset切片，提前上传不能越过前一代DFC读者。target与draft有各自model-owned通信状态，不能借target E24状态去捕获draft E8。

`_model_forward`的run_model/update位于try/finally，异常仍release metadata submission；Engram准备在try之前，不能因此宣称所有prepare异常都有同样退役保证。`_engram_capture_active`和wrapper的局部finally已读；GPUModelRunner正式capture的全局enable开关在正常出口才清，throwaway profile capture另有finally。新body capture失败需全rank一致退出/清理，不能一张卡fallback eager、其余卡继续EP图。release也不是设备末读完成；保留先前producer_done/消费者lease协议。

证据：`worker-capture-family-frontier.json`、`worker-capture-source-review.json`、`worker-capture-exploration-notes.json`、复算脚本。没有新NPU、服务请求、新profile或远端写入；另一个用户授权framework profile controller仅按交接只读获知，未操作/重复启动，也未把其静态状态文件当“当前运行中”证明。

下一条：从DP engine→execute_dummy_batch与`_input_fits_in_drafter`分支深入整inactive-DP target40+draft：先列真正活的跨卡expert服务和死输出，再按原profile整段定价。保留上述graph/native/缺失callee/启动MM budget缺口继续前进，不回到两个flag-only或C8低值dummy小方案。

后续收到另一授权任务完成的新framework profile，见[独立证据导航](worker-new-framework-profile-navigation.json)。先绑定Gloo与实际target/draft就绪；本页原C8价格未混用新cohort。
