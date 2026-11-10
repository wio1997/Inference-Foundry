# 模型guard、SP行与旋转Engram交付

介绍：从5个模型/attention入口guard追SP mask、实际router、Engram旋转和generic runner/cache两条backend派发。到了哪里：27当前源身份、21进程初始单项env、85原Python selector/布尔行/opaque控制；新增3处，原156位122受限绑定/34动态待追。结果：5处源码条件已说明，不是5处原生人口闭合；全callgraph仍未扫全，新增原语/框架准入0，每替换原语逐位一致。

已停止：把未选中的分解vision/hash fallback当A3热路径再融合（当前cap已含原生hash/vision）；将A5 unrotated gate替当前nonNone rotation；寻找一个该generic selector源链并不需要的use_compress注入；原样重跑既有独立Engram Wkv/gate/常量小配方。Wkv有quant_config参数但描述仍FLOAT，旧原生任务BF16；初版quantized标签已纠正并保留原件，整段范围未缩小。

旧原C8 first-token匹配：当前6源与旧身份相交、量化描述SHA同2f4020；完整gate全免最大0.489683%TTFT，Wkv0.615615%，常量/重复widening0.037375%。Wkv+完整gate+copy1.115388%、再含紧邻HC Pre1.207905%仍包含必要数学/残差，不是预计净收益，更不是lookup/TP/host上界。50%保留工作情景见原arrival模型，既有有限M3/M15 native fixture按历史资格保留，本轮不重跑或新增准入。

待确认、继续：更大的跨模块row来源→route/dispatch→expert/shared→HC/aux交付。旧EP0/EP8代表rank、localM3形状窗中，global24整DFC全免15.979–17.537%，norm→Post均22.505%；这只是两代表卡旧窗口，非全16卡/请求上界。若真正删掉DFC时间2.5%且每call新增5us，局部均净1.516ms/0.162240%该窗；25%为36.759ms/3.932827%。时间份额不是padding行比例，未知默认0时新增费用为负收益；请求实际eligible calls/曝光和六普通未绑，新增row请求信用默认0，不迁移1025ms诊断或原C8/TTFT。

SP helper只补模型shape对TP的余数；输入mask=None、shape8时不会自动知道actual6。当前Ascend router/BaseRouter所读路径不直接消费is_padding，不能因flag默认True就认padding expert已被跳过，也不能未闭合runner/dispatch/可选reader便删发布。maskNone的profile路径与本卡query0不同，后者保留global cache准备、TP交换、补齐和O投影。generic runner源链选MLA sibling、V41逐层cache backend独立；其updatable predicate不是GQA路径，gid12该源歧义收窄，但捕获地址/全部读span、task reset/异常与native位/净价仍待确认。

已确认可实现：本轮0；旧封闭CPU租约原型1及历史有限native常量fixture分别保留原资格。source、初始env、opaque控制均不当loaded对象/真实事件或数值逐位验收；错误读路径、AST helper扩展、2代表rank误数及量化标签纠正均有记录。

[本段判断](模型guard与SP行及旋转Engram交付判断.md) · [122/34与实际消费者](model-gate-source-review.json) · [85源码控制](model-gate-source-controls.json) · [先验大row范围](model-pad-row-parent-primary.json) · [分cohort局部/条件请求价](model-gate-parent-prices.json)。

下一条待扫：loop047 Run174546Z PLANNED：沿真实V1 _prepare_inputs/_preprocess追target/draft ID、position、hidden尾部的所有写者/末读与行映射，到M06 router/dispatch/shared/HC/aux/下一轮消费者。区分actual/FIA/graph/DP/SP，不把is_padding写入当路由已跳过padding；先核大row交付的实际人口/完整输出义务。34动态原位及递归callee未闭合，不重扫122固定约束；保留大metadata/gid12和Engram owner/input候选。source-first整段边界后用旧C8/六普通，单独核费用/曝光，不跨cohort相加；每替换原语逐位一致。此源审Run无新Torch/NPU/模型/profile/请求/服务或远端变更。

Run170051Z仅此有界源码/旧price PASS，正确性NA；Run174546Z PLANNED。6源相交不当旧19身份全部重核；量化描述8.68MB仅读JSON，权重tensor payload读取0。27身份不当27全文；756文件只是term盘点。EP0/8 shape窗不外推全16卡，更不和其它cohort收益相加。新Torch/NPU/模型/profile/请求/服务/远端写入0，未重测live健康。
