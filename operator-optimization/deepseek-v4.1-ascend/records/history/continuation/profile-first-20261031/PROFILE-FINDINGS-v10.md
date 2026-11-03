# 图执行、字段冲突与 decode 范围审查 v10

本轮补齐了一个原始 EP3 capture 的 CSV/计算记录对应关系，并把图提交关联到 replay 控制对和已记录调度；同时完整核对同次16卡的 MC2物理形状。整体仍未完成，0个整模块关闭。

## 242条记录的对应已解决，冲突保留

242条Triton记录的name、type、stream、task、start、end六字段全部严格一致，且一一对应。CSV把Model ID写为4294967295，数据库TASK写为49；此差异保留在独立overlay，没有改原始数据，也没有对所有sentinel统一替换。当前EP3的36521条计算执行均已找到CSV对应：36279条原四字段对应，加242条显式冲突对应。

完整CSV仍按层级分账：36521条计算对应、1849条COMM区间、278条AICPU任务，共38648条。COMM区间与内部任务可能重叠，不能把两层时长相加。该修正只验证了此EP3；其他rank或capture不能自动套用。

## Replay与调度证据

11次原生graph execute各有同connection的MODEL_EXECUTE/WAIT控制对。仅从WAIT开始取区间会遗漏先启动的图任务；使用同一控制对的EXECUTE起点至WAIT终点后，每次1650条计算均唯一归属，合计18150条，0遗漏/重复。1650个定义在11次执行中保持一致，各stream的定义顺序签名也全部一致。

每次均有40组HcPre/HcPost/HcPre/HcPost，且每组各1个attention、1个MC2，共440个结构检查通过的块。原生PyTorch connection没有记录到提交父级；另通过同线程自定义annotation的确切call_id与配对sidecar，绑定了其中10次提交的调度：同4个请求均已缓存至少32768token，各调度6token、spec5，总24。请求HMAC集合与两次已分析eager调用一致。该关联由call ID与同线程包含域组成，不是时间最近匹配，也不是客户端request_id证明。

首次图提交没有对应的scheduled-model annotation，继续保留阶段未知。调度逻辑token数不能替代物理有效mask、接受数或专家接收计数。

## MC2的不同物理规模

16卡共8944条MC2记录，其中捕获图内7040条，每卡440条，物理M均为3、local experts24、K6。eager target M643/1023和draft M2/3分别保留；不能用一种形状的瓶颈解释整个C8负载。

这一结果补充了F21的小M decode范围，但没有展开原生dispatch/FFN/combine内部阶段。实际有效mask、专家负载、同调用EP参与者、权重流量和净收益仍缺证据。166observer计数与这些169历史时延不作同调用拼接。

## 计数器与模块边界

EP3的10次已绑定decode提交有400个marker块。按同域AIC周期加权MAC代理，块20最低，约0.0681534；这是这个单卡、这个decode子集的记录代理，不是全模型最低NPU利用率、MFU或硬件峰值效率。AIV向量代理单独保留，不能与AIC或CPU占核混为一个利用率。

每次1650条计算中，1539条位于marker块内；余111条分为块前21、块间/跨块82、块后8，共57种形状/类型签名。它们是实际工作。定点复用外层model.forward确认：辅助hidden mean先于Engram，外层循环完成lookup就绪等待及Engram，再调用decoder layer；最后还有fullHC MTP复制、collapse和norm。因此跨层HC融合必须保留这些前后消费者，块外不能直接称为idle或并入某一decoder。111条仍需逐definition归属；源码的控制流回答不能替代实际alias/代际证据。

下一步先利用已经保存的57签名及v6 Engram/tail合同，完成块外工作与消费者对应，再处理未定阶段和需要有界补采的框架成本。未改目标实现、未操作服务、未发新推理/NPU负载。

本轮三个分析错误均保留：WAIT-only容器假设不足、metadata目录层级错误、全16卡结果保存后的stdout展示预算错误。最后一项仅重新读取已完成结果，没有重查CSV。
