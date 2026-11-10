# worker状态协议与早期CPU坐标交付判断

介绍：沿真实execute→状态更新/CPU copy lease→异步修正→metadata provider→forward/错误→12字段暂态→sample消费协议，排除固定配置不合法gate。到了哪里：24当前源身份；61原worker guard源码leg走读，13原方法opaque协议控制+13字面量/factory控制；新增29受限冻结约束，原156位119绑定/37动态待追。结果：源码leg不等于分支人口或完整callee覆盖，全链路仍未闭合，新增原语/框架准入0。

已停止：仅同配平C8详细诊断下，保持CPU计数生产/最终记账与GPU metadata原语不变的“删CPU位置影子/结构索引共享/早期等待后移”独立小方案。先定义完整父交付，再放宽到整个execute输入准备self，其prepare-Free仅1.687078ms/11循环（0.164512%诊断窗）；全命中/全转移/零新增的条件整批代理仅35.272ms/0.220889%。50%实删暴露、每frame新增0.02ms，局部净0.623539ms/0.060803%诊断窗；全命中/50%诊断转移的请求代理13.036ms/0.081637%。非实测收益，不迁移原不同prompt C8/TTFT/prefill/六普通；不作为删除CPU计数D2H整个协议、CP共同frame或gid12大段的上界，不与旧候选相加。

待确认：37位仍有动态输入/人口或后续参数缺口；其中32worker位已有条件leg记录，另外5个model/attention guard继续。递归input/preprocess/lookback/MM/DBO/cascade及数值callee尚须追到全部消费者。A3 V1 model-call错误会release metadata；preprocess/Engram准备错误在该try之前，不能把V2 join当V1全错误保证。空scheduler/有效批提前返回与全局活跃但本CP query0不同，后者仍有global cache准备与O交换/补齐；CPU H2D guard也不等于producer_done。gid12整段和大共同frame继续保留图读span、task reset/异常、完整packet/逐位与无探针净价缺口。

已确认可实现：本轮0，旧封闭CPU租约原型1保留。逐位门槛不放宽。控制架两次初版遗漏字段/误数13字段已按原源码修为12并留记录，不是后端错误或native数值失败。

[本段判断](worker状态协议与早期CPU坐标交付判断.md) · [119/37逐位与61源码leg](worker-protocol-source-review.json) · [13原协议控制](worker-execution-protocol-controls.json) · [先验整段范围](worker-position-parent-primary.json) · [整段放宽上界/条件价](worker-position-parent-price.json)。

下一条待扫：loop047 Run170051Z PLANNED：先核5个model/attention guard的初始化与实际读者（MoE skip-padding、Engram rotation、no-metadata/空query），再沿未闭合递归callee走实际_prepare_inputs/_preprocess/lookback/Engram及M03/M06/M08大交付。32个worker动态位已分列源码leg与运行人口，不重扫29个冻结gate。补actual OOT selector patch/use_compress注入与graph backend，影响gid12整段候选合法性。先独立源范围再匹配旧C8/六普通；保留大frame/gid12、全部CPU/producer/consumer租约与native逐位缺口。此Run无新Torch/NPU/模型/profile/请求/服务或远端变更。

Run163605Z仅有界源码/旧预算PASS、正确性NA；Run170051Z PLANNED。C8 detailed17.632761s与light15.967999s分别保留，后者229/231真实metadata frame只用于显式条件代理；默认未知命中/转移0。24身份不当24全文，13协议控制不当真实事件或native数值。原诊断2未匹配FT区间在其它已标记metadata parent内，未补造。新Torch/NPU/模型/profile/请求/服务/远端写入0，未重测live健康。
