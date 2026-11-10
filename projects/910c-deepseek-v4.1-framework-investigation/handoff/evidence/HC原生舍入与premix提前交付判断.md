# HC原生舍入与premix提前交付判断

介绍：扩大 O 投影交付到 HcPost→FFN HcPreV3→RmsNormCast→MLP→FFN Post，核实能否共享行数据、减少等待。每替换原语逐位一致；先保留原系数计算，判断输出间是否真有依赖。

到了哪里：当前20份源码 identity，逐文件审查行段单列，原始body和SHA保留；100个symbol文件/340个路径只作导航。追到 OOT→aclnnHcPre、A3 K-split mixed1:2的stage1全核同步→stage2、HcPost D-split、RMS BF16与FP32两路、MLP两种路由入口和FFN Post。两代表rank各516条源一致链，目标480条、草稿36条，价到实际下一FFN Post。源一致顺序不是指针/事件或活跃binary证明；未读完的系数尾部、nextdecoder/Engram/final读者和FULL分流资格留作下一Run。

结果：发现 `hasPreMix=1` 的真实提前交付机会。FFN HcPreV3收到的是 attention HcPre先前产出的 `attn_pre`。其 ProcessY 用这个旧系数 collapse当前四路残差；当前 HcPre新产出的 `pre/post/comb` 分别供后续pre_mix与FFN Post。因而可以让原序collapse＋norm先交付MLP，并让当前系数支路并行、延迟到FFN Post及下一pre_mix消费处join。`hasPreMix=0` 的首段仍需等新pre；不能靠开关推广到所有层/分支。

原数学边界已具体化：HcPost先FP32乘，再按hcIndex逐项Add，最后BF16 RINT；ProcessY将这个已舍入残差扩宽，FP32乘系数、按0→1→2→3左折叠，再BF16 RINT。RMS随后先逐元素平方乘1/D，按原64lane累积/WholeReduceSum归约，加epsilon、sqrt/div，再FP32乘rstd及gamma、BF16 RINT。其FP32输出是最终舍入BF16再扩宽，供内部router或外部F.linear，不能直接拿舍入前FP32替代。

成本按fork/join计算，保留完整原HcPre系数支路，连原本重复的Y计算也不先讨信用。另付提前collapse的e、支路新增暴露h、MLP争用delta、系数支路慢化gamma：新join时长为`max(C*(1+gamma),e+N+MLP+delta)+h`。两个目标短窗的C总计17.396/17.522ms，乐观全隐藏上限1.8503%/1.8848%窗口。e5/h5且delta0时条件净省1.33975%/1.36849%；若MLP额外慢40us则为−0.70238%/−0.69680%。全部144情景含草稿和负收益保留。这不是免费重叠或请求预测，也不把MLPspan里的跨卡/其它等待算作可删计算。

资源边界：现有HcPre stage1是mixed24AIC/48AIV启动，K分割、FP32 cast workspace、HF32 AB与原partial归约均仍保留。208MiB是源码分配下限，不是实测physical peak；保留系数支路就没有该容量释放信用。与gate/DFC并行的真实争用能改变去留，且FULL graph要有独立输出事件和正确的后续join，不能把同一op的tuple输出误当独立完成。原rounded四路残差还需给FFN Post，跨层pre代次也必须保护。

待确认：提前交付方案的原语位、实际UB/flags/事件、FULL/capture与跨卡collective顺序，当前coeff尾部与下一decoder/Engram/final读者，实际提前成本及MLP争用，大prefill/六普通/草稿/no-premix回退和完整请求曝光。值得继续，缺口具体保留后推进下一读者；本轮未做NPU参数扫或新采样。

已停止：用舍入前FP32直接喂router；只因norm输出已生成就删除后来FFN Post所读残差；无premix时原样提前collapse或不计算仍活跃的coeff输出。停止范围不包括保持原舍入的新consumer融合、付费重算/owner重排。

各情景互斥，且与O投影/QKV/head/frame方案有重叠与资源相互作用，不能把这些百分比相加。

已确认可实现：新增替换原语0。源码依赖可分离和条件净价都不是native逐位准入；全链路未扫全。新NPU/模型进口/请求/profile/服务动作/显式远端写入均0。

证据：`hc-delivery-{catalog,dependencies,source-review,source-excerpts,frontier}.json`、`hc-early-delivery-price.json`。原始逐链、每条成本参数及所有负情景保留。
