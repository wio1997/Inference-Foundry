# Engram全局行Nowner几何与付成本请求判断

介绍：从现存全局complete-K lookup bank，沿原WKV到rotated gate、持久四路残差和下一HC消费者，判断改输出列owner能少算/少搬什么。先价全段，旧独立小预算继续有效；每个替换原语逐位一致。

到了哪里：9当前源全匹配并限定body行段；原replicated/custom方法、FLOAT选择、完整权重加载和现有通信入口。56有限几何情景，4组CPU纯U16搬运共1,638,400个原输出word，各含全部65,536种编码，0搬运位差；不是GEMM、HCCL、gate或NPU精度测试。128原C8首token WKV/gate记录、48新配平短窗完整链、108 C8/216短窗条件价；六普通响应代理和544原生component比例逐项保留。不是全链路或全native路径穷尽证明。

当前replicated不是漏设TP开关：`AscendReplicatedLinear`通常用`CustomReplicatedOp`，但create_weights和weight_loader仍是完整N；FLOAT返回None，走原unquantized_gemm/F.linear。仅改tp_size或disable_tp不会自动把两份300MiB权重切开。旧trace BF16/ND形状也不能指定当前loaded binary或K/Fixpipe顺序。

全局24行K6144输入在SP分行前已经存在，可供Nowner计算；每rank新产24×3200后，仍需按原SP行恢复成3×25600再交原gate。N128示例新增每rank总交换150KiB，其中远端131.25KiB。原TP head AG仍然需要，不能同时认领direct-SP-owner省掉全局bank的信用。两层table/权重及代次各自独立，不能因同ID复用结果。

保留八个原M3行面板、各按M16说明时，总padded MAC不变，不能记8倍计算省。若逐面板重新调原完整GEMM，B操作数读取又回到每rank300MiB；只有K外层、八面板交错复用B才有少搬空间。原每输出K累积、BF16舍入要保持。N128的八个FP32 M16 panel需64KiB L0C，双buffer128KiB只是静态示例，未证明其它flags/alloc；N512/1024单buffer已超过128KiB，停止这个单AIC全八panel同驻配方，须另付分组/重读成本。N256以后等宽3200会切原Ntile，要用ragged或padding交换；不能默认无额外tail。

原C8全free包络复算仍为最大请求TTFT 1.115388%。加入新交换时，112个首token父段可取自身下一attention的A2A作参考；另16段原来只有AG→zero→RS，无A2A/投影，不能借一个不存在的原算子当零新增成本。本轮首次一律要求A2A的解析失败及原脚本已保留，修正为同rank/count/dtype其它父段的明确proxy。这是原dummy/no-metadata分支证据，不等于任意local Q0。HCCL/AIV参考含到达等待，只是成本敏感性，不是新transport实测或链路下界；3200/4096 payload比也不直接换成时间比。

结果：WKV剩1/8、gate剩1/4、其它oldbody免费、每处另付5us时，新交换免费/.25/1倍旧参考对应最大C8条件TTFT信用约0.922%/0.865%/0.709%。即使gate全部免费，付.25倍交换及5us也仅0.984%。只停止旧1%排序参考下这些独立子情景，不把1%变成用户新增验收阈值，更不剪改producer/全局输入/HC的大parent。只有更高且付成本后仍可能改变去留的大范围，才值得补原生参数。

六普通的完整旧响应代理全免范围核准为0.824710%–1.418414%；此前commentary粗略范围以此更正。保存的B1尾部target/Engram仍是24行物理图桶，不能拿逻辑6行推成WKV M1。用128 eager、128 graph和288 padded原生样本，最大`WKV/(WKV+gate)=0.704206803`。在原代理的正权重插值、rank-max、median和verification次数规则内，独立WKV全免响应上限至多0.9988565%，剩1/8时至多0.8739995%，尚未付新成本；因此不为同一WKV-only预算补native key或新参数。近1%的全免模型值不具硬件置信度，亦不包含input copy、lookup、gate、host或改调度，不能用它停止更大parent。

原gate的inverse block32 FP32 rotation、RMS、左结合dot、signbit/clamp/sqrt、mask、`0*V`与BF16 residual保留；最终Torch sum和HCPre左折叠不按公式互换。一个FP32有限公式0与1的反例只否定普遍split-K重关联放行，不冒充本WKV原生位差。现有globalbank原本可能在SP clone之后退役；新Nowner要持有到分布式WKV完成，lookup ready不是其最后读者。跨层提前WKV需另建bank/weight代次和失败退役，并重新价完整producer。

待确认：更大lookup/globalowner/WKV/gate/HC或跨层提交的原始K/M/N/Fixpipe、全部资源/TP图通信、真实新成本、globalbank最后读者和完整请求曝光。已停止：上述明确低值子情景、flag-only分片、声称保原行面板还能省8倍乘加或八次完整kernel还能省8倍B读取，以及N512/1024单AIC八FP32 panel同驻。已确认可实现：本轮新原语0。

环境：171统一hostname已核，期间owned容器变为exited、exit0，同image/id；首次running guard失败已留原件，随后用docker cp逐文件stdout tar只读内存解析，9源全SHA匹配。未启动/重启/探测服务健康。新NPU、框架/模型进口、请求、profile、显式远端写入或服务动作均0。

下一处转较大MoE输入交付：动态quant、TopK行复制、dispatch到首个grouped GEMM的真实物理读者与共享条件。已有最大DFC父段价格优先；不复跑旧empty-credit、stock AGMM位差或direct-UB失败配方。证据：`engram-row-owner-{dependencies,source-review,source-excerpts,frontier,cpu,price}.json`，两个失败原件与container-state单列。
