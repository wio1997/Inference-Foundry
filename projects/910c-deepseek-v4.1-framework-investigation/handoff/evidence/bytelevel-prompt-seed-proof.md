# ByteLevel prompt seed：受限等价论证与准入边界

输入域：本点原型只覆盖list prompt IDs。None/prompt_embeds入口必须先回退原stream；未在此点验收。下文P均指list，不能把list fixture假设推广为全Request schema。

介绍：要删的是首次step对完整prompt的两次native decode，不删模型计算、输入tokenization或已有原构造的ID转换。候选仍先执行原DecodeStream(ids=P,skip)；只在不可变ByteLevel且最后非空单token decode以ASCII结束时弃用其seed、建空stream。其它情况原样返回。原ids转换异常发生在guard之前，故保留；原stream的额外分配/guard成本进入CPU局部比较。

到了哪里：上游tokenizers v0.22.2的ByteLevel.decode_chain按token将字节串接，再String::from_utf8_lossy。最后一个有效token的最后字节为ASCII，确保完整prompt字节串B(P)结束在稳定UTF8边界，且decode(P)不以替换字符结尾。此前即使有非法字节/内部替换字符，也不能被未来字节修复；ASCII后追加的字节不能改变P的解码。逐token native decode取尾ASCII是这个充分条件，不是“最后固定k个token总能代替prompt”的假设。

在首个非空有效输出前，decode(P+O)=decode(P)+decode(O)，两边都在decode(O)以替换字符结尾时返回None，并保留未完成UTF8；空/被跳过special也都返回None。首个有效输出返回decode(O)的完全相同UTF8字节。原算法随后drain P，保留新输出prefix/ids；候选可比原算法提前丢掉零解码字节的special/unknown IDs，但投影字节串及被消费的边界相同。之后前缀字节、保留suffix字节和prefix-index所指字节边界相同，按step归纳，返回字节/None相同。不同的零字节ID物理buffer不能当完全内存state逐位相同；验收对象是原语可观察返回值、异常及后续消费行为。

Python绑定先提取u32或Vec<u32>才调用step：无效类型/负数/溢出在状态变更前抛出。保留原构造使P参数的全量提取异常也一致。ByteLevel concat边界下不会因删P新增InvalidPrefix；若不是这个decoder/发生词表decoder mutation或不能确定尾ASCII，直接原路径。不声称其它decoder、fastokens shim或任意tokenizer共享此定理。当前fast类通过module取DecodeStream，新接入必须核实实际类/binary版本，不能绕过patched backend。

结果：这是一个限定数学依据；已装binary的点证、真实输入IDs/flags身份和完整wrapper/SSE回归另见point/report。有限fixture不证明所有未来序列，源码tag不证明已装binary来源。未完成UTF8 seed盲删有反例，因此不扩大guard。字节/None逐步相同可让未改的spacing/stop/holdback/DTO逻辑沿同状态运行，但仍需原方法wrapper回归以及真实应用接线，才能列为已确认可实现。当前不部署，不把局部CPU时间换名为实测TTFT。
