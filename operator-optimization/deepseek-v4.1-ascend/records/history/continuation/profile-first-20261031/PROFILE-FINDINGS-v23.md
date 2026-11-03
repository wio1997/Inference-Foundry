本轮继续复用profile，没有重采、运行模型/NPU或变更服务。新增内容是属性证据质量核对，整体仍未完成，属性归属问题也未宣称关闭。

此前取回的attrInfo字符串不是凭字段名猜测解码：官方CANN9.1.0-beta.1表结构明确给出attrInfo通过STRING_IDS映射，内容用于算子shape及自定义参数描述。这支持解码约定；该文档不是已安装9.2导出器的精确实现证明，尝试的9.2同路径页面不可访问，已记录而未反复重试。[官方DB表结构](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/devaids/Profiling/atlasprofiling_16_1144.html)

对保存的132,000次图计算记录新增相邻属性ID核对，共18种op-pair、14,480次非空同ID相邻配对，其中13,840次跨算子类型。这里“相邻”只表示按startNs/endNs/TASKrowid排序后的记录，不能当作跨流生产依赖或API父子关系。相同字符串合法地共用ID也很常见，不能仅据ID相同判定错误。原先由结构精确绑定的3,200个ForeachAddList->FFN HcPost仍全部同ID；另有rotary/scatter、matmul/cast、indexer/prepare等跨类型复用。因此在绑定具体归属前，不将这些文本直接升级成自身算子的native参数、tiling或二进制证据。

本轮代码范围保持定点：按attrInfo/COMPUTE_TASK_INFO两个字面量，在两个已安装profiler Python analysis目录做有界定位，只找到表名常量和读取函数，没有定位到writer。这不证明其他写入路径不存在。随后仅查导出入口：Torch的CANNExportParser.run在Db分支调用外部msprof --export=on --type=db；精确CANN9.2的msprof.py只有28行，通过msinterface.msprof_entrance.MsprofEntrance().main()动态进入下一层。这是源码分支契约，不自动证明历史具体导出命令或writer规则。

属性为何复用、真正的own-op参数归属和loaded native算术仍保留UNKNOWN。没有断言exporter bug或“必然复制前一API属性”。如后续某个候选必须依赖属性值，下一步仅允许沿已确定的MsprofEntrance.main点查；若不影响候选定性，则保留未知并推进其他profile问题，避免扩展成整个SDK源码审计。v22的alpha默认1、pre-add缩放位置、Tensor分支、行域与BF16边界来自独立的固定源码/shape契约，不受这批原始属性文本替代。

初次定位因容器无rg停止，已保留失败记录并使用现有grep；另一次复核因主控写错保存字段名停止，已按原结构修正；一个本地任务路径输入错误发生在远程执行前。失败及修正均登记，没有安装软件或更改目标环境。新证据、报告、候选与模块台账已归档；仍为33模型候选、7框架候选，26整模块没有宣称闭合。Goal保持active，最后健康观测仍是旧v12post。
