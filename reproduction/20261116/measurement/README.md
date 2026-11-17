# G131–G134当前对比测量

[源码与复现步骤](../../../docs/005-current-comparison/README.md) · [逐轮数值](rounds.json) · [主比较](comparison.json)

22阶段、528请求：G131早期128K（约418s预热→正式间隔，非主表）、G132有效32K/64K/128K、G133原256K和确认128K/256K、G134当前native完整四档；每档16warm/32formal。原始token与时间值不改，提示词/预测正文不入本批。

SG相对当前native主比较−1.42%/−9.39%/+7.92%/+8.51%，长档为固定确认；原轮并列保留。平均TTFT四档仍较高；四档目标、全模型质量和稳定领先均未验收。64K请求时间线证据仅定位后续补入延迟，不能推断当前kernel根因。

原件和脱敏件SHA见[evidence/manifest.json](evidence/manifest.json)；原始时间数组从经过manifest校验的SQLite无损展开，保留输入内容SHA。原始大包由资料保管方保存，未提供可公开下载地址，不伪造链接。

从仓根执行`python3 reproduction/20261116/measurement/evidence/recompute.py`可离线核验全部阶段和主表。机器输出的容差是官方舍入精度，不是新机重测容差。
