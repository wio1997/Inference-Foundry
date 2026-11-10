#!/usr/bin/env python3
"""Record frame retirement evidence and move to the larger O projection consumer."""
from pathlib import Path
import json,re

ROOT=Path('audit/execution-search')
TITLE='2026-11-10 共同frame双端退役与packet净价：继续O投影大交付'
SUMMARY='介绍：同代CP共同frame沿metadata producer/consumer走到下一代覆写，并补head8 query packet成本。到了哪里：15当前source identities、11限定body行段，Python回调提交时点与device/CPU存储寿命分开；20场景1033顺序模型、两rank各12帧/144成本情景；另7当前OA/OB→SP→mHC/FFN入口源已核。结果：release-only/devicewait-only pinned复用缺边，明确acquire须早于runner持久输入覆写，CPU快照与producer/consumer完成分别保护。三目标packet旧参考616–620us/迭代，不按层重复；替换与共存成本不同。大frame/head候选保留待确认，转O投影大段；新原语准入0，全链路未扫全，每替换原语逐位一致。'
NEXT='loop041 Run run-20261110T070506Z：实际OA TransposeBatchMatMul、所选OB BF16的TorchNPU/CANN tile/fixpipe/舍入→SP collective→mHC/FFN双读者；保留大prefill原M/全部权重集与rounded残差，按匹配旧数据价整段。7源入口/局部调用链不当native准入；不重测已停M3 OA/小逆RoPE打包。只读源码/已有数据，无新NPU/框架进口/profile/服务动作。'
BODY=SUMMARY+'\n\n待确认：common frame的native事件代次/alias/逐原语位与净价，head全局packet/压缩mask/graph资格；较大OA/OB→SP→mHC/FFN原生几何与净价。已确认可实现：本轮0。已停止：release-only输入复用、devicewait-only pinned源复用，以及挪走commonwriter但保留eager依赖读者的原样方案；不停止完整owner/fence或更大parent。\n\n[真实覆写/双端退役与成本](evidence/共同frame的真实覆写与双端退役判断.md) · [顺序模型](evidence/metadata-frame-order-cpu.json) · [packet价格](evidence/global-query-packet-price.json) · [下一段入口](evidence/o-projection-native-entry-review.json)。\n\n下一条待扫：'+NEXT+'\n'
def insert(path,body):
    p=Path(path);s=p.read_text();h='## '+TITLE
    if h in s:return
    i=s.find('\n')+1;p.write_text(s[:i]+'\n'+h+'\n\n'+body+'\n'+s[i:])
def main():
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nloop040 PIVOTED仅有边界review；loop041入口Run070221Z PASS/正确性NA，新Run070506Z仍PLANNED。171本轮hostname核对成功，同owned image/id，只读新源，未健康探测。其它流程详细性能/恢复回执不改。公开同步回执另记。')
    for m in ['M00','M01','M03','M04','M05','M10','M11']:insert('audit/modules/'+m+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('))
    insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT)
    insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    insert('ENVIRONMENT.md','适用冻结DeepSeek-V4.1-Flash-1017 W8A8/CANN9.2/910C。171统一入口hostname核对成功；同owned容器3be44e7b5c29/image ce42d65b，current磁盘source identities15（4份新取body均SHA记录）、另7O投影入口全match。本轮仅python -B -I -S只读source和本机stdlib旧数据/有限DAG，未进口Torch/NPU/模型；新NPU/请求/profile/显式远端写入/服务动作0。running身份不作health证明；保留其它流程账单/恢复历史回执。')
    p=ROOT/'source-frontier.json';d=json.loads(p.read_text());d.update(status='ACTIVE_LOOP041_O_PROJECTION_NATIVE_TO_SP_HC_FFN',region='Common frame overwrite/retirement bounded; larger O projection through SP and rounded HC/FFN consumers',next_action=NEXT,completeness_established=False,common_frame_lifetime_review=json.loads((ROOT/'common-frame-lifetime-frontier.json').read_text()),global_query_packet_price=dict(path='global-query-packet-price.json',reference_target_packet_us_per_iteration=[615.94,619.52575],scenarios=144,not_native_prediction=True),o_projection_native_entry=json.loads((ROOT/'o-projection-native-entry-review.json').read_text()));p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,frame='audit/execution-search/common-frame-lifetime-frontier.json',next_entry='audit/execution-search/o-projection-native-entry-review.json',whole_chain_complete=False,new_primitive_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
        p=Path(name);s=p.read_text();banner='<aside id="frame-o-projection-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>当前覆盖：全链路未扫全 · 每个替换原语逐位一致</strong><p>'+SUMMARY+'</p><p>'+NEXT+'</p><p><a href="audit/execution-search/共同frame的真实覆写与双端退役判断.md">frame与packet判断</a> · <a href="HANDOFF.md">当前断点</a></p></aside>'
        s,n=re.subn(r'<aside id="(?:head-common-metadata-followup|frame-o-projection-followup)".*?</aside>',lambda m:banner,s,count=1,flags=re.S);assert n==1;p.write_text(s)
if __name__=='__main__':main()
