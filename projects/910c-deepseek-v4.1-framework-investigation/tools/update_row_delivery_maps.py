#!/usr/bin/env python3
"""Update only this review's map status; preserve other worktree bytes."""
from pathlib import Path
from hashlib import sha256
import json,re
from review_row_delivery import ROOT,REPORT,TITLE,SUMMARY,DETAIL,LINKS,NEXT

BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:model-gate-default-entry|row-delivery-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
    ident='row-delivery-default-entry' if default else 'row-delivery-followup'
    return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前：输入尾部→MC2→原生DFC；全链路未扫全</strong><p>30当前源、38原Python行控制。target旧尾部、DSpark真实query/显式尾部、SP新补零分列；独立mc2_mask已传入DFC，不能重复计padding跳过收益。继续shared/routed归约→scale/add→HC完整交付；原122约束/34动态未闭合，新原语/框架准入0，每替换原语逐位一致。</p><a href="'+link+'">真实写者、原生mask与整段剩余包络</a></aside>'
def frontier():
    r=json.loads((ROOT/'row-delivery-review.json').read_text())
    return dict(status=r['status'],classification='待确认，继续全链路',current_source_pins=30,original_Python_row_controls=38,original_static_sites=156,scoped_source_bindings=122,pending_original_sites=34,new_original_bindings=0,whole_callgraph_closed=False,whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,isolated_CPU_prototypes_passed=1,decisions=r['classification'],next_action=NEXT,evidence=[REPORT,'row-delivery-review.json','row-delivery-controls.json','row-delivery-price.json','row-delivery-reviewed-source.json'])
def insert(path,body):
    p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def main():
    shared=['HANDOFF.md','171-推理数据流与已查节点.html','171-框架Profile热点.html','171-融合算子全盘排查进度图.html']
    (ROOT/'row-delivery-shared-before.json').write_text(json.dumps({n:sha256(Path(n).read_bytes()).hexdigest() for n in shared},ensure_ascii=False,indent=2)+'\n')
    f=frontier();(ROOT/'row-delivery-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nRun174546Z有界PASS/正确性NA，Run182623Z PLANNED。公开同步回执待补；loop047活动，无live任务等待。\n')
    for n in ['M00','M01','M02','M03','M06','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('))
    insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n')
    insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    for p in [ROOT/'source-frontier.json',Path('audit/first-pass.json')]:
        d=json.loads(p.read_text());d.update(row_delivery_review=f,status=f['status'],next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name,default in [('171-推理数据流与已查节点.html',False),('171-融合算子全盘排查进度图.html',False),('171-框架Profile热点.html',True)]:
        p=Path(name);old=p.read_text();pat=PATTERN if default else r'<aside id="(?:model-gate-followup|row-delivery-followup)"[^>]*>.*?</aside>'
        s,n=re.subn(pat,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S)
        assert n==1 and re.sub(pat,'',old,flags=re.S)==re.sub(pat,'',s,flags=re.S);p.write_text(s)
    for p in [Path('tasks/execution-derived-fusion-search/index.json'),Path('tasks/execution-derived-fusion-search/loops/loop-047/state.json')]:
        d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('tasks/execution-derived-fusion-search/loops/loop-047/status.md');p.write_text(re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,p.read_text(),flags=re.M))
    p=Path('tasks/execution-derived-fusion-search/loops/loop-047/report.md');p.write_text(p.read_text()+'\n\n## Run174546Z · 输入尾部/独立MC2 mask/完整MoE输出（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run182623Z PLANNED。\n')
    print(json.dumps(dict(sources=30,controls=38,new_original_bindings=0,full_chain_complete=False)))
if __name__=='__main__': main()
