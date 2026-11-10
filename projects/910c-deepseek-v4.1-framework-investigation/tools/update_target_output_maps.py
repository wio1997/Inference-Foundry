#!/usr/bin/env python3
"""Add only this bounded output/coverage review to maps and handoff."""
from pathlib import Path
from hashlib import sha256
import json,re
from review_target_output import R,REPORT,TITLE,SUMMARY,DETAIL,NEXT

LINKS='[本段判断](evidence/'+REPORT+') · [原语全部输出与缺口](evidence/target-output-review.json) · [12原Python协议控制](evidence/target-output-controls.json) · [完整父段与付成本价格](evidence/target-output-price.json) · [递归callee工作表](evidence/recursive-source-worklist.json) · [明确走读/旧原文交集/未闭合边](evidence/recursive-source-disposition.json)。'
BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:draft-output-default-entry|target-output-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
    ident='target-output-default-entry' if default else 'target-output-followup'
    return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前：target完整score读者→CPU/RPC/Engine下一轮；递归边仍待扫</strong><p>27当前源、64具名body走读、12协议控制；bonus全score/MinTokens全局恢复/thinking状态与异步交付已分列。固定投影小方案沿原预算停止，大owner/Markov交付保留。319 body/2739 callsite仅保守静态工作表，34未定输入不是34条未扫源码；新准入0，每替换原语逐位一致，全链路未闭合。</p><a href="'+link+'">完整交付、条件净价与逐边待扫</a></aside>'
def frontier():
    d=json.loads((R/'target-output-review.json').read_text())
    return dict(status=d['status'],current_source_pins=27,named_source_body_walks=64,original_protocol_controls=12,classification=d['classification'],recursive_frontier=d['recursive_frontier'],whole_chain_complete=False,whole_recursive_callgraph_closed=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,next_action=NEXT,evidence=[REPORT,'target-output-review.json','target-output-controls.json','target-output-price.json','recursive-source-worklist.json','recursive-source-disposition.json'])
def insert(path,body):
    p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def main():
    shared=['HANDOFF.md','171-推理数据流与已查节点.html','171-框架Profile热点.html','171-融合算子全盘排查进度图.html']
    (R/'target-output-shared-before.json').write_text(json.dumps({p:sha256(Path(p).read_bytes()).hexdigest() for p in shared},ensure_ascii=False,indent=2)+'\n')
    f=frontier();(R/'target-output-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
    run=json.loads(Path('/tmp/target-output-next-run.json').read_text())['run_id'];footer='Run191838Z有界PASS/正确性NA，'+run+' PLANNED。公开同步回执待补；loop047活动，无live任务等待。'
    Path('/tmp/target-output-handoff-footer.txt').write_text(footer)
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\n'+footer+'\n')
    for name in ['M00','M08','M09','M10','M11']:insert('audit/modules/'+name+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(R/'搜索范围与当前断点.md',BODY.replace('(evidence/','('))
    insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n')
    insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    for p in [R/'source-frontier.json',Path('audit/first-pass.json')]:
        d=json.loads(p.read_text());d.update(status=f['status'],target_output_review=f,next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name,default in [('171-推理数据流与已查节点.html',False),('171-框架Profile热点.html',True),('171-融合算子全盘排查进度图.html',False)]:
        p=Path(name);old=p.read_text();pat=PATTERN if default else r'<aside id="(?:draft-output-followup|target-output-followup)"[^>]*>.*?</aside>'
        s,n=re.subn(pat,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S);assert n==1 and re.sub(pat,'',old,flags=re.S)==re.sub(pat,'',s,flags=re.S);p.write_text(s)
    task=Path('tasks/execution-derived-fusion-search');loop=task/'loops/loop-047'
    for p in [task/'index.json',loop/'state.json']:
        d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for p in [task/'status.md',loop/'status.md']:
        p.write_text(re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,p.read_text(),flags=re.M))
    for p in [loop/'journal.md',loop/'report.md']:p.write_text(p.read_text()+'\n\n## Run191838Z · target完整输出与递归前沿（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run '+run+' PLANNED。\n')
    print(json.dumps(dict(current_sources=27,controls=12,source_walk_bodies=64,next_run=run,whole_chain_complete=False)))
if __name__=='__main__':main()
