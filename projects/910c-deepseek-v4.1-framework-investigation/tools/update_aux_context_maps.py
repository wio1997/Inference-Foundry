#!/usr/bin/env python3
"""Preserve foreign worktree bytes while updating this source delivery frontier."""
from pathlib import Path
from hashlib import sha256
import json,re
from review_aux_context import R,REPORT,TITLE,SUMMARY,DETAIL,NEXT
LINKS='[本段判断](evidence/'+REPORT+') · [真实receiver与分支](evidence/aux-context-review.json) · [逐边派发](evidence/aux-context-edge-dispositions.json) · [15原Python/有限控制](evidence/aux-context-controls.json) · [完整分支与付成本价](evidence/aux-context-price.json)。'
BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:target-output-default-entry|aux-context-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
    ident='aux-context-default-entry' if default else 'aux-context-followup'
    return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前：三aux→main/三context cache→SMLA实读；独立小方案沿原预算停止</strong><p>19当前源+3旧SHA回调、24条receiver派发、15原Python/有限布局控制。A3 main不量化；重复RoPE准备可共享但完整native分支仅0.893095%旧代表窗，付成本联合情景约0.23%，非请求收益。更大host/context与既有高值候选保留。原122/34未变，native准入0，全链路未闭合；下一条扫Compressor输入/发布消费者。</p><a href="'+link+'">实际消费者、价格与未闭合边界</a></aside>'
def frontier():
    d=json.loads((R/'aux-context-review.json').read_text());return dict(status=d['status'],current_source_pins=19,prior_exact_SHA_callback_sources=3,source_walks=39,receiver_routes=24,matched_original_edges=17,original_or_finite_controls=15,original_bound_inputs=122,original_unbound_inputs=34,classification=d['classification'],whole_chain_complete=False,whole_recursive_callgraph_closed=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,next_action=NEXT,evidence=[REPORT,'aux-context-review.json','aux-context-edge-dispositions.json','aux-context-controls.json','aux-context-price.json'])
def insert(path,body):
    p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def main():
    names=['HANDOFF.md','171-推理数据流与已查节点.html','171-框架Profile热点.html','171-融合算子全盘排查进度图.html'];(R/'aux-context-shared-before.json').write_text(json.dumps({n:sha256(Path(n).read_bytes()).hexdigest() for n in names},ensure_ascii=False,indent=2)+'\n')
    f=frontier();(R/'aux-context-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n');new=json.loads(Path('/tmp/aux-context-next-run.json').read_text())['run_id'];footer='Run194757Z有界PASS/正确性NA，'+new+' PLANNED。公开同步回执待补；loop047活动，无live任务等待。';Path('/tmp/aux-context-handoff-footer.txt').write_text(footer)
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\n'+footer+'\n')
    for m in ['M00','M08','M10']:insert('audit/modules/'+m+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(R/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n');insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    for p in [R/'source-frontier.json',Path('audit/first-pass.json')]:
        d=json.loads(p.read_text());d.update(status=f['status'],aux_context_review=f,next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name,default in [('171-推理数据流与已查节点.html',False),('171-框架Profile热点.html',True),('171-融合算子全盘排查进度图.html',False)]:
        p=Path(name);old=p.read_text();pat=PATTERN if default else r'<aside id="(?:target-output-followup|aux-context-followup)"[^>]*>.*?</aside>'
        s,n=re.subn(pat,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S);assert n==1 and re.sub(pat,'',s,flags=re.S)==re.sub(pat,'',old,flags=re.S);p.write_text(s)
    task=Path('tasks/execution-derived-fusion-search');loop=task/'loops/loop-047'
    for p in [task/'index.json',loop/'state.json']:
        d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for p in [task/'status.md',loop/'status.md']:p.write_text(re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,p.read_text(),flags=re.M))
    for p in [loop/'journal.md',loop/'report.md']:p.write_text(p.read_text()+'\n\n## Run194757Z · aux与三context cache（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run '+new+' PLANNED。\n')
    print(json.dumps(dict(current_sources=19,receiver_routes=24,controls=15,next_run=new,whole_chain_complete=False)))
if __name__=='__main__':main()
