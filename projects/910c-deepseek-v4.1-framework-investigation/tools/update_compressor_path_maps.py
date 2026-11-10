#!/usr/bin/env python3
"""Preserve foreign worktree bytes while updating this source delivery frontier."""
from pathlib import Path
from hashlib import sha256
import json,re
from review_compressor_path import R,REPORT,TITLE,SUMMARY,DETAIL,NEXT
LINKS='[本段判断](evidence/'+REPORT+') · [实际源与分支](evidence/compressor-path-review.json) · [逐边派发](evidence/compressor-path-edge-dispositions.json) · [15原Python/有限控制](evidence/compressor-path-controls.json) · [完整分支与付成本价](evidence/compressor-path-price.json)。'
BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:aux-context-default-entry|compressor-path-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
    ident='compressor-path-default-entry' if default else 'compressor-path-followup'
    return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前：C1/C2投影→完整ring与索引K/scale、长KV→QLI/SMLA</strong><p>16当前源码身份，30具名body/range、11条receiver路由、15原Python/有限控制；96完整源交付已匹配。C2绕过forward；原FP32 ring和BF16零行/舍入仍保留。同旧预算独立后处理、pool/ring/compact小方案停止，大输入/query/SWA/QLI/SMLA父段保留。原122/34未变，每替换原语逐位一致，native新增0，全链路未闭合。下一段Engram embedding/UVA交付。</p><a href="'+link+'">到了哪里、价格与具体缺口</a></aside>'

def frontier():
    d=json.loads((R/'compressor-path-review.json').read_text());return dict(status=d['status'],current_source_pins=16,source_walks=30,receiver_routes=11,matched_original_edges=15,original_or_finite_controls=15,original_bound_inputs=122,original_unbound_inputs=34,classification=d['classification'],whole_chain_complete=False,whole_recursive_callgraph_closed=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,next_action=NEXT,evidence=[REPORT,'compressor-path-review.json','compressor-path-edge-dispositions.json','compressor-path-controls.json','compressor-path-price.json'])
def insert(path,body):
    p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def main():
    names=['HANDOFF.md','171-推理数据流与已查节点.html','171-框架Profile热点.html','171-融合算子全盘排查进度图.html'];(R/'compressor-path-shared-before.json').write_text(json.dumps({n:sha256(Path(n).read_bytes()).hexdigest() for n in names},ensure_ascii=False,indent=2)+'\n')
    f=frontier();(R/'compressor-path-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n');new=Path('/tmp/compressor-path-next-run.json').read_text().strip().split('/')[-1];footer='Run202744Z有界PASS/正确性NA，'+new+' PLANNED。公开同步回执待补；loop047活动，无live任务等待。';Path('/tmp/compressor-path-handoff-footer.txt').write_text(footer)
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\n'+footer+'\n')
    for m in ['M00','M03','M10']:insert('audit/modules/'+m+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(R/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n');insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    for p in [R/'source-frontier.json',Path('audit/first-pass.json')]:
        d=json.loads(p.read_text());d.update(status=f['status'],compressor_path_review=f,next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name,default in [('171-推理数据流与已查节点.html',False),('171-框架Profile热点.html',True),('171-融合算子全盘排查进度图.html',False)]:
        p=Path(name);old=p.read_text();pat=PATTERN if default else r'<aside id="(?:aux-context-followup|compressor-path-followup)"[^>]*>.*?</aside>'
        s,n=re.subn(pat,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S);assert n==1 and re.sub(pat,'',s,flags=re.S)==re.sub(pat,'',old,flags=re.S);p.write_text(s)
    task=Path('tasks/execution-derived-fusion-search');loop=task/'loops/loop-047'
    for p in [task/'index.json',loop/'state.json']:
        d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for p in [task/'status.md',loop/'status.md']:p.write_text(re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,p.read_text(),flags=re.M))
    for p in [loop/'journal.md',loop/'report.md']:p.write_text(p.read_text()+'\n\n## Run202744Z · C1/C2完整源交付（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run '+new+' PLANNED。\n')
    print(json.dumps(dict(current_sources=16,receiver_routes=11,controls=15,next_run=new,whole_chain_complete=False)))
if __name__=='__main__':main()
